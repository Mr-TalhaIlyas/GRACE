import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader

from tools.eval_sys import SeizureEvaluator, print_evaluation_results, delong_test
from data.utils import video_transform
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_inference_with_evaluation(model,
                                 dataloader: DataLoader,
                                 evaluator: SeizureEvaluator,
                                 device: torch.device,
                                 save_results: bool = True,
                                 results_dir: str = "results/") -> Dict:
    """
    Run inference and comprehensive evaluation
    
    Returns:
        Dictionary containing all evaluation results and predictions
    """
    model.eval()
    
    # Storage for predictions and metadata
    all_window_predictions = []
    all_window_ground_truth = []
    all_window_timestamps = []
    all_filenames = []
    all_window_probabilities = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
            # Prepare inputs
            frames = video_transform(batch['frames']).to(device, non_blocking=True)
            body = batch['body'].permute(0,4,2,3,1).float().to(device, non_blocking=True)
            face = batch['face'].permute(0,4,2,3,1).float().to(device, non_blocking=True)
            rh = batch['rh'].permute(0,4,2,3,1).float().to(device, non_blocking=True)
            lh = batch['lh'].permute(0,4,2,3,1).float().to(device, non_blocking=True)
            hrv = batch['hrv'].to(torch.float).to(device, non_blocking=True)
            
            # Ground truth
            gt_labels = torch.argmax(batch['super_lbls'], dim=1).cpu().numpy()
            
            # Model inference
            outputs = model(frames, body, face, rh, lh, hrv)
            fusion_logits = outputs['fusion_outputs'].cpu().numpy()
            fusion_probs = torch.softmax(outputs['fusion_outputs'], dim=1).cpu().numpy()
            
            # Store results
            for i in range(len(fusion_logits)):
                all_window_predictions.append(fusion_logits[i])
                all_window_probabilities.append(fusion_probs[i])
                all_window_ground_truth.append(gt_labels[i])
                all_filenames.append(batch['filename'][i])
                
                # Calculate timestamp (assuming sequential windows)
                timestamp = batch_idx * dataloader.batch_size * evaluator.aggregator.stride + i * evaluator.aggregator.stride
                all_window_timestamps.append(timestamp)
    
    print(f"Inference complete. Processed {len(all_window_predictions)} windows.")
    
    # Convert to numpy arrays
    window_predictions = np.array(all_window_predictions)
    window_probabilities = np.array(all_window_probabilities)
    window_ground_truth = np.array(all_window_ground_truth)
    
    print("\nEvaluating at window level...")
    # Window-level evaluation
    window_results = evaluator.evaluate_windows(window_probabilities, window_ground_truth)
    print_evaluation_results(window_results, "Window-Level Results")
    
    print("\nEvaluating at event level...")
    # Event-level evaluation with different aggregation methods
    event_results = {}
    aggregation_methods = ['average', 'majority', 'max_prob']
    
    for method in aggregation_methods:
        print(f"\nUsing aggregation method: {method}")
        event_result, per_file_results = evaluator.evaluate_events(
            [pred for pred in window_probabilities],
            [gt for gt in window_ground_truth],
            all_window_timestamps,
            all_filenames,
            aggregation_method=method,
            apply_postprocessing=True
        )
        event_results[method] = {
            'results': event_result,
            'per_file': per_file_results
        }
        print_evaluation_results(event_result, f"Event-Level Results ({method})")
        
        # Print additional event-level metrics
        summary = per_file_results['summary']
        print(f"False Positives per Hour: {summary['fp_per_hour']:.2f}")
        print(f"Total Files Evaluated: {summary['total_files']}")
        print(f"Total Duration: {summary['total_duration_hours']:.2f} hours")
    
    # Create comprehensive results dictionary
    results = {
        'window_level': window_results,
        'event_level': event_results,
        'raw_predictions': {
            'window_predictions': window_predictions,
            'window_probabilities': window_probabilities,
            'window_ground_truth': window_ground_truth,
            'filenames': all_filenames,
            'timestamps': all_window_timestamps
        }
    }
    
    # Save results if requested
    if save_results:
        save_evaluation_results(results, results_dir)
    
    return results

def save_evaluation_results(results: Dict, results_dir: str):
    """Save evaluation results to files"""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Save window-level results
    window_df = pd.DataFrame([{
        'metric': 'sensitivity', 'value': results['window_level'].sensitivity
    }, {
        'metric': 'specificity', 'value': results['window_level'].specificity
    }, {
        'metric': 'precision', 'value': results['window_level'].precision
    }, {
        'metric': 'f1_score', 'value': results['window_level'].f1_score
    }, {
        'metric': 'accuracy', 'value': results['window_level'].accuracy
    }, {
        'metric': 'auroc', 'value': results['window_level'].auroc
    }, {
        'metric': 'cohen_kappa', 'value': results['window_level'].cohen_kappa
    }])
    window_df.to_csv(results_path / 'window_level_metrics.csv', index=False)
    
    # Save event-level results
    event_data = []
    for method, data in results['event_level'].items():
        event_data.append({
            'aggregation_method': method,
            'sensitivity': data['results'].sensitivity,
            'precision': data['results'].precision,
            'f1_score': data['results'].f1_score,
            'true_positives': data['results'].true_positives,
            'false_positives': data['results'].false_positives,
            'false_negatives': data['results'].false_negatives,
            'fp_per_hour': data['per_file']['summary']['fp_per_hour']
        })
    
    event_df = pd.DataFrame(event_data)
    event_df.to_csv(results_path / 'event_level_metrics.csv', index=False)
    
    # Save raw predictions
    raw_data = results['raw_predictions']
    np.savez(results_path / 'raw_predictions.npz',
             predictions=raw_data['window_predictions'],
             probabilities=raw_data['window_probabilities'],
             ground_truth=raw_data['window_ground_truth'],
             filenames=raw_data['filenames'],
             timestamps=raw_data['timestamps'])
    
    print(f"Results saved to {results_path}")

def compare_models_statistical(results1: Dict, 
                              results2: Dict,
                              model1_name: str = "Model 1",
                              model2_name: str = "Model 2") -> Dict:
    """
    Statistical comparison between two models using DeLong test
    """
    # Extract probabilities for seizure class
    probs1 = results1['raw_predictions']['window_probabilities'][:, 1]
    probs2 = results2['raw_predictions']['window_probabilities'][:, 1]
    gt = results1['raw_predictions']['window_ground_truth']
    
    # Ensure same ground truth
    assert np.array_equal(gt, results2['raw_predictions']['window_ground_truth']), \
        "Ground truth mismatch between models"
    
    # Perform DeLong test
    z_stat, p_value = delong_test(gt, probs1, probs2)
    
    # Get AUROCs
    auroc1 = results1['window_level'].auroc
    auroc2 = results2['window_level'].auroc
    
    comparison_results = {
        'model1_name': model1_name,
        'model2_name': model2_name,
        'auroc1': auroc1,
        'auroc2': auroc2,
        'auroc_difference': auroc1 - auroc2,
        'delong_z_statistic': z_stat,
        'delong_p_value': p_value,
        'significant': p_value < 0.05
    }
    
    print(f"\nStatistical Comparison: {model1_name} vs {model2_name}")
    print("=" * 50)
    print(f"{model1_name} AUROC: {auroc1:.4f}")
    print(f"{model2_name} AUROC: {auroc2:.4f}")
    print(f"Difference: {auroc1 - auroc2:.4f}")
    print(f"DeLong Z-statistic: {z_stat:.4f}")
    print(f"DeLong p-value: {p_value:.4f}")
    print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
    
    return comparison_results

def plot_evaluation_results(results: Dict, save_path: str = None):
    """Create visualization plots for evaluation results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Window-level metrics
    window_metrics = {
        'Sensitivity': results['window_level'].sensitivity,
        'Specificity': results['window_level'].specificity,
        'Precision': results['window_level'].precision,
        'F1-Score': results['window_level'].f1_score,
        'Accuracy': results['window_level'].accuracy,
        'AUROC': results['window_level'].auroc
    }
    
    axes[0, 0].bar(window_metrics.keys(), window_metrics.values())
    axes[0, 0].set_title('Window-Level Metrics')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Event-level comparison across aggregation methods
    methods = list(results['event_level'].keys())
    f1_scores = [results['event_level'][method]['results'].f1_score for method in methods]
    fp_per_hour = [results['event_level'][method]['per_file']['summary']['fp_per_hour'] for method in methods]
    
    axes[0, 1].bar(methods, f1_scores)
    axes[0, 1].set_title('Event-Level F1-Score by Aggregation Method')
    axes[0, 1].set_ylim(0, 1)
    
    # Plot 3: False positives per hour
    axes[1, 0].bar(methods, fp_per_hour)
    axes[1, 0].set_title('False Positives per Hour by Aggregation Method')
    
    # Plot 4: Confusion matrix heatmap
    conf_matrix = np.array([
        [results['window_level'].true_negatives, results['window_level'].false_positives],
        [results['window_level'].false_negatives, results['window_level'].true_positives]
    ])
    
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Non-Seizure', 'Predicted Seizure'],
                yticklabels=['Actual Non-Seizure', 'Actual Seizure'],
                ax=axes[1, 1])
    axes[1, 1].set_title('Confusion Matrix (Window-Level)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {save_path}")
    
    plt.show()

# Example usage function
def example_evaluation_pipeline(model, val_loader, config):
    """Example of how to use the evaluation system"""
    
    # Initialize evaluator
    evaluator = SeizureEvaluator(
        window_duration=config['sample_duration'],
        overlap=config['window_overlap'],
        sampling_rate=1.0,  # 1 Hz output
        min_seizure_duration=5.0,  # 5 seconds minimum
        min_overlap_ratio=0.1  # 10% overlap for event matching
    )
    
    # Run inference and evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = run_inference_with_evaluation(
        model=model,
        dataloader=val_loader,
        evaluator=evaluator,
        device=device,
        save_results=True,
        results_dir="evaluation_results/"
    )
    
    # Create visualization
    plot_evaluation_results(results, "evaluation_results/evaluation_plots.png")
    
    return results