import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score
)
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from data.utils import video_transform
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModalityResults:
    """Container for individual modality evaluation results"""
    name: str
    predictions: np.ndarray
    probabilities: np.ndarray
    ground_truth: np.ndarray
    timestamps: np.ndarray
    filenames: List[str]
    
    # Computed metrics
    confusion_matrix: np.ndarray = None
    accuracy: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0
    precision: float = 0.0
    f1_score: float = 0.0
    auroc: float = 0.0
    auprc: float = 0.0

@dataclass
class FileResults:
    """Container for per-file evaluation results"""
    filename: str
    total_windows: int
    seizure_windows: int
    non_seizure_windows: int
    transition_point: int  # Window index where seizure begins
    
    # Per-modality results for this file
    modality_results: Dict[str, ModalityResults] = None

class ComprehensiveEvaluator:
    """
    Comprehensive evaluation pipeline for multi-modal seizure detection
    Designed for files with single seizure events (non-seizure → seizure)
    """
    
    def __init__(self, 
                 threshold: float = 0.5,
                 window_duration: float = 10.0,
                 smoothing_window: int = 3):
        """
        Args:
            threshold: Decision threshold for binary classification
            window_duration: Duration of each analysis window in seconds
            smoothing_window: Number of windows for temporal smoothing
        """
        self.threshold = threshold
        self.window_duration = window_duration
        self.smoothing_window = smoothing_window
        
        # Modality configurations
        self.modality_config = {
            'fusion_outputs': {'name': 'Fusion', 'color': '#1f77b4'},
            'ecg_outputs': {'name': 'ECG', 'color': '#ff7f0e'},
            'flow_outputs': {'name': 'Optical Flow', 'color': '#2ca02c'},
            'joint_pose_outputs': {'name': 'Pose (Joint)', 'color': '#d62728'},
            'body_outputs': {'name': 'Body Pose', 'color': '#9467bd'},
            'face_outputs': {'name': 'Face Pose', 'color': '#8c564b'},
            'rhand_outputs': {'name': 'Right Hand', 'color': '#e377c2'},
            'lhand_outputs': {'name': 'Left Hand', 'color': '#7f7f7f'}
        }
        
        # Results storage
        self.results = {}
        self.file_results = {}
        self.processed_results = {}  # Add this line
        
    def run_evaluation(self, 
                      model,
                      dataloader,
                      device: torch.device,
                      save_dir: str = "evaluation_results/") -> Dict:
        """
        Main evaluation pipeline
        
        Returns:
            Dictionary containing all evaluation results
        """
        print("🚀 Starting Comprehensive Multi-Modal Evaluation...")
        
        # Create save directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Step 1: Run model inference and collect predictions
        predictions_data = self._run_inference(model, dataloader, device)
        
        # Step 2: Process predictions by file and modality
        self.processed_results = self._process_predictions(predictions_data)  # Store as instance variable
        
        # Step 3: Compute metrics for each modality
        modality_metrics = self._compute_modality_metrics(self.processed_results)
        
        # Step 4: Analyze per-file performance
        file_analysis = self._analyze_per_file_performance(self.processed_results)
        
        # Step 5: Generate comprehensive visualizations
        self._generate_visualizations(modality_metrics, file_analysis, save_dir)
        
        # Step 6: Generate detailed reports
        self._generate_reports(modality_metrics, file_analysis, save_dir)
        
        results = {
            'modality_metrics': modality_metrics,
            'file_analysis': file_analysis,
            'processed_results': self.processed_results,
            'config': {
                'threshold': self.threshold,
                'window_duration': self.window_duration,
                'smoothing_window': self.smoothing_window
            }
        }
        
        print(f"✅ Evaluation complete! Results saved to {save_dir}")
        return results
    
    def _run_inference(self, model, dataloader, device) -> Dict:
        """Run model inference and collect all predictions"""
        print("🔄 Running model inference...")
        
        model.eval()
        all_predictions = defaultdict(list)
        all_ground_truth = []
        all_filenames = []
        all_timestamps = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Prepare inputs
                frames = video_transform(batch['frames']).to(device, non_blocking=True)
                body = batch['body'].permute(0,4,2,3,1).float().to(device, non_blocking=True)
                face = batch['face'].permute(0,4,2,3,1).float().to(device, non_blocking=True)
                rh = batch['rh'].permute(0,4,2,3,1).float().to(device, non_blocking=True)
                lh = batch['lh'].permute(0,4,2,3,1).float().to(device, non_blocking=True)
                hrv = batch['hrv'].to(torch.float).to(device, non_blocking=True)
                
                # Ground truth - handle both hard and soft labels
                if 'super_lbls' in batch:
                    # For soft labels, take argmax
                    gt_labels = torch.argmax(batch['super_lbls'], dim=1).cpu().numpy()
                elif 'labels' in batch:
                    # For hard labels
                    gt_labels = batch['labels'].cpu().numpy()
                else:
                    # Fallback
                    gt_labels = np.zeros(len(batch['frames']))
                
                # Model inference
                outputs = model(frames, body, face, rh, lh, hrv)
                
                # Process each modality output
                for modality_key, logits in outputs.items():
                    if modality_key in self.modality_config:
                        # Convert 2-node softmax to single seizure probability
                        if logits.shape[1] == 2:  # 2-node output [non-seizure, seizure]
                            probs = torch.softmax(logits, dim=1).cpu().numpy()
                            seizure_probs = probs[:, 1]  # Take seizure probability
                            preds = torch.argmax(logits, dim=1).cpu().numpy()
                        else:  # Single node output (sigmoid)
                            seizure_probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                            probs = np.column_stack([1-seizure_probs, seizure_probs])  # Create 2-column format
                            preds = (seizure_probs >= self.threshold).astype(int)
                        
                        all_predictions[modality_key].extend([
                            {
                                'predictions': preds[i],
                                'seizure_prob': seizure_probs[i],  # Single seizure probability for sklearn
                                'full_probabilities': probs[i],
                                'filename': batch['filename'][i],
                                'batch_idx': batch_idx,
                                'sample_idx': i
                            }
                            for i in range(len(logits))
                        ])
                
                # Store common data
                all_ground_truth.extend(gt_labels)
                all_filenames.extend(batch['filename'])
                
                # Calculate timestamps (approximation based on batch index)
                batch_timestamps = [batch_idx * len(batch['filename']) + i 
                                for i in range(len(batch['filename']))]
                all_timestamps.extend(batch_timestamps)
        
        return {
            'predictions': dict(all_predictions),
            'ground_truth': np.array(all_ground_truth),
            'filenames': all_filenames,
            'timestamps': np.array(all_timestamps)
        }

    
    def _process_predictions(self, predictions_data) -> Dict:
        """Process predictions by organizing them by file and modality"""
        print("📊 Processing predictions by file and modality...")
        
        processed = defaultdict(lambda: defaultdict(list))
        
        # Group predictions by filename and modality
        for modality_key, modality_preds in predictions_data['predictions'].items():
            for pred_info in modality_preds:
                filename = pred_info['filename']
                processed[filename][modality_key].append(pred_info)
        
        # Add ground truth information
        for i, (filename, gt) in enumerate(zip(predictions_data['filenames'], 
                                              predictions_data['ground_truth'])):
            if 'ground_truth' not in processed[filename]:
                processed[filename]['ground_truth'] = []
            processed[filename]['ground_truth'].append(gt)
        
        return dict(processed)
    
    def _apply_temporal_smoothing(self, predictions: np.ndarray, probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply temporal smoothing to predictions and probabilities
        
        Args:
            predictions: Binary predictions array
            probabilities: Probability scores array
            
        Returns:
            Tuple of (smoothed_predictions, smoothed_probabilities)
        """
        if self.smoothing_window <= 1 or len(predictions) < self.smoothing_window:
            return predictions, probabilities
        
        # Apply moving average smoothing to probabilities
        smoothed_probs = np.convolve(probabilities, 
                                    np.ones(self.smoothing_window) / self.smoothing_window, 
                                    mode='same')
        
        # Apply majority voting smoothing to predictions
        smoothed_preds = np.zeros_like(predictions)
        half_window = self.smoothing_window // 2
        
        for i in range(len(predictions)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(predictions), i + half_window + 1)
            window_preds = predictions[start_idx:end_idx]
            # Majority vote
            smoothed_preds[i] = 1 if np.sum(window_preds) > len(window_preds) / 2 else 0
        
        return smoothed_preds, smoothed_probs

    def _compute_modality_metrics(self, processed_results) -> Dict[str, ModalityResults]:
        """Compute comprehensive metrics for each modality"""
        print("📈 Computing metrics for each modality...")
        
        modality_metrics = {}
        
        for modality_key in self.modality_config.keys():
            # Collect all predictions for this modality across all files
            all_preds = []
            all_seizure_probs = []  # Single probability for sklearn
            all_full_probs = []     # Full probability array for visualization
            all_gt = []
            all_timestamps = []
            all_filenames = []
            
            for filename, file_data in processed_results.items():
                if modality_key in file_data:
                    for pred_info in file_data[modality_key]:
                        all_preds.append(pred_info['predictions'])
                        all_seizure_probs.append(pred_info['seizure_prob'])
                        all_full_probs.append(pred_info['full_probabilities'])
                        all_filenames.append(filename)
                    
                    # Add corresponding ground truth
                    all_gt.extend(file_data['ground_truth'])
                    all_timestamps.extend(range(len(file_data['ground_truth'])))
            
            if len(all_preds) == 0:
                continue
                
            # Convert to numpy arrays
            all_preds = np.array(all_preds)
            all_seizure_probs = np.array(all_seizure_probs)  # Use this for sklearn metrics
            all_full_probs = np.array(all_full_probs)
            all_gt = np.array(all_gt)
            all_timestamps = np.array(all_timestamps)
            
            # Compute metrics using single seizure probability
            cm = confusion_matrix(all_gt, all_preds)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            # AUROC and AUPRC using single seizure probability
            auroc = 0.0
            auprc = 0.0
            if len(np.unique(all_gt)) > 1:
                try:
                    fpr, tpr, _ = roc_curve(all_gt, all_seizure_probs)
                    auroc = auc(fpr, tpr)
                    auprc = average_precision_score(all_gt, all_seizure_probs)
                except Exception as e:
                    print(f"Warning: Could not compute AUROC/AUPRC for {modality_key}: {e}")
            
            # Create ModalityResults object
            modality_results = ModalityResults(
                name=self.modality_config[modality_key]['name'],
                predictions=all_preds,
                probabilities=all_full_probs,  # Store full probabilities for visualization
                ground_truth=all_gt,
                timestamps=all_timestamps,
                filenames=all_filenames,
                confusion_matrix=cm,
                accuracy=accuracy,
                sensitivity=sensitivity,
                specificity=specificity,
                precision=precision,
                f1_score=f1,
                auroc=auroc,
                auprc=auprc
            )
            
            modality_metrics[modality_key] = modality_results
        
        return modality_metrics
    

    def _analyze_per_file_performance(self, processed_results) -> Dict[str, FileResults]:
        """Analyze performance for each individual file"""
        print("📁 Analyzing per-file performance...")
        
        file_analysis = {}
        
        for filename, file_data in processed_results.items():
            if 'ground_truth' not in file_data:
                continue
                
            gt = np.array(file_data['ground_truth'])
            total_windows = len(gt)
            seizure_windows = np.sum(gt == 1)
            non_seizure_windows = np.sum(gt == 0)
            
            # Find transition point (first seizure window)
            transition_point = np.where(gt == 1)[0][0] if seizure_windows > 0 else total_windows
            
            # Process modality results for this file
            file_modality_results = {}
            for modality_key in self.modality_config.keys():
                if modality_key in file_data:
                    preds = np.array([p['predictions'] for p in file_data[modality_key]])
                    probs = np.array([p['seizure_prob'] for p in file_data[modality_key]])
                    
                    # Apply temporal smoothing
                    smoothed_preds, smoothed_probs = self._apply_temporal_smoothing(preds, probs)
                    
                    file_modality_results[modality_key] = {
                        'predictions': preds,  # Original predictions
                        'smoothed_predictions': smoothed_preds,  # Smoothed predictions
                        'seizure_probabilities': probs,  # Original probabilities
                        'smoothed_probabilities': smoothed_probs,  # Smoothed probabilities
                        'ground_truth': gt
                    }
            
            file_analysis[filename] = FileResults(
                filename=filename,
                total_windows=total_windows,
                seizure_windows=seizure_windows,
                non_seizure_windows=non_seizure_windows,
                transition_point=transition_point,
                modality_results=file_modality_results
            )
        
        return file_analysis
    
    def _generate_visualizations(self, 
                            modality_metrics: Dict[str, ModalityResults],
                            file_analysis: Dict[str, FileResults],
                            save_dir: str):
        """Generate comprehensive visualizations"""
        print("🎨 Generating comprehensive visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        # 1. Overall Performance Comparison
        self._plot_overall_performance_comparison(modality_metrics, save_dir, colors)
        
        # 2. Confusion Matrices
        self._plot_confusion_matrices(modality_metrics, save_dir)
        
        # 3. ROC Curves
        self._plot_roc_curves(modality_metrics, save_dir, colors)
        
        # 4. Precision-Recall Curves
        self._plot_precision_recall_curves(modality_metrics, save_dir, colors)
        
        # 5. Per-file Analysis
        self._plot_per_file_analysis(file_analysis, save_dir)
        
        # 6. Comprehensive Temporal Analysis for ALL files (NEW!)
        self._plot_temporal_analysis_all_files(file_analysis, save_dir)
        
        # 7. Interactive Dashboard
        self._create_interactive_dashboard(modality_metrics, file_analysis, save_dir)
    
    def _plot_overall_performance_comparison(self, 
                                           modality_metrics: Dict[str, ModalityResults],
                                           save_dir: str,
                                           colors: List[str]):
        """Plot overall performance comparison across modalities"""
        
        # Prepare data
        modalities = []
        metrics_data = {
            'Accuracy': [],
            'Sensitivity': [],
            'Specificity': [],
            'Precision': [],
            'F1-Score': [],
            'AUROC': [],
            'AUPRC': []
        }
        
        for modality_key, results in modality_metrics.items():
            modalities.append(results.name)
            metrics_data['Accuracy'].append(results.accuracy)
            metrics_data['Sensitivity'].append(results.sensitivity)
            metrics_data['Specificity'].append(results.specificity)
            metrics_data['Precision'].append(results.precision)
            metrics_data['F1-Score'].append(results.f1_score)
            metrics_data['AUROC'].append(results.auroc)
            metrics_data['AUPRC'].append(results.auprc)
        
        # Create subplot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Modal Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Bar chart of main metrics
        ax1 = axes[0, 0]
        x = np.arange(len(modalities))
        width = 0.12
        
        for i, (metric, values) in enumerate(list(metrics_data.items())[:5]):
            ax1.bar(x + i*width, values, width, label=metric, color=colors[i])
        
        ax1.set_xlabel('Modality')
        ax1.set_ylabel('Score')
        ax1.set_title('Classification Metrics Comparison')
        ax1.set_xticks(x + width*2)
        ax1.set_xticklabels(modalities, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: AUROC vs AUPRC
        ax2 = axes[0, 1]
        for i, (modality, results) in enumerate(modality_metrics.items()):
            ax2.scatter(results.auroc, results.auprc, s=100, 
                       color=colors[i], label=results.name, alpha=0.7)
        
        ax2.set_xlabel('AUROC')
        ax2.set_ylabel('AUPRC')
        ax2.set_title('AUROC vs AUPRC')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Sensitivity vs Specificity
        ax3 = axes[1, 0]
        for i, (modality, results) in enumerate(modality_metrics.items()):
            ax3.scatter(results.specificity, results.sensitivity, s=100,
                       color=colors[i], label=results.name, alpha=0.7)
        
        ax3.set_xlabel('Specificity')
        ax3.set_ylabel('Sensitivity')
        ax3.set_title('Sensitivity vs Specificity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: F1-Score ranking
        ax4 = axes[1, 1]
        f1_scores = [results.f1_score for results in modality_metrics.values()]
        modality_names = [results.name for results in modality_metrics.values()]
        
        # Sort by F1-score
        sorted_data = sorted(zip(f1_scores, modality_names), reverse=True)
        sorted_f1, sorted_names = zip(*sorted_data)
        
        bars = ax4.barh(range(len(sorted_names)), sorted_f1, color=colors[:len(sorted_names)])
        ax4.set_yticks(range(len(sorted_names)))
        ax4.set_yticklabels(sorted_names)
        ax4.set_xlabel('F1-Score')
        ax4.set_title('F1-Score Ranking')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, sorted_f1)):
            ax4.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/overall_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, 
                               modality_metrics: Dict[str, ModalityResults],
                               save_dir: str):
        """Plot confusion matrices for all modalities"""
        
        n_modalities = len(modality_metrics)
        cols = 3
        rows = (n_modalities + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Confusion Matrices by Modality', fontsize=16, fontweight='bold')
        
        for idx, (modality_key, results) in enumerate(modality_metrics.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            cm = results.confusion_matrix
            if cm is not None and cm.size > 0:
                # Normalize confusion matrix
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                # Plot
                im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
                ax.set_title(f'{results.name}\nAccuracy: {results.accuracy:.3f}')
                
                # Add text annotations
                thresh = cm_norm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2f})',
                               ha="center", va="center",
                               color="white" if cm_norm[i, j] > thresh else "black")
                
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(['Non-Seizure', 'Seizure'])
                ax.set_yticklabels(['Non-Seizure', 'Seizure'])
        
        # Hide empty subplots
        for idx in range(n_modalities, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/confusion_matrices.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, 
                        modality_metrics: Dict[str, ModalityResults],
                        save_dir: str,
                        colors: List[str]):
        """Plot ROC curves for all modalities"""
        
        plt.figure(figsize=(10, 8))
        
        for idx, (modality_key, results) in enumerate(modality_metrics.items()):
            if len(np.unique(results.ground_truth)) > 1:
                # Extract seizure probabilities from stored processed_results
                seizure_probs = []
                for filename, file_data in self.processed_results.items():
                    if modality_key in file_data:
                        for pred_info in file_data[modality_key]:
                            seizure_probs.append(pred_info['seizure_prob'])
                
                if len(seizure_probs) > 0:
                    seizure_probs = np.array(seizure_probs)
                    
                    fpr, tpr, _ = roc_curve(results.ground_truth, seizure_probs)
                    roc_auc = auc(fpr, tpr)
                    
                    plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                            label=f'{results.name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multi-Modal Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f"{save_dir}/roc_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curves(self, 
                                     modality_metrics: Dict[str, ModalityResults],
                                     save_dir: str,
                                     colors: List[str]):
        """Plot Precision-Recall curves for all modalities"""
        
        plt.figure(figsize=(10, 8))
        
        for idx, (modality_key, results) in enumerate(modality_metrics.items()):
            if len(np.unique(results.ground_truth)) > 1:
                # Extract seizure probabilities from stored processed_results
                seizure_probs = []
                for filename, file_data in self.processed_results.items():
                    if modality_key in file_data:
                        for pred_info in file_data[modality_key]:
                            seizure_probs.append(pred_info['seizure_prob'])
                
                if len(seizure_probs) > 0:
                    seizure_probs = np.array(seizure_probs)
                    
                    precision, recall, _ = precision_recall_curve(results.ground_truth, seizure_probs)
                    avg_precision = average_precision_score(results.ground_truth, seizure_probs)
                    
                    plt.plot(recall, precision, color=colors[idx % len(colors)], lw=2,
                            label=f'{results.name} (AP = {avg_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Multi-Modal Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f"{save_dir}/precision_recall_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_file_analysis(self, 
                              file_analysis: Dict[str, FileResults],
                              save_dir: str):
        """Plot per-file performance analysis"""
        
        # Prepare data for plotting
        filenames = []
        seizure_windows = []
        transition_points = []
        fusion_accuracies = []
        
        for filename, file_result in file_analysis.items():
            filenames.append(filename[:15] + '...' if len(filename) > 15 else filename)
            seizure_windows.append(file_result.seizure_windows)
            transition_points.append(file_result.transition_point / file_result.total_windows)
            
            # Calculate fusion accuracy for this file
            if 'fusion_outputs' in file_result.modality_results:
                fusion_data = file_result.modality_results['fusion_outputs']
                accuracy = np.mean(fusion_data['predictions'] == fusion_data['ground_truth'])
                fusion_accuracies.append(accuracy)
            else:
                fusion_accuracies.append(0)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Per-File Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Seizure window distribution
        ax1 = axes[0, 0]
        ax1.hist(seizure_windows, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Number of Seizure Windows')
        ax1.set_ylabel('Number of Files')
        ax1.set_title('Distribution of Seizure Windows per File')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Transition point distribution
        ax2 = axes[0, 1]
        ax2.hist(transition_points, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Seizure Onset (Fraction of File)')
        ax2.set_ylabel('Number of Files')
        ax2.set_title('Distribution of Seizure Onset Points')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Fusion accuracy per file
        ax3 = axes[1, 0]
        bars = ax3.bar(range(len(fusion_accuracies)), fusion_accuracies, 
                      color='lightgreen', alpha=0.7)
        ax3.set_xlabel('File Index')
        ax3.set_ylabel('Fusion Accuracy')
        ax3.set_title('Fusion Model Accuracy per File')
        ax3.grid(True, alpha=0.3)
        
        # Add horizontal line for average accuracy
        avg_acc = np.mean(fusion_accuracies)
        ax3.axhline(y=avg_acc, color='red', linestyle='--', 
                   label=f'Average: {avg_acc:.3f}')
        ax3.legend()
        
        # Plot 4: Seizure windows vs accuracy
        ax4 = axes[1, 1]
        ax4.scatter(seizure_windows, fusion_accuracies, alpha=0.6, s=50)
        ax4.set_xlabel('Number of Seizure Windows')
        ax4.set_ylabel('Fusion Accuracy')
        ax4.set_title('Accuracy vs Seizure Window Count')
        ax4.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = np.corrcoef(seizure_windows, fusion_accuracies)[0, 1]
        ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax4.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/per_file_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_temporal_analysis_all_files(self, 
                                        file_analysis: Dict[str, FileResults],
                                        save_dir: str):
        """Plot temporal analysis for ALL files in a signal-like format"""
        
        print("🎨 Generating comprehensive temporal analysis for all files...")
        
        # Get all filenames and sort them for consistent ordering
        all_filenames = sorted(list(file_analysis.keys()))
        
        # Calculate figure dimensions
        n_files = len(all_filenames)
        fig_height = max(20, n_files * 1.5)  # Minimum 20 inches, 1.5 inch per file
        
        # Create the figure with signal-like aspect ratio (wide x, narrow y)
        fig, axes = plt.subplots(n_files, 1, figsize=(20, fig_height))
        
        # Handle single file case
        if n_files == 1:
            axes = [axes]
        
        fig.suptitle('Comprehensive Temporal Analysis - All Test Files', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Process each file
        for idx, filename in enumerate(all_filenames):
            ax = axes[idx]
            file_result = file_analysis[filename]
            
            # Get fusion predictions and ground truth
            if 'fusion_outputs' in file_result.modality_results:
                fusion_data = file_result.modality_results['fusion_outputs']
                
                # Get data
                seizure_probs = fusion_data['seizure_probabilities']
                predictions = fusion_data['predictions']
                ground_truth = fusion_data['ground_truth']
                
                # Get smoothed data if available
                if 'smoothed_probabilities' in fusion_data:
                    smoothed_probs = fusion_data['smoothed_probabilities']
                    smoothed_preds = fusion_data['smoothed_predictions']
                else:
                    smoothed_probs = seizure_probs
                    smoothed_preds = predictions
                
                windows = np.arange(len(ground_truth))
                
                # Create time axis in seconds (assuming 10-second windows)
                time_seconds = windows * self.window_duration
                
                # Plot ground truth as background
                ax.fill_between(time_seconds, -0.1, ground_truth, alpha=0.2, color='red', 
                            label='Ground Truth', step='mid')
                
                # Plot seizure probabilities
                ax.plot(time_seconds, seizure_probs, color='blue', linewidth=1.5, 
                    label='Raw Probability', alpha=0.7)
                
                # Plot smoothed probabilities if different
                if not np.array_equal(seizure_probs, smoothed_probs):
                    ax.plot(time_seconds, smoothed_probs, color='darkblue', linewidth=2, 
                        label=f'Smoothed Prob (w={self.smoothing_window})')
                
                # Plot predictions as step function
                ax.step(time_seconds, predictions, color='green', linewidth=1.5, 
                    label='Raw Predictions', alpha=0.7, where='mid')
                
                # Plot smoothed predictions if different
                if not np.array_equal(predictions, smoothed_preds):
                    ax.step(time_seconds, smoothed_preds, color='darkgreen', linewidth=2, 
                        label=f'Smoothed Pred (w={self.smoothing_window})', where='mid')
                
                # Add threshold line
                ax.axhline(y=self.threshold, color='orange', linestyle=':', 
                        linewidth=1, label=f'Threshold ({self.threshold})')
                
                # Mark transition point
                transition_time = file_result.transition_point * self.window_duration
                ax.axvline(x=transition_time, color='red', linestyle='--',
                        alpha=0.8, linewidth=1.5, label='Seizure Onset')
                
                # Calculate metrics for both raw and smoothed predictions
                raw_accuracy = np.mean(predictions == ground_truth)
                raw_tp = np.sum((predictions == 1) & (ground_truth == 1))
                raw_fp = np.sum((predictions == 1) & (ground_truth == 0))
                raw_fn = np.sum((predictions == 0) & (ground_truth == 1))
                
                smoothed_accuracy = np.mean(smoothed_preds == ground_truth)
                smoothed_tp = np.sum((smoothed_preds == 1) & (ground_truth == 1))
                smoothed_fp = np.sum((smoothed_preds == 1) & (ground_truth == 0))
                smoothed_fn = np.sum((smoothed_preds == 0) & (ground_truth == 1))
                
                # Calculate file duration
                total_duration = len(ground_truth) * self.window_duration
                seizure_duration = np.sum(ground_truth) * self.window_duration
                
                # Create compact title with key metrics
                short_filename = filename.split('/')[-1][:30] + '...' if len(filename.split('/')[-1]) > 30 else filename.split('/')[-1]
                
                title_parts = [
                    f"{short_filename}",
                    f"Dur: {total_duration:.0f}s",
                    f"Sz: {seizure_duration:.0f}s",
                    f"Raw Acc: {raw_accuracy:.3f}",
                    f"TP/FP/FN: {raw_tp}/{raw_fp}/{raw_fn}"
                ]
                
                if not np.array_equal(predictions, smoothed_preds):
                    title_parts.extend([
                        f"Smooth Acc: {smoothed_accuracy:.3f}",
                        f"TP/FP/FN: {smoothed_tp}/{smoothed_fp}/{smoothed_fn}"
                    ])
                
                ax.set_title(" | ".join(title_parts), fontsize=9, pad=5)
                
                # Formatting for signal-like appearance
                ax.set_ylim(-0.15, 1.15)
                ax.set_xlim(0, time_seconds[-1])
                ax.grid(True, alpha=0.3, linewidth=0.5)
                
                # X-axis formatting
                if total_duration > 300:  # More than 5 minutes
                    ax.set_xlabel('Time (minutes)', fontsize=8)
                    # Convert to minutes for long recordings
                    minute_ticks = np.arange(0, time_seconds[-1] + 60, 60)
                    ax.set_xticks(minute_ticks)
                    ax.set_xticklabels([f'{t/60:.0f}' for t in minute_ticks], fontsize=8)
                else:
                    ax.set_xlabel('Time (seconds)', fontsize=8)
                    ax.tick_params(axis='x', labelsize=8)
                
                # Y-axis formatting
                ax.set_ylabel('Prob/Pred', fontsize=8)
                ax.tick_params(axis='y', labelsize=8)
                ax.set_yticks([0, 0.5, 1])
                
                # Legend only for first subplot to save space
                if idx == 0:
                    ax.legend(loc='upper right', fontsize=8, ncol=3, 
                            bbox_to_anchor=(1, 1), framealpha=0.8)
            else:
                # Handle case where fusion_outputs is not available
                ax.text(0.5, 0.5, f'No fusion data available for\n{filename}', 
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{filename} - No Data", fontsize=10)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.subplots_adjust(top=0.98, bottom=0.02, hspace=0.8)
        
        # Save with high DPI for clarity
        output_path = f"{save_dir}/temporal_analysis_all_files.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"📊 Comprehensive temporal analysis saved to {output_path}")
        
        # Also create a summary statistics plot
        self._plot_temporal_summary_statistics(file_analysis, save_dir)
    
    def _plot_temporal_summary_statistics(self, 
                                        file_analysis: Dict[str, FileResults],
                                        save_dir: str):
        """Plot summary statistics from temporal analysis"""
        
        # Collect statistics
        filenames = []
        raw_accuracies = []
        smoothed_accuracies = []
        file_durations = []
        seizure_durations = []
        transition_points = []
        improvement_scores = []
        
        for filename, file_result in file_analysis.items():
            if 'fusion_outputs' not in file_result.modality_results:
                continue
                
            fusion_data = file_result.modality_results['fusion_outputs']
            ground_truth = fusion_data['ground_truth']
            predictions = fusion_data['predictions']
            
            # Calculate metrics
            raw_acc = np.mean(predictions == ground_truth)
            
            if 'smoothed_predictions' in fusion_data:
                smoothed_preds = fusion_data['smoothed_predictions']
                smoothed_acc = np.mean(smoothed_preds == ground_truth)
            else:
                smoothed_acc = raw_acc
            
            # Store data
            filenames.append(filename.split('/')[-1][:20])
            raw_accuracies.append(raw_acc)
            smoothed_accuracies.append(smoothed_acc)
            file_durations.append(len(ground_truth) * self.window_duration)
            seizure_durations.append(np.sum(ground_truth) * self.window_duration)
            transition_points.append(file_result.transition_point / len(ground_truth))
            improvement_scores.append(smoothed_acc - raw_acc)
        
        # Create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temporal Analysis Summary Statistics', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy comparison
        ax1 = axes[0, 0]
        x_pos = np.arange(len(filenames))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, raw_accuracies, width, 
                    label='Raw Accuracy', alpha=0.7, color='skyblue')
        bars2 = ax1.bar(x_pos + width/2, smoothed_accuracies, width, 
                    label=f'Smoothed Accuracy (w={self.smoothing_window})', 
                    alpha=0.7, color='lightcoral')
        
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Raw vs Smoothed Accuracy by File')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(filenames, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Improvement distribution
        ax2 = axes[0, 1]
        ax2.hist(improvement_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Accuracy Improvement (Smoothed - Raw)')
        ax2.set_ylabel('Number of Files')
        ax2.set_title('Distribution of Smoothing Improvement')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_improvement = np.mean(improvement_scores)
        std_improvement = np.std(improvement_scores)
        improved_files = np.sum(np.array(improvement_scores) > 0)
        ax2.text(0.02, 0.98, f'Mean: {mean_improvement:.4f}\nStd: {std_improvement:.4f}\nImproved: {improved_files}/{len(improvement_scores)}',
                transform=ax2.transAxes, va='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        # Plot 3: File characteristics
        ax3 = axes[1, 0]
        scatter = ax3.scatter(file_durations, seizure_durations, 
                            c=raw_accuracies, s=50, alpha=0.7, cmap='viridis')
        ax3.set_xlabel('Total Duration (seconds)')
        ax3.set_ylabel('Seizure Duration (seconds)')
        ax3.set_title('File Characteristics vs Accuracy')
        plt.colorbar(scatter, ax=ax3, label='Raw Accuracy')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Transition point analysis
        ax4 = axes[1, 1]
        ax4.scatter(transition_points, raw_accuracies, alpha=0.6, s=50, 
                color='blue', label='Raw Accuracy')
        ax4.scatter(transition_points, smoothed_accuracies, alpha=0.6, s=50, 
                color='red', label='Smoothed Accuracy')
        ax4.set_xlabel('Seizure Onset (Fraction of File)')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Accuracy vs Seizure Onset Timing')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add correlation info
        corr_raw = np.corrcoef(transition_points, raw_accuracies)[0, 1]
        corr_smoothed = np.corrcoef(transition_points, smoothed_accuracies)[0, 1]
        ax4.text(0.02, 0.98, f'Raw Corr: {corr_raw:.3f}\nSmooth Corr: {corr_smoothed:.3f}',
                transform=ax4.transAxes, va='top', 
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/temporal_summary_statistics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Temporal summary statistics saved to {save_dir}/temporal_summary_statistics.png")
    
    def _create_interactive_dashboard(self, 
                                    modality_metrics: Dict[str, ModalityResults],
                                    file_analysis: Dict[str, FileResults],
                                    save_dir: str):
        """Create interactive Plotly dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Metrics', 'ROC Curves', 
                          'File-wise Accuracy', 'Confusion Matrix'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Performance metrics radar chart equivalent
        modalities = [results.name for results in modality_metrics.values()]
        metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score']
        
        for idx, (modality_key, results) in enumerate(modality_metrics.items()):
            values = [results.accuracy, results.sensitivity, results.specificity, 
                     results.precision, results.f1_score]
            
            fig.add_trace(
                go.Scatter(x=metrics, y=values, mode='lines+markers',
                          name=results.name, line=dict(width=3)),
                row=1, col=1
            )
        
        # Plot 2: ROC Curves
        for idx, (modality_key, results) in enumerate(modality_metrics.items()):
            if len(np.unique(results.ground_truth)) > 1:
                # Extract seizure probabilities from stored processed_results
                seizure_probs = []
                for filename, file_data in self.processed_results.items():
                    if modality_key in file_data:
                        for pred_info in file_data[modality_key]:
                            seizure_probs.append(pred_info['seizure_prob'])
                
                if len(seizure_probs) > 0:
                    seizure_probs = np.array(seizure_probs)
                    fpr, tpr, _ = roc_curve(results.ground_truth, seizure_probs)
                    
                    fig.add_trace(
                        go.Scatter(x=fpr, y=tpr, mode='lines',
                                  name=f'{results.name} ROC', line=dict(width=3)),
                        row=1, col=2
                    )
        
        # Add diagonal line for ROC
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                      line=dict(dash='dash', color='gray'),
                      name='Random', showlegend=False),
            row=1, col=2
        )
        
        # Plot 3: File-wise accuracy
        filenames = list(file_analysis.keys())[:20]  # First 20 files
        fusion_accuracies = []
        
        for filename in filenames:
            if 'fusion_outputs' in file_analysis[filename].modality_results:
                fusion_data = file_analysis[filename].modality_results['fusion_outputs']
                accuracy = np.mean(fusion_data['predictions'] == fusion_data['ground_truth'])
                fusion_accuracies.append(accuracy)
            else:
                fusion_accuracies.append(0)
        
        fig.add_trace(
            go.Bar(x=list(range(len(filenames))), y=fusion_accuracies,
                   name='File Accuracy', marker_color='lightblue'),
            row=2, col=1
        )
        
        # Plot 4: Sample confusion matrix (fusion)
        if 'fusion_outputs' in modality_metrics:
            cm = modality_metrics['fusion_outputs'].confusion_matrix
            if cm is not None:
                fig.add_trace(
                    go.Heatmap(z=cm, x=['Non-Seizure', 'Seizure'], 
                              y=['Non-Seizure', 'Seizure'],
                              colorscale='Blues', showscale=True),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="Multi-Modal Seizure Detection Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Metrics", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)
        fig.update_xaxes(title_text="File Index", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=1)
        
        # Save interactive plot
        fig.write_html(f"{save_dir}/interactive_dashboard.html")
        print(f"📊 Interactive dashboard saved to {save_dir}/interactive_dashboard.html")
    
    def _generate_reports(self, 
                         modality_metrics: Dict[str, ModalityResults],
                         file_analysis: Dict[str, FileResults],
                         save_dir: str):
        """Generate detailed text and CSV reports"""
        
        print("📝 Generating detailed reports...")
        
        # 1. Summary report
        with open(f"{save_dir}/evaluation_summary.txt", 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE MULTI-MODAL SEIZURE DETECTION EVALUATION\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Evaluation Configuration:\n")
            f.write(f"- Decision Threshold: {self.threshold}\n")
            f.write(f"- Window Duration: {self.window_duration} seconds\n")
            f.write(f"- Smoothing Window: {self.smoothing_window} windows\n\n")
            
            f.write(f"Dataset Overview:\n")
            f.write(f"- Total Files: {len(file_analysis)}\n")
            total_windows = sum(fr.total_windows for fr in file_analysis.values())
            total_seizure_windows = sum(fr.seizure_windows for fr in file_analysis.values())
            f.write(f"- Total Windows: {total_windows}\n")
            f.write(f"- Seizure Windows: {total_seizure_windows} ({total_seizure_windows/total_windows*100:.1f}%)\n\n")
            
            f.write("MODALITY PERFORMANCE SUMMARY:\n")
            f.write("-"*50 + "\n")
            
            for modality_key, results in modality_metrics.items():
                f.write(f"\n{results.name}:\n")
                f.write(f"  Accuracy:    {results.accuracy:.4f}\n")
                f.write(f"  Sensitivity: {results.sensitivity:.4f}\n")
                f.write(f"  Specificity: {results.specificity:.4f}\n")
                f.write(f"  Precision:   {results.precision:.4f}\n")
                f.write(f"  F1-Score:    {results.f1_score:.4f}\n")
                f.write(f"  AUROC:       {results.auroc:.4f}\n")
                f.write(f"  AUPRC:       {results.auprc:.4f}\n")
                
                if results.confusion_matrix is not None:
                    tn, fp, fn, tp = results.confusion_matrix.ravel()
                    f.write(f"  TP: {tp:4d}  FP: {fp:4d}\n")
                    f.write(f"  FN: {fn:4d}  TN: {tn:4d}\n")
        
        # 2. CSV report for metrics
        metrics_df = []
        for modality_key, results in modality_metrics.items():
            metrics_df.append({
                'Modality': results.name,
                'Accuracy': results.accuracy,
                'Sensitivity': results.sensitivity,
                'Specificity': results.specificity,
                'Precision': results.precision,
                'F1_Score': results.f1_score,
                'AUROC': results.auroc,
                'AUPRC': results.auprc
            })
        
        pd.DataFrame(metrics_df).to_csv(f"{save_dir}/modality_metrics.csv", index=False)
        
        # 3. Per-file results CSV
        file_df = []
        for filename, file_result in file_analysis.items():
            row = {
                'Filename': filename,
                'Total_Windows': file_result.total_windows,
                'Seizure_Windows': file_result.seizure_windows,
                'Non_Seizure_Windows': file_result.non_seizure_windows,
                'Transition_Point': file_result.transition_point,
                'Seizure_Ratio': file_result.seizure_windows / file_result.total_windows
            }
            
            # Add fusion accuracy if available
            if 'fusion_outputs' in file_result.modality_results:
                fusion_data = file_result.modality_results['fusion_outputs']
                accuracy = np.mean(fusion_data['predictions'] == fusion_data['ground_truth'])
                row['Fusion_Accuracy'] = accuracy
            
            file_df.append(row)
        
        pd.DataFrame(file_df).to_csv(f"{save_dir}/per_file_results.csv", index=False)
        
        print(f"📋 Reports saved to {save_dir}/")

def run_comprehensive_evaluation(model, dataloader, device, save_dir="evaluation_results/"):
    """
    Main function to run comprehensive evaluation
    
    Usage:
        results = run_comprehensive_evaluation(model, val_loader, device)
    """
    evaluator = ComprehensiveEvaluator(
        threshold=0.5,
        window_duration=10.0,
        smoothing_window=7
    )
    
    return evaluator.run_evaluation(model, dataloader, device, save_dir)

# # Example usage in main_eval.py
# if __name__ == "__main__":
#     # This would be called from your main_eval.py
#     # results = run_comprehensive_evaluation(model, val_loader, device)
#     pass