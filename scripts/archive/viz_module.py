import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class ClinicalVisualizer:
    """
    Visualization suite for clinical seizure detection evaluation
    """
    
    def __init__(self, save_dir: str = "clinical_visualizations/"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        self.modality_config = {
            'fusion_outputs': {'name': 'Fusion', 'color': '#1f77b4'},
            'ecg_outputs': {'name': 'ECG', 'color': '#ff7f0e'},
            'flow_outputs': {'name': 'Optical Flow', 'color': '#2ca02c'},
            'joint_pose_outputs': {'name': 'Pose (Joint)', 'color': '#d62728'},
        }
    
    def plot_overall_performance_comparison(self, clinical_metrics: Dict):
        """Plot comprehensive performance comparison"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Clinical Performance Comparison - OVLP Methodology', 
                    fontsize=16, fontweight='bold')
        
        # Prepare data
        modalities = []
        metrics_data = {
            'Event Sensitivity': [],
            'Event Specificity': [],
            'Event Precision': [],
            'Event F1-Score': [],
            'Pre-ictal Specificity': [],
            'FP Rate/Hour': []
        }
        
        for modality_key, metrics in clinical_metrics.items():
            modalities.append(self.modality_config[modality_key]['name'])
            metrics_data['Event Sensitivity'].append(metrics.event_sensitivity)
            metrics_data['Event Specificity'].append(metrics.event_specificity)
            metrics_data['Event Precision'].append(metrics.event_precision)
            metrics_data['Event F1-Score'].append(metrics.event_f1_score)
            metrics_data['Pre-ictal Specificity'].append(metrics.pre_ictal_specificity)
            metrics_data['FP Rate/Hour'].append(metrics.false_positive_rate_per_hour)
        
        # Plot 1: Main clinical metrics
        ax1 = axes[0, 0]
        x = np.arange(len(modalities))
        width = 0.15
        
        main_metrics = ['Event Sensitivity', 'Event Specificity', 'Event Precision', 'Event F1-Score']
        for i, metric in enumerate(main_metrics):
            ax1.bar(x + i*width, metrics_data[metric], width, 
                   label=metric, color=self.colors[i], alpha=0.8)
        
        ax1.set_xlabel('Modality')
        ax1.set_ylabel('Score')
        ax1.set_title('Main Clinical Metrics')
        ax1.set_xticks(x + width*1.5)
        ax1.set_xticklabels(modalities, rotation=45)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Plot 2: Sensitivity vs Specificity
        ax2 = axes[0, 1]
        for i, (modality_key, metrics) in enumerate(clinical_metrics.items()):
            ax2.scatter(metrics.event_specificity, metrics.event_sensitivity, 
                       s=150, color=self.modality_config[modality_key]['color'],
                       label=self.modality_config[modality_key]['name'], alpha=0.8)
        
        # Add diagonal reference line
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
        ax2.set_xlabel('Event Specificity')
        ax2.set_ylabel('Event Sensitivity')
        ax2.set_title('Clinical ROC Space')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1.05)
        ax2.set_ylim(0, 1.05)
        
        # Plot 3: Sensitivity vs FP Rate/Hour (Clinical Curve)
        ax3 = axes[0, 2]
        for i, (modality_key, metrics) in enumerate(clinical_metrics.items()):
            ax3.scatter(metrics.false_positive_rate_per_hour, metrics.event_sensitivity,
                       s=150, color=self.modality_config[modality_key]['color'],
                       label=self.modality_config[modality_key]['name'], alpha=0.8)
        
        ax3.set_xlabel('False Positives per Hour')
        ax3.set_ylabel('Event Sensitivity')
        ax3.set_title('Sensitivity vs FP Rate (Clinical)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.05)
        
        # Plot 4: Detection latency
        ax4 = axes[1, 0]
        latencies = [metrics.mean_detection_latency for metrics in clinical_metrics.values()]
        modality_names = [self.modality_config[key]['name'] for key in clinical_metrics.keys()]
        colors = [self.modality_config[key]['color'] for key in clinical_metrics.keys()]
        
        bars = ax4.bar(modality_names, latencies, color=colors, alpha=0.8)
        ax4.set_ylabel('Detection Latency (seconds)')
        ax4.set_title('Mean Detection Latency')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, latency in zip(bars, latencies):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{latency:.1f}s', ha='center', va='bottom')
        
        # Plot 5: Detailed counts breakdown
        ax5 = axes[1, 1]
        
        # Prepare stacked bar data
        tp_counts = [metrics.true_positives for metrics in clinical_metrics.values()]
        fn_counts = [metrics.false_negatives for metrics in clinical_metrics.values()]
        tn_counts = [metrics.pre_ictal_true_negatives + metrics.file_true_negatives 
                    for metrics in clinical_metrics.values()]
        fp_counts = [metrics.pre_ictal_false_positives + metrics.file_false_positives 
                    for metrics in clinical_metrics.values()]
        
        x = np.arange(len(modality_names))
        width = 0.35
        
        ax5.bar(x, tp_counts, width, label='True Positives', color='green', alpha=0.8)
        ax5.bar(x, fn_counts, width, bottom=tp_counts, label='False Negatives', color='red', alpha=0.8)
        ax5.bar(x + width, tn_counts, width, label='True Negatives', color='lightgreen', alpha=0.8)
        ax5.bar(x + width, fp_counts, width, bottom=tn_counts, label='False Positives', color='pink', alpha=0.8)
        
        ax5.set_xlabel('Modality')
        ax5.set_ylabel('Count')
        ax5.set_title('Event Outcome Breakdown')
        ax5.set_xticks(x + width/2)
        ax5.set_xticklabels(modality_names, rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Pre-ictal analysis
        ax6 = axes[1, 2]
        pre_ictal_tn = [metrics.pre_ictal_true_negatives for metrics in clinical_metrics.values()]
        pre_ictal_fp = [metrics.pre_ictal_false_positives for metrics in clinical_metrics.values()]
        pre_ictal_spec = [metrics.pre_ictal_specificity for metrics in clinical_metrics.values()]
        
        # Bar plot for pre-ictal specificity
        bars = ax6.bar(modality_names, pre_ictal_spec, color=colors, alpha=0.8)
        ax6.set_ylabel('Pre-ictal Specificity')
        ax6.set_title('Pre-ictal Period Performance\n(60s before seizure onset)')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 1.1)
        
        # Add TN/FP counts as text
        for i, (bar, tn, fp, spec) in enumerate(zip(bars, pre_ictal_tn, pre_ictal_fp, pre_ictal_spec)):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{spec:.3f}\n(TN:{tn}, FP:{fp})', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'overall_performance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Overall performance comparison saved to {self.save_dir}")
    
    def plot_roc_and_pr_curves(self, file_predictions: Dict, clinical_metrics: Dict):
        """Plot ROC and Precision-Recall curves"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC Curves
        ax1 = axes[0]
        
        for modality_key, metrics in clinical_metrics.items():
            if modality_key not in self.modality_config:
                continue
                
            # Collect probabilities and ground truth
            all_probs = []
            all_gt = []
            
            for filename, data in file_predictions.items():
                if modality_key in data['probabilities']:
                    probs = data['probabilities'][modality_key]
                    gt = data['ground_truth']
                    
                    if len(probs) > 0:
                        # Use seizure probability (class 1)
                        if len(probs.shape) > 1 and probs.shape[1] == 2:
                            seizure_probs = probs[:, 1]
                        else:
                            seizure_probs = probs
                            
                        all_probs.extend(seizure_probs)
                        all_gt.extend(gt)
            
            if len(all_probs) > 0 and len(np.unique(all_gt)) > 1:
                fpr, tpr, _ = roc_curve(all_gt, all_probs)
                roc_auc = auc(fpr, tpr)
                
                ax1.plot(fpr, tpr, linewidth=2, 
                        color=self.modality_config[modality_key]['color'],
                        label=f'{self.modality_config[modality_key]["name"]} (AUC = {roc_auc:.3f})')
        
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves - Window Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curves
        ax2 = axes[1]
        
        for modality_key, metrics in clinical_metrics.items():
            if modality_key not in self.modality_config:
                continue
                
            # Collect probabilities and ground truth (same as ROC)
            all_probs = []
            all_gt = []
            
            for filename, data in file_predictions.items():
                if modality_key in data['probabilities']:
                    probs = data['probabilities'][modality_key]
                    gt = data['ground_truth']
                    
                    if len(probs) > 0:
                        if len(probs.shape) > 1 and probs.shape[1] == 2:
                            seizure_probs = probs[:, 1]
                        else:
                            seizure_probs = probs
                            
                        all_probs.extend(seizure_probs)
                        all_gt.extend(gt)
            
            if len(all_probs) > 0 and len(np.unique(all_gt)) > 1:
                precision, recall, _ = precision_recall_curve(all_gt, all_probs)
                avg_precision = auc(recall, precision)
                
                ax2.plot(recall, precision, linewidth=2,
                        color=self.modality_config[modality_key]['color'],
                        label=f'{self.modality_config[modality_key]["name"]} (AP = {avg_precision:.3f})')
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves - Window Level')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'roc_and_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 ROC and PR curves saved to {self.save_dir}")
    
    def plot_temporal_analysis_all_files(self, file_predictions: Dict, modality_key: str = 'fusion_outputs'):
        """Plot temporal analysis for all files"""
        
        print(f"🎨 Generating temporal analysis for {modality_key}...")
        
        # Filter files that have the specified modality
        valid_files = {k: v for k, v in file_predictions.items() 
                      if modality_key in v['predictions']}
        
        if not valid_files:
            print(f"⚠️ No files found with {modality_key} data")
            return
        
        n_files = len(valid_files)
        fig_height = max(20, n_files * 1.5)
        
        fig, axes = plt.subplots(n_files, 1, figsize=(20, fig_height))
        if n_files == 1:
            axes = [axes]
        
        fig.suptitle(f'Temporal Analysis - All Files ({self.modality_config[modality_key]["name"]})', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        for idx, (filename, data) in enumerate(valid_files.items()):
            ax = axes[idx]
            
            # Get data
            predictions = data['predictions'][modality_key]
            probabilities = data['probabilities'][modality_key]
            ground_truth = data['ground_truth']
            
            # Handle probability format
            if len(probabilities.shape) > 1 and probabilities.shape[1] == 2:
                seizure_probs = probabilities[:, 1]
            else:
                seizure_probs = probabilities
            
            windows = np.arange(len(ground_truth))
            time_seconds = windows * 10  # Assuming 10-second windows
            
            # Plot ground truth as background
            ax.fill_between(time_seconds, -0.1, ground_truth, alpha=0.2, 
                          color='red', label='Ground Truth', step='mid')
            
            # Plot seizure probabilities
            ax.plot(time_seconds, seizure_probs, color='blue', linewidth=1.5, 
                   label='Seizure Probability', alpha=0.8)
            
            # Plot predictions
            ax.step(time_seconds, predictions, color='green', linewidth=1.5, 
                   label='Predictions', alpha=0.8, where='mid')
            
            # Add threshold line
            ax.axhline(y=0.5, color='orange', linestyle=':', linewidth=1, 
                      label='Threshold (0.5)')
            
            # Mark seizure onset if present
            seizure_onset_idx = np.where(ground_truth == 1)[0]
            if len(seizure_onset_idx) > 0:
                onset_time = seizure_onset_idx[0] * 10
                ax.axvline(x=onset_time, color='red', linestyle='--', 
                          alpha=0.8, linewidth=1.5, label='Seizure Onset')
                
                # Mark pre-ictal period
                pre_ictal_start = max(0, onset_time - 60)
                ax.axvspan(pre_ictal_start, onset_time, alpha=0.1, color='yellow', 
                          label='Pre-ictal (60s)')
            
            # Calculate metrics
            accuracy = np.mean(predictions == ground_truth)
            tp = np.sum((predictions == 1) & (ground_truth == 1))
            fp = np.sum((predictions == 1) & (ground_truth == 0))
            fn = np.sum((predictions == 0) & (ground_truth == 1))
            
            # Create title
            short_filename = filename.split('/')[-1][:40] + '...' if len(filename.split('/')[-1]) > 40 else filename.split('/')[-1]
            total_duration = len(ground_truth) * 10
            seizure_duration = np.sum(ground_truth) * 10
            
            title = f"{short_filename} | Duration: {total_duration}s | Seizure: {seizure_duration}s | Acc: {accuracy:.3f} | TP/FP/FN: {tp}/{fp}/{fn}"
            ax.set_title(title, fontsize=10, pad=5)
            
            # Formatting
            ax.set_ylim(-0.15, 1.15)
            ax.set_xlim(0, time_seconds[-1])
            ax.grid(True, alpha=0.3, linewidth=0.5)
            
            # Labels
            ax.set_ylabel('Prob/Pred', fontsize=9)
            if total_duration > 300:
                ax.set_xlabel('Time (minutes)', fontsize=9)
                minute_ticks = np.arange(0, time_seconds[-1] + 60, 60)
                ax.set_xticks(minute_ticks)
                ax.set_xticklabels([f'{t/60:.0f}' for t in minute_ticks], fontsize=8)
            else:
                ax.set_xlabel('Time (seconds)', fontsize=9)
                ax.tick_params(axis='x', labelsize=8)
            
            ax.tick_params(axis='y', labelsize=8)
            ax.set_yticks([0, 0.5, 1])
            
            # Legend only for first subplot
            if idx == 0:
                ax.legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.8)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.98, bottom=0.02, hspace=0.8)
        
        filename_safe = modality_key.replace('_', '-')
        output_path = self.save_dir / f'temporal_analysis_all_files_{filename_safe}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Temporal analysis saved to {output_path}")
    
    def generate_all_visualizations(self, clinical_metrics: Dict, file_predictions: Dict):
        """Generate all visualization plots"""
        
        print("🎨 Generating comprehensive clinical visualizations...")
        
        # 1. Overall performance comparison
        self.plot_overall_performance_comparison(clinical_metrics)
        
        # 2. ROC and PR curves
        self.plot_roc_and_pr_curves(file_predictions, clinical_metrics)
        
        # 3. Temporal analysis for each modality
        for modality_key in clinical_metrics.keys():
            if modality_key in self.modality_config:
                self.plot_temporal_analysis_all_files(file_predictions, modality_key)
        
        print(f"✅ All visualizations completed! Check {self.save_dir}/")

# Usage example:
def run_complete_clinical_evaluation(file_predictions: Dict, save_dir: str = "clinical_results/"):
    """Run complete clinical evaluation with separated metrics and visualization"""
    
    from .clinical_eval import calculate_all_modality_metrics
    
    # 1. Calculate metrics
    print("📊 Calculating clinical metrics...")
    clinical_metrics = calculate_all_modality_metrics(file_predictions)
    
    # 2. Generate visualizations
    print("🎨 Generating visualizations...")
    visualizer = ClinicalVisualizer(save_dir=f"{save_dir}/visualizations/")
    visualizer.generate_all_visualizations(clinical_metrics, file_predictions)
    
    # 3. Save metrics to CSV
    metrics_df = []
    for modality_key, metrics in clinical_metrics.items():
        modality_name = visualizer.modality_config[modality_key]['name']
        metrics_df.append({
            'Modality': modality_name,
            'Event_Sensitivity': metrics.event_sensitivity,
            'Event_Specificity': metrics.event_specificity,
            'Event_Precision': metrics.event_precision,
            'Event_F1_Score': metrics.event_f1_score,
            'Pre_ictal_Specificity': metrics.pre_ictal_specificity,
            'Mean_Detection_Latency': metrics.mean_detection_latency,
            'FP_Rate_per_Hour': metrics.false_positive_rate_per_hour,
            'True_Positives': metrics.true_positives,
            'False_Negatives': metrics.false_negatives,
            'Pre_ictal_TN': metrics.pre_ictal_true_negatives,
            'Pre_ictal_FP': metrics.pre_ictal_false_positives,
            'File_TN': metrics.file_true_negatives,
            'File_FP': metrics.file_false_positives,
        })
    
    df = pd.DataFrame(metrics_df)
    df.to_csv(f"{save_dir}/clinical_metrics_summary.csv", index=False)
    
    print(f"📋 Clinical metrics saved to {save_dir}/clinical_metrics_summary.csv")
    
    return clinical_metrics