import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

class ClinicalVisualizer:
    """Visualization suite for clinical seizure detection evaluation"""
    
    def __init__(self, save_dir: str = "clinical_visualizations/"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure plotting style
        plt.style.use('default')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        self.modality_config = {
            'fusion_outputs': {'name': 'Fusion', 'color': '#1f77b4'},
            'ecg_outputs': {'name': 'ECG', 'color': '#ff7f0e'},
            'flow_outputs': {'name': 'Optical Flow', 'color': '#2ca02c'},
            'joint_pose_outputs': {'name': 'Pose', 'color': '#d62728'},
        }
    
    def plot_performance_overview(self, clinical_metrics: Dict):
        """Plot comprehensive performance overview"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Clinical Performance Overview - OVLP Methodology', 
                    fontsize=16, fontweight='bold')
        
        # Prepare data
        modalities = []
        sensitivity_data = []
        specificity_data = []
        precision_data = []
        f1_data = []
        fp_rate_data = []
        latency_data = []
        
        for modality_key, metrics in clinical_metrics.items():
            if modality_key in self.modality_config:
                modalities.append(self.modality_config[modality_key]['name'])
                sensitivity_data.append(metrics.event_sensitivity)
                specificity_data.append(metrics.event_specificity)
                precision_data.append(metrics.event_precision)
                f1_data.append(metrics.event_f1_score)
                fp_rate_data.append(metrics.false_positive_rate_per_hour)
                latency_data.append(metrics.mean_detection_latency)
        
        if not modalities:
            print("⚠️ No valid modalities found for visualization")
            return
        
        colors = [self.modality_config[key]['color'] for key in clinical_metrics.keys() 
                 if key in self.modality_config]
        
        # Plot 1: Main clinical metrics
        ax1 = axes[0, 0]
        x = np.arange(len(modalities))
        width = 0.2
        
        ax1.bar(x - width*1.5, sensitivity_data, width, label='Sensitivity', alpha=0.8)
        ax1.bar(x - width*0.5, specificity_data, width, label='Specificity', alpha=0.8)
        ax1.bar(x + width*0.5, precision_data, width, label='Precision', alpha=0.8)
        ax1.bar(x + width*1.5, f1_data, width, label='F1-Score', alpha=0.8)
        
        ax1.set_xlabel('Modality')
        ax1.set_ylabel('Score')
        ax1.set_title('Main Clinical Metrics')
        ax1.set_xticks(x)
        ax1.set_xticklabels(modalities, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Plot 2: Sensitivity vs Specificity
        ax2 = axes[0, 1]
        for i, (modality_key, metrics) in enumerate(clinical_metrics.items()):
            if modality_key in self.modality_config:
                ax2.scatter(metrics.event_specificity, metrics.event_sensitivity, 
                           s=150, color=self.modality_config[modality_key]['color'],
                           label=self.modality_config[modality_key]['name'], alpha=0.8)
        
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
        ax2.set_xlabel('Event Specificity')
        ax2.set_ylabel('Event Sensitivity')
        ax2.set_title('ROC Space')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1.05)
        ax2.set_ylim(0, 1.05)
        
        # Plot 3: Sensitivity vs FP Rate/Hour
        ax3 = axes[0, 2]
        for i, (modality_key, metrics) in enumerate(clinical_metrics.items()):
            if modality_key in self.modality_config:
                ax3.scatter(metrics.false_positive_rate_per_hour, metrics.event_sensitivity,
                           s=150, color=self.modality_config[modality_key]['color'],
                           label=self.modality_config[modality_key]['name'], alpha=0.8)
        
        ax3.set_xlabel('False Positives per Hour')
        ax3.set_ylabel('Event Sensitivity')
        ax3.set_title('Sensitivity vs FP Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.05)
        
        # Plot 4: Detection latency
        ax4 = axes[1, 0]
        bars = ax4.bar(modalities, latency_data, color=colors[:len(modalities)], alpha=0.8)
        ax4.set_ylabel('Detection Latency (seconds)')
        ax4.set_title('Mean Detection Latency')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.get_xticklabels(), rotation=45)
        
        for bar, latency in zip(bars, latency_data):
            if latency > 0:  # Only show non-zero latencies
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{latency:.1f}s', ha='center', va='bottom')
        
        # Plot 5: Event outcomes
        ax5 = axes[1, 1]
        tp_counts = [metrics.true_positives for metrics in clinical_metrics.values() 
                    if hasattr(metrics, 'true_positives')]
        fn_counts = [metrics.false_negatives for metrics in clinical_metrics.values()
                    if hasattr(metrics, 'false_negatives')]
        tn_counts = [metrics.true_negatives for metrics in clinical_metrics.values()
                    if hasattr(metrics, 'true_negatives')]
        fp_counts = [metrics.false_positives for metrics in clinical_metrics.values()
                    if hasattr(metrics, 'false_positives')]
        
        if tp_counts and len(tp_counts) == len(modalities):
            x = np.arange(len(modalities))
            width = 0.35
            
            ax5.bar(x, tp_counts, width, label='TP', color='green', alpha=0.8)
            ax5.bar(x, fn_counts, width, bottom=tp_counts, label='FN', color='red', alpha=0.8)
            ax5.bar(x + width, tn_counts, width, label='TN', color='lightgreen', alpha=0.8)
            ax5.bar(x + width, fp_counts, width, bottom=tn_counts, label='FP', color='pink', alpha=0.8)
            
            ax5.set_xlabel('Modality')
            ax5.set_ylabel('Count')
            ax5.set_title('Event Outcome Counts')
            ax5.set_xticks(x + width/2)
            ax5.set_xticklabels(modalities, rotation=45)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Event counts not available', 
                    ha='center', va='center', transform=ax5.transAxes)
        
        # Plot 6: Pre-seizure analysis
        ax6 = axes[1, 2]
        pre_spec = [metrics.pre_seizure_specificity for metrics in clinical_metrics.values()]
        bars = ax6.bar(modalities, pre_spec, color=colors[:len(modalities)], alpha=0.8)
        ax6.set_ylabel('Pre-seizure Specificity')
        ax6.set_title('Pre-seizure Period Performance')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 1.1)
        plt.setp(ax6.get_xticklabels(), rotation=45)
        
        for bar, spec in zip(bars, pre_spec):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{spec:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'clinical_performance_overview.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Performance overview saved to {self.save_dir}")
    
    def plot_roc_and_pr_curves(self, file_predictions: Dict, clinical_metrics: Dict):
        """Plot ROC and Precision-Recall curves"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC Curves
        ax1 = axes[0]
        for modality_key, metrics in clinical_metrics.items():
            if modality_key not in self.modality_config:
                continue
                
            all_probs = []
            all_gt = []
            
            for filename, data in file_predictions.items():
                if (modality_key in data.get('probabilities', {}) and 
                    'ground_truth' in data):
                    all_probs.extend(data['probabilities'][modality_key])
                    all_gt.extend(data['ground_truth'])
            
            if len(all_probs) > 0 and len(np.unique(all_gt)) > 1:
                fpr, tpr, _ = roc_curve(all_gt, all_probs)
                roc_auc = auc(fpr, tpr)
                ax1.plot(fpr, tpr, color=self.modality_config[modality_key]['color'],
                        label=f'{self.modality_config[modality_key]["name"]} (AUC = {roc_auc:.3f})',
                        linewidth=2)
        
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curves
        ax2 = axes[1]
        for modality_key, metrics in clinical_metrics.items():
            if modality_key not in self.modality_config:
                continue
                
            all_probs = []
            all_gt = []
            
            for filename, data in file_predictions.items():
                if (modality_key in data.get('probabilities', {}) and 
                    'ground_truth' in data):
                    all_probs.extend(data['probabilities'][modality_key])
                    all_gt.extend(data['ground_truth'])
            
            if len(all_probs) > 0 and len(np.unique(all_gt)) > 1:
                precision, recall, _ = precision_recall_curve(all_gt, all_probs)
                pr_auc = auc(recall, precision)
                ax2.plot(recall, precision, color=self.modality_config[modality_key]['color'],
                        label=f'{self.modality_config[modality_key]["name"]} (AUC = {pr_auc:.3f})',
                        linewidth=2)
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'roc_and_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 ROC and PR curves saved to {self.save_dir}")
    
    def plot_temporal_analysis(self, file_predictions: Dict, modality_key: str = 'fusion_outputs', max_files: int = 10):
        """Plot temporal analysis for selected files"""
        
        print(f"🎨 Generating temporal analysis for {modality_key}...")
        
        valid_files = {k: v for k, v in file_predictions.items() 
                      if (modality_key in v.get('predictions', {}) and 
                          'ground_truth' in v)}
        
        if not valid_files:
            print(f"⚠️ No files found with {modality_key} data")
            return
        
        # Select subset of files for visualization
        selected_files = dict(list(valid_files.items())[:max_files])
        n_files = len(selected_files)
        
        fig, axes = plt.subplots(n_files, 1, figsize=(20, n_files * 3))
        if n_files == 1:
            axes = [axes]
        
        fig.suptitle(f'Temporal Analysis - {self.modality_config[modality_key]["name"]}', 
                    fontsize=16, fontweight='bold')
        
        for idx, (filename, data) in enumerate(selected_files.items()):
            ax = axes[idx]
            
            predictions = data['predictions'][modality_key]
            probabilities = data['probabilities'][modality_key]
            ground_truth = data['ground_truth']
            
            windows = np.arange(len(ground_truth))
            time_seconds = windows * 10  # 10-second windows
            
            # Plot ground truth as background
            ax.fill_between(time_seconds, -0.1, ground_truth, alpha=0.2, 
                          color='red', label='Ground Truth', step='mid')
            
            # Plot probabilities
            ax.plot(time_seconds, probabilities, 
                   color=self.modality_config[modality_key]['color'], 
                   linewidth=2, label='Probabilities', alpha=0.8)
            
            # Plot predictions
            ax.fill_between(time_seconds, 0, predictions * 0.5, alpha=0.4, 
                          color='blue', label='Predictions', step='mid')
            
            # Plot threshold line
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Threshold')
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Probability / Label')
            ax.set_title(f'File: {filename}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.2, 1.2)
        
        plt.tight_layout()
        filename_safe = modality_key.replace('_', '-')
        plt.savefig(self.save_dir / f'temporal_analysis_{filename_safe}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Temporal analysis saved to {self.save_dir}")
    
    def generate_summary_table(self, clinical_metrics: Dict):
        """Generate summary table as image"""
        
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        headers = ['Modality', 'Event Sens.', 'Event Spec.', 'Event Prec.', 'Event F1', 
                  'Window Sens.', 'Window Spec.', 'Window F1', 'Latency (s)', 'FP/Hour']
        
        for modality_key, metrics in clinical_metrics.items():
            if modality_key in self.modality_config:
                row = [
                    self.modality_config[modality_key]['name'],
                    f"{metrics.event_sensitivity:.3f}",
                    f"{metrics.event_specificity:.3f}",
                    f"{metrics.event_precision:.3f}",
                    f"{metrics.event_f1_score:.3f}",
                    f"{metrics.window_sensitivity:.3f}",
                    f"{metrics.window_specificity:.3f}",
                    f"{metrics.window_f1_score:.3f}",
                    f"{metrics.mean_detection_latency:.1f}",
                    f"{metrics.false_positive_rate_per_hour:.2f}",
                ]
                table_data.append(row)
        
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code performance cells
        for i in range(1, len(table_data) + 1):
            for j in range(1, 5):  # Event metrics columns
                cell_value = float(table_data[i-1][j])
                if cell_value >= 0.8:
                    table[(i, j)].set_facecolor('#C8E6C9')  # Light green
                elif cell_value >= 0.6:
                    table[(i, j)].set_facecolor('#FFF9C4')  # Light yellow
                else:
                    table[(i, j)].set_facecolor('#FFCDD2')  # Light red
        
        plt.title('Clinical Evaluation Summary', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(self.save_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📋 Summary table saved to {self.save_dir}")
    
    def generate_all_plots(self, clinical_metrics: Dict, file_predictions: Dict):
        """Generate all visualization plots and save metrics"""
        
        print("🎨 Generating comprehensive clinical visualizations...")
        
        if not clinical_metrics:
            print("⚠️ No clinical metrics available for visualization")
            return
        
        try:
            # 1. Performance overview
            self.plot_performance_overview(clinical_metrics)
            
            # 2. ROC and PR curves
            self.plot_roc_and_pr_curves(file_predictions, clinical_metrics)
            
            # 3. Temporal analysis for each modality
            for modality_key in clinical_metrics.keys():
                if modality_key in self.modality_config:
                    self.plot_temporal_analysis(file_predictions, modality_key, max_files=5)
            
            # 4. Summary table
            self.generate_summary_table(clinical_metrics)
            
            # 5. Save metrics to CSV
            self.save_metrics_to_csv(clinical_metrics)
            
            print(f"✅ All visualizations completed! Check {self.save_dir}/")
            
        except Exception as e:
            print(f"⚠️ Error generating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def save_metrics_to_csv(self, clinical_metrics: Dict):
        """Save clinical metrics to CSV file"""
        
        metrics_data = []
        for modality_key, metrics in clinical_metrics.items():
            if modality_key in self.modality_config:
                row = {
                    'Modality': self.modality_config[modality_key]['name'],
                    'Event_Sensitivity': metrics.event_sensitivity,
                    'Event_Specificity': metrics.event_specificity,
                    'Event_Precision': metrics.event_precision,
                    'Event_F1_Score': metrics.event_f1_score,
                    'Event_Accuracy': metrics.event_accuracy,
                    'Window_Sensitivity': metrics.window_sensitivity,
                    'Window_Specificity': metrics.window_specificity,
                    'Window_Precision': metrics.window_precision,
                    'Window_F1_Score': metrics.window_f1_score,
                    'Window_Accuracy': metrics.window_accuracy,
                    'Mean_Detection_Latency': metrics.mean_detection_latency,
                    'Median_Detection_Latency': metrics.median_detection_latency,
                    'Pre_seizure_Specificity': metrics.pre_seizure_specificity,
                    'FP_Rate_per_Hour': metrics.false_positive_rate_per_hour,
                    'True_Positives': metrics.true_positives,
                    'False_Positives': metrics.false_positives,
                    'False_Negatives': metrics.false_negatives,
                    'True_Negatives': metrics.true_negatives,
                    'Total_Files': metrics.total_files,
                    'Seizure_Files': metrics.seizure_files,
                    'Non_Seizure_Files': metrics.non_seizure_files,
                }
                metrics_data.append(row)
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            df.to_csv(self.save_dir / 'clinical_metrics_detailed.csv', index=False)
            print(f"📊 Detailed metrics saved to {self.save_dir}/clinical_metrics_detailed.csv")
        else:
            print("⚠️ No metrics data to save")