import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def plot_model_comparison_old(
    ground_truth,
    model_probs_list,
    model_names=None,
    interval_sec=10,
    time_axis=None,
    time_unit="seconds",
    figsize=None,
    colors=None,
    threshold=0.5,
    seizure_color="red",
    seizure_alpha=0.2,
    yticks=False,
    xticks=False,
    remove_spines=False
):
    """
    Plot ground truth and multiple model probability traces in an ECG‐style layout,
    scaling the x-axis so that each sample index represents 'interval_sec' seconds,
    and optionally removing subplot borders ("spines") for a cleaner look.

    Parameters:
    -----------
    ground_truth : array‐like of shape (N,)
        Binary ground‐truth labels (0 = non‐seizure, 1 = seizure) for each time interval.
    model_probs_list : list of array‐like, each of shape (N,)
        A list of length M, each entry containing probability outputs (floats in [0,1]) 
        from one model, aligned with ground_truth.
    model_names : list of str, optional
        Names of the M models for labeling each panel. Defaults to ["Model 1", "Model 2", …].
    interval_sec : float, default=10
        Duration in seconds represented by each index in ground_truth. The time_axis is
        computed as index * interval_sec unless a custom time_axis is provided.
    time_axis : array‐like of shape (N,), optional
        Explicit time stamps. If None, uses np.arange(N) * interval_sec.
    time_unit : str, either "seconds" or "minutes", default="seconds"
        Unit label for the x-axis. If "minutes", computed time_axis is divided by 60.
    figsize : tuple (width, height), optional
        Size of the figure in inches. If None, defaults to (12, 1.5*(M+1)).
    colors : list of color specs, optional
        List of length M specifying the line color for each model. If None, uses
        plt.get_cmap("tab10") cycling through up to 10 distinct colors.
    threshold : float, default=0.5
        Probability threshold for drawing a horizontal dashed line in each model panel.
    seizure_color : str or RGB, default="red"
        Color used to shade seizure periods (where ground_truth==1).
    seizure_alpha : float, default=0.2
        Transparency for the seizure shading in the ground‐truth row (0.2) and
        weaker (0.1) in model rows.
    yticks : bool, default=False
        Whether to show y‐axis tick labels on model panels. Usually set to False for clarity.
    xticks : bool, default=False
        Whether to show x‐axis tick labels on model panels. Usually set to False for clarity.
    remove_spines : bool, default=False
        If True, hide all subplot borders (spines) for a cleaner, minimal look.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The Figure object containing the multi‐panel ECG‐style plot.
    """
    # Validate input lengths
    N = len(ground_truth)
    M = len(model_probs_list)
    assert all(len(probs) == N for probs in model_probs_list), \
        "All model probability arrays must have the same length as ground_truth."
    
    # Compute default time axis if not provided
    if time_axis is None:
        base_time = np.arange(N) * interval_sec  # in seconds
        if time_unit == "minutes":
            time_axis = base_time / 60.0
        else:
            time_axis = base_time.copy()
    else:
        assert len(time_axis) == N, "time_axis must have length N."
        # Assume user-provided time_axis is already in correct units
    
    # Decide x-axis label based on unit
    if time_unit == "minutes":
        x_label = "Time (minutes)"
    else:
        x_label = "Time (seconds)"
    
    # Default model names
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(M)]
    else:
        assert len(model_names) == M, "model_names must be a list of length M."
    
    # Default colors using Matplotlib's tab10 palette
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(M)]
    else:
        assert len(colors) == M, "colors must be a list of length M."
    
    # Default figure size: width=12 inches, height = 1.5*(M+1) inches
    if figsize is None:
        fig_width = 12
        fig_height = 1.3 * (M + 1)
        figsize = (fig_width, fig_height)
    
    # Create figure and GridSpec layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows=M + 1, ncols=1, height_ratios=[0.5] + [1] * M, hspace=0.05)
    
    # Boolean mask for seizure periods
    gt_mask = np.array(ground_truth, dtype=bool)
    
    # -----------------------------
    # Top Panel: Ground‐Truth Plot
    # -----------------------------
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.fill_between(
        time_axis,
        0,
        1,
        where=gt_mask,
        color=seizure_color,
        alpha=seizure_alpha,
        step="post"
    )
    ax0.set_ylim(-0.1, 1.1)
    ax0.set_yticks([0, 1] if yticks else [])
    ax0.set_yticklabels(["Non‐Sz", "Sz"] if yticks else [])
    ax0.set_xticks([])
    ax0.set_ylabel("Human\nExpert", fontsize=10, rotation=0, labelpad=50, va="center")
    ax0.tick_params(axis="both", which="major", labelsize=8)
    if remove_spines:
        for spine in ax0.spines.values():
            spine.set_visible(False)
    
    # ------------------------------------------------
    # Middle Panels: One Row per Model Probability Trace
    # ------------------------------------------------
    for i in range(M):
        axi = fig.add_subplot(gs[i + 1, 0], sharex=ax0)
        
        # Plot model probability line
        axi.plot(
            time_axis,
            model_probs_list[i],
            color=colors[i],
            linewidth=1.0,
            label=model_names[i]
        )
        
        # Draw threshold line at y = threshold
        axi.axhline(
            y=threshold,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7
        )
        
        # Shade seizure regions (weaker alpha)
        axi.fill_between(
            time_axis,
            0,
            1,
            where=gt_mask,
            color=seizure_color,
            alpha=seizure_alpha / 2,
            step="post"
        )
        
        axi.set_ylim(0, 1.15)
        if not yticks:
            axi.set_yticks([])
        else:
            axi.set_yticks([0.01, threshold]) # ,1 
            axi.set_yticklabels(["Non-Sz", "Sz"]) #, None
        
        axi.set_ylabel(
            model_names[i],
            fontsize=10,
            rotation=0,
            labelpad=45,
            va="center"
        )
        axi.tick_params(axis="both", which="major", labelsize=8)
        
        # Hide x‐tick labels for all but the bottom panel
        if i < M - 1:
            plt.setp(axi.get_xticklabels(), visible=False)
        
        if remove_spines:
            for spine in axi.spines.values():
                spine.set_visible(False)
    
    # ------------------------
    # Bottom Panel: X‐Axis with Ticks
    # ------------------------
    if xticks:
        ax_last = fig.axes[-1]
        ax_last.set_xlabel(x_label, fontsize=12)
        ax_last.set_xlim(time_axis[0], time_axis[-1])
        
        # Use AutoLocator to choose "appropriate" tick intervals
        ax_last.xaxis.set_major_locator(AutoLocator())
        ax_last.tick_params(axis="x", which="major", labelsize=8)
    
    if remove_spines:
        for spine in ax_last.spines.values():
            spine.set_visible(False)
    
    # Improve layout
    plt.tight_layout()
    return fig

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



def plot_metric_with_significance(core_metrics, agreement_metrics, metric_name, 
                                 fusion_key='fusion_outputs', figsize=(12, 8), 
                                 title=None, modality_config=modality_config, use_legend=False):
    """
    Plots a bar chart of metric values with confidence intervals and significance brackets based on DeLong's test p-values.

    Args:
        core_metrics (dict): Dictionary containing metric dictionaries (e.g., core_metrics['auroc']).
        agreement_metrics (dict): Dictionary containing 'delong' results with (Z, p-value) tuples.
        metric_name (str): Name of the metric to plot (e.g., 'auroc', 'recall', 'precision').
        fusion_key (str): Key for the fusion model in the dictionaries.
        figsize (tuple): Figure size (width, height).
        title (str): Plot title. If None, uses default.
        modality_config (dict): Configuration dict with modality info.
        use_legend (bool): If True, use legend instead of x-tick labels for modality names.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    
    # Extract metrics and confidence intervals
    metrics = core_metrics[metric_name]
    metrics_CI = core_metrics.get(f'{metric_name}_CI', None)
    delong_results = agreement_metrics.get('delong', {})
    
    # Prepare data
    modalities = list(metrics.keys())
    values = [metrics[m] for m in modalities]
    
    # Get names and colors
    names = [modality_config.get(m, {'name': m})['name'] for m in modalities]
    colors = [modality_config.get(m, {'color': '#333333'})['color'] for m in modalities]
    
    # Create edge colors (darker versions)
    edge_colors = []
    for color in colors:
        if color == '#377eb8':  # Fusion color
            edge_colors.append('navy')
        else:
            edge_colors.append('black')
    
    # Prepare errors from confidence intervals
    errors_lower = []
    errors_upper = []
    if metrics_CI is not None:
        for i, m in enumerate(modalities):
            ci = metrics_CI.get(m, (values[i], values[i]))
            errors_lower.append(values[i] - ci[0])  # Distance from value to lower CI
            errors_upper.append(ci[1] - values[i])  # Distance from value to upper CI
    else:
        # No CIs available, use small default errors
        errors_lower = [0.01] * len(values)
        errors_upper = [0.01] * len(values)
    
    errors = [errors_lower, errors_upper]

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use numerical x-positions
    x_positions = range(len(modalities))
    bars = ax.bar(x_positions, values, yerr=errors, capsize=4,
                  color=colors, edgecolor=edge_colors, linewidth=1.5, alpha=0.8)

    # Highlight fusion bar
    fusion_idx = modalities.index(fusion_key)
    bars[fusion_idx].set_linewidth(2.5)
    bars[fusion_idx].set_alpha(1.0)

    def add_significance_bracket(ax, x1, x2, y, p_value, height_offset=0.02):
        """Add significance bracket with stars based on p-value"""
        if p_value < 0.001:
            sig_symbol = '***'
        elif p_value < 0.01:
            sig_symbol = '**'
        elif p_value < 0.05:
            sig_symbol = '*'
        else:
            sig_symbol = 'ns'
        
        bracket_height = y + height_offset
        # Draw bracket
        ax.plot([x1, x1, x2, x2], 
                [y + height_offset/3, bracket_height, bracket_height, y + height_offset/3], 
                'k-', linewidth=1.2)
        
        # Add significance text
        ax.text((x1 + x2) / 2, bracket_height + 0.005, sig_symbol, 
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add significance brackets based on DeLong's test results
    max_val = max([v + e for v, e in zip(values, errors_upper)])
    y_start = max_val + 0.02
    y_increment = 0.03
    
    bracket_y = y_start
    for i, m in enumerate(modalities):
        if m == fusion_key:
            continue
        
        # Get p-value from delong_results
        if m in delong_results:
            z_score, p_val = delong_results[m]
        else:
            p_val = 1.0  # Default to non-significant if missing
        
        add_significance_bracket(ax, i, fusion_idx, bracket_y, p_val)
        bracket_y += y_increment

    # Handle x-axis labels and legend based on use_legend parameter
    if use_legend:
        # Remove x-tick labels and create legend
        ax.set_xticks('off')
        ax.set_xticklabels([f'M{i+1}' for i in range(len(modalities))])  # Simple labels like M1, M2, etc.
        
        # Create legend patches
        legend_patches = []
        for i, (name, color) in enumerate(zip(names, colors)):
            # Add special marker for fusion model
            if i == fusion_idx:
                legend_patches.append(Patch(facecolor=color, edgecolor='navy', 
                                          linewidth=2, label=f'{name}*')) 
            else:
                legend_patches.append(Patch(facecolor=color, edgecolor='black', 
                                          linewidth=1, label=name))
        
        # Position legend outside the plot area
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', 
                 fontsize=10, title='Models', title_fontsize=12)
        
    else:
        # Use traditional x-tick labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(names, rotation=15, ha='right')

    # Styling
    ylabel = metric_name.upper() if metric_name.lower() in ['auroc', 'auc'] else metric_name.title()
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    
    # if title is None:
    #     title = f'Model Performance Comparison: {ylabel} with Statistical Significance\n(DeLong\'s Test)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Set y-axis limits dynamically
    y_min = min(values) * 0.9
    y_max = bracket_y + 0.05
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', alpha=0.3)

    # # Add legend for significance levels
    # legend_text = "Significance: *** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant"
    # if use_legend:
    #     # Position significance legend at the bottom when using model legend
    #     ax.text(0.5, -0.1, legend_text, ha='center', va='top', transform=ax.transAxes, 
    #             fontsize=10, style='italic')
    # else:
    #     ax.text(0.5, -0.15, legend_text, ha='center', va='top', transform=ax.transAxes, 
    #             fontsize=10, style='italic')

    # # Highlight the fusion bar with annotation
    # fusion_value = values[fusion_idx]
    # ax.annotate('Best\nPerforming', xy=(fusion_idx, fusion_value + errors_upper[fusion_idx] + 0.01), 
    #             xytext=(fusion_idx, fusion_value + errors_upper[fusion_idx] + 0.05),
    #             ha='center', va='bottom', fontweight='bold', fontsize=12,
    #             arrowprops=dict(arrowstyle='->', color='navy', lw=2))
    # plt.xticks('off')#rotation=15, ha='right')
    plt.tight_layout()
    return fig



##########################################
# LEGACY PLOT FUNCTIONS
########################################## 

def plot_model_comparison_oldv2(
    ground_truth,
    model_probs_list,
    model_names=None,
    interval_sec=10,
    time_axis=None,
    time_unit="auto",
    figsize=None,
    colors=None,
    threshold=0.5,
    seizure_color="#FF6B6B",  # Softer red
    seizure_alpha=0.3,
    yticks=False,
    remove_spines=True,
    grid=True,
    time_format="auto",
    save_dir=None,
    filename=None
):
    """
    Plot ground truth and model probabilities in a clean, scalable layout.
    Features dynamic time formatting, optimized spacing, and enhanced visuals.
    """
    # Validate input
    N = len(ground_truth)
    M = len(model_probs_list)
    assert all(len(probs) == N for probs in model_probs_list), \
        "All model probability arrays must match ground truth length."
    
    # Generate time axis if not provided
    if time_axis is None:
        base_time = np.arange(N) * interval_sec
    else:
        base_time = time_axis.copy()
        interval_sec = base_time[1] - base_time[0] if len(base_time) > 1 else 1

    # Auto-detect best time unit if requested
    total_duration = base_time[-1] - base_time[0]
    if time_unit == "auto":
        if total_duration >= 3600:
            time_unit, time_divisor = "hours", 3600
        elif total_duration >= 120:
            time_unit, time_divisor = "minutes", 60
        else:
            time_unit, time_divisor = "seconds", 1
    else:
        time_divisor = 3600 if time_unit == "hours" else 60 if time_unit == "minutes" else 1
    
    scaled_time = base_time / time_divisor
    
    # Default model names and colors
    model_names = model_names or [f"Model {i+1}" for i in range(M)]
    colors = colors or [plt.cm.tab10(i % 10) for i in range(M)]
    
    # Dynamic figure sizing
    row_height = max(0.8, 3 - 0.05 * N)  # Adjust for signal length
    if figsize is None:
        fig_width = min(14, 4 + total_duration/1800)  # Scale width with duration
        fig_height = 0.7 + M * row_height
        figsize = (fig_width, fig_height)
    
    fig = plt.figure(figsize=figsize, dpi=100)
    gs = GridSpec(M + 1, 1, height_ratios=[0.3] + [row_height] * M, hspace=0.08)
    
    # Boolean seizure mask
    gt_mask = np.array(ground_truth, dtype=bool)
    
    # -----------------------------
    # Ground Truth Panel
    # -----------------------------
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.fill_between(
        scaled_time,
        0,
        1,
        where=gt_mask,
        color=seizure_color,
        alpha=seizure_alpha,
        step="post",
        edgecolor='none'
    )
    ax0.set_ylim(-0.1, 1.1)
    ax0.set_yticks([0, 1] if yticks else [])
    ax0.set_yticklabels(["Non-Seiz", "Seiz"] if yticks else [], fontsize=9)
    ax0.set_xticks([])
    ax0.set_ylabel("Human\nExpert", fontsize=10, rotation=0, 
                  labelpad=30, va='center', ha='right')
    
    # -----------------------------
    # Model Probability Panels
    # -----------------------------
    axes = []
    for i, (probs, name, color) in enumerate(zip(model_probs_list, model_names, colors)):
        ax = fig.add_subplot(gs[i+1, 0], sharex=ax0)
        axes.append(ax)
        
        # Background seizure regions
        ax.fill_between(
            scaled_time,
            0,
            1,
            where=gt_mask,
            color=seizure_color,
            alpha=seizure_alpha/2,
            step="post",
            zorder=0
        )
        
        # Threshold line
        ax.axhline(
            threshold, 
            color='#7f7f7f', 
            linestyle=':', 
            linewidth=0.8, 
            alpha=0.9,
            zorder=1
        )
        
        # Probability trace
        ax.plot(
            scaled_time, 
            probs, 
            color=color,
            linewidth=1.0,
            alpha=0.9,
            zorder=2,
            label=name
        )
        
        # Formatting
        ax.set_ylim(-0.05, 1.05)
        if yticks:
            ax.set_yticks([0, threshold]) # ,1
            # ax.set_yticklabels([0, f"{threshold:.1f}", 1], fontsize=8)
            ax.set_yticklabels(["Non-Seiz", "Seiz"], fontsize=8)
        else:
            ax.set_yticks([])
        
        ax.set_ylabel(
            name, 
            fontsize=10,
            rotation=0, 
            labelpad=35, 
            va='center',
            ha='right'
        )
        
        # Add grid if requested
        if grid:
            ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.4, zorder=0)
        
        # Remove top/right spines
        if remove_spines:
            ax.spines[['top', 'right']].set_visible(False)
            if i < M-1:
                ax.spines['bottom'].set_visible(False)
    
    # -----------------------------
    # X-Axis Formatting
    # -----------------------------
    # Smart tick locator
    max_ticks = 12 if total_duration/time_divisor > 60 else 8
    axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=max_ticks, min_n_ticks=6))
    
    # Auto time formatting
    if time_format == "auto":
        if time_unit == "hours":
            axes[-1].xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f'{x:.1f}h' if x >= 1 else f'{x*60:.0f}min')
            )
        elif time_unit == "minutes":
            axes[-1].xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f'{x:.0f}min' if x >= 1 else f'{x*60:.0f}s')
            )
        else:
            axes[-1].xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f'{x:.0f}s')
            )
    
    # Axis labels
    time_unit_label = ("hours" if time_unit == "hours" else
                      "minutes" if time_unit == "minutes" else "seconds")
    axes[-1].set_xlabel(f"Time ({time_unit_label})", fontsize=10, labelpad=8)
    axes[-1].tick_params(axis='x', labelsize=9)
    
    # Final layout adjustments
    plt.subplots_adjust(left=0.1, right=0.98, top=0.97, bottom=0.08)
    
    if save_dir:
        if filename is not None:
            fig.savefig(f"{save_dir}/modality_comparison_{filename}.png", bbox_inches='tight')
        else:
            fig.savefig(f"{save_dir}/modality_comparison.png", bbox_inches='tight')
    return fig