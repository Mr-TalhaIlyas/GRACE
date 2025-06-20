import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoLocator
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
# set figure plotting and saving dpi
mpl.rcParams['figure.dpi'] = 600
mpl.rcParams['savefig.dpi'] = 600

from evals.tools.evaluators import TestTimeEvaluator
from evals.tools.utils import (apply_temporal_smoothing_probs,
                                apply_temporal_smoothing_preds,
                                hysteresis_thresholding)
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score, f1_score,
    recall_score, accuracy_score, precision_score
)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

desired_order = [
        'eeg_outputs_is', 'fusion_outputs', 'ecg_outputs', 'flow_outputs', 
        'joint_pose_outputs', 'hrv_outputs_is', 'ecg_outputs_is', 
    ]

modality_config = {
    # 🏆 Tier 1: Gold Standard & Equivalent (Bold, professional)
    'eeg_outputs':     {'name': 'EEG ★', 'color': '#e41a1c'},    # Set1 red
    'fusion_outputs':  {'name': 'Fusion(▲+●+■)', 'color': '#377eb8'}, # Set1 blue

    # 🎯 Tier 2: Strong Fusion Components (Saturated, confident colors)
    'ecg_outputs':     {'name': 'CHRVS ▲', 'color': '#4daf4a'},              # Set1 green
    'flow_outputs':    {'name': 'Flow ●', 'color': '#984ea3'},       # Set1 purple
    'joint_pose_outputs': {'name': 'Pose ■', 'color': '#ff7f00'},            # Set1 orange
    'body_outputs':    {'name': 'Body', 'color': '#ffff33'},               # Set1 yellow
    'face_outputs':    {'name': 'Face', 'color': '#a65628'},               # Set1 brown
    'rhand_outputs':   {'name': 'Right Hand', 'color': '#f781bf'},         # Set1 pink
    'lhand_outputs':   {'name': 'Left Hand', 'color': '#999999'},          # Set1 gray

    # 🔸 Tier 3: Inferior In-Silo Models (Reusing colors for consistency)
    'eeg_outputs_is':  {'name': 'EEG ★', 'color': '#e41a1c'},      # Set1 red (same as EEG)
    'ecg_outputs_is':  {'name': 'ECG (In-Silo)', 'color': '#999999'},      # Set1 blue
    'hrv_outputs_is':  {'name': 'CHRVS (In-Silo)', 'color': '#4daf4a'},    # Set1 green
}

def plot_roc_curves(all_preds, all_probs, all_targets, core_metrics_auroc_CI, 
                    smoothing_window=5, config=None, filename=None):
    """
    Ultra-minimalist modern ROC plot with clean aesthetic, grid lines, and compact CI text.
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
    
    desired_order = [
        'eeg_outputs_is', 'fusion_outputs', 'ecg_outputs', 'flow_outputs', 
        'joint_pose_outputs', 'hrv_outputs_is', 'ecg_outputs_is', 
    ]
    
    ordered_keys = [key for key in desired_order if key in all_preds.keys()]
    remaining_keys = [key for key in all_preds.keys() if key not in ordered_keys]
    ordered_keys.extend(remaining_keys)

    # Ultra-clean plotting
    for idx, modality_key in enumerate(ordered_keys):
        probs_ap = all_probs[modality_key] 
        probs_ap = apply_temporal_smoothing_probs(probs_ap, smoothing_window) 
        fpr, tpr, _ = roc_curve(all_targets, probs_ap)
        roc_auc = auc(fpr, tpr)
        
        color = modality_config[modality_key]['color']
        name = modality_config[modality_key]['name']
        
        # Compact label with smaller CI text on same line
        if modality_key in core_metrics_auroc_CI:
            lower_ci, upper_ci = core_metrics_auroc_CI[modality_key]
            label = f'{name} {roc_auc:.3f} [{lower_ci:.3f}–{upper_ci:.3f}]'
        else:
            label = f'{name} {roc_auc:.3f}'

        # Clean lines with hierarchy
        linewidth = 3 if idx < 2 else 2  # Emphasis on EEG/Fusion
        alpha = 1.0 if idx < 5 else 0.7  # De-emphasize in-silo
        
        ax.plot(fpr, tpr, color=color, label=label, linewidth=linewidth, alpha=alpha)

    # Minimal diagonal
    ax.plot([0, 1], [0, 1], color='#cccccc', linestyle='--', linewidth=1, alpha=0.8)
    
    # Clean axes
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('FPR', fontsize=12, color='#333')
    ax.set_ylabel('TPR', fontsize=12, color='#333')
    
    # Add subtle grid lines
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='#cccccc')
    ax.set_axisbelow(True)  # Put grid behind the curves
    
    # Minimal styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#ddd')
    ax.spines['left'].set_color('#ddd')
    ax.tick_params(colors='#666', which='both')
    
    # Clean legend with smaller font to accommodate CI text
    ax.legend(frameon=False, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    if config:
        save_dir = config.get('output_dir', '.')
        if filename is None:
            fig.savefig(f"{save_dir}/roc_curve_minimalist.png", bbox_inches='tight')
        else:
            fig.savefig(f"{save_dir}/roc_curve_minimalist_{filename}.png", bbox_inches='tight')
    
    return fig


def plot_ap_curves(all_preds, all_probs, all_targets, config=None):
    """
    Plots Average Precision (AP) curves for each modality in `all_preds` and `all_probs`.
    
    Parameters:
        all_preds (dict): Dictionary of predictions for each modality.
        all_probs (dict): Dictionary of probabilities for each modality.
        all_targets (torch.Tensor): Ground truth labels.
    """
    fig = plt.figure(figsize=(5,5))

    for modality_key in all_preds.keys():
        probs_ap = all_probs[modality_key] 
        probs_ap = apply_temporal_smoothing_probs(probs_ap, 5) 
        precision, recall, _ = precision_recall_curve(all_targets, probs_ap)
        ap_score = average_precision_score(all_targets, probs_ap)

        plt.plot(
            recall, precision,
            color=modality_config[modality_key]['color'],
            label=f'{modality_config[modality_key]["name"]} (AP = {ap_score:.3f})',
            linewidth=2
        )

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Average Precision Curves')

    # Compact legend settings
    plt.legend(loc='lower left', fontsize='small', frameon=False)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    if config:
        save_dir = config.get('output_dir', '.')
        fig.savefig(f"{save_dir}/ap_curve.png", bbox_inches='tight')
    return plt

def plot_confusion_matrix_grid(all_preds, all_probs, all_targets, normalize='true', config=None):
    """
    Plots confusion matrices for each modality in `all_preds` and `all_probs`.
    
    Parameters:
        all_preds (dict): Dictionary of predictions for each modality.
        all_probs (dict): Dictionary of probabilities for each modality.
        all_targets (torch.Tensor): Ground truth labels.
    """
    modalities = list(all_preds.keys())
    num_modalities = len(modalities)
    
    fig, axes = plt.subplots(1, num_modalities, figsize=(num_modalities * 4, 4))
    
    for i, modality_key in enumerate(modalities):
        preds = all_preds[modality_key]
        probs = all_probs[modality_key]
        
        # Apply temporal smoothing
        probs_smoothed = apply_temporal_smoothing_probs(probs, 5)
        preds_smoothed = hysteresis_thresholding(probs_smoothed, 0.5, 0.3, only_pos_probs=True)
        
        cm = confusion_matrix(all_targets, preds_smoothed, normalize=normalize)
        
        ax = axes[i] if num_modalities > 1 else axes
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(modality_config[modality_key]['name'])
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        
        # Show color bar
        plt.colorbar(im, ax=ax)
        
        # Annotate confusion matrix
        thresh = cm.max() / 2.
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                value = cm[j, k]
                text = format(value, '.2f') if isinstance(value, float) else format(value, 'd')
                ax.text(
                        k, j, text,
                        # k, j, format(cm[j, k], 'd'),
                        ha="center", va="center",
                        color="white" if cm[j, k] > thresh else "black")
    
    plt.tight_layout()
    plt.show()
    
    if config:
        save_dir = config.get('output_dir', '.')
        fig.savefig(f"{save_dir}/cm_grid.png", bbox_inches='tight')
    return plt


def plot_model_comparison(
    ground_truth,
    model_probs_list,
    model_names=None,
    interval_sec=10,
    time_axis=None,
    time_unit="auto",
    figsize=None,
    colors=None,
    threshold=0.5,
    seizure_color="#FF6B6B",  # Softer red for seizure type 1
    seizure_alpha=0.3,
    seizure_color2="#4C9F70",  # Different color for seizure type 2 (green)
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
    Supports two seizure types with different highlight colors.
    Human annotations shown as square wave for 0 and 1 only.
    Adds right side y-axis with confidence scale and figure-level label.
    Legend placed at bottom left of figure.
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
    
    # Boolean seizure masks for two types
    gt_mask_1 = (ground_truth == 1)  # Seizure type 1
    gt_mask_2 = (ground_truth == 2)  # Seizure type 2
    
    # Create mask for human annotation square wave (0 and 1 only) - both seizure types show as 1
    gt_human_wave = np.where((ground_truth == 1) | (ground_truth == 2), 1, 0)
    
    # -----------------------------
    # Ground Truth Panel
    # -----------------------------
    ax0 = fig.add_subplot(gs[0, 0])
    
    # Plot square wave for human annotations (0 and 1 only)
    ax0.step(scaled_time, gt_human_wave, where='post', color='black', linewidth=1, alpha=0.8)
    
    # Fill seizure type 1 regions
    ax0.fill_between(
        scaled_time,
        0,
        1,
        where=gt_mask_1,
        color=seizure_color,
        alpha=seizure_alpha,
        step="post",
        edgecolor='none',
        label='Seizure Type 1'
    )
    
    # Fill seizure type 2 regions
    ax0.fill_between(
        scaled_time,
        0,
        1,
        where=gt_mask_2,
        color=seizure_color2,
        alpha=seizure_alpha,
        step="post",
        edgecolor='none',
        label='Seizure Type 2'
    )
    
    ax0.set_ylim(-0.1, 1.3)  # Increased upper limit for more white space
    ax0.set_yticks([0, 1] if yticks else [])
    ax0.set_yticklabels(["Non-Seiz", "Seizure"] if yticks else [], fontsize=9)
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
        
        # Background seizure regions for type 1
        ax.fill_between(
            scaled_time,
            0,
            1,
            where=gt_mask_1,
            color=seizure_color,
            alpha=seizure_alpha/2,
            step="post",
            zorder=0
        )
        
        # Background seizure regions for type 2
        ax.fill_between(
            scaled_time,
            0,
            1,
            where=gt_mask_2,
            color=seizure_color2,
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
            # ax.set_yticks([0, threshold, 1])
            # ax.set_yticklabels(["0", f"{threshold:.1f}", "1"], fontsize=8)
            ax.set_yticks([0, threshold])
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
    # Add secondary y-axis on right side for confidence scale
    # -----------------------------
    for ax in axes:
        secax = ax.twinx()
        secax.set_ylim(ax.get_ylim())
        secax.set_yticks([0, 0.5, 1])
        secax.set_yticklabels(['0', '0.5', '1'])
        secax.tick_params(axis='y', colors='gray', labelsize=8)
        secax.spines['right'].set_visible(True)
        secax.spines['right'].set_color('gray')
        secax.spines['right'].set_linewidth(1)
    
    # -----------------------------
    # X-Axis Formatting
    # -----------------------------
    if len(axes) > 0:
        max_ticks = 12 if total_duration/time_divisor > 60 else 8
        axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=max_ticks, min_n_ticks=6))
        
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
        
        time_unit_label = ("hours" if time_unit == "hours" else
                          "minutes" if time_unit == "minutes" else "seconds")
        axes[-1].set_xlabel(f"Time ({time_unit_label})", fontsize=10, labelpad=8)
        axes[-1].tick_params(axis='x', labelsize=9)
    
    # -----------------------------
    # Add figure-level label for right side secondary y-axis
    # -----------------------------
    fig.text(0.98, 0.5, 'Seizure Confidence', va='center', rotation=270, fontsize=12, )# fontweight='bold'
    
    # -----------------------------
    # Add legend at bottom left of figure
    # -----------------------------
    if np.any(gt_mask_1) and np.any(gt_mask_2):
        legend_elements = [
            mpatches.Rectangle((0,0),1,1, facecolor=seizure_color, alpha=seizure_alpha, label='TCS'),
            mpatches.Rectangle((0,0),1,1, facecolor=seizure_color2, alpha=seizure_alpha, label='PNES')
        ]
        fig.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(-0.03, 0.03),
                   fontsize=8, framealpha=0.8, frameon=True)
    
    # Final layout adjustments - make room for right y-axis label and bottom legend
    plt.subplots_adjust(left=0.1, right=0.94, top=0.97, bottom=0.12)
    
    if save_dir:
        if filename is not None:
            fig.savefig(f"{save_dir}/modality_comparison_{filename}.png", bbox_inches='tight')
        else:
            fig.savefig(f"{save_dir}/modality_comparison.png", bbox_inches='tight')
    return fig

