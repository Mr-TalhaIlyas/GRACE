import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoLocator
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, FuncFormatter
# set figure plotting and saving dpi
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 600

from tools.eval_utils import (TestTimeEvaluator,
                              apply_temporal_smoothing_probs,
                              apply_temporal_smoothing_preds,
                              hysteresis_thresholding)
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score, f1_score,
    recall_score, accuracy_score, precision_score
)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
modality_config = {
    'fusion_outputs': {'name': 'Fusion', 'color': '#1f77b4'},
    'ecg_outputs': {'name': 'ECG', 'color': '#ff7f0e'},
    'flow_outputs': {'name': 'Optical Flow', 'color': '#2ca02c'},
    'joint_pose_outputs': {'name': 'Pose', 'color': '#d62728'},
    'body_outputs': {'name': 'Body', 'color': '#9467bd'},
    'face_outputs': {'name': 'Face', 'color': '#8c564b'},
    'rhand_outputs': {'name': 'Right Hand', 'color': '#e377c2'},
    'lhand_outputs': {'name': 'Left Hand', 'color': '#7f7f7f'}
}

def plot_roc_curves(all_preds, all_probs, all_targets, config=None):
    """
    Plots ROC curves for each modality in `all_preds` and `all_probs`.
    
    Parameters:
        all_preds (dict): Dictionary of predictions for each modality.
        all_probs (dict): Dictionary of probabilities for each modality.
        all_targets (torch.Tensor): Ground truth labels.
    """
    fig = plt.figure(figsize=(5,5))

    for modality_key in all_preds.keys():
        probs_ap = all_probs[modality_key] 
        probs_ap = apply_temporal_smoothing_probs(probs_ap, 5) 
        fpr, tpr, _ = roc_curve(all_targets, probs_ap)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr, tpr,
            color=modality_config[modality_key]['color'],
            label=f'{modality_config[modality_key]["name"]} (AUC = {roc_auc:.3f})',
            linewidth=2
        )

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')

    # Compact legend settings
    plt.legend(loc='lower right', fontsize='small', frameon=False)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    if config:
        save_dir = config.get('output_dir', '.')
        fig.savefig(f"{save_dir}/roc_curve.png", bbox_inches='tight')
    return plt

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
    seizure_color="#FF6B6B",  # Softer red
    seizure_alpha=0.3,
    yticks=False,
    remove_spines=True,
    grid=True,
    time_format="auto",
    save_dir=None
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
        fig.savefig(f"{save_dir}/modality_comparison.png", bbox_inches='tight')
    return fig

def draw_temporal_seizure_plots(all_targets, probs_list, labels_list, eval_config, threshold=0.7):

    fig = plot_model_comparison(
        ground_truth=all_targets,
        model_probs_list=probs_list,
        model_names=labels_list,
        interval_sec=10,
        time_unit="auto",
        figsize=None,
        colors=None,#["tab:blue", "tab:green", "tab:orange", "tab:cyan"],
        threshold=threshold,
        seizure_color="red",
        seizure_alpha=0.2,
        yticks=True,
        grid=True,
        time_format="auto",
        remove_spines=True,
        save_dir=eval_config['output_dir'],
    )
    return fig
# r = all_probs['fusion_outputs']
# fusion = apply_temporal_smoothing_probs(r, 5)
# processed = hysteresis_thresholding(fusion, 0.6, 0.4, only_pos_probs=True)

# p = all_probs['ecg_outputs']
# ecg = apply_temporal_smoothing_probs(p, 3)

# p = all_probs['flow_outputs']
# flow = apply_temporal_smoothing_probs(p, 3)

# p = all_probs['joint_pose_outputs']
# pose = apply_temporal_smoothing_probs(p, 3)

# p=all_probs['fusion_outputs']
# probs_conf = apply_temporal_smoothing_probs(p, 5)
# test = hysteresis_thresholding(probs_conf, 0.8, 0.5, only_pos_probs=True)

# fig = plot_model_comparison(
#     ground_truth=all_targets,
#     model_probs_list=[fusion,test , ecg,flow,pose],
#     model_names=['Fusion', 'Test', 'ECG','Flow', 'Pose'],
#     interval_sec=10,
#     time_unit="auto",
#     figsize=None,
#     colors=None,#["tab:blue", "tab:green", "tab:orange", "tab:cyan"],
#     threshold=0.7,
#     seizure_color="red",
#     seizure_alpha=0.2,
#     yticks=True,
#     grid=True,
#     time_format="auto",
#     remove_spines=True,
#     save_dir=eval_config['output_dir'],
# )

################################################
# Legacy function for backward compatibility
################################################

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