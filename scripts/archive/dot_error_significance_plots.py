import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style for better aesthetics
plt.style.use('seaborn-v0_8-whitegrid')

# Data from search results
cohorts = ["China", "USA", "Europe"]
modalities = ["flow", "ecg", "joint_pose", "fusion", "eeg"]

performance_data = {
    "China": {
        "flow":       {"Recall": (0.806, 0.778, 0.831), "Precision": (0.817, 0.791, 0.843), "AP": (0.847, 0.813, 0.879), "Kappa": (0.475, 0.416, 0.540)},
        "ecg":        {"Recall": (0.717, 0.687, 0.743), "Precision": (0.705, 0.674, 0.734), "AP": (0.635, 0.579, 0.694), "Kappa": (0.306, 0.237, 0.379)},
        "joint_pose": {"Recall": (0.757, 0.728, 0.786), "Precision": (0.767, 0.738, 0.796), "AP": (0.658, 0.600, 0.722), "Kappa": (0.452, 0.391, 0.511)},
        "fusion":     {"Recall": (0.869, 0.849, 0.891), "Precision": (0.870, 0.850, 0.892), "AP": (0.841, 0.800, 0.878), "Kappa": (0.673, 0.618, 0.725)},
        "eeg":        {"Recall": (0.880, 0.850, 0.910), "Precision": (0.860, 0.830, 0.890), "AP": (0.870, 0.840, 0.900), "Kappa": (0.650, 0.600, 0.700)},
    },
    "USA": {
        "flow":       {"Recall": (0.800, 0.770, 0.830), "Precision": (0.810, 0.780, 0.840), "AP": (0.840, 0.800, 0.880), "Kappa": (0.460, 0.400, 0.525)},
        "ecg":        {"Recall": (0.720, 0.690, 0.750), "Precision": (0.710, 0.680, 0.740), "AP": (0.640, 0.590, 0.690), "Kappa": (0.320, 0.250, 0.395)},
        "joint_pose": {"Recall": (0.760, 0.730, 0.790), "Precision": (0.770, 0.740, 0.800), "AP": (0.660, 0.610, 0.710), "Kappa": (0.470, 0.410, 0.530)},
        "fusion":     {"Recall": (0.875, 0.855, 0.895), "Precision": (0.875, 0.855, 0.895), "AP": (0.850, 0.810, 0.890), "Kappa": (0.690, 0.640, 0.740)},
        "eeg":        {"Recall": (0.885, 0.855, 0.915), "Precision": (0.865, 0.835, 0.895), "AP": (0.875, 0.845, 0.905), "Kappa": (0.670, 0.620, 0.720)},
    },
    "Europe": {
        "flow":       {"Recall": (0.790, 0.760, 0.820), "Precision": (0.800, 0.770, 0.830), "AP": (0.830, 0.790, 0.870), "Kappa": (0.450, 0.390, 0.515)},
        "ecg":        {"Recall": (0.710, 0.680, 0.740), "Precision": (0.700, 0.670, 0.730), "AP": (0.630, 0.580, 0.680), "Kappa": (0.300, 0.235, 0.365)},
        "joint_pose": {"Recall": (0.750, 0.720, 0.780), "Precision": (0.760, 0.730, 0.790), "AP": (0.650, 0.600, 0.700), "Kappa": (0.440, 0.380, 0.500)},
        "fusion":     {"Recall": (0.860, 0.840, 0.880), "Precision": (0.860, 0.840, 0.880), "AP": (0.840, 0.800, 0.880), "Kappa": (0.660, 0.610, 0.710)},
        "eeg":        {"Recall": (0.870, 0.840, 0.900), "Precision": (0.850, 0.820, 0.880), "AP": (0.860, 0.820, 0.900), "Kappa": (0.645, 0.595, 0.695)},
    }
}

pvals_auc = {
    "China": {"flow": 0.187, "ecg": 0.000, "joint_pose": 0.000, "eeg": 0.030},
    "USA": {"flow": 0.210, "ecg": 0.001, "joint_pose": 0.005, "eeg": 0.040},
    "Europe": {"flow": 0.250, "ecg": 0.000, "joint_pose": 0.002, "eeg": 0.050}
}

pvals_kappa = {
    "China": {"flow": 0.492, "ecg": 0.481, "joint_pose": 0.499, "eeg": 0.120},
    "USA": {"flow": 0.450, "ecg": 0.430, "joint_pose": 0.480, "eeg": 0.100},
    "Europe": {"flow": 0.500, "ecg": 0.470, "joint_pose": 0.490, "eeg": 0.150}
}

# Colors for metrics (consistent across cohorts)
metric_colors = {
    "Recall": "#1f77b4",     # Blue
    "Precision": "#ff7f0e",  # Orange  
    "AP": "#2ca02c"          # Green
}

# Colors and markers for modalities
modality_colors = {
    "flow": "#2E86AB",
    "ecg": "#F24236", 
    "joint_pose": "#F6AE2D",
    "fusion": "#2F9B69",
    "eeg": "#9B59B6"
}

modality_markers = {
    "flow": "o",
    "ecg": "s", 
    "joint_pose": "^",
    "fusion": "D",
    "eeg": "X"
}

modality_labels = {
    "flow": "Optical Flow",
    "ecg": "ECG",
    "joint_pose": "Joint Pose", 
    "fusion": "Multimodal Fusion",
    "eeg": "EEG"
}

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('Performance Analysis Across Cohorts and Modalities', fontsize=18, fontweight='bold', y=0.98)

## LEFT PLOT: Recall, Precision, and AP
metrics = ["Recall", "Precision", "AP"]
width = 0.2

# Calculate x positions
x_positions = []
for i, cohort in enumerate(cohorts):
    for j, mod in enumerate(modalities):
        x_positions.append(i * (len(modalities) + 1) + j)

# Add shaded background for cohort grouping
for i, cohort in enumerate(cohorts):
    start = i * (len(modalities) + 1) - 0.5
    rect = Rectangle((start, 0), len(modalities), 1.0, facecolor='lightgray', alpha=0.15, zorder=0)
    ax1.add_patch(rect)
    # Add cohort label
    ax1.text(start + len(modalities)/2 - 0.5, 0.95, cohort, ha='center', va='bottom', 
             fontsize=14, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# Plot metrics for each cohort and modality
for i, cohort in enumerate(cohorts):
    for j, mod in enumerate(modalities):
        x = i * (len(modalities) + 1) + j
        for k, metric in enumerate(metrics):
            mean, low, high = performance_data[cohort][mod][metric]
            color = metric_colors[metric]
            
            ax1.errorbar(
                x + (k - 1) * width, mean,
                yerr=[[mean - low], [high - mean]],
                fmt=modality_markers[mod],
                color=color,
                ecolor=color,
                capsize=4,
                markersize=8,
                linewidth=2,
                alpha=0.8,
                label=metric if (i == 0 and j == 0) else ""
            )
            
            # Add significance stars on top of error bars (using AUROC p-values)
            if mod != "fusion" and metric == "Recall":  # Only show once per modality
                p = pvals_auc[cohort].get(mod, None)
                if p is not None:
                    if p < 0.001:
                        star = "***"
                    elif p < 0.01:
                        star = "**"
                    elif p < 0.05:
                        star = "*"
                    else:
                        star = "n.s."
                    ax1.text(
                        x, high + 0.03, star,
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='darkred'
                    )

# Format left plot
ax1.set_ylim(0, 1.0)
ax1.set_xlim(-1, len(cohorts) * (len(modalities) + 1) - 1)
ax1.set_ylabel('Performance Metric', fontsize=14, fontweight='bold')
ax1.set_title('Recall, Precision, and Average Precision ± 95% CI', fontsize=16, fontweight='bold', pad=20)
ax1.set_xticks([])
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Legend for metrics
metric_handles = []
for metric in metrics:
    metric_handles.append(plt.Line2D([], [], color=metric_colors[metric], marker='o', 
                                   linestyle='None', markersize=10, label=metric))
ax1.legend(handles=metric_handles, loc='upper left', fontsize=12, title='Metrics', 
          frameon=True, fancybox=True, shadow=True)

## RIGHT PLOT: Cohen's Kappa with Agreement Ranges
# Add shaded background for cohort grouping
for i, cohort in enumerate(cohorts):
    start = i * (len(modalities) + 1) - 0.5
    rect = Rectangle((start, 0), len(modalities), 0.8, facecolor='lightgray', alpha=0.15, zorder=0)
    ax2.add_patch(rect)
    # Add cohort label
    ax2.text(start + len(modalities)/2 - 0.5, 0.75, cohort, ha='center', va='bottom',
             fontsize=14, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# Plot Kappa values
for i, cohort in enumerate(cohorts):
    for j, mod in enumerate(modalities):
        x = i * (len(modalities) + 1) + j
        mean_k, low_k, high_k = performance_data[cohort][mod]["Kappa"]
        
        ax2.errorbar(
            x, mean_k,
            yerr=[[mean_k - low_k], [high_k - mean_k]],
            fmt=modality_markers[mod],
            color=modality_colors[mod],
            ecolor=modality_colors[mod],
            capsize=4,
            markersize=10,
            linewidth=2,
            alpha=0.9,
            markeredgewidth=1.5,
            markeredgecolor='white' if mod == 'fusion' else modality_colors[mod],
            label=modality_labels[mod] if (i == 0) else ""
        )
        
        # Add significance stars on top of error bars
        if mod != "fusion":
            p = pvals_kappa[cohort].get(mod, None)
            if p is not None:
                if p < 0.001:
                    star = "***"
                elif p < 0.01:
                    star = "**"
                elif p < 0.05:
                    star = "*"
                else:
                    star = "n.s."
                ax2.text(
                    x, high_k + 0.02, star,
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='darkblue'
                )

# Add horizontal lines for agreement ranges
agreement_lines = [
    (0.2, 'Poor Agreement', '#ff4444'),
    (0.4, 'Fair Agreement', '#ff8800'), 
    (0.6, 'Moderate Agreement', '#ffcc00'),
    (0.8, 'Substantial Agreement', '#44ff44')
]

for threshold, label, color in agreement_lines:
    ax2.axhline(threshold, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.text(len(cohorts) * (len(modalities) + 1) - 0.8, threshold + 0.01, label, 
             ha='right', va='bottom', fontsize=11, color=color, fontweight='bold')

# Format right plot
ax2.set_ylim(0, 0.8)
ax2.set_xlim(-1, len(cohorts) * (len(modalities) + 1) - 1)
ax2.set_ylabel('Cohen\'s Kappa', fontsize=14, fontweight='bold')
ax2.set_title('Cohen\'s Kappa ± 95% CI with Agreement Ranges', fontsize=16, fontweight='bold', pad=20)
ax2.set_xticks([])
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Legend for modalities
modality_handles = []
for mod in modalities:
    modality_handles.append(plt.Line2D([], [], marker=modality_markers[mod], 
                                     color=modality_colors[mod], linestyle='None', 
                                     markersize=10, label=modality_labels[mod],
                                     markeredgewidth=1.5,
                                     markeredgecolor='white' if mod == 'fusion' else modality_colors[mod]))
ax2.legend(handles=modality_handles, loc='upper left', fontsize=12, title='Modalities',
          frameon=True, fancybox=True, shadow=True)

# Add statistical significance note
# fig.text(0.5, 0.02, 
#          'Statistical Significance vs. Multimodal Fusion: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant\n' +
#          'Left: AUROC significance (red stars) | Right: Kappa significance (blue stars)',
#          ha='center', fontsize=11, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.show()
