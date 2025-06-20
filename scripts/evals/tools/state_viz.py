import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy import stats
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 600

desired_order = [
        'eeg_outputs_is', 'fusion_outputs', 'ecg_outputs', 'flow_outputs', 
        'joint_pose_outputs', 'hrv_outputs_is', 'ecg_outputs_is', 
    ]

modality_config = {
    # 🏆 Tier 1: Gold Standard & Equivalent (Bold, professional)
    'eeg_outputs':     {'name': 'EEG\n(★)', 'color': "#e41a1c"},    # Set1 red
    'fusion_outputs':  {'name': 'Fusion\n(▲+●+■)', 'color': '#377eb8'}, # Set1 blue

    # 🎯 Tier 2: Strong Fusion Components (Saturated, confident colors)
    'ecg_outputs':     {'name': 'CHRVS\n(▲)', 'color': '#4daf4a'},              # Set1 green
    'flow_outputs':    {'name': 'Flow\n(●)', 'color': '#984ea3'},       # Set1 purple
    'joint_pose_outputs': {'name': 'Pose\n(■)', 'color': '#ff7f00'},            # Set1 orange
    'body_outputs':    {'name': 'Body', 'color': '#ffff33'},               # Set1 yellow
    'face_outputs':    {'name': 'Face', 'color': '#a65628'},               # Set1 brown
    'rhand_outputs':   {'name': 'Right\nHand', 'color': '#f781bf'},         # Set1 pink
    'lhand_outputs':   {'name': 'Left\nHand', 'color': '#999999'},          # Set1 gray

    # 🔸 Tier 3: Inferior In-Silo Models (Reusing colors for consistency)
    'eeg_outputs_is':  {'name': 'EEG\n(★)', 'color': '#e41a1c'},      # Set1 red (same as EEG)
    'ecg_outputs_is':  {'name': 'ECG\n(In-Silo)', 'color': '#999999'},      # Set1 blue
    'hrv_outputs_is':  {'name': 'CHRVS\n(In-Silo)', 'color': '#f781bf'},    # Set1 green
}


def p_to_stars(p):
    """Converts a p-value to a significance star string."""
    if p <= 0.001:
        return '***'
    elif p <= 0.01:
        return '**'
    elif p <= 0.05:
        return '*'
    else:
        return 'ns'
    
def plot_metric_with_error_bars(results_dict, ci_dict,
                                modality_config=modality_config,
                                desired_order=desired_order, 
                                metric_name='Metric', figure_size=(11, 9),
                                sort_by_value=True, stat_test_pairs=None,
                                p_values_dict=None):
    """
    Plots a metric with error bars and manually drawn statistical significance.
    - Uses modality names as colored x-axis labels.
    - Draws significance brackets outside the plot area using pre-calculated p-values.
    """
    # Filter and sort modalities
    available_modalities = [mod for mod in desired_order if mod in results_dict]
    
    if sort_by_value:
        modalities_data = sorted(
            [(mod, results_dict[mod]) for mod in available_modalities],
            key=lambda x: x[1]
        )
        modalities_order = [mod for mod, _ in modalities_data]
    else:
        modalities_order = available_modalities

    fig, ax = plt.subplots(figsize=figure_size)

    # --- Plot Mean and CI ---
    means = [results_dict[mod] for mod in modalities_order]
    ci_tuples = [ci_dict[mod] for mod in modalities_order]
    lowers = [ci[0] for ci in ci_tuples]
    uppers = [ci[1] for ci in ci_tuples]
    err_lower = np.array(means) - np.array(lowers)
    err_upper = np.array(uppers) - np.array(means)
    x_pos = np.arange(len(modalities_order))

    for i, mod in enumerate(modalities_order):
        color = modality_config[mod]['color']
        ax.errorbar(x_pos[i], means[i],
                    yerr=[[err_lower[i]], [err_upper[i]]],
                    fmt='o', color='black', ecolor=color,
                    elinewidth=3, capsize=6, capthick=2, markersize=10,
                    markerfacecolor=color, markeredgecolor='black', markeredgewidth=1.5,
                    zorder=10)
        
        ax.text(x_pos[i], uppers[i] + 0.015, f'{means[i]:.3f}',
                ha='center', va='bottom', fontsize=10, zorder=11)

    # --- Manual Statistical Annotations (drawn outside the axes) ---
    if stat_test_pairs and p_values_dict:
        # Get plot limits to calculate bracket positions in data coordinates
        y_axis_top = ax.get_ylim()[1]
        y_range = y_axis_top - ax.get_ylim()[0]
        
        # Define vertical spacing for brackets
        bracket_level_height = y_range * 0.09
        tip_height = bracket_level_height * 0.2
        
        # Start drawing the first bracket a bit above the plot area
        current_y = y_axis_top + bracket_level_height * 0.5

        for pair in stat_test_pairs:
            mod1, mod2 = pair
            try:
                idx1 = modalities_order.index(mod1)
                idx2 = modalities_order.index(mod2)
            except ValueError:
                continue

            p_key = mod2 if mod1 == 'fusion_outputs' else mod1
            if p_key not in p_values_dict:
                continue
            
            p_val = p_values_dict[p_key]
            stars = p_to_stars(p_val)

            # Draw bracket lines using clip_on=False to allow drawing outside the axes
            bracket_y = current_y
            ax.plot([idx1, idx1, idx2, idx2], 
                    [bracket_y, bracket_y + tip_height, bracket_y + tip_height, bracket_y],
                    lw=1.5, c='black', clip_on=False)

            # Add star annotation, also with clip_on=False
            ax.text((idx1 + idx2) / 2, bracket_y + tip_height, stars,
                    ha='center', va='bottom', color='black', fontsize=12, clip_on=False)
            
            # Increment y-level for the next bracket
            current_y += bracket_level_height
        
        # IMPORTANT: Do not change y-limits. The brackets are now outside the plot area.
        # The calling function will adjust the figure margins to make them visible.

    # --- Formatting ---
    ax.set_xticks(x_pos)
    xtick_labels = [modality_config[mod]['name'] for mod in modalities_order]
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=11)
    for ticklabel, mod in zip(ax.get_xticklabels(), modalities_order):
        ticklabel.set_color(modality_config[mod]['color'])
        ticklabel.set_fontweight('bold')

    ax.set_xlabel('')
    ax.set_ylabel(metric_name, fontsize=14, fontweight='600')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    # We don't call tight_layout here; it's handled by the parent function `plot_kappa`
    return fig

# %%
def plot_kappa(kappa_dict, kappa_ci_dict, p_values_dict=None,
               modality_config=modality_config,
               desired_order=desired_order,
               sort_by_value=True, figure_size=(11, 9)):
    """
    Plot Cohen's kappa with error bars, statistical significance, and a 
    secondary axis for agreement levels.
    """
    # 1. Determine pairs for statistical comparison
    stat_test_pairs = None
    if p_values_dict and 'fusion_outputs' in kappa_dict:
        other_modalities = {
            m: v for m, v in kappa_dict.items() 
            if m != 'fusion_outputs' and m in desired_order
        }
        sorted_others = sorted(other_modalities.items(), key=lambda item: item[1], reverse=True)
        top_3_mods = [mod[0] for mod in sorted_others[:3]]
        stat_test_pairs = [('fusion_outputs', mod) for mod in top_3_mods]

    # 2. Call the base plotting function
    fig = plot_metric_with_error_bars(
        results_dict=kappa_dict,
        ci_dict=kappa_ci_dict,
        p_values_dict=p_values_dict,
        stat_test_pairs=stat_test_pairs,
        modality_config=modality_config,
        desired_order=desired_order,
        metric_name="Cohen's κ",
        sort_by_value=sort_by_value,
        figure_size=figure_size
    )
    
    ax = fig.axes[0]
    ax.set_ylim([0, 0.8])
    # 3. Add secondary y-axis for kappa agreement levels
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    
    kappa_levels = [0.21, 0.41, 0.61, 0.81]
    kappa_labels = ['Fair', 'Moderate', 'Substantial', 'Almost Perfect']
    
    ax2.set_yticks(kappa_levels)
    ax2.set_yticklabels(kappa_labels, fontsize=10, color='#444444')
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params(axis='y', length=0, pad=5)
    ax2.set_ylabel("Agreement Level", fontsize=12, labelpad=20)

    ax.grid(axis='y', linestyle=':', linewidth=1, alpha=0.5)
    for level in kappa_levels:
        ax.axhline(y=level, color='#cccccc', linestyle=':', linewidth=1.5, alpha=0.6, zorder=0)
    
    # 4. Adjust layout to make space for annotations and axis labels
    # Calculate required top margin based on the number of brackets
    num_brackets = len(stat_test_pairs) if stat_test_pairs else 0
    top_margin = 1.0 - (num_brackets * 0.06) # Heuristic: 6% of figure height per bracket
    top_margin = min(0.9, top_margin) # Ensure there's always at least 10% margin

    plt.tight_layout(rect=[0.05, 0.05, 0.88, top_margin]) # [left, bottom, right, top]
    
    plt.show()
    return fig


def plot_model_comparison_with_sig(all_preds, all_probs, all_targets, metric='auroc', 
                                    figsize=(7, 6), modality_config=modality_config, 
                                    n_bootstrap=1000, test_type='Mann-Whitney',
                                    correction='bonferroni',
                                    y_limit=[0.6, 1],):
    """
    Version using pure matplotlib boxplot for absolute color control with significance brackets and * annotations
    """
    
    def calculate_metric(preds, probs, targets, metric_name):
        """Calculate the specified metric"""
        if metric_name == 'auroc':
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(targets, probs)
        elif metric_name == 'recall':
            return np.mean(preds[targets == 1] == 1)
        elif metric_name == 'precision':
            tp = np.sum((preds == 1) & (targets == 1))
            fp = np.sum((preds == 1) & (targets == 0))
            return tp / (tp + fp) if (tp + fp) > 0 else 0
        elif metric_name == 'specificity':
            return np.mean(preds[targets == 0] == 0)
        elif metric_name == 'f1':
            tp = np.sum((preds == 1) & (targets == 1))
            fp = np.sum((preds == 1) & (targets == 0))
            fn = np.sum((preds == 0) & (targets == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Prepare data
    bootstrap_results = {}
    model_keys_ordered = [k for k in all_preds.keys() if modality_config and k in modality_config]
    
    for model_key in model_keys_ordered:
        model_name = modality_config[model_key]['name']
        preds = all_preds[model_key]
        probs = all_probs[model_key]
        targets = all_targets
        
        # Bootstrap sampling
        n_samples = len(targets)
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            score = calculate_metric(preds[indices], probs[indices], targets[indices], metric)
            bootstrap_scores.append(score)
        
        bootstrap_results[model_name] = bootstrap_scores
    
    # Create plot - CHANGE 2: More compact figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data and colors for matplotlib boxplot
    plot_data = [bootstrap_results[modality_config[k]['name']] for k in model_keys_ordered]
    colors = [modality_config[k]['color'] for k in model_keys_ordered]
    labels = [modality_config[k]['name'] for k in model_keys_ordered]
    
    # Create boxplot
    bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, showfliers=True)
    
    # Apply exact colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.0)
    
    # Style other boxplot elements
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_color('black')
            item.set_linewidth(1.0 if element == 'medians' else 1.0)
    
    for flier in bp['fliers']:
        flier.set_markerfacecolor('white')
        flier.set_markeredgecolor('black')
        flier.set_alpha(0.8)
        flier.set_markersize(4)
    
    # CHANGE 1: Set y-axis limit to [0, 1] and keep it fixed
    ax.set_ylim(y_limit)
    
    # Statistical significance testing and annotation
    fusion_name = modality_config.get('fusion_outputs', {}).get('name', 'Fusion')
    
    def get_significance_symbol(p_value):
        """Convert p-value to significance symbols"""
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'ns'
    
    def add_significance_bracket(ax, x1, x2, y, text, height_offset=0.02):
        """Add significance bracket with text annotation - positioned outside plot area"""
        bracket_height = y + height_offset
        # Convert 0-based indices to 1-based positions for matplotlib boxplot
        x1_pos = x1 + 1
        x2_pos = x2 + 1
        
        # Draw bracket - positioned above y=1 but clipped to stay outside
        ax.plot([x1_pos, x1_pos, x2_pos, x2_pos], 
                [y + height_offset/3, bracket_height, bracket_height, y + height_offset/3], 
                'k-', linewidth=1.0, clip_on=False)  # clip_on=False allows drawing outside plot area
        
        # Add significance text - positioned outside plot area
        ax.text((x1_pos + x2_pos) / 2, bracket_height + 0.003, text, 
                ha='center', va='bottom', fontsize=10, fontweight='bold', clip_on=False)
    
    # Perform statistical testing if fusion exists
    if fusion_name in bootstrap_results:
        fusion_scores = bootstrap_results[fusion_name]
        
        # CHANGE 1: Position brackets outside the y=1 limit
        y_start = y_limit[1] + 0.02  # Start brackets just above y=1
        y_offset = y_start
        
        # Find fusion index
        try:
            fusion_idx = labels.index(fusion_name)
        except ValueError:
            fusion_idx = 0  # Default if fusion not found
        
        from scipy import stats
        
        # Compare each model with fusion
        for i, model_name in enumerate(labels):
            if model_name != fusion_name and model_name in bootstrap_results:
                other_scores = bootstrap_results[model_name]
                
                # Perform statistical test
                if test_type == 'Mann-Whitney':
                    stat, p_val = stats.mannwhitneyu(fusion_scores, other_scores, alternative='two-sided')
                elif test_type == 'Wilcoxon':
                    stat, p_val = stats.wilcoxon(fusion_scores, other_scores, alternative='two-sided')
                elif test_type == 't-test_ind':
                    stat, p_val = stats.ttest_ind(fusion_scores, other_scores)
                else:
                    p_val = 1.0  # Default to non-significant
                
                # Apply correction
                if correction == 'bonferroni':
                    p_val_corrected = min(p_val * (len(labels) - 1), 1.0)
                elif correction == 'fdr_bh':
                    p_val_corrected = p_val
                else:
                    p_val_corrected = p_val
                
                # Get significance symbol
                sig_symbol = get_significance_symbol(p_val_corrected)
                
                # Add bracket and annotation outside plot area
                add_significance_bracket(ax, i, fusion_idx, y_offset, sig_symbol, height_offset=0.015)
                y_offset += 0.025  # Smaller increment for compact spacing
                
                # Optional: print p-values for debugging
                print(f"{model_name} vs {fusion_name}: p={p_val:.4f}, corrected p={p_val_corrected:.4f}, sig={sig_symbol}")
    
    # Styling - CHANGE 2: More compact styling
    metric_label = metric.title() if metric != 'auroc' else 'AUROC'
    metric_label = metric.title() if metric != 'recall' else 'Sensitivity'
    ax.set_ylabel(metric_label, fontsize=12, )  # Smaller font fontweight='600'
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('')
    
    plt.xticks(rotation=0, ha='center', fontsize=10)  # Smaller font
    plt.yticks(fontsize=10)  # Smaller font
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Leave space for brackets above the plot
    plt.show()
    return fig


def plot_model_comparison_with_stats(all_preds, all_probs, all_targets, metric='auroc', 
                                   figsize=(8, 8), test_type='Mann-Whitney', 
                                   correction='bonferroni', modality_config=modality_config, 
                                   n_bootstrap=1000, use_statannotations=True,
                                   plot_type='boxplot', orientation='v',
                                   txt_format='star'):
    """
    Plot model comparison with statistical significance using your data structure.
    
    Args:
        all_preds (dict): Dictionary with model keys and prediction arrays
        all_probs (dict): Dictionary with model keys and probability arrays  
        all_targets (array): Array of true labels (shared across all models)
        metric (str): Metric to compare ('auroc', 'precision', 'recall', 'f1')
        figsize (tuple): Figure size
        test_type (str): Statistical test ('Mann-Whitney', 'Wilcoxon', 't-test_ind')
        correction (str): Multiple comparison correction ('bonferroni', 'holm', 'BH')
        modality_config (dict): Configuration for colors and names
        n_bootstrap (int): Number of bootstrap samples
        use_statannotations (bool): Whether to use statannotations library (if available)
        plot_type (str): Type of plot plot="violinplot",'boxplot' 'stripplot' 'swarmplot'
        text_format (str): Format for statistical annotations ('star', simple, full)
        orientation (str): Orientation of the plot ('v' for vertical, 'h' for horizontal)
    """
    
    def calculate_metric(preds, probs, targets, metric_name):
        """Calculate the specified metric"""
        if metric_name == 'auroc':
            return roc_auc_score(targets, probs)
        elif metric_name == 'precision':
            precision, recall, _ = precision_recall_curve(targets, probs)
            return auc(recall, precision)
        elif metric_name == 'recall':
            return np.mean(preds[targets == 1] == 1)  # Sensitivity
        elif metric_name == 'specificity':
            return np.mean(preds[targets == 0] == 0)  # Specificity
        elif metric_name == 'f1':
            tp = np.sum((preds == 1) & (targets == 1))
            fp = np.sum((preds == 1) & (targets == 0))
            fn = np.sum((preds == 0) & (targets == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Prepare data for plotting with bootstrap
    plot_data = []
    bootstrap_results = {}
    
    for model_key in all_preds.keys():
        if model_key in modality_config:
            model_name = modality_config[model_key]['name']
            preds = all_preds[model_key]
            probs = all_probs[model_key]
            targets = all_targets
            
            # Bootstrap sampling
            n_samples = len(targets)
            bootstrap_scores = []
            
            for _ in range(n_bootstrap):
                indices = np.random.choice(n_samples, n_samples, replace=True)
                score = calculate_metric(preds[indices], probs[indices], targets[indices], metric)
                bootstrap_scores.append(score)
            
            # Store bootstrap results for statistical testing
            bootstrap_results[model_name] = bootstrap_scores
            
            # Add to plot data
            for score in bootstrap_scores:
                plot_data.append({'Model': model_name, 'Score': score, 'Model_Key': model_key})
    
    # Create DataFrame
    df = pd.DataFrame(plot_data)
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Get model order and colors
    model_keys_ordered = [k for k in all_preds.keys() if k in modality_config]
    # colors_ordered = [modality_config[k]['color'] for k in model_keys_ordered]
    # Create color palette dictionary mapping model names to colors
    color_palette = {modality_config[k]['name']: modality_config[k]['color'] 
                     for k in model_keys_ordered}
    model_names_ordered = [modality_config[k]['name'] for k in model_keys_ordered]
    
    ax = sns.boxplot(data=df, x='Model', y='Score', 
                     order=model_names_ordered,
                     palette=color_palette)
    
    # Statistical testing and annotation
    fusion_name = modality_config.get('fusion_outputs', {}).get('name', 'Fusion (All Modalities)')
    
    if use_statannotations:
        try:
            from statannotations.Annotator import Annotator
            print('Using statannotations for statistical significance')
            # Define pairs for comparison (all vs fusion)
            pairs = [(fusion_name, name) for name in model_names_ordered if name != fusion_name]
            
            # Add statistical annotations
            annotator = Annotator(ax, pairs, data=df, x='Model', y='Score',
                                  plot=plot_type, orient=orientation)
            annotator.configure(test=test_type, text_format=txt_format, loc='outside', 
                               comparisons_correction=correction, verbose=False)
            annotator.apply_and_annotate()
            
        except ImportError:
            print("statannotations not available, using manual annotations")
            use_statannotations = False
    
    if not use_statannotations:
        # Manual statistical testing and annotation
        def get_significance_symbol(p_value):
            if p_value < 0.001:
                return '***'
            elif p_value < 0.01:
                return '**'
            elif p_value < 0.05:
                return '*'
            else:
                return 'ns'
        
        def add_significance_bracket(ax, x1, x2, y, text, height_offset=0.02):
            bracket_height = y + height_offset
            ax.plot([x1, x1, x2, x2], 
                    [y + height_offset/3, bracket_height, bracket_height, y + height_offset/3], 
                    'k-', linewidth=1.2)
            ax.text((x1 + x2) / 2, bracket_height + 0.005, text, 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Perform statistical tests
        if fusion_name in bootstrap_results:
            fusion_scores = bootstrap_results[fusion_name]
            max_score = df['Score'].max()
            y_offset = max_score + 0.02
            
            fusion_idx = model_names_ordered.index(fusion_name)
            
            for i, model_name in enumerate(model_names_ordered):
                if model_name != fusion_name and model_name in bootstrap_results:
                    other_scores = bootstrap_results[model_name]
                    
                    # Perform statistical test
                    if test_type == 'Mann-Whitney':
                        stat, p_val = stats.mannwhitneyu(fusion_scores, other_scores, alternative='two-sided')
                    elif test_type == 'Wilcoxon':
                        stat, p_val = stats.wilcoxon(fusion_scores, other_scores, alternative='two-sided')
                    elif test_type == 't-test_ind':
                        stat, p_val = stats.ttest_ind(fusion_scores, other_scores)
                    else:
                        p_val = 1.0  # Default to non-significant
                    
                    # Apply correction
                    if correction == 'bonferroni':
                        p_val_corrected = min(p_val * (len(model_names_ordered) - 1), 1.0)
                    else:
                        p_val_corrected = p_val  # Simplified, could add more corrections
                    
                    # Add bracket
                    sig_symbol = get_significance_symbol(p_val_corrected)
                    add_significance_bracket(ax, i, fusion_idx, y_offset, sig_symbol)
                    y_offset += 0.03
    
    # Styling
    if metric == 'auroc':
        metric = 'Sensitivity'
    ax.set_ylabel(metric, fontsize=14)# , fontweight='bold'
    # ax.set_title(f'Model Comparison: {metric.upper()} with Statistical Significance\n({test_type} Test)', 
    #             fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # turn off x-axis title
    ax.set_xlabel('')
    plt.xticks()#rotation=15, ha='right')
    plt.tight_layout()
    return plt.gcf()


def plot_multiple_metrics_single_plot(all_preds, all_probs, all_targets,
                                      metrics=['auroc', 'recall', 'precision'], 
                                    figsize=(12, 8), test_type='Mann-Whitney', 
                                    correction='bonferroni', modality_config=None, 
                                    n_bootstrap=500, use_statannotations=True,
                                    plot_type='boxplot', txt_format='star'):
    """
    Plot multiple metrics in the same plot with different colors for each metric.
    
    Args:
        all_preds (dict): Dictionary with model keys and prediction arrays
        all_probs (dict): Dictionary with model keys and probability arrays  
        all_targets (array): Array of true labels (shared across all models)
        metrics (list): List of metrics to compare ['auroc', 'precision', 'recall', 'f1', 'specificity']
        figsize (tuple): Figure size
        test_type (str): Statistical test ('Mann-Whitney', 'Wilcoxon', 't-test_ind')
        correction (str): Multiple comparison correction ('bonferroni', 'holm', 'BH')
        modality_config (dict): Configuration for colors and names
        n_bootstrap (int): Number of bootstrap samples
        use_statannotations (bool): Whether to use statannotations library
        plot_type (str): 'boxplot', 'violinplot', 'stripplot', 'swarmplot', 'pointplot'
        txt_format (str): 'star', 'simple', 'full'
    """
    
    if modality_config is None:
        modality_config = {
            'eeg_outputs': {'name': 'EEG (Gold Standard)', 'color': '#e41a1c'},
            'fusion_outputs': {'name': 'Fusion (All Modalities)', 'color': '#377eb8'},
            'ecg_outputs': {'name': 'CHRVS', 'color': '#4daf4a'},
            'flow_outputs': {'name': 'Optical Flow', 'color': '#984ea3'},
            'joint_pose_outputs': {'name': 'Pose', 'color': '#ff7f00'},
            'body_outputs': {'name': 'Body', 'color': '#ffff33'},
            'face_outputs': {'name': 'Face', 'color': '#a65628'},
            'rhand_outputs': {'name': 'Right Hand', 'color': '#f781bf'},
            'lhand_outputs': {'name': 'Left Hand', 'color': '#999999'},
            'eeg_outputs_is': {'name': 'EEG (In-Silo)', 'color': '#e41a1c'},
            'ecg_outputs_is': {'name': 'ECG (In-Silo)', 'color': '#377eb8'},
            'hrv_outputs_is': {'name': 'CHRVS (In-Silo)', 'color': '#4daf4a'},
        }
    
    # Define color palette for metrics (not modalities)
    metric_colors = {
        'auroc': '#1f77b4',      # Blue
        'recall': '#ff7f0e',     # Orange  
        'precision': '#2ca02c',  # Green
        'f1': '#d62728',         # Red
        'specificity': '#9467bd', # Purple
        'accuracy': '#8c564b',   # Brown
        'auc_pr': '#e377c2'      # Pink
    }
    
    def calculate_metric(preds, probs, targets, metric_name):
        """Calculate the specified metric"""
        if metric_name == 'auroc':
            return roc_auc_score(targets, probs)
        elif metric_name == 'precision':
            precision, recall, _ = precision_recall_curve(targets, probs)
            return auc(recall, precision)
        elif metric_name == 'recall':
            return np.mean(preds[targets == 1] == 1)  # Sensitivity
        elif metric_name == 'specificity':
            return np.mean(preds[targets == 0] == 0)  # Specificity
        elif metric_name == 'f1':
            tp = np.sum((preds == 1) & (targets == 1))
            fp = np.sum((preds == 1) & (targets == 0))
            fn = np.sum((preds == 0) & (targets == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        elif metric_name == 'accuracy':
            return np.mean(preds == targets)
    
    # Prepare data for plotting with bootstrap
    plot_data = []
    bootstrap_results = {}
    
    for metric in metrics:
        bootstrap_results[metric] = {}
        for model_key in all_preds.keys():
            if model_key in modality_config:
                model_name = modality_config[model_key]['name']
                preds = all_preds[model_key]
                probs = all_probs[model_key]
                targets = all_targets
                
                # Bootstrap sampling
                n_samples = len(targets)
                bootstrap_scores = []
                
                for _ in range(n_bootstrap):
                    indices = np.random.choice(n_samples, n_samples, replace=True)
                    score = calculate_metric(preds[indices], probs[indices], targets[indices], metric)
                    bootstrap_scores.append(score)
                
                # Store bootstrap results for statistical testing
                bootstrap_results[metric][model_name] = bootstrap_scores
                
                # Add to plot data
                for score in bootstrap_scores:
                    plot_data.append({
                        'Model': model_name, 
                        'Score': score, 
                        'Metric': metric.upper(),
                        'Model_Key': model_key
                    })
    
    # Create DataFrame
    df = pd.DataFrame(plot_data)
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Get model order 
    model_names_ordered = [modality_config[k]['name'] for k in all_preds.keys() if k in modality_config]
    
    # Create metric-based color palette
    metrics_upper = [m.upper() for m in metrics]
    palette = [metric_colors.get(m.lower(), '#333333') for m in metrics]
    
    # Create the plot based on plot_type - NOW COLORED BY METRIC
    if plot_type == 'boxplot':
        ax = sns.boxplot(data=df, x='Model', y='Score', hue='Metric', 
                        order=model_names_ordered, hue_order=metrics_upper, palette=palette)
    elif plot_type == 'violinplot':
        ax = sns.violinplot(data=df, x='Model', y='Score', hue='Metric',
                           order=model_names_ordered, hue_order=metrics_upper, palette=palette)
    elif plot_type == 'stripplot':
        ax = sns.stripplot(data=df, x='Model', y='Score', hue='Metric',
                          order=model_names_ordered, hue_order=metrics_upper, palette=palette,
                          dodge=True, alpha=0.7, size=4)
    elif plot_type == 'swarmplot':
        ax = sns.swarmplot(data=df, x='Model', y='Score', hue='Metric',
                          order=model_names_ordered, hue_order=metrics_upper, palette=palette,
                          dodge=True, alpha=0.7, size=4)
    elif plot_type == 'pointplot':
        ax = sns.pointplot(data=df, x='Model', y='Score', hue='Metric',
                          order=model_names_ordered, hue_order=metrics_upper, palette=palette,
                          dodge=0.3, join=False, markers=['o', 's', '^', 'D', 'v'][:len(metrics)])
    
    # Statistical testing and annotation
    fusion_name = modality_config.get('fusion_outputs', {}).get('name', 'Fusion (All Modalities)')
    
    if use_statannotations and fusion_name in model_names_ordered:
        try:
            from statannotations.Annotator import Annotator
            
            # Define pairs for comparison - compare fusion vs other models for each metric
            pairs = []
            for metric in metrics_upper:
                for model_name in model_names_ordered:
                    if model_name != fusion_name:
                        pairs.append(((fusion_name, metric), (model_name, metric)))
            
            # Add statistical annotations
            annotator = Annotator(ax, pairs, data=df, x='Model', y='Score', hue='Metric',
                                 order=model_names_ordered, hue_order=metrics_upper, verbose=False)
            annotator.configure(test=test_type, text_format=txt_format, loc='outside', 
                               comparisons_correction=correction)
            annotator.apply_and_annotate()
            
        except ImportError:
            print("statannotations not available, install with: pip install statannotations")
    
    # Styling
    ax.set_ylabel('Score', fontsize=14, )# fontweight='bold'
    ax.set_xlabel('Model', fontsize=14,)#  fontweight='bold'
    # ax.set_title(f'Multi-Metric Model Comparison\n(Colors represent different metrics)', 
    #             fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Improve legend - now shows metrics, not models
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return plt.gcf()

# Alternative: Scatter plot version (similar to the search result example)
def plot_metrics_scatter_style(all_preds, all_probs, all_targets, metrics=['auroc', 'recall', 'precision'], 
                              figsize=(12, 8), modality_config=None, n_bootstrap=100):
    """
    Create a scatter plot where each metric has a different color and each model has a different marker.
    """
    
    if modality_config is None:
        modality_config = {
            'eeg_outputs': {'name': 'EEG (Gold Standard)', 'color': '#e41a1c'},
            'fusion_outputs': {'name': 'Fusion (All Modalities)', 'color': '#377eb8'},
            'ecg_outputs': {'name': 'CHRVS', 'color': '#4daf4a'},
            'flow_outputs': {'name': 'Optical Flow', 'color': '#984ea3'},
            'joint_pose_outputs': {'name': 'Pose', 'color': '#ff7f00'},
            'body_outputs': {'name': 'Body', 'color': '#ffff33'},
            'face_outputs': {'name': 'Face', 'color': '#a65628'},
            'rhand_outputs': {'name': 'Right Hand', 'color': '#f781bf'},
            'lhand_outputs': {'name': 'Left Hand', 'color': '#999999'},
        }
    
    def calculate_metric(preds, probs, targets, metric_name):
        if metric_name == 'auroc':
            return roc_auc_score(targets, probs)
        elif metric_name == 'precision':
            precision, recall, _ = precision_recall_curve(targets, probs)
            return auc(recall, precision)
        elif metric_name == 'recall':
            return np.mean(preds[targets == 1] == 1)
        elif metric_name == 'specificity':
            return np.mean(preds[targets == 0] == 0)
        elif metric_name == 'f1':
            tp = np.sum((preds == 1) & (targets == 1))
            fp = np.sum((preds == 1) & (targets == 0))
            fn = np.sum((preds == 0) & (targets == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate mean scores for each model-metric combination
    plot_data = []
    for model_key in all_preds.keys():
        if model_key in modality_config:
            model_name = modality_config[model_key]['name']
            for metric in metrics:
                # Bootstrap for confidence
                scores = []
                for _ in range(n_bootstrap):
                    indices = np.random.choice(len(all_targets), len(all_targets), replace=True)
                    score = calculate_metric(all_preds[model_key][indices], 
                                           all_probs[model_key][indices], 
                                           all_targets[indices], metric)
                    scores.append(score)
                
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                plot_data.append({
                    'Model': model_name,
                    'Metric': metric.upper(), 
                    'Score': mean_score,
                    'Std': std_score
                })
    
    df = pd.DataFrame(plot_data)
    
    # Define colors for metrics and markers for models
    metric_colors = {'AUROC': '#1f77b4', 'RECALL': '#ff7f0e', 'PRECISION': '#2ca02c', 
                    'F1': '#d62728', 'SPECIFICITY': '#9467bd'}
    model_markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', '*', 'h']
    
    model_names = df['Model'].unique()
    model_marker_map = dict(zip(model_names, model_markers[:len(model_names)]))
    
    plt.figure(figsize=figsize)
    
    # Plot each combination
    for _, row in df.iterrows():
        plt.scatter(row['Metric'], row['Score'], 
                   color=metric_colors.get(row['Metric'], '#333333'),
                   marker=model_marker_map[row['Model']], 
                   s=120, alpha=0.8, edgecolors='black', linewidth=1)
        
        # Add error bars
        plt.errorbar(row['Metric'], row['Score'], yerr=row['Std'], 
                    color=metric_colors.get(row['Metric'], '#333333'), 
                    alpha=0.5, capsize=3)
    
    # Create custom legend
    metric_legend = [plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=color, markersize=10, label=metric)
                    for metric, color in metric_colors.items() if metric in df['Metric'].values]
    
    model_legend = [plt.Line2D([0], [0], marker=marker, color='w', 
                              markerfacecolor='gray', markersize=10, label=model)
                   for model, marker in model_marker_map.items()]
    
    # Add both legends
    legend1 = plt.legend(handles=metric_legend, title='Metrics', 
                        bbox_to_anchor=(1.05, 1), loc='upper left')
    legend2 = plt.legend(handles=model_legend, title='Models', 
                        bbox_to_anchor=(1.05, 0.5), loc='upper left')
    plt.gca().add_artist(legend1)  # Add first legend back
    
    plt.title('Model Performance Across Multiple Metrics\n(Colors = Metrics, Markers = Models)', 
             fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=14, fontweight='bold')
    plt.xlabel('Metric', fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    
    return plt.gcf()
