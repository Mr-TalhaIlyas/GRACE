import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    precision_score,
    roc_curve,
    auc,
    average_precision_score,
    cohen_kappa_score
)
from evals.tools.eval_utils import (apply_temporal_smoothing_probs,
                                            apply_temporal_smoothing_preds,
                                            hysteresis_thresholding,)
from MLstatkit.stats import Bootstrapping, Delong_test

def compute_kappas_and_delta(rater1, rater2, rater3):
    """
    Compute Cohen’s kappa between:
      - rater1 vs. rater2  (κ₁₂)
      - rater1 vs. rater3  (κ₁₃)
    Then return Δκ = κ₁₃ − κ₁₂.
    """
    kappa_12 = cohen_kappa_score(rater1, rater2)
    kappa_13 = cohen_kappa_score(rater1, rater3)
    delta_kappa = kappa_13 - kappa_12
    return kappa_12, kappa_13, delta_kappa
    

def bootstrap_kappa(rater1, rater2, n_bootstraps=1000, confidence_level=0.95, random_seed=None):
    """
    Bootstrap Cohen's kappa between rater1 and rater2 to get 95% CI.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    r1 = np.asarray(rater1)
    r2 = np.asarray(rater2)
    N = len(r1)
    assert len(r2) == N, "Inputs must have same length"

    # Observed kappa
    kappa_obs = cohen_kappa_score(r1, r2)

    # Bootstrap distribution
    kappa_boot = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        idx = np.random.randint(0, N, size=N)
        kappa_boot[i] = cohen_kappa_score(r1[idx], r2[idx])

    lower, upper = np.percentile(kappa_boot, [(1 - confidence_level) / 2 * 100,
                                               (1 + confidence_level) / 2 * 100])
    return kappa_obs, (lower, upper)

def bootstrap_kappa_delta(rater1, rater2, rater3, n_bootstraps=1000, random_seed=None):
    """
    Perform bootstrap to estimate 95% CIs and p-value for κ₁₂, κ₁₃, and Δκ.
    
    Parameters:
    -----------
    rater1, rater2, rater3 : array‐like of shape (N,)
        Ground truth labels and two sets of predictions (binary or categorical).
    n_bootstraps : int
        Number of bootstrap iterations (e.g., 1000).
    random_seed : int or None
        If provided, sets the RNG seed for reproducibility.
    
    Returns:
    --------
    results : dict with keys:
        'kappa_12_obs', 'kappa_13_obs', 'delta_kappa_obs' : observed values on full data
        'kappa_12_ci'   : (lower_2.5%, upper_97.5%) for κ₁₂
        'kappa_13_ci'   : (lower_2.5%, upper_97.5%) for κ₁₃
        'delta_kappa_ci': (lower_2.5%, upper_97.5%) for Δκ
        'delta_kappa_p' : two‐sided p‐value for Δκ ≠ 0 (bootstrap approximation)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Convert to numpy arrays if not already
    r1 = np.asarray(rater1)
    r2 = np.asarray(rater2)
    r3 = np.asarray(rater3)
    N = len(r1)
    assert len(r2) == N and len(r3) == N, "All three inputs must have the same length"
    
    # 1. Compute observed κ₁₂, κ₁₃, Δκ on the full dataset
    k12_obs, k13_obs, delta_obs = compute_kappas_and_delta(r1, r2, r3)
    
    # 2. Allocate arrays to store bootstrap estimates
    k12_boot = np.empty(n_bootstraps)
    k13_boot = np.empty(n_bootstraps)
    delta_boot = np.empty(n_bootstraps)
    
    # 3. Bootstrap loop
    for i in range(n_bootstraps):
        # Sample indices with replacement from [0, 1, ..., N-1]
        idx = np.random.randint(0, N, size=N)
        
        # Subsample all three rating arrays
        r1_bs = r1[idx]
        r2_bs = r2[idx]
        r3_bs = r3[idx]
        
        # Recompute κ₁₂, κ₁₃, Δκ on the bootstrap sample
        k12_bs, k13_bs, delta_bs = compute_kappas_and_delta(r1_bs, r2_bs, r3_bs)
        k12_boot[i] = k12_bs
        k13_boot[i] = k13_bs
        delta_boot[i] = delta_bs
    
    # 4. Compute 95% confidence intervals from percentiles
    k12_lower, k12_upper = np.percentile(k12_boot, [2.5, 97.5])
    k13_lower, k13_upper = np.percentile(k13_boot, [2.5, 97.5])
    delta_lower, delta_upper = np.percentile(delta_boot, [2.5, 97.5])
    
    # 5. Approximate two‐sided p-value for Δκ ≠ 0
    #    p = fraction of |delta_boot| ≥ |delta_obs|
    p_val = np.mean(np.abs(delta_boot) >= abs(delta_obs))
    
    results = {
        "kappa_12_obs": k12_obs,
        "kappa_13_obs": k13_obs,
        "delta_kappa_obs": delta_obs,
        "kappa_12_ci":   (k12_lower, k12_upper),
        "kappa_13_ci":   (k13_lower, k13_upper),
        "delta_kappa_ci":(delta_lower, delta_upper),
        "delta_kappa_p": p_val
    }
    return results

def calculate_epoch_level_metrics_extended(
    all_preds,
    all_probs,
    all_targets,
    eval_config,
    n_bootstraps=1000,
    confidence_level=0.95
):
    """
    Compute epoch-level metrics (Recall, Precision, AUROC, AP, Confusion Matrix, FA/h),
    plus Delong's test for AUC differences (fusion vs others),
    and Cohen's kappa with bootstrap CI for each modality against targets,
    and Delta κ (fusion vs others) with bootstrap CI + p-value.

    Returns dictionaries of point estimates and CIs, and prints formatted results.
    """
    modalities = list(all_preds.keys())

    # Containers for point estimates
    recall_dict = {}
    prec_dict   = {}
    auroc_dict  = {}
    ap_dict     = {}
    cm          = {}

    # Containers for confidence intervals
    recall_ci_dict = {}
    prec_ci_dict   = {}
    auroc_ci_dict  = {}
    ap_ci_dict     = {}

    # Containers for kappa and delta kappa
    kappa_dict       = {}
    kappa_ci_dict    = {}
    delta_kappa_dict = {}
    delta_kappa_ci   = {}
    delta_kappa_p    = {}

    # First pass: compute per-modality metrics and confusion matrices
    for modality in modalities:
        # Precision & Recall (bootstrap)
        probs_pr = all_probs[modality]  # shape: (N,)
        prebs_pr = hysteresis_thresholding(probs_pr, 0.8, 0.2, only_pos_probs=True)
        prebs_pr = apply_temporal_smoothing_preds(prebs_pr, 5)

        p_est, p_lo, p_hi = Bootstrapping(
            all_targets,
            prebs_pr,
            metric_str='precision',
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            average='weighted'
        )
        prec_dict[modality] = p_est
        prec_ci_dict[modality] = (p_lo, p_hi)

        r_est, r_lo, r_hi = Bootstrapping(
            all_targets,
            prebs_pr,
            metric_str='recall',
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            average='weighted'
        )
        recall_dict[modality] = r_est
        recall_ci_dict[modality] = (r_lo, r_hi)

        # AUROC & Average Precision (bootstrap)
        probs_ap = all_probs[modality]
        probs_ap = apply_temporal_smoothing_probs(probs_ap, 5)

        auc_est, auc_lo, auc_hi = Bootstrapping(
            all_targets,
            probs_ap,
            metric_str='roc_auc',
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            average='weighted'
        )
        auroc_dict[modality] = auc_est
        auroc_ci_dict[modality] = (auc_lo, auc_hi)

        ap_est, ap_lo, ap_hi = Bootstrapping(
            all_targets,
            probs_ap,
            metric_str='average_precision',
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            average='weighted'
        )
        ap_dict[modality] = ap_est
        ap_ci_dict[modality] = (ap_lo, ap_hi)

        # Confusion Matrix (single run)
        probs_conf = all_probs[modality]
        probs_conf = apply_temporal_smoothing_probs(probs_conf, 3)
        prebs_conf = hysteresis_thresholding(probs_conf, 0.6, 0.4, only_pos_probs=True)
        tn, fp, fn, tp = confusion_matrix(all_targets, prebs_conf).ravel()
        cm[modality] = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

    # Cohen's kappa per modality (vs. expert/all_targets)
    for modality in modalities:
        # Generate final predictions for this modality using same hysteresis parameters
        probs_for_kappa = all_probs[modality]
        preds_for_kappa = hysteresis_thresholding(probs_for_kappa, 0.8, 0.2, only_pos_probs=True)
        preds_for_kappa = apply_temporal_smoothing_preds(preds_for_kappa, 5)

        kappa_obs, kappa_ci = bootstrap_kappa(
            all_targets,
            preds_for_kappa,
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            random_seed=123
        )
        kappa_dict[modality] = kappa_obs
        kappa_ci_dict[modality] = kappa_ci

    # Delta kappa: compare fusion ("fusion_outputs") vs every other modality
    if 'fusion_outputs' in all_preds.keys():
        fusion_mod = 'fusion_outputs'
    else:
        fusion_mod = modalities[0]  # Fallback to first modality
        print(f"Error: 'fusion_outputs' modality not found in all_preds.\n Falling back to {modalities[0]}")
    # Prepare fusion preds
    fusion_preds = hysteresis_thresholding(all_probs[fusion_mod], 0.8, 0.2, only_pos_probs=True)
    fusion_preds = apply_temporal_smoothing_preds(fusion_preds, 5)

    for modality in modalities:
        if modality == fusion_mod:
            continue
        # Predictions for other modality
        other_preds = hysteresis_thresholding(all_probs[modality], 0.8, 0.2, only_pos_probs=True)
        other_preds = apply_temporal_smoothing_preds(other_preds, 5)

        # Compute observed kappa_12, kappa_13, delta
        k12_obs, k13_obs, delta_obs = compute_kappas_and_delta(all_targets, other_preds, fusion_preds)

        # Bootstrap for delta kappa
        results_delta = bootstrap_kappa_delta(
            all_targets,
            other_preds,
            fusion_preds,
            n_bootstraps=n_bootstraps,
            random_seed=123
        )
        delta_kappa_dict[modality] = results_delta['delta_kappa_obs']
        delta_kappa_ci[modality] = results_delta['delta_kappa_ci']
        delta_kappa_p[modality] = results_delta['delta_kappa_p']

    # Delong's test: compare AUC between Fusion and each other modality
    delong_dict = {}
    fusion_probs_sm = apply_temporal_smoothing_probs(all_probs[fusion_mod], 5)

    for modality in modalities:
        if modality == fusion_mod:
            continue
        other_probs_sm = apply_temporal_smoothing_probs(all_probs[modality], 5)
        z_score, p_value = Delong_test(all_targets, fusion_probs_sm, other_probs_sm)
        delong_dict[modality] = (z_score, p_value)

    # Printing results
    print("="*80)
    header = (
        f"{'Modality':<20} {'Recall (CI)':<20} {'Precision (CI)':<20} "
        f"{'AUROC (CI)':<20} {'AP (CI)':<20} {'Kappa (CI)':<20}"
    )
    print(header)
    print("-"*80)
    for modality in modalities:
        r_est, (r_lo, r_hi) = recall_dict[modality], recall_ci_dict[modality]
        p_est, (p_lo, p_hi) = prec_dict[modality], prec_ci_dict[modality]
        auc_est, (auc_lo, auc_hi) = auroc_dict[modality], auroc_ci_dict[modality]
        ap_est, (ap_lo, ap_hi) = ap_dict[modality], ap_ci_dict[modality]
        k_est, (k_lo, k_hi) = kappa_dict[modality], kappa_ci_dict[modality]

        print(
            f"{modality:<20} "
            f"{r_est:.3f} [{r_lo:.3f},{r_hi:.3f}]     "
            f"{p_est:.3f} [{p_lo:.3f},{p_hi:.3f}]     "
            f"{auc_est:.3f} [{auc_lo:.3f},{auc_hi:.3f}]     "
            f"{ap_est:.3f} [{ap_lo:.3f},{ap_hi:.3f}]     "
            f"{k_est:.3f} [{k_lo:.3f},{k_hi:.3f}]"
        )
    print("="*80)

    # Print Confusion Matrix
    print(f"{'Modality':<20} {'TN':<10} {'FP':<10} {'FN':<10} {'TP':<10}")
    for modality in modalities:
        tn = cm[modality]['tn']
        fp = cm[modality]['fp']
        fn = cm[modality]['fn']
        tp = cm[modality]['tp']
        print(
            f"{modality:<20} {tn:<10} {fp:<10} {fn:<10} {tp:<10}"
        )

    # False Alarms per hour
    total_duration = len(all_targets) * eval_config['window_duration']  # in seconds
    total_duration_hours = total_duration / 3600.0
    print("="*80)
    print(f"\nTotal Duration: {total_duration_hours:.2f} hours")
    print(f"{'Modality':<20} {'False Alarms/h':<10}")
    for modality in modalities:
        fa_per_h = cm[modality]['fp'] / total_duration_hours
        fa_per_h = np.round(fa_per_h, 2)
        print(f"{modality:<20} {fa_per_h:.2f} FPR/h")
    print("="*80)

    # Delong's test results
    print("\nDelong's Test (Fusion vs Other Modalities):")
    for modality, (z, p) in delong_dict.items():
        print(f"  (Fusion, {modality}): Z = {z:.3f}, p = {p:.3f}")

    print("\nDelta Kappa (Fusion vs Other Modalities):")
    for modality in delta_kappa_dict:
        d_obs = delta_kappa_dict[modality]
        d_lo, d_hi = delta_kappa_ci[modality]
        p_val = delta_kappa_p[modality]
        print(
            f"  Δκ(Fusion, {modality}) = {d_obs:.3f} "
            f"[{d_lo:.3f},{d_hi:.3f}], p = {p_val:.3f}"
        )
    print("="*80)

    return {
        'recall': recall_dict,
        'precision': prec_dict,
        'auroc': auroc_dict,
        'ap': ap_dict,
        'confusion_matrix': cm,
        'recall_CI': recall_ci_dict,
        'precision_CI': prec_ci_dict,
        'auroc_CI': auroc_ci_dict,
        'ap_CI': ap_ci_dict,
        'kappa': kappa_dict,
        'kappa_CI': kappa_ci_dict,
        'delta_kappa': delta_kappa_dict,
        'delta_kappa_CI': delta_kappa_ci,
        'delta_kappa_p': delta_kappa_p,
        'delong': delong_dict
    }

def calculate_epoch_level_metrics_bootstrap(
    all_preds,
    all_probs,
    all_targets,
    eval_config,
    n_bootstraps=1000,
    confidence_level=0.95
):
    """
    Compute epoch-level metrics (Recall, Precision, AUROC, AP, Confusion Matrix, FA/h)
    with bootstrap confidence intervals for Recall, Precision, AUROC, and Average Precision.

    Arguments:
        all_preds (dict):      modality -> array of model predictions (unused directly here)
        all_probs (dict):      modality -> array of shape (N,) (pos-class probabilities)
        all_targets (np.array):1D array of shape (N,) with ground-truth labels (0 or 1)
        eval_config (dict):    must contain 'window_duration' (in seconds)
        n_bootstraps (int):    number of bootstrap samples
        confidence_level (float): e.g. 0.95 for 95% CI

    Returns:
        recall_dict, prec_dict, auroc_dict, ap_dict, cm,
        recall_ci_dict, prec_ci_dict, auroc_ci_dict, ap_ci_dict
    """
    modalities = all_preds.keys()

    # Containers for point estimates
    recall_dict = {}
    prec_dict   = {}
    auroc_dict  = {}
    ap_dict     = {}
    cm          = {}

    # Containers for confidence intervals
    recall_ci_dict = {}
    prec_ci_dict   = {}
    auroc_ci_dict  = {}
    ap_ci_dict     = {}

    for modality in modalities:
        # ----------------------------------------
        # 1) Precision & Recall (bootstrap)
        # ----------------------------------------
        # 1a) Apply hysteresis + smoothing to get discrete predictions
        probs_pr = all_probs[modality]  # shape: (N,)
        prebs_pr = hysteresis_thresholding(probs_pr, 0.8, 0.2, only_pos_probs=True)
        prebs_pr = apply_temporal_smoothing_preds(prebs_pr, 5)

        # 1b) Bootstrap Precision
        p_est, p_lo, p_hi = Bootstrapping(
            all_targets,
            prebs_pr,
            metric_str='precision',
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            average='weighted'
        )
        prec_dict[modality] = p_est
        prec_ci_dict[modality] = (p_lo, p_hi)

        # 1c) Bootstrap Recall
        r_est, r_lo, r_hi = Bootstrapping(
            all_targets,
            prebs_pr,
            metric_str='recall',
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            average='weighted'
        )
        recall_dict[modality] = r_est
        recall_ci_dict[modality] = (r_lo, r_hi)

        # ----------------------------------------
        # 2) AUROC & Average Precision (bootstrap)
        # ----------------------------------------
        # 2a) Use smoothed probabilities for AUROC / AP
        probs_ap = all_probs[modality]
        probs_ap = apply_temporal_smoothing_probs(probs_ap, 5)

        # 2b) Bootstrap AUROC
        auc_est, auc_lo, auc_hi = Bootstrapping(
            all_targets,
            probs_ap,
            metric_str='roc_auc',
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            average='weighted'
        )
        auroc_dict[modality] = auc_est
        auroc_ci_dict[modality] = (auc_lo, auc_hi)

        # 2c) Bootstrap Average Precision
        ap_est, ap_lo, ap_hi = Bootstrapping(
            all_targets,
            probs_ap,
            metric_str='average_precision',
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            average='weighted'
        )
        ap_dict[modality] = ap_est
        ap_ci_dict[modality] = (ap_lo, ap_hi)

        # ----------------------------------------
        # 3) Confusion Matrix (single run)
        # ----------------------------------------
        probs_conf = all_probs[modality]
        probs_conf = apply_temporal_smoothing_probs(probs_conf, 3)
        prebs_conf = hysteresis_thresholding(probs_conf, 0.6, 0.4, only_pos_probs=True)
        tn, fp, fn, tp = confusion_matrix(all_targets, prebs_conf).ravel()
        cm[modality] = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

    # ----------------------------------------
    # 4) Print results
    # ----------------------------------------
    print("="*60)
    header = f"{'Modality':<20} {'Recall (CI)':<20} {'Precision (CI)':<20} {'AUROC (CI)':<20} {'AP (CI)':<20}"
    print(header)
    print("-"*60)
    for modality in modalities:
        r_est = recall_dict[modality]
        r_lo, r_hi = recall_ci_dict[modality]
        p_est = prec_dict[modality]
        p_lo, p_hi = prec_ci_dict[modality]
        auc_est = auroc_dict[modality]
        auc_lo, auc_hi = auroc_ci_dict[modality]
        ap_est = ap_dict[modality]
        ap_lo, ap_hi = ap_ci_dict[modality]

        print(
            f"{modality:<20} "
            f"{r_est:.3f} [{r_lo:.3f}, {r_hi:.3f}]     "
            f"{p_est:.3f} [{p_lo:.3f}, {p_hi:.3f}]     "
            f"{auc_est:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]     "
            f"{ap_est:.3f} [{ap_lo:.3f}, {ap_hi:.3f}]"
        )
    print("="*60)

    # Print confusion matrix per modality
    print(f"{'Modality':<20} {'TN':<10} {'FP':<10} {'FN':<10} {'TP':<10}")
    for modality in modalities:
        tn = cm[modality]['tn']
        fp = cm[modality]['fp']
        fn = cm[modality]['fn']
        tp = cm[modality]['tp']
        print(
            f"{modality:<20} {tn:<10} {fp:<10} {fn:<10} {tp:<10}"
        )

    # ----------------------------------------
    # 5) False Alarms per Hour
    # ----------------------------------------
    total_duration = len(all_targets) * eval_config['window_duration']  # in seconds
    total_duration_hours = total_duration / 3600.0

    false_alarms = {}
    print("="*60)
    print(f"\nTotal Duration: {total_duration_hours:.2f} hours")
    print("_"*60)
    print(f"{'Modality':<20} {'False Alarms/h':<10}")
    for modality in modalities:
        fa_per_h = cm[modality]['fp'] / total_duration_hours
        fa_per_h = np.round(fa_per_h, 2)
        false_alarms[modality] = fa_per_h
        print(f"{modality:<20} {fa_per_h:.2f} FPR/h")
    print("="*60)

    return (
        recall_dict,
        prec_dict,
        auroc_dict,
        ap_dict,
        cm,
        recall_ci_dict,
        prec_ci_dict,
        auroc_ci_dict,
        ap_ci_dict
    )