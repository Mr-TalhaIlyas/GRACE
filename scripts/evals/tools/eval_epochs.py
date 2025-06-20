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
from evals.tools.utils import (apply_temporal_smoothing_probs,
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
    
    # Calculate two-sided p-value for H0: kappa = 0
    # This is twice the proportion of bootstrap estimates that are on the other side of 0.
    p_val = 2 * np.minimum(np.mean(kappa_boot <= 0), np.mean(kappa_boot >= 0))
    
    return kappa_obs, (lower, upper), p_val

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


def calculate_classification_metrics(
    all_probs,
    all_targets,
    eval_config,
    smoothing_window=5,
    hysteresis_high=0.8,
    hysteresis_low=0.2,
    n_bootstraps=1000,
    confidence_level=0.95
):
    """
    Computes and prints core classification metrics:
    Recall, Precision, F1-score, Specificity, NPV, PPV, AUROC, AP,
    Confusion Matrix, and False Alarms per hour.
    Includes bootstrap CIs for Recall, Precision, AUROC, AP.
    """
    modalities = list(all_probs.keys())

    # Containers for point estimates
    recall_dict = {}
    prec_dict = {}  # PPV is the same as precision
    f1_dict = {}
    spec_dict = {}
    npv_dict = {}
    auroc_dict = {}
    ap_dict = {}
    cm_dict = {} # For storing TN, FP, FN, TP

    # Containers for confidence intervals
    recall_ci_dict = {}
    prec_ci_dict = {}
    auroc_ci_dict = {}
    ap_ci_dict = {}

    for modality in modalities:
        # 1. Predictions for threshold-based metrics (Recall, Precision, F1, Spec, NPV, CM)
        probs_for_threshold_metrics = all_probs[modality]
        preds_for_threshold_metrics = hysteresis_thresholding(
            probs_for_threshold_metrics, hysteresis_high, hysteresis_low, only_pos_probs=True
        )
        preds_for_threshold_metrics = apply_temporal_smoothing_preds(
            preds_for_threshold_metrics, smoothing_window
        )

        # 2. Bootstrap Recall
        r_est, r_lo, r_hi = Bootstrapping(
            all_targets,
            preds_for_threshold_metrics,
            metric_str='recall',
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            average='weighted'
        )
        recall_dict[modality] = r_est
        recall_ci_dict[modality] = (r_lo, r_hi)

        # 3. Bootstrap Precision (PPV)
        p_est, p_lo, p_hi = Bootstrapping(
            all_targets,
            preds_for_threshold_metrics,
            metric_str='precision',
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            average='weighted'
        )
        prec_dict[modality] = p_est
        prec_ci_dict[modality] = (p_lo, p_hi)

        # 4. Calculate Confusion Matrix based on preds_for_threshold_metrics
        tn, fp, fn, tp = confusion_matrix(all_targets, preds_for_threshold_metrics).ravel()
        cm_dict[modality] = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

        # 5. Calculate F1, Specificity, NPV from the same CM components
        # Sensitivity is r_est, PPV is p_est
        sensitivity = r_est
        ppv = p_est

        if (ppv + sensitivity) > 0:
            f1_dict[modality] = 2 * (ppv * sensitivity) / (ppv + sensitivity)
        else:
            f1_dict[modality] = 0.0

        if (tn + fp) > 0:
            spec_dict[modality] = tn / (tn + fp)
        else:
            spec_dict[modality] = 0.0 # Or np.nan, depending on desired behavior for edge case

        if (tn + fn) > 0:
            npv_dict[modality] = tn / (tn + fn)
        else:
            npv_dict[modality] = 0.0 # Or np.nan

        # 6. AUROC & Average Precision (bootstrap) using smoothed probabilities
        probs_for_roc_ap = all_probs[modality]
        probs_for_roc_ap = apply_temporal_smoothing_probs(probs_for_roc_ap, smoothing_window)

        auc_est, auc_lo, auc_hi = Bootstrapping(
            all_targets,
            probs_for_roc_ap,
            metric_str='roc_auc',
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            average='weighted'
        )
        auroc_dict[modality] = auc_est
        auroc_ci_dict[modality] = (auc_lo, auc_hi)

        ap_est, ap_lo, ap_hi = Bootstrapping(
            all_targets,
            probs_for_roc_ap,
            metric_str='average_precision',
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            average='weighted'
        )
        ap_dict[modality] = ap_est
        ap_ci_dict[modality] = (ap_lo, ap_hi)

    # Printing results
    print("="*120)
    header = (
        f"{'Modality':<20} {'Recall (CI)':<20} {'Precision (CI)':<20} {'F1-Score':<10} "
        f"{'Specificity':<10} {'NPV':<10} {'AUROC (CI)':<20} {'AP (CI)':<20}"
    )
    print(header)
    print("-"*120)
    for modality in modalities:
        r_est, (r_lo, r_hi) = recall_dict[modality], recall_ci_dict[modality]
        p_est, (p_lo, p_hi) = prec_dict[modality], prec_ci_dict[modality]
        f1_val = f1_dict[modality]
        spec_val = spec_dict[modality]
        npv_val = npv_dict[modality]
        auc_est, (auc_lo, auc_hi) = auroc_dict[modality], auroc_ci_dict[modality]
        ap_est, (ap_lo, ap_hi) = ap_dict[modality], ap_ci_dict[modality]

        print(
            f"{modality:<20} "
            f"{r_est:.3f} [{r_lo:.3f},{r_hi:.3f}]     "
            f"{p_est:.3f} [{p_lo:.3f},{p_hi:.3f}]     "
            f"{f1_val:<10.3f} {spec_val:<10.3f} {npv_val:<10.3f} "
            f"{auc_est:.3f} [{auc_lo:.3f},{auc_hi:.3f}]     "
            f"{ap_est:.3f} [{ap_lo:.3f},{ap_hi:.3f}]"
        )
    print("="*120)

    # Print Confusion Matrix
    print(f"\n{'Modality':<20} {'TN':<10} {'FP':<10} {'FN':<10} {'TP':<10}")
    for modality in modalities:
        tn = cm_dict[modality]['tn']
        fp = cm_dict[modality]['fp']
        fn = cm_dict[modality]['fn']
        tp = cm_dict[modality]['tp']
        print(
            f"{modality:<20} {tn:<10} {fp:<10} {fn:<10} {tp:<10}"
        )
    print("="*80)

    # False Alarms per hour
    total_duration = len(all_targets) * eval_config['window_duration']  # in seconds
    total_duration_hours = total_duration / 3600.0
    print(f"\nTotal Duration: {total_duration_hours:.2f} hours")
    print(f"{'Modality':<20} {'False Alarms/h':<15}")
    for modality in modalities:
        fa_per_h = cm_dict[modality]['fp'] / total_duration_hours if total_duration_hours > 0 else 0
        fa_per_h = np.round(fa_per_h, 2)
        print(f"{modality:<20} {fa_per_h:<15.2f} FPR/h")
    print("="*80)

    return {
        'recall': recall_dict, 'recall_CI': recall_ci_dict,
        'precision': prec_dict, 'precision_CI': prec_ci_dict, # Precision is PPV
        'f1_score': f1_dict,
        'specificity': spec_dict,
        'npv': npv_dict,
        'auroc': auroc_dict, 'auroc_CI': auroc_ci_dict,
        'ap': ap_dict, 'ap_CI': ap_ci_dict,
        'confusion_matrix': cm_dict
    }

def calculate_agreement_metrics(
    all_probs,
    all_targets,
    smoothing_window=5,
    hysteresis_high=0.8,
    hysteresis_low=0.2,
    n_bootstraps=1000,
    confidence_level=0.95,
    random_seed=123, # For kappa reproducibility
    fusion_key_name='fusion_outputs'
):
    """
    Computes and prints comparative and agreement metrics:
    Cohen's Kappa (with CIs) for each modality against targets,
    DeLong's test for AUC differences (fusion vs others),
    and Delta κ (fusion vs others) with bootstrap CI + p-value.
    """
    modalities = list(all_probs.keys())

    kappa_dict = {}
    kappa_ci_dict = {}
    kappa_p_dict = {}
    delta_kappa_dict = {}
    delta_kappa_ci = {}
    delta_kappa_p_dict = {}
    delong_dict = {}

    # 1. Cohen's kappa per modality (vs. expert/all_targets)
    for modality in modalities:
        probs_for_kappa = all_probs[modality]
        preds_for_kappa = hysteresis_thresholding(
            probs_for_kappa, hysteresis_high, hysteresis_low, only_pos_probs=True
        )
        preds_for_kappa = apply_temporal_smoothing_preds(preds_for_kappa, smoothing_window)

        kappa_obs, kappa_ci, kappa_p = bootstrap_kappa(
            all_targets,
            preds_for_kappa,
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            random_seed=random_seed
        )
        kappa_dict[modality] = kappa_obs
        kappa_ci_dict[modality] = kappa_ci
        kappa_p_dict[modality] = kappa_p
    # 2. Determine fusion modality
    if fusion_key_name in all_probs:
        fusion_mod = fusion_key_name
    elif modalities:
        fusion_mod = modalities[0]  # Fallback to first modality
        print(f"Warning: '{fusion_key_name}' modality not found. Falling back to '{modalities[0]}' for comparisons.")
    else:
        print("Error: No modalities found for comparative metrics.")
        return {}


    # Prepare fusion preds and probs once if fusion_mod is valid
    if fusion_mod:
        fusion_preds_for_delta_k = hysteresis_thresholding(
            all_probs[fusion_mod], hysteresis_high, hysteresis_low, only_pos_probs=True
        )
        fusion_preds_for_delta_k = apply_temporal_smoothing_preds(
            fusion_preds_for_delta_k, smoothing_window
        )
        fusion_probs_sm_for_delong = apply_temporal_smoothing_probs(
            all_probs[fusion_mod], smoothing_window
        )

    # 3. Delta kappa & DeLong's test: compare fusion vs every other modality
    for modality in modalities:
        if not fusion_mod or modality == fusion_mod:
            continue

        # Predictions for other modality (for Delta Kappa)
        other_preds_for_delta_k = hysteresis_thresholding(
            all_probs[modality], hysteresis_high, hysteresis_low, only_pos_probs=True
        )
        other_preds_for_delta_k = apply_temporal_smoothing_preds(
            other_preds_for_delta_k, smoothing_window
        )

        # Bootstrap for delta kappa
        results_delta = bootstrap_kappa_delta(
            all_targets,          # rater1 (ground truth)
            other_preds_for_delta_k, # rater2 (other modality)
            fusion_preds_for_delta_k, # rater3 (fusion modality)
            n_bootstraps=n_bootstraps,
            random_seed=random_seed
        )
        delta_kappa_dict[modality] = results_delta['delta_kappa_obs']
        delta_kappa_ci[modality] = results_delta['delta_kappa_ci']
        delta_kappa_p_dict[modality] = results_delta['delta_kappa_p']

        # Smoothed probabilities for other modality (for DeLong's test)
        other_probs_sm_for_delong = apply_temporal_smoothing_probs(
            all_probs[modality], smoothing_window
        )
        z_score, p_value = Delong_test(
            all_targets,
            fusion_probs_sm_for_delong, # Pass fusion probs first
            other_probs_sm_for_delong   # Pass other modality probs second
        )
        # Storing as (Fusion vs Other), so if Delong(ref, test) gives Z for AUC(test) - AUC(ref)
        # and we want to see if Fusion is better, then Delong(Other, Fusion)
        # Or, if Delong_test(targets, preds1, preds2) compares AUC(preds1) vs AUC(preds2)
        # then Z for AUC(preds2) - AUC(preds1).
        # Assuming Delong_test(targets, model1_probs, model2_probs) returns Z for AUC(model2) - AUC(model1)
        # To test Fusion vs Other (AUC_fusion - AUC_other), call Delong_test(targets, other_probs, fusion_probs)
        # The original code was Delong_test(all_targets, fusion_probs_sm, other_probs_sm)
        # Let's assume this means Z for AUC(other_probs_sm) - AUC(fusion_probs_sm)
        # So a negative Z means fusion is better. This matches the previous interpretation.
        delong_dict[modality] = (z_score, p_value)


    # Printing results
    print("="*80)
    print("="*80)
    print("Cohen's Kappa (vs Ground Truth):")
    header_kappa = f"{'Modality':<20} {'Kappa (CI)':<20} {'p-value':<10}"
    print(header_kappa)
    print("-"*80)
    for modality in modalities:
        k_est, (k_lo, k_hi) = kappa_dict[modality], kappa_ci_dict[modality]
        p_val = kappa_p_dict[modality]
        print(f"{modality:<20} {k_est:.3f} [{k_lo:.3f},{k_hi:.3f}] {p_val:<10.4f}")
    print("="*80)

    if fusion_mod and delong_dict:
        print("\nDeLong's Test (Fusion vs Other Modalities - Z for AUC_other - AUC_fusion):")
        # A negative Z means AUC_fusion is higher
        for modality, (z, p) in delong_dict.items():
            print(f"  (Fusion, {modality}): Z = {z:.3f}, p = {p:.3f} "
                  f"{'(Fusion AUC higher)' if z < 0 and p < 0.05 else '(Fusion AUC lower or not sig. different)' if z > 0 and p < 0.05 else '(No sig. diff. in AUC)'}")
        print("="*80)


    if fusion_mod and delta_kappa_dict:
        print("\nDelta Kappa (κ_fusion - κ_other, vs Ground Truth):")
        # A positive Delta Kappa means Fusion's agreement is higher
        for modality in delta_kappa_dict:
            d_obs = delta_kappa_dict[modality]
            d_lo, d_hi = delta_kappa_ci[modality]
            p_val = delta_kappa_p_dict[modality]
            # Significance based on CI not including 0 for Delta Kappa
            sig_note = ""
            if d_lo > 0 and d_hi > 0: # CI is all positive
                 sig_note = "(Fusion agreement sig. higher)"
            elif d_lo < 0 and d_hi < 0: # CI is all negative
                 sig_note = "(Fusion agreement sig. lower)"
            else: # CI includes 0
                 sig_note = "(No sig. diff. in agreement)"

            print(
                f"  Δκ(Fusion, {modality}) = {d_obs:.3f} "
                f"[{d_lo:.3f},{d_hi:.3f}], p_boot = {p_val:.3f} {sig_note}"
            )
        print("="*80)

    return {
        'kappa': kappa_dict,
        'kappa_CI': kappa_ci_dict,
        'delta_kappa': delta_kappa_dict,
        'delta_kappa_CI': delta_kappa_ci,
        'delta_kappa_p': delta_kappa_p_dict, # Note: p-value interpretation discussed previously
        'delong': delong_dict
    }



