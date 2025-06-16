import numpy as np
from scipy.stats import mode


from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score, f1_score,
    recall_score, accuracy_score, precision_score, cohen_kappa_score
)



def apply_temporal_smoothing_probs(probabilities, smoothing_window=3):
    """Apply temporal smoothing to 1D probability array"""
    if len(probabilities) < smoothing_window:
        return probabilities
    
    # Use valid mode to avoid edge effects
    smoothed = np.convolve(probabilities, 
                            np.ones(smoothing_window) / smoothing_window, 
                            mode='same')
    return smoothed

def apply_temporal_smoothing_preds(preds, window_size=3):
    """
    Applies temporal smoothing to the predictions using a moving mode filter.
    
    Args:
        preds (np.ndarray): Predictions of shape (N,).
        window_size (int): Size of the smoothing window.
        
    Returns:
        np.ndarray: Smoothed predictions.
    """
    if window_size < 1:
        return preds

    smoothed_preds = np.copy(preds)
    half_window = window_size // 2

    for i in range(len(preds)):
        start = max(0, i - half_window)
        end = min(len(preds), i + half_window + 1)
        smoothed_preds[i] = mode(preds[start:end], keepdims=False).mode

    return smoothed_preds

def hysteresis_thresholding(probs, high_thresh=0.7, low_thresh=0.3,
                            initial_state=0, only_pos_probs=False):
    
    """
    Apply hysteresis thresholding to a sequence of binary classification probabilities.

    Args:
        probs (np.ndarray): Array of shape (T, 2), where each row is [p_neg, p_pos] from softmax,
                            or shape (T,) if only positive probabilities are provided.
        high_thresh (float): Threshold to switch from negative to positive.
        low_thresh (float): Threshold to switch from positive to negative.
        initial_state (int): Starting state (0=negative, 1=positive).
        only_pos_probs (bool): If True, `probs` is expected to be a 1D array of positive class probabilities.

    Returns:
        np.ndarray: Array of shape (T,), with values 0 or 1 representing the predicted class at each time step.
    """
    # probs = np.asarray(probs)  # Ensure input is a NumPy array

    if only_pos_probs:
        if probs.ndim != 1:
            raise ValueError("Expected 1D array for positive probabilities when only_pos_probs=True.")
        pos_probs = probs
    else:
        if probs.ndim != 2 or probs.shape[1] != 2:
            raise ValueError("Expected 2D array of shape (T, 2) when only_pos_probs=False.")
        pos_probs = probs[:, 1]
        
    n = len(pos_probs)

    # Initialize predictions array
    preds = np.zeros(n, dtype=int)
    current_state = initial_state

    for i, p in enumerate(pos_probs):
        if current_state == 0:
            # Only switch to positive if we exceed the high threshold
            if p >= high_thresh:
                current_state = 1
        else:
            # Only switch to negative if we fall below the low threshold
            if p < low_thresh:
                current_state = 0
        preds[i] = current_state

    return preds


def calculate_epoch_level_metrics(all_preds, all_probs, all_targets, eval_config):

    modalities = all_preds.keys()

    recall_dict, prec_dict, auroc_dict, ap_dict, cm = {}, {}, {}, {}, {}

    for modality in modalities:
        # Calculate Precision and Recall First
        probs_pr = all_probs[modality] # shape: (N,) we only get pos_probs from evaluator
        
        prebs_pr = hysteresis_thresholding(probs_pr, 0.8, 0.2, only_pos_probs=True)
        prebs_pr = apply_temporal_smoothing_preds(prebs_pr, 5)
        
        recall_dict[modality] = recall_score(all_targets, prebs_pr, pos_label=1,
                                                average='weighted', zero_division='warn')
        prec_dict[modality] = precision_score(all_targets, prebs_pr,  pos_label=1,
                                                average='weighted', zero_division='warn')
        
        # Calculate AUROC and Average Precision   
        probs_ap = all_probs[modality] 
        probs_ap = apply_temporal_smoothing_probs(probs_ap, 5) # pass this one for plotting AP and AUROC
        fpr, tpr, _ = roc_curve(all_targets, probs_ap, pos_label=1)
        
        auroc_dict[modality] = auc(fpr, tpr)
        ap_dict[modality] = average_precision_score(all_targets, probs_ap)
        
        # get tp, fp, fn, tn
        probs_conf = all_probs[modality]
        probs_conf = apply_temporal_smoothing_probs(probs_conf, 3)
        # makeing high and low thresholds up and tighter gives lower false alarms
        prebs_conf = hysteresis_thresholding(probs_conf, 0.6, 0.4, only_pos_probs=True)
        tn, fp, fn, tp = confusion_matrix(all_targets, prebs_conf).ravel()
        cm[modality] = {
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }
    print(60*"=")
    # Print results in formatted way
    print(f"{'Modality':<20} {'Recall':<10} {'Precision':<10} {'AUROC':<10} {'AP':<10}")
    for modality in modalities:
        print(f"{modality:<20} {recall_dict[modality]:<10.4f} "
            f"{prec_dict[modality]:<10.4f} {auroc_dict[modality]:<10.4f} "
            f"{ap_dict[modality]:<10.4f}")
    print(60*"=")
    # Print confusion matrix in formatted way
    print(f"{'Modality':<20} {'TN':<10} {'FP':<10} {'FN':<10} {'TP':<10}")
    for modality in modalities:
        print(f"{modality:<20} {cm[modality]['tn']:<10} "
            f"{cm[modality]['fp']:<10} {cm[modality]['fn']:<10} "
            f"{cm[modality]['tp']:<10}")
    
    # get total duration of all targets
    total_duration = len(all_targets) * eval_config['window_duration']
    # get duration in hours
    total_duration_hours = total_duration / 3600.0
    # get FALSE POSITIVE RATE/h
    false_alrams = {}
    for modality in all_preds.keys():
        false_alarms = cm[modality]['fp'] / total_duration_hours
        false_alarms = np.round(false_alarms, 2)
        false_alrams[modality] = false_alarms
    print(60*"=")
    print(f"\nTotal Duration: {total_duration_hours:.2f} hours")
    print(f"{'Modality':<20} {'False Alarms/h':<10}")
    for modality, fpr in false_alrams.items():
        print(f"{modality:<20} {fpr:.2f} FPR/h")
    print(60*"=")
    return recall_dict, prec_dict, auroc_dict, ap_dict, cm

    
def seprate_synchronize_events(x, y):
    """
    Extracts non-seizure → seizure events from ground truth (x) and aligns them with model outputs (y).
    
    Args:
        x (list or np.ndarray): Ground truth labels (0 = non-seizure, 1 = seizure).
        y (list or np.ndarray): Model outputs (same length as x).

    Returns:
        List of dicts, each containing:
            - 'start': start index of the event
            - 'end': end index of the event
            - 'ground_truth': list of ground truth labels for the event
            - 'model_output': list of model outputs for the event
    """

    x = np.array(x)
    y = np.array(y)

    events = []
    i = 0
    n = len(x)

    while i < n:
        # Find start of a non-seizure segment
        if x[i] == 0:
            start = i
            while i < n and x[i] == 0:
                i += 1
            # Now i is at the start of a seizure segment (1s)
            if i < n and x[i] == 1:
                while i < n and x[i] == 1:
                    i += 1
                end = i  # i now points to the end of seizure segment
                event = {
                    'start': start,
                    'end': end,
                    'ground_truth': x[start:end],#.tolist(),
                    'model_output': y[start:end]#.tolist()
                }
                events.append(event)
        else:
            i += 1  # Skip any 1s not preceded by a 0

    return events

def split_by_patient_id(patient_ids, pred_array):
    """
    Splits the predictions into a list of tuples (patient_id, array) for each unique patient_id.
    
    Args:
        patient_ids (np.ndarray): Array of patient IDs.
        pred_array (np.ndarray): Array of predictions corresponding to patient IDs.
        
    Returns:
        List of tuples (patient_id, array).
    """
    # check if patient_ids is a numpy array else convert it
    if not isinstance(patient_ids, np.ndarray):
        patient_ids = np.array(patient_ids)
    seen = set()
    unique_ids = []
    for pid in patient_ids:
        if pid not in seen:
            unique_ids.append(pid)
            seen.add(pid)

    # Build two lists of (patient_id, array) tuples
    patient_pred_events = [
        (pid, pred_array[patient_ids == pid])
        for pid in unique_ids
    ]
    
    return patient_pred_events