import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from scipy.stats import mode
from data.utils import video_transform
from MLstatkit.stats import Bootstrapping, Delong_test

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score, f1_score,
    recall_score, accuracy_score, precision_score, cohen_kappa_score
)
class TestTimeEvaluator:
    """
    Evaluator for test-time inference that collects predictions, probabilities,
    and ground-truth labels for all modalities. Uses a tqdm progress bar and
    does not compute any metrics internally.
    """
    def __init__(self, model, device=torch.device("cuda")):
        self.model = model.to(device)
        self.device = device
        # Specify the output keys from the model and corresponding modality names
        self.output_keys = [
            "flow_outputs",
            "body_outputs",
            "face_outputs",
            "rhand_outputs",
            "lhand_outputs",
            "ecg_outputs",
            "joint_pose_outputs",
            "fusion_outputs",
        ]

    def evaluate(self, loader):
        """
        Runs inference on `loader` (e.g., a DataLoader) and returns dictionaries
        of predictions, probabilities, and labels for each modality.
        
        Returns:
            all_preds:   { modality_name: torch.Tensor of shape (N,) }
            all_probs:   { modality_name: torch.Tensor of shape (N, num_classes) }
            all_targets: torch.Tensor of shape (N,)
        """
        self.model.eval()

        # Prepare containers
        all_preds = {key: [] for key in self.output_keys}
        all_probs = {key: [] for key in self.output_keys}
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(loader, desc="Test-time evaluation", unit="batch")
            for batch in pbar:
                # ---------------------------------------------------------
                # 1. Preprocess inputs (copy the same transforms as during training)
                # ---------------------------------------------------------
                frames = video_transform(batch["frames"]).to(self.device, non_blocking=True)
                body   = batch["body"].permute(0,4,2,3,1).float().to(self.device, non_blocking=True)
                face   = batch["face"].permute(0,4,2,3,1).float().to(self.device, non_blocking=True)
                rh     = batch["rh"].permute(0,4,2,3,1).float().to(self.device, non_blocking=True)
                lh     = batch["lh"].permute(0,4,2,3,1).float().to(self.device, non_blocking=True)
                hrv    = batch["hrv"].to(torch.float).to(self.device, non_blocking=True)
                targets = torch.argmax(batch["super_lbls"], dim=1).long().to(self.device, non_blocking=True)

                # ---------------------------------------------------------
                # 2. Forward pass
                # ---------------------------------------------------------
                outputs = self.model(frames, body, face, rh, lh, hrv)
                # outputs is expected to be a dict with keys matching self.output_keys

                # ---------------------------------------------------------
                # 3. Extract predicted class indices and softmax probabilities
                # ---------------------------------------------------------
                for key in self.output_keys:
                    logits = outputs[key]               # shape: (batch_size, num_classes)
                    prob   = F.softmax(logits, dim=1)   # shape: (batch_size, num_classes)
                    pred   = torch.argmax(prob, dim=1)  # shape: (batch_size,)

                    all_preds[key].append(pred.cpu())
                    all_probs[key].append(prob.cpu())

                # ---------------------------------------------------------
                # 4. Collect ground-truth labels
                # ---------------------------------------------------------
                all_targets.append(targets.cpu())

        # ---------------------------------------------------------
        # 5. Concatenate tensors across all batches
        # ---------------------------------------------------------
        for key in self.output_keys:
            all_preds[key] = torch.cat(all_preds[key], dim=0).numpy()   # shape: (N,)
            # all_probs[key] = torch.cat(all_probs[key], dim=0).numpy()   # shape: (N, num_classes)
            all_probs[key] = torch.cat(all_probs[key], dim=0)[:, 1].numpy()  # shape: (N,)
        all_targets = torch.cat(all_targets, dim=0).numpy()             # shape: (N,)

        return all_preds, all_probs, all_targets

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
    import numpy as np

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
