import numpy as np
import pandas as pd
from MLstatkit.stats import Delong_test
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, cohen_kappa_score
)
from scipy import stats
from scipy.ndimage import binary_opening, binary_closing
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import warnings

@dataclass
class EvaluationResults:
    """Container for evaluation results"""
    sensitivity: float
    specificity: float
    precision: float
    f1_score: float
    accuracy: float
    auroc: float
    ppv: float  # Positive Predictive Value (same as precision)
    npv: float  # Negative Predictive Value
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int
    cohen_kappa: float
    
# class WindowAggregator:
#     """Combines overlapping window predictions into continuous sequences"""
    
#     def __init__(self, 
#                  window_duration: float,
#                  overlap: float,
#                  sampling_rate: float = 1.0,
#                  aggregation_method: str = 'average'):
#         """
#         Args:
#             window_duration: Duration of each window in seconds
#             overlap: Overlap between windows in seconds  
#             sampling_rate: Sampling rate of the output sequence (Hz)
#             aggregation_method: 'average', 'majority', 'max_prob'
#         """
#         self.window_duration = window_duration
#         self.overlap = overlap
#         self.stride = window_duration - overlap
#         self.sampling_rate = sampling_rate
#         self.aggregation_method = aggregation_method
        
#     def aggregate_windows(self, 
#                          window_predictions: List[np.ndarray],
#                          window_timestamps: List[float],
#                          total_duration: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Aggregate overlapping window predictions into a continuous sequence
        
#         Args:
#             window_predictions: List of prediction arrays (probabilities or logits)
#             window_timestamps: List of start times for each window
#             total_duration: Total duration of the sequence (if known)
            
#         Returns:
#             Tuple of (continuous_predictions, timestamps)
#         """
#         if not window_predictions:
#             return np.array([]), np.array([])
            
#         # Determine output sequence length
#         if total_duration is None:
#             max_end_time = max(ts + self.window_duration for ts in window_timestamps)
#             total_duration = max_end_time
            
#         output_length = int(total_duration * self.sampling_rate)
        
#         # Initialize accumulators
#         prob_sum = np.zeros((output_length, window_predictions[0].shape[-1]))
#         count = np.zeros(output_length)
        
#         # Accumulate predictions
#         for pred, start_time in zip(window_predictions, window_timestamps):
#             start_idx = int(start_time * self.sampling_rate)
#             window_length = int(self.window_duration * self.sampling_rate)
#             end_idx = min(start_idx + window_length, output_length)
            
#             # Handle case where prediction might be shorter than expected
#             pred_length = min(pred.shape[0] if pred.ndim > 1 else len(pred), end_idx - start_idx)
            
#             if self.aggregation_method == 'average':
#                 prob_sum[start_idx:start_idx + pred_length] += pred[:pred_length]
#                 count[start_idx:start_idx + pred_length] += 1
#             elif self.aggregation_method == 'max_prob':
#                 prob_sum[start_idx:start_idx + pred_length] = np.maximum(
#                     prob_sum[start_idx:start_idx + pred_length], pred[:pred_length])
#                 count[start_idx:start_idx + pred_length] = 1
                
#         # Avoid division by zero
#         count = np.maximum(count, 1)
        
#         if self.aggregation_method in ['average', 'max_prob']:
#             continuous_pred = prob_sum / count[:, np.newaxis] if prob_sum.ndim > 1 else prob_sum / count
#         elif self.aggregation_method == 'majority':
#             # For majority voting, we need binary predictions
#             binary_sum = np.zeros(output_length)
#             for pred, start_time in zip(window_predictions, window_timestamps):
#                 start_idx = int(start_time * self.sampling_rate)
#                 window_length = int(self.window_duration * self.sampling_rate)
#                 end_idx = min(start_idx + window_length, output_length)
                
#                 # Convert to binary if needed
#                 if pred.ndim > 1:
#                     binary_pred = np.argmax(pred, axis=-1)
#                 else:
#                     binary_pred = (pred > 0.5).astype(int)
                    
#                 pred_length = min(len(binary_pred), end_idx - start_idx)
#                 binary_sum[start_idx:start_idx + pred_length] += binary_pred[:pred_length]
                
#             continuous_pred = (binary_sum / count > 0.5).astype(int)
        
#         # Generate timestamps
#         timestamps = np.arange(output_length) / self.sampling_rate
        
#         return continuous_pred, timestamps

# ...existing imports and classes...

class WindowAggregator:
    """Combines overlapping window predictions into continuous sequences"""
    
    def __init__(self, 
                 window_duration: float,
                 overlap: float,
                 sampling_rate: float = 1.0,
                 aggregation_method: str = 'average'):
        """
        Args:
            window_duration: Duration of each window in seconds
            overlap: Overlap between windows in seconds  
            sampling_rate: Sampling rate of the output sequence (Hz)
            aggregation_method: 'average', 'majority', 'max_prob'
        """
        self.window_duration = window_duration
        self.overlap = overlap
        self.stride = window_duration - overlap
        self.sampling_rate = sampling_rate
        self.aggregation_method = aggregation_method
        
    def aggregate_windows(self, 
                         window_predictions: List[np.ndarray],
                         window_timestamps: List[float],
                         total_duration: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregate overlapping window predictions into a continuous sequence
        """
        if not window_predictions:
            return np.array([]), np.array([])
            
        # Check if predictions are scalars or arrays
        first_pred = window_predictions[0]
        is_scalar = np.isscalar(first_pred) or (isinstance(first_pred, np.ndarray) and first_pred.ndim == 0)
        
        # Determine output sequence length based on actual window timestamps
        if total_duration is None:
            if len(window_timestamps) > 0:
                max_end_time = max(ts + self.window_duration for ts in window_timestamps)
                total_duration = max_end_time
            else:
                total_duration = self.window_duration
            
        output_length = int(total_duration * self.sampling_rate)
        
        if output_length == 0:
            return np.array([]), np.array([])
        
        if is_scalar:
            # Handle scalar predictions (e.g., ground truth labels)
            prob_sum = np.zeros(output_length)
            count = np.zeros(output_length)
            
            # Accumulate predictions
            for pred, start_time in zip(window_predictions, window_timestamps):
                start_idx = int(start_time * self.sampling_rate)
                window_length = int(self.window_duration * self.sampling_rate)
                end_idx = min(start_idx + window_length, output_length)
                
                if end_idx <= start_idx:
                    continue
                    
                if self.aggregation_method == 'average':
                    prob_sum[start_idx:end_idx] += pred
                    count[start_idx:end_idx] += 1
                elif self.aggregation_method == 'max_prob':
                    prob_sum[start_idx:end_idx] = np.maximum(prob_sum[start_idx:end_idx], pred)
                    count[start_idx:end_idx] = 1
                elif self.aggregation_method == 'majority':
                    prob_sum[start_idx:end_idx] += pred
                    count[start_idx:end_idx] += 1
            
            # Avoid division by zero
            count = np.maximum(count, 1)
            
            if self.aggregation_method in ['average', 'max_prob']:
                continuous_pred = prob_sum / count
            elif self.aggregation_method == 'majority':
                continuous_pred = (prob_sum / count > 0.5).astype(int)
                
        else:
            # Handle array predictions (e.g., probability vectors)
            num_classes = window_predictions[0].shape[-1] if window_predictions[0].ndim > 0 else 1
            prob_sum = np.zeros((output_length, num_classes))
            count = np.zeros(output_length)
            
            # Accumulate predictions
            for pred, start_time in zip(window_predictions, window_timestamps):
                start_idx = int(start_time * self.sampling_rate)
                window_length = int(self.window_duration * self.sampling_rate)
                end_idx = min(start_idx + window_length, output_length)
                
                if end_idx <= start_idx:
                    continue
                
                if self.aggregation_method == 'average':
                    if pred.ndim == 0:  # scalar
                        prob_sum[start_idx:end_idx, 0] += pred
                    else:
                        prob_sum[start_idx:end_idx] += pred
                    count[start_idx:end_idx] += 1
                elif self.aggregation_method == 'max_prob':
                    if pred.ndim == 0:  # scalar
                        prob_sum[start_idx:end_idx, 0] = np.maximum(
                            prob_sum[start_idx:end_idx, 0], pred)
                    else:
                        prob_sum[start_idx:end_idx] = np.maximum(
                            prob_sum[start_idx:end_idx], pred)
                    count[start_idx:end_idx] = 1
                    
            # Avoid division by zero
            count = np.maximum(count, 1)
            
            if self.aggregation_method in ['average', 'max_prob']:
                continuous_pred = prob_sum / count[:, np.newaxis]
            elif self.aggregation_method == 'majority':
                # For majority voting, convert to binary predictions first
                binary_sum = np.zeros(output_length)
                count = np.zeros(output_length)
                
                for pred, start_time in zip(window_predictions, window_timestamps):
                    start_idx = int(start_time * self.sampling_rate)
                    window_length = int(self.window_duration * self.sampling_rate)
                    end_idx = min(start_idx + window_length, output_length)
                    
                    if end_idx <= start_idx:
                        continue
                    
                    # Convert to binary if needed
                    if pred.ndim > 0 and len(pred) > 1:
                        binary_pred = np.argmax(pred)
                    else:
                        binary_pred = 1 if (pred > 0.5 if np.isscalar(pred) else pred[1] > 0.5) else 0
                        
                    binary_sum[start_idx:end_idx] += binary_pred
                    count[start_idx:end_idx] += 1
                    
                count = np.maximum(count, 1)
                continuous_pred = (binary_sum / count > 0.5).astype(int)
        
        # Generate timestamps
        timestamps = np.arange(output_length) / self.sampling_rate
        
        return continuous_pred, timestamps



class PostProcessor:
    """Post-processing methods to reduce false positives"""
    
    @staticmethod
    def temporal_filter(predictions: np.ndarray, 
                       min_duration: float,
                       sampling_rate: float) -> np.ndarray:
        """Remove seizure predictions shorter than min_duration"""
        min_samples = int(min_duration * sampling_rate)
        
        # Convert probabilities to binary if needed
        if predictions.ndim > 1:
            binary_pred = np.argmax(predictions, axis=-1)
        else:
            binary_pred = (predictions > 0.5).astype(int)
            
        # Use morphological opening to remove short events
        if min_samples > 1:
            binary_pred = binary_opening(binary_pred, structure=np.ones(min_samples))
            
        return binary_pred
    
    @staticmethod
    def morphological_filter(predictions: np.ndarray,
                           opening_size: int = 3,
                           closing_size: int = 5) -> np.ndarray:
        """Apply morphological operations to smooth predictions"""
        if predictions.ndim > 1:
            binary_pred = np.argmax(predictions, axis=-1)
        else:
            binary_pred = (predictions > 0.5).astype(int)
            
        # Remove small objects (opening)
        if opening_size > 1:
            binary_pred = binary_opening(binary_pred, structure=np.ones(opening_size))
            
        # Fill small gaps (closing)
        if closing_size > 1:
            binary_pred = binary_closing(binary_pred, structure=np.ones(closing_size))
            
        return binary_pred
    
    @staticmethod
    def hysteresis_threshold(predictions: np.ndarray,
                           high_threshold: float = 0.7,
                           low_threshold: float = 0.3) -> np.ndarray:
        """Apply hysteresis thresholding to reduce noise"""
        if predictions.ndim > 1:
            probs = predictions[:, 1]  # Seizure probabilities
        else:
            probs = predictions
            
        result = np.zeros_like(probs, dtype=int)
        in_seizure = False
        
        for i, prob in enumerate(probs):
            if not in_seizure and prob >= high_threshold:
                in_seizure = True
                result[i] = 1
            elif in_seizure and prob <= low_threshold:
                in_seizure = False
                result[i] = 0
            elif in_seizure:
                result[i] = 1
                
        return result

class EventMatcher:
    """Matches predicted events with ground truth events"""
    
    def __init__(self, min_overlap: float = 0.1):
        """
        Args:
            min_overlap: Minimum overlap ratio for a true positive
        """
        self.min_overlap = min_overlap
        
    def find_events(self, binary_sequence: np.ndarray, 
                   timestamps: np.ndarray) -> List[Tuple[float, float]]:
        """Find continuous events in binary sequence"""
        events = []
        in_event = False
        start_time = None
        
        for i, (value, timestamp) in enumerate(zip(binary_sequence, timestamps)):
            if value == 1 and not in_event:
                in_event = True
                start_time = timestamp
            elif value == 0 and in_event:
                in_event = False
                events.append((start_time, timestamp))
            elif i == len(binary_sequence) - 1 and in_event:
                # Handle case where event extends to end
                events.append((start_time, timestamp))
                
        return events
    
    def compute_overlap(self, event1: Tuple[float, float], 
                       event2: Tuple[float, float]) -> float:
        """Compute overlap ratio between two events"""
        start1, end1 = event1
        start2, end2 = event2
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_end <= overlap_start:
            return 0.0
            
        overlap_duration = overlap_end - overlap_start
        event1_duration = end1 - start1
        
        return overlap_duration / event1_duration if event1_duration > 0 else 0.0
    
    def match_events(self, 
                    predicted_events: List[Tuple[float, float]],
                    true_events: List[Tuple[float, float]]) -> Tuple[int, int, int]:
        """
        Match predicted events with true events using one-to-one matching
        
        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        used_true_events = set()
        true_positives = 0
        
        for pred_event in predicted_events:
            best_match = None
            best_overlap = 0.0
            
            for i, true_event in enumerate(true_events):
                if i in used_true_events:
                    continue
                    
                overlap = self.compute_overlap(pred_event, true_event)
                if overlap >= self.min_overlap and overlap > best_overlap:
                    best_match = i
                    best_overlap = overlap
                    
            if best_match is not None:
                used_true_events.add(best_match)
                true_positives += 1
                
        false_positives = len(predicted_events) - true_positives
        false_negatives = len(true_events) - true_positives
        
        return true_positives, false_positives, false_negatives

class SeizureEvaluator:
    """Main evaluation class for seizure detection models"""
    
    def __init__(self, 
                 window_duration: float = 10.0,
                 overlap: float = 9.0,
                 sampling_rate: float = 1.0,
                 min_seizure_duration: float = 5.0,
                 min_overlap_ratio: float = 0.1):
        """
        Args:
            window_duration: Duration of each window in seconds
            overlap: Overlap between windows in seconds
            sampling_rate: Output sampling rate (Hz)
            min_seizure_duration: Minimum duration for valid seizure events
            min_overlap_ratio: Minimum overlap for event matching
        """
        self.aggregator = WindowAggregator(window_duration, overlap, sampling_rate)
        self.postprocessor = PostProcessor()
        self.event_matcher = EventMatcher(min_overlap_ratio)
        self.min_seizure_duration = min_seizure_duration
        self.sampling_rate = sampling_rate
        
    def evaluate_windows(self, 
                        predictions: np.ndarray,
                        ground_truth: np.ndarray) -> EvaluationResults:
        """Evaluate predictions at window level"""
        # Convert to binary if needed
        if predictions.ndim > 1:
            pred_probs = predictions[:, 1]  # Seizure probabilities
            pred_binary = np.argmax(predictions, axis=1)
        else:
            pred_probs = predictions
            pred_binary = (predictions > 0.5).astype(int)
            
        if ground_truth.ndim > 1:
            gt_binary = np.argmax(ground_truth, axis=1)
        else:
            gt_binary = ground_truth.astype(int)
            
        return self._compute_metrics(pred_binary, gt_binary, pred_probs)
    
    def evaluate_events(self,
                       window_predictions: List[np.ndarray],
                       window_ground_truth: List[np.ndarray], 
                       window_timestamps: List[float],
                       filenames: List[str],
                       aggregation_method: str = 'average',
                       apply_postprocessing: bool = True) -> Tuple[EvaluationResults, Dict]:
        """
        Evaluate predictions at event level
        
        Returns:
            Tuple of (overall_results, per_file_results)
        """
        # Group by filename
        file_groups = defaultdict(list)
        for pred, gt, ts, fname in zip(window_predictions, window_ground_truth, 
                                     window_timestamps, filenames):
            file_groups[fname].append((pred, gt, ts))
            
        all_pred_events = []
        all_true_events = []
        per_file_results = {}
        
        for fname, windows in file_groups.items():
            # Separate predictions, ground truth, and timestamps
            file_preds = [w[0] for w in windows]
            file_gts = [w[1] for w in windows]
            file_timestamps = [w[2] for w in windows]
            
            # Aggregate windows
            self.aggregator.aggregation_method = aggregation_method
            cont_pred, timestamps = self.aggregator.aggregate_windows(
                file_preds, file_timestamps)
            cont_gt, _ = self.aggregator.aggregate_windows(
                file_gts, file_timestamps)
            
            # Convert to binary
            if cont_pred.ndim > 1:
                pred_binary = np.argmax(cont_pred, axis=-1)
            else:
                pred_binary = (cont_pred > 0.5).astype(int)
                
            if cont_gt.ndim > 1:
                gt_binary = np.argmax(cont_gt, axis=-1)
            else:
                gt_binary = cont_gt.astype(int)
            
            # Apply post-processing
            if apply_postprocessing:
                pred_binary = self.postprocessor.temporal_filter(
                    pred_binary, self.min_seizure_duration, self.sampling_rate)
                pred_binary = self.postprocessor.morphological_filter(pred_binary)
            
            # Find events
            pred_events = self.event_matcher.find_events(pred_binary, timestamps)
            true_events = self.event_matcher.find_events(gt_binary, timestamps)
            
            # Match events for this file
            tp, fp, fn = self.event_matcher.match_events(pred_events, true_events)
            
            per_file_results[fname] = {
                'predicted_events': pred_events,
                'true_events': true_events,
                'tp': tp, 'fp': fp, 'fn': fn
            }
            
            all_pred_events.extend(pred_events)
            all_true_events.extend(true_events)
        
        # Compute overall event-level metrics
        total_tp = sum(r['tp'] for r in per_file_results.values())
        total_fp = sum(r['fp'] for r in per_file_results.values())
        total_fn = sum(r['fn'] for r in per_file_results.values())
        total_tn = 0  # Not directly applicable for event-level evaluation
        
        # Calculate metrics
        sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        ppv = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        f1 = 2 * (sensitivity * ppv) / (sensitivity + ppv) if (sensitivity + ppv) > 0 else 0.0
        
        # False positives per hour (assuming timestamps are in seconds)
        total_duration = sum(max(ts + self.aggregator.window_duration 
                               for _, _, ts in windows) for windows in file_groups.values())
        fp_per_hour = (total_fp * 3600) / total_duration if total_duration > 0 else 0.0
        
        results = EvaluationResults(
            sensitivity=sensitivity,
            specificity=0.0,  # Not applicable for event-level
            precision=ppv,
            f1_score=f1,
            accuracy=0.0,  # Not applicable for event-level
            auroc=0.0,  # Not applicable for event-level
            ppv=ppv,
            npv=0.0,  # Not applicable for event-level
            false_positives=total_fp,
            false_negatives=total_fn,
            true_positives=total_tp,
            true_negatives=total_tn,
            cohen_kappa=0.0  # Not applicable for event-level
        )
        
        per_file_results['summary'] = {
            'fp_per_hour': fp_per_hour,
            'total_files': len(file_groups),
            'total_duration_hours': total_duration / 3600
        }
        
        return results, per_file_results
    
    def _compute_metrics(self, 
                        predictions: np.ndarray,
                        ground_truth: np.ndarray,
                        probabilities: Optional[np.ndarray] = None) -> EvaluationResults:
        """Compute evaluation metrics"""
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(ground_truth, predictions, labels=[0, 1]).ravel()
        
        # Basic metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = f1_score(ground_truth, predictions, average='binary', pos_label=1)
        
        # AUROC
        auroc = 0.0
        if probabilities is not None and len(np.unique(ground_truth)) > 1:
            try:
                auroc = roc_auc_score(ground_truth, probabilities)
            except ValueError:
                auroc = 0.0
                
        # Cohen's Kappa
        kappa = cohen_kappa_score(ground_truth, predictions)
        
        return EvaluationResults(
            sensitivity=sensitivity,
            specificity=specificity,
            precision=precision,
            f1_score=f1,
            accuracy=accuracy,
            auroc=auroc,
            ppv=precision,
            npv=npv,
            false_positives=int(fp),
            false_negatives=int(fn),
            true_positives=int(tp),
            true_negatives=int(tn),
            cohen_kappa=kappa
        )

def delong_test(y_true: np.ndarray, 
               y_score1: np.ndarray, 
               y_score2: np.ndarray) -> Tuple[float, float]:
    """
    Perform DeLong test to compare two AUROC scores
    
    Returns:
        Tuple of (z_statistic, p_value)
    """
    try:
        z_score, p_value = Delong_test(y_true, y_score1, y_score2)
        
        return z_score, p_value
        
    except Exception as e:
        warnings.warn(f"DeLong test failed: {e}")
        return 0.0, 1.0

def print_evaluation_results(results: EvaluationResults, 
                           title: str = "Evaluation Results"):
    """Pretty print evaluation results"""
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Sensitivity (Recall): {results.sensitivity:.4f}")
    print(f"Specificity:          {results.specificity:.4f}")
    print(f"Precision (PPV):      {results.precision:.4f}")
    print(f"NPV:                  {results.npv:.4f}")
    print(f"F1-Score:             {results.f1_score:.4f}")
    print(f"Accuracy:             {results.accuracy:.4f}")
    print(f"AUROC:                {results.auroc:.4f}")
    print(f"Cohen's Kappa:        {results.cohen_kappa:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"TP: {results.true_positives:4d} | FN: {results.false_negatives:4d}")
    print(f"FP: {results.false_positives:4d} | TN: {results.true_negatives:4d}")