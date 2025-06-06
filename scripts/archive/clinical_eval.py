import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SeizureEvent:
    """Represents a single seizure event in a file"""
    file_id: str
    onset_window: int
    offset_window: int
    onset_time: float
    offset_time: float
    duration: float

@dataclass
class PredictionEvent:
    """Represents a predicted seizure event"""
    file_id: str
    start_window: int
    end_window: int
    start_time: float
    end_time: float
    max_probability: float
    mean_probability: float
    confidence_score: float

@dataclass
class EventResults:
    """Results for event-based evaluation with corrected TN logic"""
    file_id: str
    
    # Event-level outcomes (±60s tolerance zone)
    true_positive: bool
    false_negative: bool
    detection_latency: Optional[float]
    
    # Pre-seizure period outcomes (CORRECTED: before onset-60s)
    pre_seizure_true_negative: bool
    pre_seizure_false_positive: bool
    pre_seizure_fp_count: int  # Number of FP windows in pre-seizure period
    pre_seizure_total_windows: int  # Total windows in pre-seizure evaluation zone
    
    # Non-seizure file outcomes
    file_true_negative: bool
    file_false_positive: bool
    
    # Additional info
    prediction_event: Optional[PredictionEvent]
    ground_truth_event: Optional[SeizureEvent]
    has_seizure: bool
    
    # Zone boundaries for debugging
    detection_zone_start: Optional[int]  # onset - 60s (in windows)
    detection_zone_end: Optional[int]    # onset + 60s (in windows)
    pre_seizure_zone_end: Optional[int]  # onset - 60s (in windows)

@dataclass
class ClinicalMetrics:
    """Comprehensive clinical evaluation metrics with corrected TN"""
    # Event-level detection metrics
    event_sensitivity: float  # TP / (TP + FN) - seizure detection rate
    event_specificity: float  # (TN_file + TN_pre) / (TN_file + TN_pre + FP_file + FP_pre)
    event_precision: float    # TP / (TP + FP_total)
    event_f1_score: float
    event_accuracy: float
    
    # Clinical performance metrics
    mean_detection_latency: float
    median_detection_latency: float
    false_positive_rate_per_hour: float
    pre_seizure_specificity: float  # TN_pre / (TN_pre + FP_pre) - pre-seizure period performance
    
    # Detailed counts
    total_files: int
    seizure_files: int
    non_seizure_files: int
    
    # Event outcomes (±60s zone)
    true_positives: int
    false_negatives: int
    
    # Pre-seizure outcomes (CORRECTED: before onset-60s zone)
    pre_seizure_true_negatives: int
    pre_seizure_false_positives: int
    
    # File outcomes (non-seizure files)
    file_true_negatives: int
    file_false_positives: int
    
    # Window-level metrics for comparison
    window_sensitivity: float
    window_specificity: float
    window_f1_score: float

class ClinicalMetricsCalculator:
    """
    Calculate clinical seizure detection metrics with corrected TN handling
    """
    
    def __init__(self, 
                 window_duration: float = 10.0,
                 detection_tolerance: float = 60.0,
                 min_prediction_duration: float = 5.0,
                 probability_threshold: float = 0.5,
                 pre_seizure_fp_threshold: int = 3):
        """
        Args:
            window_duration: Duration of each analysis window in seconds
            detection_tolerance: Time tolerance for detection (±seconds from onset)
            min_prediction_duration: Minimum duration for a valid prediction
            probability_threshold: Threshold for binary classification
            pre_seizure_fp_threshold: Max FP windows allowed in pre-seizure period for TN
        """
        self.window_duration = window_duration
        self.detection_tolerance = detection_tolerance
        self.min_prediction_duration = min_prediction_duration
        self.probability_threshold = probability_threshold
        self.pre_seizure_fp_threshold = pre_seizure_fp_threshold
        
        # Calculate tolerance window count
        self.tolerance_windows = int(self.detection_tolerance / self.window_duration)
        
        print(f"Clinical Metrics Configuration:")
        print(f"  Detection tolerance: ±{self.detection_tolerance}s (±{self.tolerance_windows} windows)")
        print(f"  Pre-seizure FP threshold: ≤{self.pre_seizure_fp_threshold} windows for TN")
        print(f"  Pre-seizure evaluation: from start until (onset - {self.detection_tolerance}s)")
    
    def extract_ground_truth_events(self, file_predictions: Dict) -> Dict[str, SeizureEvent]:
        """Extract ground truth seizure events from each file"""
        ground_truth_events = {}
        
        for filename, data in file_predictions.items():
            gt = data['ground_truth']
            seizure_windows = np.where(gt == 1)[0]
            
            if len(seizure_windows) > 0:
                onset_window = seizure_windows[0]
                offset_window = seizure_windows[-1]
                
                onset_time = onset_window * self.window_duration
                offset_time = (offset_window + 1) * self.window_duration
                duration = offset_time - onset_time
                
                ground_truth_events[filename] = SeizureEvent(
                    file_id=filename,
                    onset_window=onset_window,
                    offset_window=offset_window,
                    onset_time=onset_time,
                    offset_time=offset_time,
                    duration=duration
                )
        
        return ground_truth_events
    
    def extract_predicted_events(self, file_predictions: Dict, modality_key: str) -> Dict[str, List[PredictionEvent]]:
        """Extract predicted seizure events for a specific modality"""
        predicted_events = {}
        
        for filename, data in file_predictions.items():
            predicted_events[filename] = []
            
            if modality_key not in data['probabilities']:
                continue
            
            probs = data['probabilities'][modality_key]
            preds = data['predictions'][modality_key]
            
            if len(probs) == 0 or len(preds) == 0:
                continue
            
            # Find continuous prediction segments
            prediction_segments = self._find_prediction_segments(preds, probs)
            
            # Convert segments to PredictionEvent objects
            for start_idx, end_idx, max_prob, mean_prob in prediction_segments:
                duration = (end_idx - start_idx + 1) * self.window_duration
                if duration >= self.min_prediction_duration:
                    pred_event = PredictionEvent(
                        file_id=filename,
                        start_window=start_idx,
                        end_window=end_idx,
                        start_time=start_idx * self.window_duration,
                        end_time=(end_idx + 1) * self.window_duration,
                        max_probability=max_prob,
                        mean_probability=mean_prob,
                        confidence_score=max_prob
                    )
                    predicted_events[filename].append(pred_event)
        
        return predicted_events
    
    def _find_prediction_segments(self, predictions: np.ndarray, probabilities: np.ndarray) -> List[Tuple]:
        """Find continuous segments of positive predictions"""
        segments = []
        in_segment = False
        start_idx = 0
        
        for i, pred in enumerate(predictions):
            if pred == 1 and not in_segment:
                in_segment = True
                start_idx = i
            elif pred == 0 and in_segment:
                in_segment = False
                end_idx = i - 1
                
                if start_idx <= end_idx < len(probabilities):
                    segment_probs = probabilities[start_idx:end_idx+1]
                    max_prob = np.max(segment_probs)
                    mean_prob = np.mean(segment_probs)
                    segments.append((start_idx, end_idx, max_prob, mean_prob))
        
        # Handle segment continuing to end
        if in_segment and start_idx < len(predictions):
            end_idx = len(predictions) - 1
            if start_idx <= end_idx < len(probabilities):
                segment_probs = probabilities[start_idx:end_idx+1]
                max_prob = np.max(segment_probs)
                mean_prob = np.mean(segment_probs)
                segments.append((start_idx, end_idx, max_prob, mean_prob))
        
        return segments
    
    def _evaluate_file_with_modality(self, 
                                   filename: str,
                                   gt_event: Optional[SeizureEvent],
                                   pred_events: List[PredictionEvent],
                                   file_predictions: Dict,
                                   modality_key: str) -> EventResults:
        """Evaluate a single file with corrected TN logic"""
        
        result = EventResults(
            file_id=filename,
            true_positive=False,
            false_negative=False,
            detection_latency=None,
            pre_seizure_true_negative=False,
            pre_seizure_false_positive=False,
            pre_seizure_fp_count=0,
            pre_seizure_total_windows=0,
            file_true_negative=False,
            file_false_positive=False,
            prediction_event=None,
            ground_truth_event=gt_event,
            has_seizure=gt_event is not None,
            detection_zone_start=None,
            detection_zone_end=None,
            pre_seizure_zone_end=None
        )
        
        if gt_event is None:
            # Non-seizure file evaluation
            if len(pred_events) == 0:
                result.file_true_negative = True
            else:
                result.file_false_positive = True
                result.prediction_event = max(pred_events, key=lambda x: x.confidence_score)
        else:
            # Seizure file evaluation with corrected zones
            onset_window = gt_event.onset_window
            
            # Define zones (CORRECTED LOGIC)
            detection_zone_start = max(0, onset_window - self.tolerance_windows)  # onset - 60s
            detection_zone_end = onset_window + self.tolerance_windows            # onset + 60s
            pre_seizure_zone_end = detection_zone_start                           # onset - 60s
            
            result.detection_zone_start = detection_zone_start
            result.detection_zone_end = detection_zone_end
            result.pre_seizure_zone_end = pre_seizure_zone_end
            
            # 1. Evaluate seizure detection (TP/FN) in ±60s tolerance zone
            onset_time = gt_event.onset_time
            tolerance_start = max(0, onset_time - self.detection_tolerance)
            tolerance_end = onset_time + self.detection_tolerance
            
            valid_predictions = []
            for pred_event in pred_events:
                if (pred_event.start_time <= tolerance_end and 
                    pred_event.end_time >= tolerance_start):
                    valid_predictions.append(pred_event)
            
            if len(valid_predictions) > 0:
                result.true_positive = True
                best_pred = min(valid_predictions, key=lambda x: abs(x.start_time - onset_time))
                result.prediction_event = best_pred
                result.detection_latency = best_pred.start_time - onset_time
            else:
                result.false_negative = True
            
            # 2. Evaluate pre-seizure period (CORRECTED: before onset-60s)
            if (filename in file_predictions and 
                modality_key in file_predictions[filename]['predictions']):
                
                preds = file_predictions[filename]['predictions'][modality_key]
                
                # Pre-seizure evaluation zone: from start to (onset - 60s)
                if pre_seizure_zone_end > 0:
                    pre_seizure_preds = preds[0:pre_seizure_zone_end]
                    fp_count = np.sum(pre_seizure_preds == 1)
                    total_windows = len(pre_seizure_preds)
                    
                    result.pre_seizure_fp_count = fp_count
                    result.pre_seizure_total_windows = total_windows
                    
                    if fp_count <= self.pre_seizure_fp_threshold:
                        result.pre_seizure_true_negative = True
                    else:
                        result.pre_seizure_false_positive = True
                else:
                    # No pre-seizure period to evaluate (seizure at very beginning)
                    result.pre_seizure_total_windows = 0
        
        return result
    
    def calculate_clinical_metrics(self, 
                                 file_predictions: Dict,
                                 modality_key: str) -> ClinicalMetrics:
        """Calculate comprehensive clinical metrics for a specific modality"""
        
        print(f"\n📊 Calculating clinical metrics for {modality_key}...")
        
        # Extract events
        ground_truth_events = self.extract_ground_truth_events(file_predictions)
        predicted_events = self.extract_predicted_events(file_predictions, modality_key)
        
        # Get all unique filenames
        all_files = set(ground_truth_events.keys())
        all_files.update(predicted_events.keys())
        all_files.update(file_predictions.keys())
        
        # Evaluate each file
        results_list = []
        for filename in all_files:
            gt_event = ground_truth_events.get(filename)
            pred_events = predicted_events.get(filename, [])
            
            result = self._evaluate_file_with_modality(
                filename, gt_event, pred_events, file_predictions, modality_key
            )
            results_list.append(result)
        
        # Count outcomes
        tp = sum(1 for r in results_list if r.true_positive)
        fn = sum(1 for r in results_list if r.false_negative)
        
        pre_seizure_tn = sum(1 for r in results_list if r.pre_seizure_true_negative)
        pre_seizure_fp = sum(1 for r in results_list if r.pre_seizure_false_positive)
        
        file_tn = sum(1 for r in results_list if r.file_true_negative)
        file_fp = sum(1 for r in results_list if r.file_false_positive)
        
        # Combined metrics
        total_tn = pre_seizure_tn + file_tn
        total_fp = pre_seizure_fp + file_fp
        total_files = len(results_list)
        seizure_files = tp + fn
        non_seizure_files = file_tn + file_fp
        
        # Calculate metrics
        event_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        event_specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
        event_precision = tp / (tp + total_fp) if (tp + total_fp) > 0 else 0.0
        event_accuracy = (tp + total_tn) / total_files if total_files > 0 else 0.0
        event_f1 = (2 * event_precision * event_sensitivity / 
                   (event_precision + event_sensitivity) 
                   if (event_precision + event_sensitivity) > 0 else 0.0)
        
        # Pre-seizure specificity (CORRECTED)
        pre_seizure_specificity = (pre_seizure_tn / (pre_seizure_tn + pre_seizure_fp) 
                                 if (pre_seizure_tn + pre_seizure_fp) > 0 else 0.0)
        
        # Detection latency
        latencies = [r.detection_latency for r in results_list if r.detection_latency is not None]
        mean_latency = np.mean(latencies) if latencies else 0.0
        median_latency = np.median(latencies) if latencies else 0.0
        
        # False positive rate per hour
        total_duration_hours = sum(len(data['ground_truth']) * self.window_duration / 3600
                                 for data in file_predictions.values())
        fp_per_hour = total_fp / total_duration_hours if total_duration_hours > 0 else 0.0
        
        # Window-level metrics for comparison
        all_window_gt = []
        all_window_pred = []
        
        for filename, data in file_predictions.items():
            if modality_key in data['predictions']:
                all_window_gt.extend(data['ground_truth'])
                all_window_pred.extend(data['predictions'][modality_key])
        
        if len(all_window_gt) > 0 and len(all_window_pred) > 0:
            try:
                from sklearn.metrics import confusion_matrix
                window_cm = confusion_matrix(all_window_gt, all_window_pred)
                if window_cm.size == 4:
                    w_tn, w_fp, w_fn, w_tp = window_cm.ravel()
                    window_sensitivity = w_tp / (w_tp + w_fn) if (w_tp + w_fn) > 0 else 0.0
                    window_specificity = w_tn / (w_tn + w_fp) if (w_tn + w_fp) > 0 else 0.0
                    window_precision = w_tp / (w_tp + w_fp) if (w_tp + w_fp) > 0 else 0.0
                    window_f1 = (2 * window_precision * window_sensitivity / 
                               (window_precision + window_sensitivity) 
                               if (window_precision + window_sensitivity) > 0 else 0.0)
                else:
                    window_sensitivity = window_specificity = window_f1 = 0.0
            except:
                window_sensitivity = window_specificity = window_f1 = 0.0
        else:
            window_sensitivity = window_specificity = window_f1 = 0.0
        
        # Create metrics object
        metrics = ClinicalMetrics(
            event_sensitivity=event_sensitivity,
            event_specificity=event_specificity,
            event_precision=event_precision,
            event_f1_score=event_f1,
            event_accuracy=event_accuracy,
            mean_detection_latency=mean_latency,
            median_detection_latency=median_latency,
            false_positive_rate_per_hour=fp_per_hour,
            pre_seizure_specificity=pre_seizure_specificity,
            total_files=total_files,
            seizure_files=seizure_files,
            non_seizure_files=non_seizure_files,
            true_positives=tp,
            false_negatives=fn,
            pre_seizure_true_negatives=pre_seizure_tn,
            pre_seizure_false_positives=pre_seizure_fp,
            file_true_negatives=file_tn,
            file_false_positives=file_fp,
            window_sensitivity=window_sensitivity,
            window_specificity=window_specificity,
            window_f1_score=window_f1
        )
        
        # Print detailed summary
        print(f"  Event Detection (±{self.detection_tolerance}s zone): TP={tp}, FN={fn}")
        print(f"  Pre-seizure Period (before onset-{self.detection_tolerance}s): TN={pre_seizure_tn}, FP={pre_seizure_fp}")
        print(f"  Non-seizure Files: TN={file_tn}, FP={file_fp}")
        print(f"  Event Sensitivity: {event_sensitivity:.4f}")
        print(f"  Event Specificity: {event_specificity:.4f}")
        print(f"  Pre-seizure Specificity: {pre_seizure_specificity:.4f}")
        print(f"  F1-Score: {event_f1:.4f}")
        
        # Print zone analysis for debugging
        total_pre_seizure_windows = sum(r.pre_seizure_total_windows for r in results_list)
        total_pre_seizure_fp_windows = sum(r.pre_seizure_fp_count for r in results_list)
        files_with_pre_seizure = sum(1 for r in results_list if r.pre_seizure_total_windows > 0)
        
        print(f"  Zone Analysis:")
        print(f"    Files with pre-seizure periods: {files_with_pre_seizure}")
        print(f"    Total pre-seizure windows: {total_pre_seizure_windows}")
        print(f"    Pre-seizure FP windows: {total_pre_seizure_fp_windows}")
        
        return metrics

def calculate_all_modality_metrics(file_predictions: Dict, 
                                 modality_keys: List[str] = None) -> Dict[str, ClinicalMetrics]:
    """Calculate clinical metrics for all modalities"""
    
    if modality_keys is None:
        modality_keys = ['fusion_outputs', 'ecg_outputs', 'flow_outputs', 'joint_pose_outputs']
    
    calculator = ClinicalMetricsCalculator()
    all_metrics = {}
    
    for modality_key in modality_keys:
        # Check if this modality exists in the data
        has_data = any(modality_key in data['predictions'] 
                      for data in file_predictions.values())
        
        if has_data:
            metrics = calculator.calculate_clinical_metrics(file_predictions, modality_key)
            all_metrics[modality_key] = metrics
        else:
            print(f"⚠️  No data found for {modality_key}, skipping...")
    
    return all_metrics


'''
Corrected Clinical TN Definition
For seizure files, we need to evaluate the pre-seizure period EXCLUDING the ±60 second tolerance window around onset:

Timeline for a seizure file:
[---- Pre-seizure Period ----][±60s Tolerance Zone][-- Seizure Period --]
        ^                           ^                      ^
   TN/FP evaluation          Detection window         Seizure (label=1)
      zone                   (excluded from          
                            TN calculation)
                            
Correct Logic:

TN: In the pre-seizure period (before onset-60s), ≤3 windows predicted as seizure
FP: In the pre-seizure period (before onset-60s), >3 windows predicted as seizure
Detection Zone: onset-60s to onset+60s is used ONLY for TP/FN evaluation
Seizure Period: onset+60s onwards (where label=1) - not used for TN/FP
This makes perfect clinical sense because:

The ±60s tolerance zone is reserved for seizure detection evaluation
The pre-seizure period (excluding tolerance zone) tests the model's specificity
We avoid double-counting the same time period for both detection and specificity

'''