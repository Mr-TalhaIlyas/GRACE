import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score
)
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from data.utils import video_transform
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

@dataclass
class SeizureEvent:
    """Represents a single seizure event in a file"""
    file_id: str
    onset_window: int  # Window index where seizure starts
    offset_window: int  # Window index where seizure ends (if applicable)
    onset_time: float  # Time in seconds
    offset_time: float  # Time in seconds
    duration: float    # Duration in seconds

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
    """Results for event-based evaluation"""
    file_id: str
    true_positive: bool
    false_positive: bool
    false_negative: bool
    true_negative: bool
    detection_latency: Optional[float]  # Time from onset to detection (if TP)
    prediction_event: Optional[PredictionEvent]
    ground_truth_event: Optional[SeizureEvent]

@dataclass
class ClinicalMetrics:
    """Clinical evaluation metrics"""
    # Event-level metrics
    event_sensitivity: float  # TP / (TP + FN)
    event_specificity: float  # TN / (TN + FP) 
    event_precision: float    # TP / (TP + FP)
    event_f1_score: float
    event_accuracy: float
    
    # Detection performance
    mean_detection_latency: float
    median_detection_latency: float
    false_positive_rate_per_hour: float
    
    # Raw counts
    total_files: int
    seizure_files: int
    non_seizure_files: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    
    # Window-level metrics for comparison
    window_sensitivity: float
    window_specificity: float
    window_f1_score: float

class ClinicalSeizureEvaluator:
    """
    Clinical seizure detection evaluator implementing OVLP methodology
    from "Objective evaluation metrics for automatic classification of EEG events"
    """
    
    def __init__(self, 
                 window_duration: float = 10.0,
                 detection_tolerance: float = 60.0,  # ±60 seconds tolerance
                 min_prediction_duration: float = 5.0,  # Minimum prediction duration
                 probability_threshold: float = 0.5,
                 smoothing_window: int = 3):
        """
        Args:
            window_duration: Duration of each analysis window in seconds
            detection_tolerance: Time tolerance for detection (±seconds from onset)
            min_prediction_duration: Minimum duration for a valid prediction
            probability_threshold: Threshold for binary classification
            smoothing_window: Number of windows for temporal smoothing
        """
        self.window_duration = window_duration
        self.detection_tolerance = detection_tolerance
        self.min_prediction_duration = min_prediction_duration
        self.probability_threshold = probability_threshold
        self.smoothing_window = smoothing_window
        
        # Modality configurations
        self.modality_config = {
            'fusion_outputs': {'name': 'Fusion', 'color': '#1f77b4'},
            'ecg_outputs': {'name': 'ECG', 'color': '#ff7f0e'},
            'flow_outputs': {'name': 'Optical Flow', 'color': '#2ca02c'},
            'joint_pose_outputs': {'name': 'Pose (Joint)', 'color': '#d62728'},
        }
    
    def evaluate_clinical_performance(self, 
                                    model,
                                    dataloader,
                                    device: torch.device,
                                    save_dir: str = "clinical_evaluation_results/") -> Dict:
        """
        Main clinical evaluation pipeline
        """
        print("🏥 Starting Clinical Seizure Detection Evaluation...")
        
        # Create save directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Step 1: Run inference and collect predictions by file
        file_predictions = self._run_clinical_inference(model, dataloader, device)
        
        # Step 2: Extract ground truth seizure events
        ground_truth_events = self._extract_ground_truth_events(file_predictions)
        
        # Step 3: Extract predicted seizure events for each modality
        predicted_events = self._extract_predicted_events(file_predictions)
        
        # Step 4: Perform event-based evaluation using OVLP methodology
        evaluation_results = self._evaluate_events_ovlp(ground_truth_events, predicted_events)
        
        # Step 5: Compute clinical metrics
        clinical_metrics = self._compute_clinical_metrics(evaluation_results, file_predictions)
        
        # Step 6: Generate clinical reports and visualizations
        self._generate_clinical_reports(clinical_metrics, evaluation_results, save_dir)
        
        return {
            'clinical_metrics': clinical_metrics,
            'evaluation_results': evaluation_results,
            'ground_truth_events': ground_truth_events,
            'predicted_events': predicted_events,
            'file_predictions': file_predictions
        }
    
    def _run_clinical_inference(self, model, dataloader, device) -> Dict:
        """Run inference and organize results by file"""
        print("🔄 Running clinical inference...")
        
        model.eval()
        file_predictions = defaultdict(lambda: {
            'predictions': defaultdict(list),
            'probabilities': defaultdict(list),
            'ground_truth': [],
            'timestamps': [],
            'window_indices': [],
            'filenames': []  # Track individual filenames for debugging
        })
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Clinical Inference")):
                # Prepare inputs
                frames = video_transform(batch['frames']).to(device, non_blocking=True)
                body = batch['body'].permute(0,4,2,3,1).float().to(device, non_blocking=True)
                face = batch['face'].permute(0,4,2,3,1).float().to(device, non_blocking=True)
                rh = batch['rh'].permute(0,4,2,3,1).float().to(device, non_blocking=True)
                lh = batch['lh'].permute(0,4,2,3,1).float().to(device, non_blocking=True)
                hrv = batch['hrv'].to(torch.float).to(device, non_blocking=True)
                
                # Ground truth
                if 'super_lbls' in batch:
                    gt_labels = torch.argmax(batch['super_lbls'], dim=1).cpu().numpy()
                elif 'labels' in batch:
                    gt_labels = batch['labels'].cpu().numpy()
                else:
                    gt_labels = np.zeros(len(batch['frames']))
                
                # Model inference
                outputs = model(frames, body, face, rh, lh, hrv)
                
                # Process each sample in batch
                batch_size = len(batch['frames'])
                for i in range(batch_size):
                    # Handle different ways filename might be stored
                    if 'filename' in batch:
                        if isinstance(batch['filename'], list):
                            filename = batch['filename'][i]
                        else:
                            filename = batch['filename'][i].item() if hasattr(batch['filename'][i], 'item') else str(batch['filename'][i])
                    elif 'file_id' in batch:
                        filename = batch['file_id'][i] if isinstance(batch['file_id'], list) else str(batch['file_id'][i])
                    else:
                        # Fallback: create filename from batch and sample index
                        filename = f"batch_{batch_idx:04d}_sample_{i:04d}"
                    
                    # Store ground truth and metadata
                    file_predictions[filename]['ground_truth'].append(gt_labels[i])
                    file_predictions[filename]['timestamps'].append(batch_idx * batch_size + i)
                    file_predictions[filename]['window_indices'].append(len(file_predictions[filename]['ground_truth']) - 1)
                    file_predictions[filename]['filenames'].append(filename)  # For consistency checking
                    
                    # Process each modality
                    for modality_key, logits in outputs.items():
                        if modality_key in self.modality_config:
                            # Get probabilities and predictions
                            if logits.dim() > 1 and logits.shape[1] == 2:
                                probs = torch.softmax(logits, dim=1).cpu().numpy()
                                seizure_prob = probs[i, 1]
                                pred = torch.argmax(logits, dim=1).cpu().numpy()[i]
                            elif logits.dim() > 1 and logits.shape[1] == 1:
                                seizure_prob = torch.sigmoid(logits).squeeze().cpu().numpy()[i]
                                pred = int(seizure_prob >= self.probability_threshold)
                            else:
                                # Handle case where logits is 1D
                                seizure_prob = torch.sigmoid(logits).cpu().numpy()[i]
                                pred = int(seizure_prob >= self.probability_threshold)
                            
                            file_predictions[filename]['probabilities'][modality_key].append(seizure_prob)
                            file_predictions[filename]['predictions'][modality_key].append(pred)
        
        # Convert lists to numpy arrays
        for filename in file_predictions:
            file_predictions[filename]['ground_truth'] = np.array(file_predictions[filename]['ground_truth'])
            file_predictions[filename]['timestamps'] = np.array(file_predictions[filename]['timestamps'])
            file_predictions[filename]['window_indices'] = np.array(file_predictions[filename]['window_indices'])
            
            for modality_key in self.modality_config.keys():
                if modality_key in file_predictions[filename]['probabilities']:
                    file_predictions[filename]['probabilities'][modality_key] = np.array(
                        file_predictions[filename]['probabilities'][modality_key])
                    file_predictions[filename]['predictions'][modality_key] = np.array(
                        file_predictions[filename]['predictions'][modality_key])
        
        print(f"Processed {len(file_predictions)} unique files")
        return dict(file_predictions)
    
    def _extract_ground_truth_events(self, file_predictions: Dict) -> Dict[str, SeizureEvent]:
        """Extract ground truth seizure events from each file"""
        print("📋 Extracting ground truth seizure events...")
        
        ground_truth_events = {}
        
        for filename, data in file_predictions.items():
            gt = data['ground_truth']
            
            # Find seizure onset (first window with label 1)
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
        
        print(f"Found {len(ground_truth_events)} files with seizure events")
        total_files = len(file_predictions)
        non_seizure_files = total_files - len(ground_truth_events)
        print(f"Total files: {total_files} (Seizure: {len(ground_truth_events)}, Non-seizure: {non_seizure_files})")
        
        return ground_truth_events
    
    def _extract_predicted_events(self, file_predictions: Dict) -> Dict[str, Dict[str, List[PredictionEvent]]]:
        """Extract predicted seizure events for each modality and file"""
        print("🔍 Extracting predicted seizure events...")
        
        predicted_events = {}  # Changed from defaultdict to regular dict
        
        for filename, data in file_predictions.items():
            predicted_events[filename] = {}  # Initialize file entry
            
            for modality_key in self.modality_config.keys():
                predicted_events[filename][modality_key] = []  # Initialize modality list
                
                if modality_key not in data['probabilities']:
                    continue
                
                probs = data['probabilities'][modality_key]
                preds = data['predictions'][modality_key]
                
                if len(probs) == 0 or len(preds) == 0:
                    continue
                
                # Apply temporal smoothing if specified
                if self.smoothing_window > 1 and len(probs) >= self.smoothing_window:
                    probs = self._apply_temporal_smoothing_1d(probs)
                    preds = (probs >= self.probability_threshold).astype(int)
                
                # Find continuous prediction segments
                prediction_segments = self._find_prediction_segments(preds, probs)
                
                # Convert segments to PredictionEvent objects
                for start_idx, end_idx, max_prob, mean_prob in prediction_segments:
                    # Check minimum duration requirement
                    duration = (end_idx - start_idx + 1) * self.window_duration
                    if duration >= self.min_prediction_duration:
                        
                        start_time = start_idx * self.window_duration
                        end_time = (end_idx + 1) * self.window_duration
                        confidence = max_prob  # or some other confidence measure
                        
                        pred_event = PredictionEvent(
                            file_id=filename,
                            start_window=start_idx,
                            end_window=end_idx,
                            start_time=start_time,
                            end_time=end_time,
                            max_probability=max_prob,
                            mean_probability=mean_prob,
                            confidence_score=confidence
                        )
                        
                        predicted_events[filename][modality_key].append(pred_event)
        
        # Print summary
        for modality_key in self.modality_config.keys():
            total_predictions = sum(len(predicted_events[f][modality_key]) 
                                  for f in predicted_events.keys() 
                                  if modality_key in predicted_events[f])
            files_with_predictions = sum(1 for f in predicted_events.keys() 
                                       if modality_key in predicted_events[f] and len(predicted_events[f][modality_key]) > 0)
            print(f"{self.modality_config[modality_key]['name']}: {total_predictions} predictions across {files_with_predictions} files")
        
        return predicted_events
    
    def _apply_temporal_smoothing_1d(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to 1D probability array"""
        if len(probabilities) < self.smoothing_window:
            return probabilities
        
        # Use valid mode to avoid edge effects
        smoothed = np.convolve(probabilities, 
                              np.ones(self.smoothing_window) / self.smoothing_window, 
                              mode='same')
        return smoothed
    
    def _find_prediction_segments(self, predictions: np.ndarray, probabilities: np.ndarray) -> List[Tuple]:
        """Find continuous segments of positive predictions"""
        segments = []
        in_segment = False
        start_idx = 0
        
        for i, pred in enumerate(predictions):
            if pred == 1 and not in_segment:
                # Start of new segment
                in_segment = True
                start_idx = i
            elif pred == 0 and in_segment:
                # End of current segment
                in_segment = False
                end_idx = i - 1
                
                # Calculate segment statistics
                if start_idx <= end_idx < len(probabilities):
                    segment_probs = probabilities[start_idx:end_idx+1]
                    max_prob = np.max(segment_probs)
                    mean_prob = np.mean(segment_probs)
                    segments.append((start_idx, end_idx, max_prob, mean_prob))
        
        # Handle case where segment continues to end
        if in_segment and start_idx < len(predictions):
            end_idx = len(predictions) - 1
            if start_idx <= end_idx < len(probabilities):
                segment_probs = probabilities[start_idx:end_idx+1]
                max_prob = np.max(segment_probs)
                mean_prob = np.mean(segment_probs)
                segments.append((start_idx, end_idx, max_prob, mean_prob))
        
        return segments
    
    def _evaluate_events_ovlp(self, 
                            ground_truth_events: Dict[str, SeizureEvent],
                            predicted_events: Dict[str, Dict[str, List[PredictionEvent]]]) -> Dict:
        """Evaluate using OVLP (Any Overlap) methodology"""
        print("⚖️ Performing OVLP event-based evaluation...")
        
        evaluation_results = {}  # Changed from defaultdict
        
        # Get all unique filenames
        all_files = set(ground_truth_events.keys())
        all_files.update(predicted_events.keys())
        all_files = sorted(list(all_files))  # Sort for consistent processing
        
        print(f"Evaluating {len(all_files)} total files")
        
        for modality_key in self.modality_config.keys():
            print(f"Evaluating {self.modality_config[modality_key]['name']}...")
            evaluation_results[modality_key] = []  # Initialize as list
            
            for filename in all_files:
                gt_event = ground_truth_events.get(filename)
                pred_events = predicted_events.get(filename, {}).get(modality_key, [])
                
                result = self._evaluate_single_file_ovlp(filename, gt_event, pred_events)
                evaluation_results[modality_key].append(result)
        
        return evaluation_results
    
    def _evaluate_single_file_ovlp(self, 
                                 filename: str,
                                 gt_event: Optional[SeizureEvent],
                                 pred_events: List[PredictionEvent]) -> EventResults:
        """Evaluate a single file using OVLP methodology"""
        
        # Initialize result
        result = EventResults(
            file_id=filename,
            true_positive=False,
            false_positive=False,
            false_negative=False,
            true_negative=False,
            detection_latency=None,
            prediction_event=None,
            ground_truth_event=gt_event
        )
        
        if gt_event is None:
            # No seizure in this file
            if len(pred_events) == 0:
                # Correctly predicted no seizure
                result.true_negative = True
            else:
                # False alarm - predicted seizure when none exists
                result.false_positive = True
                # Take the prediction with highest confidence as the FP event
                result.prediction_event = max(pred_events, key=lambda x: x.confidence_score)
        else:
            # Seizure exists in this file
            # Check if any prediction overlaps with tolerance window
            onset_time = gt_event.onset_time
            tolerance_start = max(0, onset_time - self.detection_tolerance)
            tolerance_end = onset_time + self.detection_tolerance
            
            # Find predictions that overlap with tolerance window
            valid_predictions = []
            for pred_event in pred_events:
                # Check if prediction overlaps with tolerance window
                if (pred_event.start_time <= tolerance_end and 
                    pred_event.end_time >= tolerance_start):
                    valid_predictions.append(pred_event)
            
            if len(valid_predictions) > 0:
                # True positive - seizure correctly detected within tolerance
                result.true_positive = True
                
                # Select the best prediction (earliest detection or highest confidence)
                best_pred = min(valid_predictions, key=lambda x: abs(x.start_time - onset_time))
                result.prediction_event = best_pred
                
                # Calculate detection latency
                detection_time = best_pred.start_time
                result.detection_latency = detection_time - onset_time
                
                # Check for additional false positives
                # Any prediction outside the tolerance window is still a FP
                for pred_event in pred_events:
                    if (pred_event.start_time > tolerance_end or 
                        pred_event.end_time < tolerance_start):
                        result.false_positive = True
                        break
                        
            else:
                # False negative - seizure missed
                result.false_negative = True
                
                # Check if there are any predictions (would be FP as well)
                if len(pred_events) > 0:
                    result.false_positive = True
                    result.prediction_event = max(pred_events, key=lambda x: x.confidence_score)
        
        return result
    
    def _compute_clinical_metrics(self, 
                                evaluation_results: Dict,
                                file_predictions: Dict) -> Dict[str, ClinicalMetrics]:
        """Compute clinical metrics for each modality"""
        print("📊 Computing clinical metrics...")
        
        clinical_metrics = {}
        
        for modality_key, results_list in evaluation_results.items():
            
            # Count event-level outcomes
            tp = sum(1 for r in results_list if r.true_positive)
            fp = sum(1 for r in results_list if r.false_positive)
            fn = sum(1 for r in results_list if r.false_negative)
            tn = sum(1 for r in results_list if r.true_negative)
            
            total_files = len(results_list)
            seizure_files = tp + fn
            non_seizure_files = tn + fp
            
            # Event-level metrics
            event_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            event_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            event_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            event_accuracy = (tp + tn) / total_files if total_files > 0 else 0.0
            event_f1 = (2 * event_precision * event_sensitivity / 
                       (event_precision + event_sensitivity) 
                       if (event_precision + event_sensitivity) > 0 else 0.0)
            
            # Detection latency metrics
            latencies = [r.detection_latency for r in results_list 
                        if r.detection_latency is not None]
            mean_latency = np.mean(latencies) if latencies else 0.0
            median_latency = np.median(latencies) if latencies else 0.0
            
            # False positive rate per hour (estimate)
            total_duration_hours = sum(len(data['ground_truth']) * self.window_duration / 3600
                                     for data in file_predictions.values())
            fp_per_hour = fp / total_duration_hours if total_duration_hours > 0 else 0.0
            
            # Window-level metrics for comparison
            all_window_gt = []
            all_window_pred = []
            
            for filename, data in file_predictions.items():
                if modality_key in data['predictions']:
                    all_window_gt.extend(data['ground_truth'])
                    all_window_pred.extend(data['predictions'][modality_key])
            
            if len(all_window_gt) > 0 and len(all_window_pred) > 0:
                try:
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
            
            clinical_metrics[modality_key] = ClinicalMetrics(
                event_sensitivity=event_sensitivity,
                event_specificity=event_specificity,
                event_precision=event_precision,
                event_f1_score=event_f1,
                event_accuracy=event_accuracy,
                mean_detection_latency=mean_latency,
                median_detection_latency=median_latency,
                false_positive_rate_per_hour=fp_per_hour,
                total_files=total_files,
                seizure_files=seizure_files,
                non_seizure_files=non_seizure_files,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                true_negatives=tn,
                window_sensitivity=window_sensitivity,
                window_specificity=window_specificity,
                window_f1_score=window_f1
            )
            
            print(f"{self.modality_config[modality_key]['name']} Event Metrics:")
            print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
            print(f"  Sensitivity: {event_sensitivity:.4f}, Specificity: {event_specificity:.4f}")
            print(f"  F1-Score: {event_f1:.4f}")
        
        return clinical_metrics
    
    def _generate_clinical_reports(self, 
                                 clinical_metrics: Dict[str, ClinicalMetrics],
                                 evaluation_results: Dict,
                                 save_dir: str):
        """Generate clinical evaluation reports"""
        print("📝 Generating clinical reports...")
        
        # Summary report
        with open(f"{save_dir}/clinical_evaluation_summary.txt", 'w') as f:
            f.write("="*80 + "\n")
            f.write("CLINICAL SEIZURE DETECTION EVALUATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("Evaluation Parameters:\n")
            f.write(f"- Detection Tolerance: ±{self.detection_tolerance} seconds\n")
            f.write(f"- Minimum Prediction Duration: {self.min_prediction_duration} seconds\n")
            f.write(f"- Probability Threshold: {self.probability_threshold}\n")
            f.write(f"- Window Duration: {self.window_duration} seconds\n\n")
            
            f.write("EVENT-LEVEL PERFORMANCE (OVLP Methodology):\n")
            f.write("-"*60 + "\n")
            
            for modality_key, metrics in clinical_metrics.items():
                modality_name = self.modality_config[modality_key]['name']
                f.write(f"\n{modality_name}:\n")
                f.write(f"  Event Sensitivity:     {metrics.event_sensitivity:.4f}\n")
                f.write(f"  Event Specificity:     {metrics.event_specificity:.4f}\n")
                f.write(f"  Event Precision:       {metrics.event_precision:.4f}\n")
                f.write(f"  Event F1-Score:        {metrics.event_f1_score:.4f}\n")
                f.write(f"  Event Accuracy:        {metrics.event_accuracy:.4f}\n")
                f.write(f"  Detection Latency:     {metrics.mean_detection_latency:.2f}s (mean)\n")
                f.write(f"  FP Rate per Hour:      {metrics.false_positive_rate_per_hour:.2f}\n")
                f.write(f"  Files: {metrics.total_files} (Seizure: {metrics.seizure_files}, Non-seizure: {metrics.non_seizure_files})\n")
                f.write(f"  TP: {metrics.true_positives}, FP: {metrics.false_positives}, FN: {metrics.false_negatives}, TN: {metrics.true_negatives}\n")
                
                f.write(f"\n  Window-level comparison:\n")
                f.write(f"    Window Sensitivity:  {metrics.window_sensitivity:.4f}\n")
                f.write(f"    Window Specificity:  {metrics.window_specificity:.4f}\n")
                f.write(f"    Window F1-Score:     {metrics.window_f1_score:.4f}\n")
        
        # CSV report
        metrics_df = []
        for modality_key, metrics in clinical_metrics.items():
            metrics_df.append({
                'Modality': self.modality_config[modality_key]['name'],
                'Event_Sensitivity': metrics.event_sensitivity,
                'Event_Specificity': metrics.event_specificity,
                'Event_Precision': metrics.event_precision,
                'Event_F1_Score': metrics.event_f1_score,
                'Event_Accuracy': metrics.event_accuracy,
                'Mean_Detection_Latency': metrics.mean_detection_latency,
                'FP_Rate_per_Hour': metrics.false_positive_rate_per_hour,
                'True_Positives': metrics.true_positives,
                'False_Positives': metrics.false_positives,
                'False_Negatives': metrics.false_negatives,
                'True_Negatives': metrics.true_negatives,
                'Window_Sensitivity': metrics.window_sensitivity,
                'Window_Specificity': metrics.window_specificity,
                'Window_F1_Score': metrics.window_f1_score
            })
        
        pd.DataFrame(metrics_df).to_csv(f"{save_dir}/clinical_metrics.csv", index=False)
        
        print(f"📋 Clinical reports saved to {save_dir}/")

def run_clinical_evaluation(model, dataloader, device, save_dir="clinical_evaluation_results/"):
    """
    Main function to run clinical evaluation with OVLP methodology
    
    Usage:
        results = run_clinical_evaluation(model, val_loader, device)
    """
    evaluator = ClinicalSeizureEvaluator(
        window_duration=10.0,
        detection_tolerance=60.0,  # ±60 seconds
        min_prediction_duration=5.0,
        probability_threshold=0.5,
        smoothing_window=3
    )
    
    return evaluator.evaluate_clinical_performance(model, dataloader, device, save_dir)