# Replace the clinical evaluation section with:

from tools.clinical_eval import calculate_all_modality_metrics
from tools.viz_module import ClinicalVisualizer, run_complete_clinical_evaluation
from tools.evaluation_interface import convert_evaluator_results_to_file_format, run_inference_by_file_direct

# Option 1: Convert your existing results
print("🔄 Converting existing results to file format...")
file_predictions = convert_evaluator_results_to_file_format(window_results, val_loader)

# Option 2: Alternative - run inference directly organized by file (more accurate)
# print("🔄 Running inference directly organized by file...")
# file_predictions = run_inference_by_file_direct(model, val_loader, DEVICE)

# Run complete clinical evaluation
clinical_metrics = run_complete_clinical_evaluation(
    file_predictions=file_predictions,
    save_dir="/home/user01/Data/npj/scripts/clinical_evaluation_results/"
)

# Print summary
print("\n🏥 Clinical Evaluation Summary:")
print("="*60)
for modality_key, metrics in clinical_metrics.items():
    modality_name = {'fusion_outputs': 'Fusion', 'ecg_outputs': 'ECG', 
                     'flow_outputs': 'Flow', 'joint_pose_outputs': 'Pose'}[modality_key]
    print(f"\n{modality_name}:")
    print(f"  Event Sensitivity: {metrics.event_sensitivity:.4f}")
    print(f"  Event Specificity: {metrics.event_specificity:.4f}")
    print(f"  Pre-seizure Specificity: {metrics.pre_seizure_specificity:.4f}")
    print(f"  Event F1-Score: {metrics.event_f1_score:.4f}")
    print(f"  Detection Latency: {metrics.mean_detection_latency:.2f}s")
    print(f"  FP Rate/Hour: {metrics.false_positive_rate_per_hour:.2f}")
    print(f"  Outcomes: TP={metrics.true_positives}, FN={metrics.false_negatives}")
    print(f"  Pre-seizure: TN={metrics.pre_seizure_true_negatives}, FP={metrics.pre_seizure_false_positives}")
    print(f"  Files: TN={metrics.file_true_negatives}, FP={metrics.file_false_positives}")