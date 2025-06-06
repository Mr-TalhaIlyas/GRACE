import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from tools.clinical_evaluation import run_clinical_evaluation
import torch
import torch.nn.functional as F
from data.utils import video_transform
from tqdm import tqdm

def run_comparison_evaluation(model, dataloader, device, save_dir="comparison_results/"):
    """
    Run both window-level and event-level evaluations for comparison
    """
    print("🔄 Running Comprehensive Evaluation Comparison...")
    
    # 1. Your original window-level evaluation
    print("\n1️⃣ Running Window-Level Evaluation...")
    window_results = run_window_level_evaluation(model, dataloader, device)
    
    # 2. Clinical event-level evaluation
    print("\n2️⃣ Running Clinical Event-Level Evaluation...")
    clinical_results = run_clinical_evaluation(model, dataloader, device, save_dir)
    
    # 3. Generate comparison report
    print("\n📊 Generating Comparison Report...")
    generate_comparison_report(window_results, clinical_results, save_dir)
    
    return {
        'window_level': window_results,
        'clinical_level': clinical_results
    }

def run_window_level_evaluation(model, dataloader, device):
    """Your original evaluation approach"""
    
    model.eval()
    
    # Key map for modalities
    key_map = {
        'flow_outputs': 'flow',
        'ecg_outputs': 'ecg',
        'joint_pose_outputs': 'pose',
        'fusion_outputs': 'fusion',
    }
    
    # Prepare containers
    all_preds = {mod: [] for mod in key_map.values()}
    all_probs = {mod: [] for mod in key_map.values()}
    all_labels = []
    
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Window Evaluation"):
            # Data preprocessing
            frames = video_transform(batch['frames']).to(device, non_blocking=True)
            body = batch['body'].permute(0, 4, 2, 3, 1).float().to(device, non_blocking=True)
            face = batch['face'].permute(0, 4, 2, 3, 1).float().to(device, non_blocking=True)
            rh = batch['rh'].permute(0, 4, 2, 3, 1).float().to(device, non_blocking=True)
            lh = batch['lh'].permute(0, 4, 2, 3, 1).float().to(device, non_blocking=True)
            hrv = batch['hrv'].to(torch.float).to(device, non_blocking=True)
            target = torch.argmax(batch['super_lbls'], dim=1).long().to(device, non_blocking=True)

            outputs = model(frames, body, face, rh, lh, hrv)

            # For each modality: compute preds, probs; store them
            for out_key, modality in key_map.items():
                logits = outputs[out_key]
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds[modality].append(preds.cpu())
                all_probs[modality].append(probs.cpu())

            all_labels.append(target.cpu())

    # Concatenate results
    labels_tensor = torch.cat(all_labels, dim=0)
    labels_np = labels_tensor.numpy()

    window_results = {}
    avg_type = 'weighted'

    for modality in key_map.values():
        preds_tensor = torch.cat(all_preds[modality], dim=0)
        probs_tensor = torch.cat(all_probs[modality], dim=0)
        
        y_true = labels_np
        y_pred = preds_tensor.numpy()
        y_prob = probs_tensor.numpy()[:, 1]  # Probability of seizure class

        # Compute metrics
        precision = precision_score(y_true, y_pred, average=avg_type)
        sensitivity = recall_score(y_true, y_pred, average=avg_type)
        f1 = f1_score(y_true, y_pred, average=avg_type)
        auroc = roc_auc_score(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        window_results[modality] = {
            'precision': precision,
            'sensitivity': sensitivity,
            'f1': f1,
            'auroc': auroc,
            'average_precision': avg_precision,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'accuracy': (tp + tn) / (tp + tn + fp + fn)
        }

    return window_results

def generate_comparison_report(window_results, clinical_results, save_dir):
    """Generate a detailed comparison report"""
    
    with open(f"{save_dir}/evaluation_comparison.txt", 'w') as f:
        f.write("="*100 + "\n")
        f.write("EVALUATION METHODOLOGY COMPARISON\n")
        f.write("="*100 + "\n\n")
        
        f.write("METHODOLOGY DIFFERENCES:\n")
        f.write("-"*50 + "\n")
        f.write("Window-Level Evaluation:\n")
        f.write("- Each window treated as independent sample\n")
        f.write("- Standard machine learning metrics\n")
        f.write("- Higher sensitivity to class imbalance\n")
        f.write("- Good for technical performance assessment\n\n")
        
        f.write("Clinical Event-Level Evaluation (OVLP):\n")
        f.write("- One prediction per file (seizure event)\n")
        f.write("- ±60 second detection tolerance\n")
        f.write("- Clinically relevant metrics\n")
        f.write("- Accounts for early detection value\n\n")
        
        f.write("PERFORMANCE COMPARISON:\n")
        f.write("="*100 + "\n")
        
        # Compare fusion model results
        if 'fusion' in window_results and 'fusion_outputs' in clinical_results['clinical_metrics']:
            f.write("\nFUSION MODEL COMPARISON:\n")
            f.write("-"*30 + "\n")
            
            window_fusion = window_results['fusion']
            clinical_fusion = clinical_results['clinical_metrics']['fusion_outputs']
            
            f.write(f"{'Metric':<25} {'Window-Level':<15} {'Clinical OVLP':<15} {'Difference':<15}\n")
            f.write("-"*70 + "\n")
            
            metrics_to_compare = [
                ('Sensitivity', 'sensitivity', 'event_sensitivity'),
                ('Specificity', None, 'event_specificity'),  # Window doesn't have direct equivalent
                ('Precision', 'precision', 'event_precision'),
                ('F1-Score', 'f1', 'event_f1_score'),
                ('AUROC', 'auroc', None),  # Clinical doesn't compute AUROC at event level
            ]
            
            for metric_name, window_key, clinical_key in metrics_to_compare:
                if window_key and clinical_key:
                    window_val = window_fusion[window_key]
                    clinical_val = getattr(clinical_fusion, clinical_key)
                    diff = clinical_val - window_val
                    f.write(f"{metric_name:<25} {window_val:<15.4f} {clinical_val:<15.4f} {diff:<15.4f}\n")
                elif window_key:
                    window_val = window_fusion[window_key]
                    f.write(f"{metric_name:<25} {window_val:<15.4f} {'N/A':<15} {'N/A':<15}\n")
                elif clinical_key:
                    clinical_val = getattr(clinical_fusion, clinical_key)
                    f.write(f"{metric_name:<25} {'N/A':<15} {clinical_val:<15.4f} {'N/A':<15}\n")
            
            f.write("\nRAW COUNTS COMPARISON:\n")
            f.write(f"Window-Level  - TP: {window_fusion['tp']:4d}, FP: {window_fusion['fp']:4d}, FN: {window_fusion['fn']:4d}, TN: {window_fusion['tn']:4d}\n")
            f.write(f"Clinical OVLP - TP: {clinical_fusion.true_positives:4d}, FP: {clinical_fusion.false_positives:4d}, FN: {clinical_fusion.false_negatives:4d}, TN: {clinical_fusion.true_negatives:4d}\n")
            
            f.write(f"\nCLINICAL-SPECIFIC METRICS:\n")
            f.write(f"- Mean Detection Latency: {clinical_fusion.mean_detection_latency:.2f} seconds\n")
            f.write(f"- False Positives per Hour: {clinical_fusion.false_positive_rate_per_hour:.2f}\n")
            f.write(f"- Total Files Evaluated: {clinical_fusion.total_files}\n")
            f.write(f"- Files with Seizures: {clinical_fusion.seizure_files}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("- Higher window-level metrics may indicate good technical performance\n")
        f.write("- Lower clinical metrics may reflect the stringent OVLP requirements\n")
        f.write("- Clinical metrics are more relevant for real-world deployment\n")
        f.write("- Detection latency indicates how early seizures are detected\n")

    print(f"📊 Comparison report saved to {save_dir}/evaluation_comparison.txt")