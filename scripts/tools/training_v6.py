import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import Accuracy, F1Score, AUROC, Recall
torch.autograd.set_detect_anomaly(False)
from configs.config import config
from data.utils import video_transform
from models.utils.loss import get_loss   # or use nn.CrossEntropyLoss
from torch_ema import ExponentialMovingAverage

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, model, optimizer, scaler=None, cfg=config, class_weights=None):
        print('TRAINER INIT-v6')
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.scaler = scaler or GradScaler()
        self.cfg = cfg
        self.class_weights = class_weights.to(DEVICE) if class_weights is not None else None

        self.ema_model = ExponentialMovingAverage(self.model.parameters(), decay=0.999)
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights) # Or your custom get_loss
        self.fusion_loss_fn_smooth = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
        self.aux_weight = cfg.get('auxiliary_loss_weight', 0.4)
        self.main_weight = cfg.get('main_loss_weight', 1.0)
        self.consistent_loss_weight = cfg.get('consistent_loss_weight', 0.2)
        self.temperature = cfg.get('temperature', 1.0)

        # --- Enhancements for Fusion Performance ---
        self.current_epoch = 0
        # Store F1 scores for adaptive weighting
        self.performance_history = {
            'flow': [], 'ecg': [], 'pose': [], 'fusion': []
        }
        self.metric_keys_for_history = ['flow', 'ecg', 'pose', 'fusion'] # Match keys in train_accs

        # Phased learning parameters
        self.fusion_warmup_epochs = cfg.get('fusion_warmup_epochs', 5) # Epochs before fusion leads consistency
        #
        self.inter_group_consistency_weight = cfg.get('inter_group_consistency_weight', 0.15)
        self.bio_signal_keys = cfg.get('bio_signal_keys', ['ecg_outputs']) 
        self.visual_signal_keys = cfg.get('visual_signal_keys', ['flow_outputs', 'joint_pose_outputs'])
        
        # Metrics (ensure keys match `key_map` in train_epoch)
        self.metric_names = ['flow', 'ecg', 'pose', 'fusion'] # Used for F1Score objects
        self.metrics_train = nn.ModuleDict({
            name: F1Score(task="multiclass", num_classes=cfg.get('num_classes', 2), average='macro').to(DEVICE)
            for name in self.metric_names
        })
        # Key mapping from model output keys to metric/history keys
        self.output_to_metric_key_map = {
            'flow_outputs': 'flow',
            'ecg_outputs': 'ecg',
            'joint_pose_outputs': 'pose',
            'fusion_outputs': 'fusion',
        }
            
    def _compute_loss(self, outputs, target):
        """Enhanced multi-modal loss computation with dynamic and phased strategies."""
        total_loss = torch.tensor(0.0, device=target.device)
        batch_size = target.size(0)
        loss_dict = {}

        individual_ce_losses = {}
        all_logits = {} # Store logits for consistency loss

        for output_key, logits in outputs.items():
            ce_loss = self.fusion_loss_fn_smooth(logits, target)
            # ce_loss = self.loss_fn(logits, target) 
            individual_ce_losses[output_key] = ce_loss
            all_logits[output_key] = logits
            loss_dict[f'ce_loss_{output_key}'] = ce_loss.item()

        for output_key, ce_loss in individual_ce_losses.items():
            # weight = self.main_weight if output_key == 'fusion_outputs' else self.aux_weight
            weight = 0.0 # Default weight
            if output_key == 'fusion_outputs':
                weight = self.cfg['main_loss_weight']
            elif output_key == 'joint_pose_outputs':
                weight = self.cfg['joint_pose_weight'] 
            elif output_key in ['body_outputs', 'face_outputs', 'rhand_outputs', 'lhand_outputs']:
                weight = self.cfg['pose_fhb_weight'] 
            elif output_key in ['flow_outputs', 'ecg_outputs']:
                weight = self.cfg['auxiliary_loss_weight'] 
            total_loss += weight * ce_loss
            
        # 5. Negative-frame margin consistency
        neg_margin_total_weight = self.cfg.get('neg_margin_weight', 0.0) # Overall weight for this block
        if neg_margin_total_weight > 0:
            base_margin = self.cfg.get('neg_margin', 0.2)
            
            neg_mask = (target == 0)
            # Skip if no negative samples in batch -or -
            # only apply margin when you have ≥ K negative samples
            min_neg_samples = batch_size * 0.25 
            if not neg_mask.any() or neg_mask.sum() < min_neg_samples: 
                return total_loss, loss_dict

            # Get logits for positive class (index 1) for negative samples
            lecg_n, lflow_n, lpose_n = None, None, None

            if 'ecg_outputs' in all_logits:
                lecg_n = all_logits['ecg_outputs'][neg_mask, 1]
            if 'flow_outputs' in all_logits:
                lflow_n = all_logits['flow_outputs'][neg_mask, 1]
            if 'joint_pose_outputs' in all_logits:
                lpose_n = all_logits['joint_pose_outputs'][neg_mask, 1]

            current_margin_loss = torch.tensor(0.0, device=target.device)
            num_margin_losses_applied = 0

            # --- (Optional) Add JSD for inter-group agreement on negative samples ---
            jsd_loss_weight_factor = self.cfg.get('neg_jsd_factor', 1.0) # Add to config if using
            if jsd_loss_weight_factor > 0 and neg_mask.any():
                bio_logits_neg = []
                if 'ecg_outputs' in all_logits and all_logits['ecg_outputs'][neg_mask].numel() > 0:
                    bio_logits_neg.append(all_logits['ecg_outputs'][neg_mask])
                # Add other bio_signals if any (e.g., hrv_outputs)

                visual_logits_neg = []
                if 'flow_outputs' in all_logits and all_logits['flow_outputs'][neg_mask].numel() > 0:
                    visual_logits_neg.append(all_logits['flow_outputs'][neg_mask])
                if 'joint_pose_outputs' in all_logits and all_logits['joint_pose_outputs'][neg_mask].numel() > 0:
                    visual_logits_neg.append(all_logits['joint_pose_outputs'][neg_mask])

                if bio_logits_neg and visual_logits_neg:
                    # Aggregate probs for bio group on negative samples
                    avg_bio_probs_neg = torch.stack(
                        [F.softmax(lgts / self.temperature, dim=1) for lgts in bio_logits_neg]
                    ).mean(dim=0)

                    # Aggregate probs for visual group on negative samples
                    avg_visual_probs_neg = torch.stack(
                        [F.softmax(lgts / self.temperature, dim=1) for lgts in visual_logits_neg]
                    ).mean(dim=0)

                    # Calculate JSD
                    log_avg_bio_probs_neg = F.log_softmax(avg_bio_probs_neg, dim=1) # Use already temp-scaled probs
                    log_avg_visual_probs_neg = F.log_softmax(avg_visual_probs_neg, dim=1)
                    
                    m_probs_neg = 0.5 * (avg_bio_probs_neg + avg_visual_probs_neg)
                    # Clamp m_probs_neg to avoid log(0) if any probability is exactly 0
                    # m_probs_neg_log = (m_probs_neg.clamp(min=1e-7)).log()
                    m_probs_neg = m_probs_neg.clamp(min=1e-7)

                    jsd = 0.5 * (F.kl_div(log_avg_bio_probs_neg, m_probs_neg, reduction='batchmean', log_target=False) + \
                                F.kl_div(log_avg_visual_probs_neg, m_probs_neg, reduction='batchmean', log_target=False))
                    
                    if not torch.isnan(jsd) and jsd > 0:
                        current_margin_loss += jsd_loss_weight_factor * jsd # Add to existing margin loss sum
                        loss_dict['margin_neg_jsd_bio_vs_visual'] = jsd.item()
                        num_margin_losses_applied +=1
            
            
            # Helper function to calculate and accumulate margin loss
            def _add_margin_loss(logit_spiking, logit_calm, factor_key, loss_name_suffix):
                nonlocal current_margin_loss, num_margin_losses_applied
                if logit_spiking is not None and logit_calm is not None and logit_spiking.numel() > 0 and logit_calm.numel() > 0:
                    
                    factor = self.cfg.get(factor_key, 1.0) # Default to 1.0 if factor not in config
                    if factor > 0:
                        diff = logit_spiking - logit_calm + base_margin
                        Lm = diff.clamp(min=0).mean()
                        if not torch.isnan(Lm) and Lm > 0: # Only add if loss is actually incurred
                            current_margin_loss += factor * Lm
                            loss_dict[f'margin_neg_{loss_name_suffix}'] = Lm.item()
                            num_margin_losses_applied +=1


            # --- ECG vs Flow ---
            _add_margin_loss(lecg_n, lflow_n, 'neg_margin_factor_ecg_vs_flow', 'ecg_vs_flow')
            _add_margin_loss(lflow_n, lecg_n, 'neg_margin_factor_flow_vs_ecg', 'flow_vs_ecg')

            # --- ECG vs Pose ---
            _add_margin_loss(lecg_n, lpose_n, 'neg_margin_factor_ecg_vs_pose', 'ecg_vs_pose')
            _add_margin_loss(lpose_n, lecg_n, 'neg_margin_factor_pose_vs_ecg', 'pose_vs_ecg')
            
            # --- Flow vs Pose ---
            # _add_margin_loss(lflow_n, lpose_n, 'neg_margin_factor_flow_vs_pose', 'flow_vs_pose')
            # _add_margin_loss(lpose_n, lflow_n, 'neg_margin_factor_pose_vs_flow', 'pose_vs_flow')

            if num_margin_losses_applied > 0: # Add weighted sum to total loss
                 total_loss += neg_margin_total_weight * (current_margin_loss / num_margin_losses_applied if num_margin_losses_applied > 0 else 0)
                 # Storing the average of the Lm values (before factor) that were non-zero
                 # Or you might want to log the sum `current_margin_loss` directly, scaled by `neg_margin_total_weight`
                 loss_dict['margin_neg_consistency_sum_factored_Lm'] = current_margin_loss.item() if isinstance(current_margin_loss, torch.Tensor) else current_margin_loss
                            
        return total_loss, loss_dict

    def train_epoch(self, loader, epoch, scheduler=None):
        self.model.train()
        num_batches = len(loader)
        self.current_epoch = epoch+1 # Update current epoch for loss calculation
        epoch_total_losses = [] # Store total loss for the epoch average

        for metric in self.metrics_train.values():
            metric.reset()

        # pbar = tqdm(enumerate(loader), total=len(loader),
        #             desc=f"[Train {epoch+1}/{self.cfg['epochs']}]")
        print(f"--- Training Epoch {epoch+1}/{self.cfg['epochs']} ---")
        # for step, batch in pbar:
        for step, batch in enumerate(loader):
            frames = video_transform(batch['frames']).to(DEVICE, non_blocking=True)
            body   = batch['body'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
            face   = batch['face'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
            rh     = batch['rh'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
            lh     = batch['lh'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
            hrv    = batch['hrv'].to(torch.float).to(DEVICE, non_blocking=True) # for inception time
            target = torch.argmax(batch['super_lbls'], dim=1).long().to(DEVICE, non_blocking=True)

            with autocast(enabled=self.scaler.is_enabled()):
                outputs = self.model(frames, body, face, rh, lh, hrv)
                loss, loss_breakdown = self._compute_loss(outputs, target)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient Clipping (before optimizer step, after unscaling)
            self.scaler.unscale_(self.optimizer) # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.ema_model.update()
            if scheduler: # + epoch * len(loader)
                if self.cfg['lr_schedule'] == 'cyclic':
                    scheduler.step()
                else:
                    scheduler(self.optimizer, step, epoch) 
                    
            epoch_total_losses.append(loss.item())

            # Update metrics
            for output_key, metric_key in self.output_to_metric_key_map.items():
                if output_key in outputs:
                    logits = outputs[output_key]
                    preds  = torch.argmax(logits, dim=1)
                    self.metrics_train[metric_key].update(preds, target)
            
            if (step + 1) % 100 == 0 or (step + 1) == num_batches:
                current_lr = self.optimizer.param_groups[0]['lr']
                avg_step_loss = np.mean(epoch_total_losses[-self.cfg.get('log_step_freq', 20):])
                accs_computed = {k: m.compute().item() for k, m in self.metrics_train.items()}
                # pbar_postfix = {**{'loss': avg_step_loss, 'lr': f"{current_lr:.2e}"}, **accs_computed}
                # pbar.set_postfix(pbar_postfix)
                log_message_parts = [
                    # f"Epoch {epoch+1}/{self.cfg['epochs']}",
                    f"Step {step+1}/{num_batches}",
                    f"Loss: {avg_step_loss:.4f}",
                    f"LR: {current_lr:.2e}"
                ]
                for name, acc_val in accs_computed.items():
                    log_message_parts.append(f"{name.capitalize()}F1: {acc_val:.4f}")
                
                # Optionally add loss breakdown to the log message
                # for loss_name, loss_val in loss_breakdown.items():
                #     if 'loss' in loss_name: # or any other condition to select relevant losses
                #         log_message_parts.append(f"{loss_name}: {loss_val:.4f}")
                print(" | ".join(log_message_parts))

        # End of epoch
        avg_epoch_loss = np.mean(epoch_total_losses)
        final_epoch_accs = {k: m.compute().item() for k, m in self.metrics_train.items()}

        # Store performance for adaptive weighting in the next epoch's _compute_loss
        for key in self.metric_keys_for_history:
            if key in final_epoch_accs:
                self.performance_history[key].append(final_epoch_accs[key])
            # else: # Handle missing key if necessary, e.g., by appending a default or previous value
            #     self.performance_history[key].append(0.0 if not self.performance_history[key] else self.performance_history[key][-1])


        return avg_epoch_loss, final_epoch_accs, loss_breakdown


class Evaluator:
    def __init__(self, model, cfg=config):
        self.model = model.to(DEVICE)
        self.cfg   = cfg
        self.metric_names = ['flow','ecg','pose','fusion']
        self.metrics_val = nn.ModuleDict({
            name: F1Score(task="multiclass", num_classes=2, average='macro').to(DEVICE)
            for name in self.metric_names
        })

    def validate(self, loader, epoch):
        num_batches = len(loader)
        self.model.eval()
        for m in self.metrics_val.values():
            m.reset()

        print(f"--- Validating Epoch {epoch+1}/{self.cfg['epochs']} ---")
        with torch.no_grad():
            # pbar = tqdm(enumerate(loader), total=len(loader),
            #             desc=f"[ Valid {epoch+1}/{self.cfg['epochs']}]")
            # for _, batch in pbar:
            for step, batch in enumerate(loader):
                # copy the same data prep as Trainer...
                frames = video_transform(batch['frames']).to(DEVICE, non_blocking=True)
                body   = batch['body'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
                face   = batch['face'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
                rh     = batch['rh'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
                lh     = batch['lh'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
                hrv    = batch['hrv'].to(torch.float).to(DEVICE, non_blocking=True) # for inception time
                target = torch.argmax(batch['super_lbls'], dim=1).long().to(DEVICE, non_blocking=True)

                outputs = self.model(frames, body, face, rh, lh, hrv)

                key_map = {
                    'flow_outputs': 'flow',
                    'ecg_outputs': 'ecg',
                    'joint_pose_outputs': 'pose',
                    'fusion_outputs': 'fusion',
                }
                for out_key, metric_key in key_map.items():
                    preds = torch.argmax(outputs[out_key], dim=1)
                    self.metrics_val[metric_key].update(preds, target)

                # Log progress at specified frequency for validation
                log_freq = 30 # Or a different frequency for validation
                if (step + 1) % log_freq == 0 or (step + 1) == num_batches:
                    accs_computed = {k: m.compute().item() for k, m in self.metrics_val.items()}
                    log_message_parts = [
                        # f"Val. Epoch {epoch+1}",
                        f"Step {step+1}/{num_batches}"
                    ]
                    for name, acc_val in accs_computed.items():
                        log_message_parts.append(f"{name.capitalize()}F1: {acc_val:.4f}")
                    print(" | ".join(log_message_parts))


            final_accs = {k: m.compute().item() for k,m in self.metrics_val.items()}
        # print(f"--- Validation Epoch {epoch+1} Complete ---")
        return final_accs

