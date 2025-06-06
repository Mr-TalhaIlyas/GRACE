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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, model, optimizer, scaler=None, cfg=config, class_weights=None):
        print('TRAINER INIT-v7')
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.scaler = scaler or GradScaler()
        self.cfg = cfg
        self.class_weights = class_weights.to(DEVICE) if class_weights is not None else None

        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        self.fusion_loss_fn_smooth = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
        self.aux_weight = cfg.get('auxiliary_loss_weight', 0.4)
        self.main_weight = cfg.get('main_loss_weight', 1.0)
        self.consistent_loss_weight = cfg.get('consistent_loss_weight', 0.2)
        self.temperature = cfg.get('temperature', 1.0)
        
        self.current_epoch = 0
        self.performance_history = {'flow': [], 'ecg': [], 'pose': [], 'fusion': []}
        self.metric_keys_for_history = ['flow', 'ecg', 'pose', 'fusion']

        self.fusion_warmup_epochs = cfg.get('fusion_warmup_epochs', 5)
        self.inter_group_consistency_weight = cfg.get('inter_group_consistency_weight', 0.15)
        self.bio_signal_keys = cfg.get('bio_signal_keys', ['ecg_outputs'])
        self.visual_signal_keys = cfg.get('visual_signal_keys', ['flow_outputs', 'joint_pose_outputs'])
        
        self.metric_names = ['flow', 'ecg', 'pose', 'fusion']
        self.metrics_train = nn.ModuleDict({
            name: F1Score(task="multiclass", num_classes=cfg.get('num_classes', 2), average='macro').to(DEVICE)
            for name in self.metric_names
        })
        self.output_to_metric_key_map = {
            'flow_outputs': 'flow',
            'ecg_outputs': 'ecg',
            'joint_pose_outputs': 'pose',
            'body_outputs': 'pose', # Individual pose parts map to 'pose' for freezing status
            'face_outputs': 'pose',
            'rhand_outputs': 'pose',
            'lhand_outputs': 'pose',
            'fusion_outputs': 'fusion',
        }

        # Initialize frozen status - this will be updated from main_train.py
        # Expected keys: 'flow', 'ecg', 'pose'
        self.encoders_frozen_status = {name: False for name in self.metric_names if name != 'fusion'}
        self.frozen_feature_dropout_rate = self.cfg.get('frozen_feature_dropout_rate', 0.2) # e.g., 0.1 or 0.2

    def update_frozen_status(self, new_status: dict):
        """Updates the status of frozen encoders and informs the model."""
        for key in self.encoders_frozen_status: # Iterate over expected keys
            if key in new_status:
                self.encoders_frozen_status[key] = new_status[key]
        
        # Propagate this status to the model if it needs to apply dropout
        if hasattr(self.model, 'update_frozen_status'):
            # Pass the overall dropout rate; model decides if/how to use it based on its training state
            current_dropout_rate = self.frozen_feature_dropout_rate if self.model.training else 0.0
            self.model.update_frozen_status(self.encoders_frozen_status, current_dropout_rate)
        # print(f"[Trainer] Updated frozen status: {self.encoders_frozen_status}")


    def _compute_loss(self, outputs, target):
        total_loss = torch.tensor(0.0, device=target.device)
        batch_size = target.size(0)
        loss_dict = {}

        individual_ce_losses = {}
        all_logits = {} 

        for output_key, logits in outputs.items():
            ce_loss = self.fusion_loss_fn_smooth(logits, target)
            individual_ce_losses[output_key] = ce_loss
            all_logits[output_key] = logits
            loss_dict[f'ce_loss_{output_key}'] = ce_loss.item()

        # Check if the 'flow' encoder is frozen
        is_flow_frozen = self.encoders_frozen_status.get('flow', False)

        for output_key, ce_loss in individual_ce_losses.items():
            original_weight = 0.0
            if output_key == 'fusion_outputs':
                original_weight = self.cfg['main_loss_weight']
            elif output_key == 'joint_pose_outputs':
                original_weight = self.cfg['joint_pose_weight']
            elif output_key in ['body_outputs', 'face_outputs', 'rhand_outputs', 'lhand_outputs']:
                original_weight = self.cfg['pose_fhb_weight']
            elif output_key in ['flow_outputs', 'ecg_outputs']:
                original_weight = self.cfg['auxiliary_loss_weight']
            
            current_weight = original_weight
            modality_metric_key = self.output_to_metric_key_map.get(output_key)

            if modality_metric_key and modality_metric_key != 'fusion': # Not fusion output
                is_current_encoder_frozen = self.encoders_frozen_status.get(modality_metric_key, False)
                
                if is_current_encoder_frozen:
                    # Optional: Reduce weight if encoder is frozen (params won't update, but good for clarity)
                    current_weight *= self.cfg.get('frozen_encoder_ce_loss_factor', 0.1) 
                elif is_flow_frozen and modality_metric_key in ['ecg', 'pose']: 
                    # If FLOW is frozen, and current is ECG or POSE (active ones)
                    current_weight *= self.cfg.get('active_encoder_ce_boost_factor', 1.5) # Boost active
            
            total_loss += current_weight * ce_loss
            loss_dict[f'weighted_ce_{output_key}'] = (current_weight * ce_loss).item()


        neg_margin_total_weight = self.cfg.get('neg_margin_weight', 0.0)
        if neg_margin_total_weight > 0 and self.current_epoch > self.cfg.get('consistency_warmup_epochs', 0):
            base_margin = self.cfg.get('neg_margin', 0.2)
            neg_mask = (target == 0)
            min_neg_samples = batch_size * self.cfg.get('min_neg_sample_ratio_for_margin', 0.1)
            
            if not neg_mask.any() or neg_mask.sum() < min_neg_samples:
                pass
            else:
                lecg_n, lflow_n, lpose_n = None, None, None
                if 'ecg_outputs' in all_logits: lecg_n = all_logits['ecg_outputs'][neg_mask, 1]
                if 'flow_outputs' in all_logits: lflow_n = all_logits['flow_outputs'][neg_mask, 1]
                if 'joint_pose_outputs' in all_logits: lpose_n = all_logits['joint_pose_outputs'][neg_mask, 1]

                current_margin_loss = torch.tensor(0.0, device=target.device)
                num_margin_losses_applied = 0
                
                # JSD Loss (Inter-group consistency)
                jsd_loss_weight_factor = self.cfg.get('neg_jsd_factor', 0.0) 
                if jsd_loss_weight_factor > 0 and neg_mask.any():
                    bio_logits_neg_list = []
                    if 'ecg_outputs' in all_logits and all_logits['ecg_outputs'][neg_mask].numel() > 0:
                        if not self.encoders_frozen_status.get('ecg', False): # Only include if not frozen
                             bio_logits_neg_list.append(all_logits['ecg_outputs'][neg_mask])

                    visual_logits_neg_list = []
                    if 'flow_outputs' in all_logits and all_logits['flow_outputs'][neg_mask].numel() > 0:
                        if not self.encoders_frozen_status.get('flow', False): # Only include if not frozen
                            visual_logits_neg_list.append(all_logits['flow_outputs'][neg_mask])
                    if 'joint_pose_outputs' in all_logits and all_logits['joint_pose_outputs'][neg_mask].numel() > 0:
                        if not self.encoders_frozen_status.get('pose', False): # Only include if not frozen
                            visual_logits_neg_list.append(all_logits['joint_pose_outputs'][neg_mask])

                    if bio_logits_neg_list and visual_logits_neg_list:
                        avg_bio_probs_neg = torch.stack(
                            [F.softmax(lgts / self.temperature, dim=1) for lgts in bio_logits_neg_list]
                        ).mean(dim=0)
                        avg_visual_probs_neg = torch.stack(
                            [F.softmax(lgts / self.temperature, dim=1) for lgts in visual_logits_neg_list]
                        ).mean(dim=0)

                        log_avg_bio_probs_neg = F.log_softmax(avg_bio_probs_neg, dim=1)
                        log_avg_visual_probs_neg = F.log_softmax(avg_visual_probs_neg, dim=1)
                        m_probs_neg = 0.5 * (avg_bio_probs_neg + avg_visual_probs_neg)
                        
                        jsd = 0.5 * (F.kl_div(log_avg_bio_probs_neg, m_probs_neg, reduction='batchmean', log_target=False) + \
                                     F.kl_div(log_avg_visual_probs_neg, m_probs_neg, reduction='batchmean', log_target=False))
                        
                        current_jsd_factor = jsd_loss_weight_factor
                        # If flow is frozen and it's part of visual group, reduce JSD impact
                        if is_flow_frozen and 'flow_outputs' in self.visual_signal_keys:
                             current_jsd_factor *= self.cfg.get('consistency_vs_frozen_flow_factor', 0.1)

                        if not torch.isnan(jsd) and jsd > 0:
                            current_margin_loss += current_jsd_factor * jsd
                            loss_dict['margin_neg_jsd_bio_vs_visual'] = jsd.item()
                            num_margin_losses_applied +=1
                
                # Margin Loss (Intra-group consistency)
                def _add_margin_loss(logit_spiking, logit_calm, factor_key, loss_name_suffix, mod_spiking_name, mod_calm_name):
                    nonlocal current_margin_loss, num_margin_losses_applied
                    if logit_spiking is not None and logit_calm is not None and logit_spiking.numel() > 0 and logit_calm.numel() > 0:
                        
                        current_pair_factor = self.cfg.get(factor_key, 1.0)
                        
                        # If flow is frozen and involved in this pair, reduce the factor
                        is_mod_spiking_frozen = self.encoders_frozen_status.get(mod_spiking_name, False)
                        is_mod_calm_frozen = self.encoders_frozen_status.get(mod_calm_name, False)

                        if is_flow_frozen and (mod_spiking_name == 'flow' or mod_calm_name == 'flow'):
                            current_pair_factor *= self.cfg.get('consistency_vs_frozen_flow_factor', 0.1)
                        # Optional: General handling if any part of pair is frozen (even if not flow)
                        # elif is_mod_spiking_frozen or is_mod_calm_frozen:
                        #    current_pair_factor *= self.cfg.get('consistency_pair_vs_any_frozen_factor', 0.5)


                        if current_pair_factor > 0:
                            diff = logit_spiking - logit_calm + base_margin
                            Lm = diff.clamp(min=0).mean()
                            if not torch.isnan(Lm) and Lm > 0:
                                current_margin_loss += current_pair_factor * Lm
                                loss_dict[f'margin_neg_{loss_name_suffix}'] = Lm.item()
                                num_margin_losses_applied +=1

                _add_margin_loss(lecg_n, lflow_n, 'neg_margin_factor_ecg_vs_flow', 'ecg_vs_flow', 'ecg', 'flow')
                _add_margin_loss(lflow_n, lecg_n, 'neg_margin_factor_flow_vs_ecg', 'flow_vs_ecg', 'flow', 'ecg')
                _add_margin_loss(lecg_n, lpose_n, 'neg_margin_factor_ecg_vs_pose', 'ecg_vs_pose', 'ecg', 'pose')
                _add_margin_loss(lpose_n, lecg_n, 'neg_margin_factor_pose_vs_ecg', 'pose_vs_ecg', 'pose', 'ecg')
                # _add_margin_loss(lflow_n, lpose_n, 'neg_margin_factor_flow_vs_pose', 'flow_vs_pose', 'flow', 'pose')
                # _add_margin_loss(lpose_n, lflow_n, 'neg_margin_factor_pose_vs_flow', 'pose_vs_flow', 'pose', 'flow')

                if num_margin_losses_applied > 0:
                     total_loss += neg_margin_total_weight * (current_margin_loss / num_margin_losses_applied) # Average effect
                     loss_dict['margin_neg_consistency_sum_factored_Lm'] = current_margin_loss.item()
                                
        return total_loss, loss_dict

    def train_epoch(self, loader, epoch, scheduler=None):
        self.model.train()
        # Crucial: Ensure model's frozen status and dropout rate are set for training
        if hasattr(self.model, 'update_frozen_status'):
            self.model.update_frozen_status(self.encoders_frozen_status, self.frozen_feature_dropout_rate if self.model.training else 0.0)

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
            self.scaler.step(self.optimizer)
            self.scaler.update()

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
                    f"Iter. {step+1}/{num_batches}",
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
            # break # testing
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
                        f"Iter. {step+1}/{num_batches}"
                    ]
                    for name, acc_val in accs_computed.items():
                        log_message_parts.append(f"{name.capitalize()}F1: {acc_val:.4f}")
                    print(" | ".join(log_message_parts))
                # break

            final_accs = {k: m.compute().item() for k,m in self.metrics_val.items()}
        # print(f"--- Validation Epoch {epoch+1} Complete ---")
        return final_accs

