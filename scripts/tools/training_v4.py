import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import Accuracy, F1Score, AUROC

from configs.config import config
from data.utils import video_transform
from models.utils.loss import get_loss   # or use nn.CrossEntropyLoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, model, optimizer, scaler=None, cfg=config):
        print('TRAINER INIT-v4')
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.scaler = scaler or GradScaler()
        self.cfg = cfg

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss() # Or your custom get_loss
        self.fusion_loss_fn_smooth = nn.CrossEntropyLoss(label_smoothing=cfg.get('label_smoothing_factor', 0.1))
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
        total_loss = 0.0
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
            weight = self.main_weight if output_key == 'fusion_outputs' else self.aux_weight
            total_loss += weight * ce_loss
            

               # --- 3. Consistency Loss (Phased Direction) ---
        if self.consistent_loss_weight > 0 and 'fusion_outputs' in all_logits:
            fusion_logits = all_logits['fusion_outputs']
            # No temperature
            fusion_probs_log = F.log_softmax(fusion_logits, dim=1)
            fusion_probs = F.softmax(fusion_logits, dim=1) # Needed if fusion is target

            consistency_loss_value = 0.0
            num_consistency_pairs = 0

            modality_logits_list = {
                key: lgts for key, lgts in all_logits.items() if key != 'fusion_outputs'
            }
            
            # Prepare modality probs/log_probs without temperature
            modality_probs_list = {k: F.softmax(v, dim=1) for k,v in modality_logits_list.items()}
            modality_probs_log_list = {k: F.log_softmax(v, dim=1) for k,v in modality_logits_list.items()}


            if self.current_epoch < self.fusion_warmup_epochs:
                # Modalities teach fusion: KL(fusion_probs || modality_probs)
                for key in modality_logits_list.keys():
                    consistency_loss_value += F.kl_div(
                        modality_probs_log_list[key], # input: log_softmax(modality_logits)
                        fusion_probs,                 # target: softmax(fusion_logits)
                        reduction='batchmean',
                        log_target=False 
                    )
                    num_consistency_pairs += 1
            else:
                # Fusion teaches modalities: KL(modality_probs || fusion_probs)
                for key in modality_logits_list.keys():
                    consistency_loss_value += F.kl_div(
                        fusion_probs_log,             # input: log_softmax(fusion_logits)
                        modality_probs_list[key],     # target: softmax(modality_logits)
                        reduction='batchmean',
                        log_target=False
                    )
                    num_consistency_pairs += 1
            
            if num_consistency_pairs > 0:
                consistency_loss_value /= num_consistency_pairs
                if isinstance(consistency_loss_value, torch.Tensor):
                    total_loss += self.consistent_loss_weight * consistency_loss_value
                    loss_dict['consistency_loss'] = consistency_loss_value.item()

        # --- 4. Inter-Group Consistency Loss (Bio-signals vs. Visual-signals) ---
        if self.inter_group_consistency_weight > 0:
            bio_group_logits_list = [all_logits[key] for key in self.bio_signal_keys if key in all_logits]
            visual_group_logits_list = [all_logits[key] for key in self.visual_signal_keys if key in all_logits]

            if bio_group_logits_list and visual_group_logits_list:
                # Aggregate predictions for bio group (using probs from raw logits)
                avg_bio_probs = torch.stack(
                    [F.softmax(lgts, dim=1) for lgts in bio_group_logits_list]
                ).mean(dim=0)

                # Aggregate predictions for visual group
                avg_visual_probs = torch.stack(
                    [F.softmax(lgts, dim=1) for lgts in visual_group_logits_list]
                ).mean(dim=0)

                eps = 1e-8 # Epsilon for numerical stability

                # KL(Bio || Visual)
                kl_bio_visual = F.kl_div(
                    (avg_visual_probs + eps).log(), # input: log(Q)
                    avg_bio_probs,                  # target: P
                    reduction='batchmean',
                    log_target=False
                )
                # KL(Visual || Bio)
                kl_visual_bio = F.kl_div(
                    (avg_bio_probs + eps).log(),    # input: log(P)
                    avg_visual_probs,               # target: Q
                    reduction='batchmean',
                    log_target=False
                )
                
                inter_group_div_loss = (kl_bio_visual + kl_visual_bio) / 2.0
                
                total_loss += self.inter_group_consistency_weight * inter_group_div_loss
                loss_dict['inter_group_consistency_loss'] = inter_group_div_loss.item()
                
        return total_loss, loss_dict

    def train_epoch(self, loader, epoch, scheduler=None):
        self.model.train()
        self.current_epoch = epoch+1 # Update current epoch for loss calculation
        epoch_total_losses = [] # Store total loss for the epoch average

        for metric in self.metrics_train.values():
            metric.reset()

        pbar = tqdm(enumerate(loader), total=len(loader),
                    desc=f"[Train {epoch+1}/{self.cfg['epochs']}]")
        
        for step, batch in pbar:
            frames = video_transform(batch['frames']).to(DEVICE, non_blocking=True)
            body   = batch['body'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
            face   = batch['face'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
            rh     = batch['rh'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
            lh     = batch['lh'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
            hrv    = batch['hrv'].to(torch.float).to(DEVICE, non_blocking=True)
            target = torch.argmax(batch['super_lbls'], dim=1).long().to(DEVICE, non_blocking=True)

            with autocast(enabled=self.scaler.is_enabled()):
                outputs = self.model(frames, body, face, rh, lh, hrv)
                loss, loss_breakdown = self._compute_loss(outputs, target)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if scheduler: # + epoch * len(loader)
                scheduler(self.optimizer, step, epoch) # Pass global step

            epoch_total_losses.append(loss.item())

            # Update metrics
            for output_key, metric_key in self.output_to_metric_key_map.items():
                if output_key in outputs:
                    logits = outputs[output_key]
                    preds  = torch.argmax(logits, dim=1)
                    self.metrics_train[metric_key].update(preds, target)
            
            if (step + 1) % self.cfg.get('log_step_freq', 20) == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                avg_step_loss = np.mean(epoch_total_losses[-self.cfg.get('log_step_freq', 20):])
                accs_computed = {k: m.compute().item() for k, m in self.metrics_train.items()}
                pbar_postfix = {**{'loss': avg_step_loss, 'lr': f"{current_lr:.2e}"}, **accs_computed}
                # Include loss breakdown in postfix if desired
                # pbar_postfix.update({f"loss_{k}": v for k,v in loss_breakdown.items() if 'loss' in k})
                pbar.set_postfix(pbar_postfix)

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
            name: F1Score(task="multiclass", num_classes=2).to(DEVICE)
            for name in self.metric_names
        })

    def validate(self, loader, epoch):
        self.model.eval()
        for m in self.metrics_val.values():
            m.reset()

        with torch.no_grad():
            pbar = tqdm(enumerate(loader), total=len(loader),
                        desc=f"[ Valid {epoch+1}/{self.cfg['epochs']}]")
            for _, batch in pbar:
                # copy the same data prep as Trainer...
                frames = video_transform(batch['frames']).to(DEVICE, non_blocking=True)
                body   = batch['body'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
                face   = batch['face'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
                rh     = batch['rh'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
                lh     = batch['lh'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
                hrv    = batch['hrv'].to(torch.float).to(DEVICE, non_blocking=True)
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

            final_accs = {k: m.compute().item() for k,m in self.metrics_val.items()}
        return final_accs

'''# In epoch loop'''
# if epoch < 5:  # First stage: train encoders
#     for name, param in model.named_parameters():
#         if 'fusion' in name:
#             param.requires_grad = False
#         else:
#             param.requires_grad = True
# elif epoch < 10:  # Second stage: train fusion
#     for name, param in model.named_parameters():
#         if 'fusion' in name:
#             param.requires_grad = True
#         else:
#             param.requires_grad = False
# else:  # Final stage: train everything
#     for param in model.parameters():
#         param.requires_grad = True
'''
fusion v2 loss no adaptive weights
'''
    # def _compute_loss(self, outputs, target):
    #     """Enhanced multi-modal loss computation."""
    #     total = 0.0
    #     losses = {}
    #     batch_size = target.size(0)
        
    #     # 1. Standard Cross Entropy Losses (as before but with corrected key name)
    #     logits_list = []
    #     for key, logits in outputs.items():
            
    #         weight = self.main_weight if key == 'fusion_outputs' else self.aux_weight
    #         ce_loss = self.loss_fn(logits, target)
    #         total += weight * ce_loss
    #         losses[key] = ce_loss.item()
    #         logits_list.append(logits)
        
    #     # 2. Add consistency loss between modalities
    #     if self.consistent_loss_weight > 0:
    #         consistency_loss = 0
    #         n_pairs = 0
            
    #         # Get softmax probabilities
    #         probs = [F.softmax(logits/self.temperature, dim=1) for logits in logits_list]
            
    #         # Compare each modality's prediction with fusion
    #         fusion_probs = probs[-1]  # Assuming fusion is the last one
    #         for i in range(len(probs)-1):  # All except fusion
    #             # KL divergence to measure prediction consistency
    #             kl_div = F.kl_div(
    #                 fusion_probs.log(), probs[i], 
    #                 reduction='batchmean'
    #             )
    #             consistency_loss += kl_div
    #             n_pairs += 1
            
    #         if n_pairs > 0:
    #             consistency_loss = consistency_loss / n_pairs
    #             total += self.consistent_loss_weight * consistency_loss
    #             losses['consistency'] = consistency_loss.item()
        
    #     # 3. Implement adaptive weighting based on confidence
    #     if self.use_adaptive_weights and self.training:
    #         # Get confidence scores for each modality
    #         confidences = []
    #         for logits in logits_list:
    #             probs = F.softmax(logits, dim=1)
    #             # Higher max probability = higher confidence
    #             conf = torch.max(probs, dim=1)[0].mean()
    #             confidences.append(conf.item())
            
    #         # Update modality weights based on their confidence
    #         with torch.no_grad():
    #             conf_tensor = torch.tensor(confidences, device=self.modality_weights.device)
    #             # Softmax to normalize confidences to weights
    #             new_weights = F.softmax(conf_tensor, dim=0)
    #             # Gradual update to prevent instability (moving average)
    #             alpha = self.cfg['loss_alpha']  # Controls update rate
    #             self.modality_weights.data = alpha * self.modality_weights.data + (1-alpha) * new_weights
                
    #         # Apply adaptive weights to loss components
    #         losses['adaptive_weights'] = self.modality_weights.data.cpu().numpy()
        
    #     return total, losses
    
    
'''
modality + aux weights included
'''

    # def _compute_loss(self, outputs, target):
    #     """Enhanced multi-modal loss computation with trainable weights."""
    #     total = 0.0
    #     losses = {}
    #     batch_size = target.size(0)
        
    #     # 1. Standard Cross Entropy Losses
    #     logits_list = []
    #     ce_losses = []  # Store individual losses for weighted combination
        
    #     for key, logits in outputs.items():
    #         ce_loss = self.loss_fn(logits, target)
    #         ce_losses.append(ce_loss)
    #         losses[key] = ce_loss.item()
    #         logits_list.append(logits)
        
    #     # 2. Add consistency loss between modalities (unchanged)
    #     if self.consistent_loss_weight > 0:
    #         consistency_loss = 0
    #         n_pairs = 0
            
    #         # Get softmax probabilities
    #         probs = [F.softmax(logits/self.temperature, dim=1) for logits in logits_list]
            
    #         # Compare each modality's prediction with fusion
    #         fusion_probs = probs[-1]  # Assuming fusion is the last one
    #         for i in range(len(probs)-1):  # All except fusion
    #             # KL divergence to measure prediction consistency
    #             kl_div = F.kl_div(
    #                 fusion_probs.log(), probs[i], 
    #                 reduction='batchmean'
    #             )
    #             consistency_loss += kl_div
    #             n_pairs += 1
            
    #         if n_pairs > 0:
    #             consistency_loss = consistency_loss / n_pairs
    #             total += self.consistent_loss_weight * consistency_loss
    #             losses['consistency'] = consistency_loss.item()
        
    #     # 3. Implement fully trainable adaptive weighting
    #     if self.use_adaptive_weights:
    #         # Calculate confidence scores directly with gradients
    #         confidences = []
    #         for logits in logits_list:
    #             probs = F.softmax(logits, dim=1)
    #             conf = torch.max(probs, dim=1)[0].mean()
    #             confidences.append(conf)
            
    #         # Stack confidences into a tensor for processing
    #         conf_tensor = torch.stack(confidences)
            
    #         # Option 1: Direct trainable weights (fully differentiable)
    #         # Remove the torch.no_grad() block to allow gradient flow
    #         new_weights = F.softmax(conf_tensor, dim=0)
            
    #         # Apply exponential moving average with alpha
    #         alpha = self.cfg['loss_alpha']
    #         updated_weights = alpha * self.modality_weights + (1-alpha) * new_weights
            
    #         # Re-normalize to ensure weights sum to 1
    #         updated_weights = F.softmax(updated_weights, dim=0)
            
    #         # Update the weights parameter
    #         self.modality_weights.data = updated_weights.data
            
    #         # Apply these weights to the losses
    #         for i, (key, loss) in enumerate(zip(outputs.keys(), ce_losses)):
    #             if key == 'fusion_outputs':
    #                 # Always give fusion a minimum weight
    #                 mod_weight = self.main_weight * self.modality_weights[i]
    #             else:
    #                 mod_weight = self.aux_weight * self.modality_weights[i]
                
    #             total += mod_weight * loss
            
    #         # Log the adaptive weights for monitoring
    #         losses['adaptive_weights'] = self.modality_weights.detach().cpu().numpy()
    #     else:
    #         # Original static weighting
    #         for key, loss in zip(outputs.keys(), ce_losses):
    #             weight = self.main_weight if key == 'fusion_outputs' else self.aux_weight
    #             total += weight * loss
        
    #     return total, losses