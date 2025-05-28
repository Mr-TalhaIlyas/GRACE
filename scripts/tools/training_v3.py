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
        self.model     = model.to(DEVICE)
        self.optimizer = optimizer
        self.scaler    = scaler or GradScaler()
        self.cfg       = cfg

        # main + aux weights
        self.loss_fn       = nn.CrossEntropyLoss()
        self.aux_weight    = cfg['auxiliary_loss_weight']  # e.g. 0.4
        self.main_weight   = cfg['main_loss_weight']       # e.g. 1.0

        # metrics only care about the 4 outputs you monitor
        self.metric_names  = ['flow', 'ecg', 'pose', 'fusion']
        self.metrics_train = nn.ModuleDict({
            name: F1Score(task="multiclass", num_classes=2).to(DEVICE)
            for name in self.metric_names
        })

        # For adaptive weighting
        self.modality_weights = nn.Parameter(torch.ones(len(self.metric_names))/len(self.metric_names))
        self.consistent_loss_weight = self.cfg['consistent_loss_weight']
        
        # Temperature for confidence scaling
        self.temperature = self.cfg['temperature']
            
    def _compute_loss(self, outputs, target):
        """Enhanced multi-modal loss computation."""
        total = 0.0
        losses = {}
        # batch_size = target.size(0)
        
        # 1. Standard Cross Entropy Losses (as before but with corrected key name)
        logits_list = []
        for key, logits in outputs.items():
            
            weight = self.main_weight if key == 'fusion_outputs' else self.aux_weight
            ce_loss = self.loss_fn(logits, target)
            total += weight * ce_loss
            losses[key] = ce_loss.item()
            logits_list.append(logits)
        
        # 2. Add consistency loss between modalities
        if self.consistent_loss_weight > 0:
            consistency_loss = 0
            n_pairs = 0
            
            # Get softmax probabilities
            probs = [F.softmax(logits/self.temperature, dim=1) for logits in logits_list]
            
            # Compare each modality's prediction with fusion
            fusion_probs = probs[-1]  # Assuming fusion is the last one
            for i in range(len(probs)-1):  # All except fusion
                # KL divergence to measure prediction consistency
                kl_div = F.kl_div(
                    fusion_probs.log(), probs[i], 
                    reduction='batchmean'
                )
                consistency_loss += kl_div
                n_pairs += 1
            
            if n_pairs > 0:
                consistency_loss = consistency_loss / n_pairs
                total += self.consistent_loss_weight * consistency_loss
                losses['consistency'] = consistency_loss.item()
        
        return total, losses

    def train_epoch(self, loader, epoch, scheduler=None):
        self.model.train()
        epoch_losses = []
        for m in self.metrics_train.values():
            m.reset()

        pbar = tqdm(enumerate(loader), total=len(loader),
                    desc=f"[Train {epoch+1}/{self.cfg['epochs']}]")
        for step, batch in pbar:
            # 1) move data
            frames = video_transform(batch['frames']).to(DEVICE, non_blocking=True)
            body   = batch['body'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
            face   = batch['face'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
            rh     = batch['rh'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
            lh     = batch['lh'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
            hrv    = batch['hrv'].to(torch.float).to(DEVICE, non_blocking=True)
            

            # target: class indices [B]
            target = torch.argmax(batch['super_lbls'], dim=1).long().to(DEVICE, non_blocking=True)

            # 2) forward + loss under autocast
            with autocast():
                outputs = self.model(frames, body, face, rh, lh, hrv)
                loss, loss_dict = self._compute_loss(outputs, target)

            # 3) backward + step
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scheduler(self.optimizer, step, epoch)

            epoch_losses.append(loss.item())

            # 4) update metrics
            # pick only the four you care about
            key_map = {
                'flow_outputs': 'flow',
                'ecg_outputs': 'ecg',
                'joint_pose_outputs': 'pose',
                'fusion_outputs': 'fusion',
            }
            for out_key, metric_key in key_map.items():
                logits = outputs[out_key]
                preds  = torch.argmax(logits, dim=1)
                self.metrics_train[metric_key].update(preds, target)

            # 5) update tqdm
            if (step+1) % self.cfg['log_step_freq'] == 0:
                avg_loss = np.mean(epoch_losses[-self.cfg['log_step_freq']:])
                accs = {k: m.compute().item() for k,m in self.metrics_train.items()}
                pbar.set_postfix({**{'loss':avg_loss}, **accs})

        # end of epoch
        avg_loss = np.mean(epoch_losses)
        final_accs = {k: m.compute().item() for k,m in self.metrics_train.items()}
        return avg_loss, final_accs


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