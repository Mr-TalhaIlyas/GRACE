import os
import numpy as np
from tqdm import tqdm

import torch, wandb
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import Accuracy, F1Score, AUROC, Recall
torch.autograd.set_detect_anomaly(False)
from configs.config import config
from data.utils import video_transform
from torch_ema import ExponentialMovingAverage
from tools.training_utils import log_model_gradients, calculate_composite_loss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, model, optimizer, scaler=None, cfg=config, class_weights=None):
        print('TRAINER INIT-v7')
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.scaler = scaler or GradScaler()
        self.cfg = cfg
        self.class_weights = class_weights.to(DEVICE) if class_weights is not None else None

        if self.cfg['USE_EMA_UPDATES']:
            print(30*"-")
            print("Using EMA updates with momentum:", self.cfg['ema_momentum'])
            print(30*"-")
            self.ema_model = ExponentialMovingAverage(self.model.parameters(), decay=self.cfg['ema_momentum'])
        # Loss function
        # Initialize loss hyperparameters that will be passed to calculate_composite_loss
        self.loss_hyperparams = {
            'loss_fn': torch.nn.CrossEntropyLoss(weight=class_weights), # Example
            'fusion_loss_fn_smooth': torch.nn.CrossEntropyLoss(
                label_smoothing=self.cfg.get('label_smoothing_factor', 0.1), # from config
                weight=class_weights
            ),
            'ce_weights': self.cfg.get('ce_weights', { # from config
                'fusion_outputs': 1.5, 'flow_outputs': 1.0, 'ecg_outputs': 1.0, 'joint_pose_outputs': 0.5
            }),
            'consistent_loss_weight': self.cfg.get('consistent_loss_weight', 0.2), # from config
            'inter_group_consistency_weight': self.cfg.get('inter_group_consistency_weight', 0.4), # from config
            'fusion_warmup_epochs': self.cfg.get('fusion_warmup_epochs', 2), # from config
            'bio_signal_keys': self.cfg.get('bio_signal_keys', ['ecg_outputs']), # from config
            'visual_signal_keys': self.cfg.get('visual_signal_keys', ['flow_outputs', 'joint_pose_outputs']), # from config
            'eps': self.cfg.get('kl_eps', 1e-8) # from config
        }
        self.current_epoch = 0 # Will be updated in train_loop
        self.performance_history = {
            'flow': [], 'ecg': [], 'pose': [], 'fusion': []
        }
        self.metric_keys_for_history = ['flow', 'ecg', 'pose', 'fusion'] # Match keys in train_accs
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
        """Computes loss by calling the utility function."""
        return calculate_composite_loss(outputs, target, self.loss_hyperparams, self.current_epoch)
                            
    def _log_gradients(self, step):
        """Logs gradients by calling the utility function."""
        log_model_gradients(self.model, self.cfg, step, wandb)
    
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
            
            if self.cfg['GRADIENT_CLIPPING']:
                # Gradient Clipping (before optimizer step, after unscaling)
                self.scaler.unscale_(self.optimizer) # Unscale gradients before clipping
                if (step + 1) == num_batches: # log at epoch end
                    self._log_gradients(self.current_epoch) 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg['grad_max_norm'])
            else:
                if (step + 1) == num_batches: # i.e., last step in epoch
                    self.scaler.unscale_(self.optimizer) # only unscale for logging.
                    self._log_gradients(self.current_epoch) 
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.cfg['USE_EMA_UPDATES']:
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
            
            if (step + 1) % 200 == 0 or (step + 1) == num_batches:
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
                log_freq = 100 # Or a different frequency for validation
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

