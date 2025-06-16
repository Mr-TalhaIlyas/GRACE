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
        self.num_super_classes = len(cfg['super_classes'])
        # main + aux weights
        self.loss_fn       = nn.CrossEntropyLoss()
        self.aux_weight    = cfg['auxiliary_loss_weight']  # e.g. 0.4
        self.main_weight   = cfg['main_loss_weight']       # e.g. 1.0

        # metrics only care about the 4 outputs you monitor
        self.metric_names  = ['body_outputs', 'face_outputs', 'joint_pose_outputs']
        self.metrics_train = nn.ModuleDict({
            name: F1Score(task="multiclass", num_classes=self.num_super_classes).to(DEVICE) # UPDATED
            for name in self.metric_names
        })

    def _compute_loss(self, outputs, target):
        """Sum up weighted auxiliary + main fusion loss."""
        total = 0.0
        losses = {}

        # aux losses
        for key, logits in outputs.items():
            weight = 1.0 if key=='joint_pose_outputs' else 0.5
            loss = self.loss_fn(logits, target)
            total += weight * loss
            losses[key] = loss.item()

        return total, losses

    def train_epoch(self, loader, epoch, scheduler=None):
        self.model.train()
        num_batches = len(loader)
        epoch_losses = []
        epoch_total_losses = []
        for m in self.metrics_train.values():
            m.reset()

        # pbar = tqdm(enumerate(loader), total=len(loader),
        #             desc=f"[Train {epoch+1}/{self.cfg['epochs']}]")
        for step, batch in enumerate(loader):
            # 1) move data
            body   = batch['body'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
            face   = batch['face'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
            rh     = batch['rh'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
            lh     = batch['lh'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)

            target = batch["super_lbls"].long().to(DEVICE, non_blocking=True)
            
            # 2) forward + loss under autocast
            with autocast():
                outputs = self.model(body, face, rh, lh)
                loss, loss_dict = self._compute_loss(outputs, target)

            # 3) backward + step
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scheduler(self.optimizer, step, epoch)

            epoch_losses.append(loss.item())
            epoch_total_losses.append(loss.item())
            # 4) update metrics
            # pick only the four you care about
            key_map = {
                'body_outputs': 'body_outputs',
                'face_outputs': 'face_outputs',
                'joint_pose_outputs': 'joint_pose_outputs',
            }
            for out_key, metric_key in key_map.items():
                logits = outputs[out_key]
                preds  = torch.argmax(logits, dim=1)
                self.metrics_train[metric_key].update(preds, target)
            # break
            # 5) update tqdm
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

        # end of epoch
        avg_loss = np.mean(epoch_losses)
        final_accs = {k: m.compute().item() for k,m in self.metrics_train.items()}
        return avg_loss, final_accs


class Evaluator:
    def __init__(self, model, cfg=config):
        self.model = model.to(DEVICE)
        self.cfg   = cfg
        self.num_super_classes = len(cfg['super_classes']) # ADD THIS LINE
        self.metric_names =  ['body_outputs', 'face_outputs', 'joint_pose_outputs']
        self.metrics_val = nn.ModuleDict({
            name: F1Score(task="multiclass", num_classes=self.num_super_classes).to(DEVICE) # UPDATED
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
            for step, batch in enumerate(loader):
                # copy the same data prep as Trainer...
                body   = batch['body'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
                face   = batch['face'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
                rh     = batch['rh'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
                lh     = batch['lh'].permute(0,4,2,3,1).float().to(DEVICE, non_blocking=True)
                
                target = batch["super_lbls"].long().to(DEVICE, non_blocking=True)

                outputs = self.model(body, face, rh, lh)

                key_map = {
                'body_outputs': 'body_outputs',
                'face_outputs': 'face_outputs',
                'joint_pose_outputs': 'joint_pose_outputs',
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
        return final_accs


