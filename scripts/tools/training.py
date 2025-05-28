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

    def _compute_loss(self, outputs, target):
        """Sum up weighted auxiliary + main fusion loss."""
        total = 0.0
        losses = {}

        # aux losses
        for key, logits in outputs.items():
            weight = self.main_weight if key=='fusion_outputs' else self.aux_weight
            loss = self.loss_fn(logits, target)
            total += weight * loss
            losses[key] = loss.item()

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
                'flow_outpus': 'flow',
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
                    'flow_outpus': 'flow',
                    'ecg_outputs': 'ecg',
                    'joint_pose_outputs': 'pose',
                    'fusion_outputs': 'fusion',
                }
                for out_key, metric_key in key_map.items():
                    preds = torch.argmax(outputs[out_key], dim=1)
                    self.metrics_val[metric_key].update(preds, target)

            final_accs = {k: m.compute().item() for k,m in self.metrics_val.items()}
        return final_accs


