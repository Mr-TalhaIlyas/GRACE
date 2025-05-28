from configs.config import config
import cv2, os, random
from tqdm import tqdm
import numpy as np
from termcolor import cprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from models.utils.loss import get_loss
from data.utils import video_transform
from torchmetrics.classification import BinaryAccuracy
from torchmetrics import Accuracy, F1Score, AUROC


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dprint(input, debug=config['DEBUG']):
    if debug:
        print(input)

def update_ema(model, ema_model, alpha=0.9, epoch=None, ema_warmup_epochs=3):
     # Optionally delay EMA updates until after a specified number of epochs
    if epoch is not None and (epoch+1) < ema_warmup_epochs:
        ema_model.load_state_dict(model.state_dict())
        print(f'EMA not updated, epoch: {epoch+1}')
        return  # Skip EMA update during warm-up period
    #  exponential moving average of model parameters
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        if not param.data.shape: # scalar tensor
            ema_param.data = alpha * ema_param.data + (1 - alpha) * param.data
        else:
            ema_param.data[:] = alpha * ema_param[:].data[:] + (1 - alpha) * param[:].data[:]
    print('EMA updated...')
    return None

class Trainer(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss = nn.CrossEntropyLoss() # CrossEntropyLoss  BCEWithLogitsLoss
        self.get_acc = Accuracy(task="multiclass", num_classes=2)
        self.get_f1 = F1Score(task="multiclass", num_classes=2)
        self.get_auc = AUROC(task="multiclass", num_classes=2)


    def calculate_loss(self, outputs, seizure_conf_label,
                        aux_weight=0.4, main_weight=1):
        all_losses = {}
        total_loss = 0.0
        # Calculate and weight loss for each auxiliary output
        for key in outputs:
            if key != 'fusion_outputs':  # Handle auxiliary outputs
                aux_loss = self.loss(outputs[key], seizure_conf_label)
                total_loss += aux_weight * aux_loss
                all_losses[key] = aux_loss.item()

        # Calculate loss for main (fusion) output
        fusion_loss = self.loss(outputs['fusion_outputs'], seizure_conf_label)
        # Add weighted main loss to total
        total_loss += main_weight * fusion_loss
        all_losses['fusion_outputs'] = fusion_loss.item()
        
        return total_loss, all_losses
    
    def calculate_accuracy(self, preds, labels):
        all_accuracy = {}
        for key in preds:
            aux_acc = self.get_acc(preds[key].softmax(dim=-1).cpu().detach().squeeze(),
                                        labels.argmax(1).cpu().detach())
            all_accuracy[key] = aux_acc.item()
        return all_accuracy
    
    def calculate_f1(self, preds, labels):
        all_f1 = {}
        for key in preds:
            aux_f1 = self.get_f1(preds[key].softmax(dim=-1).cpu().detach().squeeze(),
                                        labels.argmax(1).cpu().detach())
            all_f1[key] = aux_f1.item()
        return all_f1
    
    def calculate_auc(self, preds, labels):
        all_auc = {}
        for key in preds:
            aux_auc = self.get_auc(preds[key].softmax(dim=-1).cpu().detach().squeeze(),
                                        labels.argmax(1).cpu().detach())
            all_auc[key] = aux_auc.item()
        return all_auc
    
    
    def training_step(self, batched_data):
        # shape from BTHWC -> BCTHW
        flow = video_transform(batched_data['frames']).to(DEVICE)
        # shape from N*C*T*V*M -> N*M*T*V*C
        body = batched_data['body'].permute(0,4,2,3,1).to(torch.float).to(DEVICE)
        face = batched_data['face'].permute(0,4,2,3,1).to(torch.float).to(DEVICE)
        rhand = batched_data['rh'].permute(0,4,2,3,1).to(torch.float).to(DEVICE)
        lhand = batched_data['lh'].permute(0,4,2,3,1).to(torch.float).to(DEVICE)
        # AST ::: shape from B,S,T -> B,C,T,S
        # ecg = batched_data['ecg'].unsqueeze(1).permute(0,1,3,2).to(DEVICE)
        # ViT ::: shape from B,T -> B,1,T
        # ecg = batched_data['ecg_seg'].unsqueeze(1).type(torch.float).to(DEVICE)
        # InceptionTime ::: shape from B,19,T 
        hrv = batched_data['hrv'].to(torch.float).to(DEVICE)
        # print(joints.shape)
        # sub_lbl_batch = batched_data['sub_lbls'].type(torch.float).to(DEVICE) # shape Bx5
        sup_lbl_batch = batched_data['super_lbls'].type(torch.float).to(DEVICE) # Bx3
        
        self.model.zero_grad()
        self.optimizer.zero_grad()

        preds = self.model.forward(flow, body, face, rhand, lhand, hrv)

        loss, all_losses = self.calculate_loss(preds, sup_lbl_batch)
        
        loss.backward()
        self.optimizer.step()

        all_acc = self.calculate_accuracy(preds, sup_lbl_batch)
        all_f1 = self.calculate_f1(preds, sup_lbl_batch)
        all_auc = self.calculate_auc(preds, sup_lbl_batch)
        
        return loss.item(), all_losses, all_acc, all_f1, all_auc
    
class Evaluator(object):
    def __init__(self, model):
        self.model = model
        self.get_acc = Accuracy(task="multiclass", num_classes=2)
        self.get_f1 = F1Score(task="multiclass", num_classes=2)
        self.get_auc = AUROC(task="multiclass", num_classes=2)


    def calculate_accuracy(self, preds, labels):
        all_accuracy = {}
        for key in preds:
            aux_acc = self.sup_accuracy(preds[key].softmax(dim=-1).cpu().detach().squeeze(),
                                        labels.argmax(1).cpu().detach())
            all_accuracy[key] = aux_acc.item()
        return all_accuracy
    
    def calculate_f1(self, preds, labels):
        all_f1 = {}
        for key in preds:
            aux_f1 = self.get_f1(preds[key].softmax(dim=-1).cpu().detach().squeeze(),
                                        labels.argmax(1).cpu().detach())
            all_f1[key] = aux_f1.item()
        return all_f1
    
    def calculate_auc(self, preds, labels):
        all_auc = {}
        for key in preds:
            aux_auc = self.get_auc(preds[key].softmax(dim=-1).cpu().detach().squeeze(),
                                        labels.argmax(1).cpu().detach())
            all_auc[key] = aux_auc.item()
        return all_auc
    
    def eval_step(self, batched_data):
        # shape from BTHWC -> BCTHW
        flow = video_transform(batched_data['frames']).to(DEVICE)
        # shape from N*C*T*V*M -> N*M*T*V*C
        body = batched_data['body'].permute(0,4,2,3,1).to(torch.float).to(DEVICE)
        face = batched_data['face'].permute(0,4,2,3,1).to(torch.float).to(DEVICE)
        rhand = batched_data['rh'].permute(0,4,2,3,1).to(torch.float).to(DEVICE)
        lhand = batched_data['lh'].permute(0,4,2,3,1).to(torch.float).to(DEVICE)
        # AST ::: shape from B,S,T -> B,C,T,S
        # ecg = batched_data['ecg'].unsqueeze(1).permute(0,1,3,2).to(DEVICE)
        # ViT ::: shape from B,T -> B,1,T
        # ecg = batched_data['ecg_seg'].unsqueeze(1).type(torch.float).to(DEVICE)
        # InceptionTime ::: shape from B,19,T 
        hrv = batched_data['hrv'].to(torch.float).to(DEVICE)
        # print(joints.shape)
        # sub_lbl_batch = batched_data['sub_lbls'].type(torch.float).to(DEVICE) # shape Bx5
        sup_lbl_batch = batched_data['super_lbls'].type(torch.float).to(DEVICE) # Bx3
        # sup_lbl_batch = torch.argmax(batched_data['super_lbls'], dim=1).type(torch.LongTensor).to(DEVICE) # Bx1
        self.model.eval()
        with torch.no_grad():
            preds = self.model.forward(flow, body, face, rhand, lhand, hrv) 

        all_acc = self.calculate_accuracy(preds, sup_lbl_batch)
        all_f1 = self.calculate_f1(preds, sup_lbl_batch)
        all_auc = self.calculate_auc(preds, sup_lbl_batch)
        return all_acc, all_f1, all_auc