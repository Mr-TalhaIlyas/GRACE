from config import config
import cv2, os, random
from tqdm import tqdm
import numpy as np
from termcolor import cprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics.classification import BinaryAccuracy
from torchmetrics import Accuracy


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dprint(input, debug=config['DEBUG']):
    if debug:
        print(input)


class Trainer(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss() # CrossEntropyLoss  BCEWithLogitsLoss
        self.accuracy = Accuracy(task="multiclass", num_classes=3)

    
    def training_step(self, batched_data):
        
        # shape from B,S,T -> B,C,T,S
        # ecg = batched_data['ecg'].unsqueeze(1).permute(0,1,3,2).to(DEVICE)
        # shape from B,S,T -> B,T,S
        # ecg = batched_data['ecg'].permute(0,2,1).to(DEVICE)
        ecg = batched_data['ecg_seg'].unsqueeze(1).type(torch.float).to(DEVICE) # B,1,T
        # print(joints.shape)
        # sub_lbl_batch = batched_data['sub_lbls'].type(torch.float).to(DEVICE) # shape Bx5
        sup_lbl_batch = batched_data['super_lbls'].type(torch.float).to(DEVICE) # Bx3
        
        self.model.zero_grad()
        self.optimizer.zero_grad()

        preds = self.model.forward(ecg)

        loss = self.criterion(preds, sup_lbl_batch)
        acc = self.accuracy(preds.softmax(dim=-1).cpu().detach(),
                            sup_lbl_batch.argmax(1).cpu().detach())
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), acc.item()
    
class Evaluator(object):
    def __init__(self, model):
        self.model = model
        self.sup_accuracy = Accuracy(task="multiclass", num_classes=3)

   
    
    def eval_step(self, batched_data):
        
        # shape from B,S,T -> B,C,T,S
        # ecg = batched_data['ecg'].unsqueeze(1).permute(0,1,3,2).to(DEVICE)
        # shape from B,S,T -> B,T,S
        # ecg = batched_data['ecg'].permute(0,2,1).to(DEVICE)
        ecg = batched_data['ecg_seg'].unsqueeze(1).type(torch.float).to(DEVICE)
        # print(joints.shape)
        # sub_lbl_batch = batched_data['sub_lbls'].type(torch.float).to(DEVICE) # shape Bx5
        sup_lbl_batch = batched_data['super_lbls'].type(torch.float).to(DEVICE) # Bx3
        # sup_lbl_batch = torch.argmax(batched_data['super_lbls'], dim=1).type(torch.LongTensor).to(DEVICE) # Bx1
        
        with torch.no_grad():
            preds = self.model.forward(ecg) 

        preds = preds.softmax(dim=-1).cpu().detach()
        lbl_batch = sup_lbl_batch.argmax(1).cpu().detach()
        
        acc = self.sup_accuracy(preds, lbl_batch)

        return acc.item(), preds.numpy(), lbl_batch.numpy()