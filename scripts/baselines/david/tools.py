
from tqdm import tqdm
import numpy as np
from termcolor import cprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from utils import video_transform
from torchmetrics.classification import BinaryAccuracy
from torchmetrics import Accuracy
import sklearn.metrics as skm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss() # CrossEntropyLoss  BCEWithLogitsLoss
    
    def training_step(self, batched_data):
        # shape from BTHWC -> BCTHW
        flow = video_transform(batched_data['frames']).type(torch.float).to(DEVICE)
        b,c,t,h,w = flow.shape
        flow = flow.view(b,c*t,h,w)
        
        c_map = batched_data['center_map'].type(torch.float).to(DEVICE)
        # shape from B*200
        pose_feats = batched_data['body'].type(torch.float).to(DEVICE)
        # print(joints.shape)
        # sub_lbl_batch = batched_data['sub_lbls'].type(torch.float).to(DEVICE) # shape Bx5
        sup_lbl_batch = batched_data['super_lbls'].type(torch.float).to(DEVICE) # Bx2
        
        self.model.zero_grad()
        self.optimizer.zero_grad()

        preds = self.model(pose_feats, flow, c_map)

        loss = self.criterion(preds, sup_lbl_batch.argmax(1))
        
        loss.backward()
        self.optimizer.step()

        accuracy = skm.accuracy_score(sup_lbl_batch.argmax(1).cpu().detach().numpy(),
                                      preds.argmax(1).cpu().detach().numpy())
        
        return loss.item(), accuracy
    
class Evaluator(object):
    def __init__(self, model):
        self.model = model
        
    def eval_step(self, batched_data):
        flow = video_transform(batched_data['frames']).type(torch.float).to(DEVICE)
        b,c,t,h,w = flow.shape
        flow = flow.view(b,c*t,h,w)
        
        c_map = batched_data['center_map'].type(torch.float).to(DEVICE)
        # shape from N*C*T*V*M -> N*M*T*V*C
        pose_feats = batched_data['body'].type(torch.float).to(DEVICE)
        # print(joints.shape)
        # sub_lbl_batch = batched_data['sub_lbls'].type(torch.float).to(DEVICE) # shape Bx5
        sup_lbl_batch = batched_data['super_lbls'].type(torch.float).to(DEVICE) # Bx2
        
        self.model.eval()
        with torch.no_grad():
            preds = self.model(pose_feats, flow, c_map)
        
        probs = F.softmax(preds, dim=1).cpu().detach().numpy()
        preds = preds.argmax(1).cpu().detach().numpy()
        labels = sup_lbl_batch.argmax(1).cpu().detach().numpy()
        
        return probs, preds, labels