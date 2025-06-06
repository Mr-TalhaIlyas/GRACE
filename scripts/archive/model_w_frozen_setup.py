# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 18:06:18 2024

@author: talha
"""
#%%
import os, psutil
# os.chdir(os.path.dirname(__file__))
# os.chdir('/home/user01/Data/npj/scripts/')
from configs.config import config
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"] = "0";
from models.gcn import PoseGCN
from models.ctrgcn import PoseCTGCN
from models.slowfast import SlowFast
from models.ewt import EWT
# from models.fusion import Fusion
from models.fusion_v3 import Fusion
from models.vit import DILVIT
from models.gtn import GTN
from models.utils import graph

from tsai.models.InceptionTime import InceptionTime
from tsai.models.XCM import XCM

from models.fusion import get_pose_graph, get_batch_indices_n_types

import torch
import torch.nn as nn
import torch.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MME_Model(nn.Module):
    def __init__(self, config):
        super(MME_Model, self).__init__()
        self.fusion_type = config['fusion_type']
        self.num_classes = config['num_sub_classes']
        self.sup_classes = config['num_sup_classes']
        self.pose_batch_edge_index, self.pose_batch_vector = get_pose_graph(batch_size = config['batch_size'],)
        self.batch_vector, self.batch_edge_index, self.batch_edge_types = get_batch_indices_n_types(
                                                                        batch_size=config['batch_size'],
                                                                        num_nodes=3,
                                                                        self_loops=True)

        # INPUT: # [B, C, T, H, W]
        self.slowfast = SlowFast(self.num_classes, self.sup_classes,
                                 fusion_in_channels=config['fusion_in_channels'],
                                 dropout_ratio=config['flow_dropout'],
                                 pretrained_path=config['slowfast_pretrained_chkpts'])
        # INPUT:  [N, M, T, V, C]
        self.bodygcn = PoseCTGCN(num_classes=self.num_classes, sup_classes=self.sup_classes,
                               num_persons=config['num_persons'],
                                backbone_in_channels=config['backbone_in_channels'],
                                head_in_channels=config['head_in_channels'],
                                num_nodes=graph.coco_num_node, inward_edges=graph.coco_inward_edges,
                                dropout=config['pose_dropout'],
                                fusion_in_channels=config['fusion_in_channels'],
                                checkpoint_path=config['body_pretrainned_chkpts']
                                )
        
        self.facegcn = PoseCTGCN(num_classes=self.num_classes, sup_classes=self.sup_classes,
                               num_persons=config['num_persons'],
                                backbone_in_channels=config['backbone_in_channels'],
                                head_in_channels=config['head_in_channels'],
                                num_nodes=graph.num_nodes_face, inward_edges=graph.face_inward_edges,
                                dropout=config['pose_dropout'],
                                fusion_in_channels=config['fusion_in_channels'],
                                )
        
        self.rhgcn = PoseCTGCN(num_classes=self.num_classes, sup_classes=self.sup_classes,
                             num_persons=config['num_persons'],
                            backbone_in_channels=config['backbone_in_channels'],
                            head_in_channels=config['head_in_channels'],
                            num_nodes=graph.num_nodes_hand, inward_edges=graph.hand_inward_edges,
                            dropout=config['pose_dropout'],
                            fusion_in_channels=config['fusion_in_channels'],
                            checkpoint_path=config['hand_pretrainned_chkpts']
                            )
        
        self.lhgcn = PoseCTGCN(num_classes=self.num_classes, sup_classes=self.sup_classes,
                             num_persons=config['num_persons'],
                            backbone_in_channels=config['backbone_in_channels'],
                            head_in_channels=config['head_in_channels'],
                            num_nodes=graph.num_nodes_hand, inward_edges=graph.hand_inward_edges,
                            dropout=config['pose_dropout'],
                            fusion_in_channels=config['fusion_in_channels'],
                            checkpoint_path=config['hand_pretrainned_chkpts']
                            )
        # INPUT: # B*C*Time*(Scales or bins)
        # self.ewt = EWT(self.num_classes, self.sup_classes, in_channels = config['ewt_head_ch'],
        #                 mod_feats = config['mod_feats'], dropout_ratio= config['ewt_dropout_ratio'],
        #                 pretrained_path=config['ewt_pretrainned_chkpts'])
        # self.ewt = DILVIT(pretrained_path=config['ewt_pretrainned_chkpts'])
        # self.ewt = InceptionTime(c_in=config['hrv_channels'], c_out=self.sup_classes,
        #                          fc_dropout=config['ecg_dropout'], nf=config['fusion_in_channels']//4)
        # self.ewt = GTN(d_model=config['d_model_emb'], d_input=config['d_input_T'],
        #                 d_channel=config['hrv_channels'],
        #                 d_output=self.sup_classes,
        #                 d_hidden=config['d_hidden_ffn'],
        #                 fusion_dim=config['fusion_in_channels'],
        #                 n_heads=config['gtn_num_heads'],
        #                 N=config['gtn_num_layers'],
        #                 dropout=config['ecg_dropout'])
        print('USING XCM')
        self.ewt = XCM(c_in=config['hrv_channels'],
                       c_out=self.sup_classes, seq_len=2500,
                       fc_dropout=config['ecg_dropout'],
                       op_feat_dim=config['fusion_in_channels'],
                       nf=config['xcm_num_features'])
        
        feature_dim = config['fusion_in_channels']
        self.norm_of = nn.LayerNorm(feature_dim)
        self.norm_body = nn.LayerNorm(feature_dim)
        self.norm_face = nn.LayerNorm(feature_dim)
        self.norm_rhand = nn.LayerNorm(feature_dim)
        self.norm_lhand = nn.LayerNorm(feature_dim)
        self.norm_ecg = nn.LayerNorm(feature_dim)
        
        self.fusion = Fusion(in_channels=config['fusion_in_channels'],
                             heads = config['fusion_heads'],
                             num_relations=len(self.batch_edge_types.unique()),
                             pose_fusion_dropout=config['pose_fusion_dropout'], 
                             mod_fusion_dropout=config['mod_fusion_dropout'],
                             num_super_classes=self.sup_classes,
                             num_sub_classes=self.num_classes,
                             type = self.fusion_type)

        # For handling frozen encoder feature dropout
        self.encoders_frozen_status = {} # Expected: {'flow': False, 'ecg': False, 'pose': False}
        self.frozen_feature_dropout_rate = 0.0
        self.feature_dropout_layer = nn.Dropout(p=0.0) # Dropout layer, p will be updated
        
    def update_frozen_status(self, frozen_status: dict, dropout_rate: float):
        """
        Called by Trainer to update the model's knowledge of frozen encoders
        and the dropout rate to apply to their features.
        """
        self.encoders_frozen_status = frozen_status
        if self.training: # Only apply dropout during training
            self.frozen_feature_dropout_rate = dropout_rate
            self.feature_dropout_layer.p = dropout_rate
        else: # Ensure no dropout during evaluation
            self.frozen_feature_dropout_rate = 0.0
            self.feature_dropout_layer.p = 0.0
        print(f"[MME_Model] Updated frozen status: {self.encoders_frozen_status}, Dropout p: {self.feature_dropout_layer.p}")

    def forward(self, frames, body, face, rh, lh, ecg):
        # frames = [B, C, T, H, W]
        # body = [B, M, T, V, C]
        # face = [B, M, T, V, C]
        # rh = [B, M, T, V, C]
        # lh = [B, M, T, V, C]
        # ecg = [B, C, T, S]
        # sub_lbls = [B, N]
        # super_lbls = [B, N]

        # Slowfast
        of_cls_score, of_feats_raw = self.slowfast(frames)
        # Body GCN
        body_cls_score, body_feats_raw = self.bodygcn(body)
        # Face GCN
        face_cls_score, face_feats_raw = self.facegcn(face)
        # Right Hand GCN
        rhand_cls_score, rhand_feats_raw = self.rhgcn(rh)
        # Left Hand GCN
        lhand_cls_score, lhand_feats_raw = self.lhgcn(lh)
        # ECG
        ecg_cls_score, ecg_feats_raw = self.ewt(ecg)
        
        if self.training and self.frozen_feature_dropout_rate > 0:
            if self.encoders_frozen_status.get('flow', False):
                of_feats_raw = self.feature_dropout_layer(of_feats_raw)
            if self.encoders_frozen_status.get('ecg', False): # If ECG can also be frozen
                ecg_feats_raw = self.feature_dropout_layer(ecg_feats_raw)
            if self.encoders_frozen_status.get('pose', False): # If pose can be frozen
                body_feats_raw = self.feature_dropout_layer(body_feats_raw)
                face_feats_raw = self.feature_dropout_layer(face_feats_raw)
                rhand_feats_raw = self.feature_dropout_layer(rhand_feats_raw)
                lhand_feats_raw = self.feature_dropout_layer(lhand_feats_raw)
                

        self.of_feats = self.norm_of(of_feats_raw)
        body_feats = self.norm_body(body_feats_raw)
        face_feats = self.norm_face(face_feats_raw)
        rhand_feats = self.norm_rhand(rhand_feats_raw)
        lhand_feats = self.norm_lhand(lhand_feats_raw)
        self.ecg_feats = self.norm_ecg(ecg_feats_raw)
        # Fusion
        fusion_cls_score, pose_scores = self.fusion(body_feats, face_feats, rhand_feats, lhand_feats,
                                                    self.ecg_feats, self.of_feats,
                                                    self.pose_batch_edge_index, self.pose_batch_vector,
                                                    self.batch_edge_index, self.batch_edge_types)

        return {'flow_outputs': of_cls_score,
                'body_outputs': body_cls_score,
                'face_outputs': face_cls_score,
                'rhand_outputs': rhand_cls_score,
                'lhand_outputs': lhand_cls_score,
                'ecg_outputs': ecg_cls_score,
                'joint_pose_outputs': pose_scores,
                'fusion_outputs': fusion_cls_score}
#%%
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# config['model']['batch_size'] = 1
# config['model']['fusion_in_channels'] = 512
# # config['model']['fusion_type'] = 'cmft'
# model = MME_Model(config['model'])
# model = model.to(DEVICE)

# B = 1

# x = model(torch.randn((B,3,48,224,224)).to(DEVICE),
#           torch.randn((B,1,150,17,3)).to(DEVICE),
#           torch.randn((B,1,150,70,3)).to(DEVICE),
#           torch.randn((B,1,150,21,3)).to(DEVICE),
#           torch.randn((B,1,150,21,3)).to(DEVICE),
#         # torch.randn((4,1,2500,128)).to(DEVICE),  # for AST
#           torch.randn((B,19,2500)).to(DEVICE), # for ViT
#           # torch.randn((4, 5)),
#           # torch.randn((4,1))
#           )

# for k,v in x.items():
#     print(k, v[0].shape)#, v[1].shape)
# #%%

# print(model.ecg_feats.shape,
# model.of_feats.shape,
# model.fusion.pose_fusion.pose_feats.shape,
# model.fusion.mod_fusion.fusion_feats.shape)
# %%
