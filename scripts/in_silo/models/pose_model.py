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
from models.fusion import EnhancedPoseFusion
from models.vit import DILVIT
from models.gtn import GTN
from models.utils import graph

# from tsai.models.InceptionTime import InceptionTime
from models.incep_time import InceptionTime
from tsai.models.XCM import XCM

from models.fusion import get_pose_graph, get_batch_indices_n_types

import torch
import torch.nn as nn
import torch.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class POSE_Model(nn.Module):
    def __init__(self, config):
        super(POSE_Model, self).__init__()
        # self.fusion_type = config['fusion_type']
        self.num_classes = config['num_sub_classes']
        self.sup_classes = config['num_sup_classes']
        self.pose_batch_edge_index, self.pose_batch_vector = get_pose_graph(batch_size = config['batch_size'],)
        self.batch_vector, self.batch_edge_index, self.batch_edge_types = get_batch_indices_n_types(
                                                                        batch_size=config['batch_size'],
                                                                        num_nodes=3,
                                                                        self_loops=True)


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
       
        
        feature_dim = config['fusion_in_channels']
        self.norm_body = nn.LayerNorm(feature_dim)
        self.norm_face = nn.LayerNorm(feature_dim)
        self.norm_rhand = nn.LayerNorm(feature_dim)
        self.norm_lhand = nn.LayerNorm(feature_dim)

        self.pose_fusion = EnhancedPoseFusion(in_channels=config['fusion_in_channels'],
                                              heads=config['fusion_heads'],
                                              num_super_classes=self.sup_classes,
                                              dropout=config['pose_fusion_dropout'])


    def forward(self, body, face, rh, lh):
        # body = [B, M, T, V, C]
        # face = [B, M, T, V, C]
        # rh = [B, M, T, V, C]
        # lh = [B, M, T, V, C]
        # sub_lbls = [B, N]
        # super_lbls = [B, N]

        # Body GCN
        body_cls_score, body_feats_raw = self.bodygcn(body)
        # Face GCN
        face_cls_score, face_feats_raw = self.facegcn(face)
        # Right Hand GCN
        rhand_cls_score, rhand_feats_raw = self.rhgcn(rh)
        # Left Hand GCN
        lhand_cls_score, lhand_feats_raw = self.lhgcn(lh)


        body_feats = self.norm_body(body_feats_raw)
        face_feats = self.norm_face(face_feats_raw)
        rhand_feats = self.norm_rhand(rhand_feats_raw)
        lhand_feats = self.norm_lhand(lhand_feats_raw)

        pose = [body_feats, face_feats, rhand_feats, lhand_feats]
        pose = [p.unsqueeze(1) for p in pose]
        pose = torch.cat(pose, dim=1)
        
        pose_scores, _ = self.pose_fusion(pose,
                                          self.pose_batch_edge_index,
                                          self.pose_batch_vector)
       
        return {'body_outputs': body_cls_score,
                'face_outputs': face_cls_score,
                'rhand_outputs': rhand_cls_score,
                'lhand_outputs': lhand_cls_score,
                'joint_pose_outputs': pose_scores,
                }
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
