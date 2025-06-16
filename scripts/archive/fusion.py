#%%
# import os
# # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# # The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1";
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, TopKPooling, LayerNorm, RGCNConv
from torch_geometric.utils import (to_undirected, sort_edge_index,
                                   add_self_loops, to_dense_adj)

from models.cmft import CMFT
from models.mfi import MFI

def generate_edge_types(edge_index):
    """
    only generate edge types for a graph where (i, j) and (j, i) have the same type.
    all self loops (i,i) will have different type of relation
    """
    sorted_edges = torch.sort(edge_index, dim=0).values
    unique_edges, edge_types = torch.unique(sorted_edges, dim=1, return_inverse=True)

    return edge_types

def get_batch_indices_n_types(batch_size, num_nodes, self_loops=False):
    '''
    only generates indices and relation types of a fully connected graph.
    '''
    edge_index = torch.combinations(torch.arange(0, num_nodes),
                                    r=2, with_replacement=False).t().contiguous()
    edge_index = to_undirected(edge_index)
    if self_loops:
        edge_index, _ = add_self_loops(edge_index)
    # create a batch vector to inform layer about indices belonging to one batch
    batch_vector = torch.arange(batch_size).repeat_interleave(num_nodes)
    
    edge_types = generate_edge_types(edge_index)
    
    batch_edge_types = edge_types.repeat(batch_size)
    
    batch_edge_index = edge_index.clone()
    for i in range(1, batch_size):
        # offset the edge indices for each graph in the batch
        offset_edge_index = edge_index + i * num_nodes
        batch_edge_index = torch.cat([batch_edge_index, offset_edge_index], dim=1)
        
    return (batch_vector.to('cuda' if torch.cuda.is_available() else 'cpu'),
            batch_edge_index.to('cuda' if torch.cuda.is_available() else 'cpu'),
            batch_edge_types.to('cuda' if torch.cuda.is_available() else 'cpu'))

def get_pose_graph(batch_size = 5):
    pose_nodes = 4
    # shape: Batch * Nodes * Features
    # => (Batch*Nodes) * Features
    self_link = [(i, i) for i in range(pose_nodes)]
    
    inward = [(1, 0), # face -> body
              (2, 0), # r_hand -> body
              (3, 0)] # l_hand -> body
    
    outward = [(j, i) for (i, j) in inward]
    neighbor = self_link + inward + outward
    # Create edge index for PyTorch Geometric
    edge_index = torch.tensor(neighbor, dtype=torch.long).t().contiguous()
    
    batch_edge_index = edge_index.clone()
    for i in range(1, batch_size):
        # offset the edge indices for each graph in the batch
        offset_edge_index = edge_index + i * pose_nodes
        batch_edge_index = torch.cat([batch_edge_index, offset_edge_index], dim=1)
    
    # create a batch vector to inform layer about indices belonging to one batch
    batch_vector = torch.arange(batch_size).repeat_interleave(pose_nodes)
    
    return (batch_edge_index.to('cuda' if torch.cuda.is_available() else 'cpu'),
            batch_vector.to('cuda' if torch.cuda.is_available() else 'cpu'))

class CatFusion(nn.Module):
    def __init__(self, sub_classes, sup_classes, dropout_ratio=0.0):
        super(CatFusion, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(256 * 6, 512),  # Assuming concatenation of 6 feature vectors each of size 256
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.dropout = torch.nn.Dropout(dropout_ratio)
        # Classification head for 3-class problem
        self.head_3_classes = nn.Linear(256, sup_classes)
        self.pose_head = nn.Linear(256*4, sup_classes)
        # Classification head for 5-class problem
        # self.head_5_classes = nn.Linear(256, sub_classes)
        
    def forward(self, body, face, r_hand, l_hand, ecg, flow):
        # Concatenate the features along the feature dimension
        pose_feat = torch.cat((body, face, r_hand, l_hand), dim=1)
        concatenated_features = torch.cat((pose_feat, ecg, flow), dim=1)
        
        # Shared layers
        shared_features = self.shared_layers(concatenated_features)
        shared_features = self.dropout(shared_features)
        # Classification heads
        output_3_classes = self.head_3_classes(shared_features)
        pose_output = self.pose_head(pose_feat)
        # output_5_classes = self.head_5_classes(shared_features)
        
        self.pose_feats = shared_features # B*256
        self.fusion_feats = shared_features # B*256
        
        return pose_output, output_3_classes
    

class PoseFusion(torch.nn.Module):
    def __init__(self, in_channels, heads, num_super_classes, dropout):
        super(PoseFusion, self).__init__()
        # self.batch_edge_index = batch_edge_index,
        # self.batch_vector = batch_vector,
        self.tgcn = TransformerConv(in_channels=in_channels,
                                    out_channels=in_channels,
                                    heads=heads,
                                    dropout=dropout)  
        self.norm1 = LayerNorm(in_channels * heads)  
        self.topk_pool = TopKPooling(in_channels=in_channels * heads, ratio=0.25) # fixed 4 nodes -> 1
        self.mlp = torch.nn.Linear(in_features=in_channels * heads, out_features=in_channels)
        self.norm2 = LayerNorm(in_channels * heads)  
        self.cls_head = torch.nn.Linear(in_channels * heads, num_super_classes)

    def forward(self, pose, batch_edge_index, batch_vector):
        B, N, C = pose.shape # (Batch, Nodes, Channels)
        pose_feat = self.tgcn(pose.view(-1, C), batch_edge_index)
        pose_feat = pose_feat.view(B, N, -1)  # Reshape to (Batch, Nodes, Channels)
        pose_feat = F.relu(self.norm1(pose_feat))  # Apply ReLU after normalization
        
        # Flatten features from all nodes to simulate a 'batch' of nodes for pooling
        pose_feat_flat = pose_feat.view(-1, pose_feat.size(-1))
        
        pooled_feat, edge_index, _, batch, _, _ = self.topk_pool(x=pose_feat_flat,
                                                                 edge_index=batch_edge_index,
                                                                 batch=batch_vector)
        pooled_feat = F.relu(self.norm2(pooled_feat))  # Apply ReLU after normalization
        op = self.mlp(pooled_feat)
        cls_out = self.cls_head(pooled_feat)
        self.pose_feats = op
        return cls_out, op

class ModFusion(torch.nn.Module):
    def __init__(self, in_channels, num_relations, num_sub_classes,
                 num_super_classes, dropout_ratio=0.0):
        super(ModFusion, self).__init__()
        self.norm_before_rgcn = LayerNorm(in_channels)
        self.realation_gcn = RGCNConv(in_channels=in_channels,
                                      out_channels=in_channels,
                                      num_relations=num_relations)
        self.norm_after_rgcn = LayerNorm(in_channels)
        self.dropout = torch.nn.Dropout(dropout_ratio)  # Adjust dropout rate as needed
        # Task-specific heads
        # self.sub_class_head = torch.nn.Linear(in_channels, num_sub_classes)
        self.super_class_head = torch.nn.Linear(in_channels, num_super_classes)

    def forward(self, x, batch_edge_index, batch_edge_types):
        B, N, C = x.shape # (Batch, Nodes, Channels)
        x = self.norm_before_rgcn(x)
        x = self.realation_gcn(x.view(-1, C), batch_edge_index, batch_edge_types)
        x = F.relu(self.norm_after_rgcn(x.view(B, N, -1)))
        x = x.mean(dim=1)
        x = self.dropout(x)
        # sub_class_logits = self.sub_class_head(x)
        super_class_logits = self.super_class_head(x)
        
        self.fusion_feats = x
        return super_class_logits

class Fusion(torch.nn.Module):
    def __init__(self, in_channels, heads, num_relations, pose_fusion_dropout, 
                 mod_fusion_dropout, num_sub_classes, num_super_classes, type='graph'):
        super(Fusion, self).__init__()
        self.type = type
        # asset self type to be one of
        assert self.type in ['cat', 'trans', 'graph', 'cmft', 'mfi'], 'Invalid fusion type!'

        if self.type == 'cat': 
            self.cat_fusion = CatFusion(num_sub_classes, num_super_classes, mod_fusion_dropout)
        elif self.type == 'cmft':
            self.cmft = CMFT(backbone_dim=in_channels, c_dim=in_channels, num_c=num_super_classes)
        elif self.type == 'mfi':
            self.mfi = MFI(in_channels, num_super_classes)
        elif self.type == 'graph':
            self.pose_fusion = PoseFusion(in_channels, heads, num_super_classes, pose_fusion_dropout)
            self.mod_fusion = ModFusion(in_channels, num_relations,
                                        num_sub_classes, num_super_classes,
                                        mod_fusion_dropout)
        
    def forward(self, body, face, r_hand, l_hand, ecg, flow,
                    pose_batch_edge_index, pose_batch_vector,
                    batch_edge_index, batch_edge_types):
        
        if self.type == 'graph':
            pose = [body, face, r_hand, l_hand]
            pose = [p.unsqueeze(1) for p in pose]
            pose = torch.cat(pose, dim=1)
            
            pose_op, fused_pose = self.pose_fusion(pose, pose_batch_edge_index, pose_batch_vector)
            
            ecg = ecg.unsqueeze(1)
            flow = flow.unsqueeze(1)
            fused_pose = fused_pose.unsqueeze(1)
            
            all_mod = torch.cat([ecg, flow, fused_pose], dim=1)
            
            logits = self.mod_fusion(all_mod, batch_edge_index, batch_edge_types)
        
        elif self.type == 'cmft':
            pose_op, logits = self.cmft(body, face, r_hand, l_hand, ecg, flow)
        elif self.type == 'cat':
            pose_op, logits = self.cat_fusion(body, face, r_hand, l_hand, ecg, flow)
        elif self.type == 'mfi':
            pose_op, logits = self.mfi(body, face, r_hand, l_hand, ecg, flow)
        
        return logits, pose_op
#%%
# pose_batch_edge_index, pose_batch_vector = get_pose_graph(batch_size = 5)
# batch_vector, batch_edge_index, batch_edge_types = get_batch_indices_n_types(batch_size=2,
#                                                                   num_nodes=3,
#                                                                   self_loops=True)

# body = torch.randn((5, 256)) #  node-0
# face = torch.randn((5, 256)) # node-1 
# r_hand = torch.randn((5, 256))# node-2
# l_hand = torch.randn((5, 256)) # node-3




# ecg = torch.randn((5, 256))
# flow = torch.randn((5, 256))

# fuse = Fusion(in_channels=256, heads=3, num_relations=len(batch_edge_types.unique()),
#               dropout_ratio=0.0, num_sub_classes=2, num_super_classes=5)

# op = fuse(body, face, r_hand, l_hand, ecg, flow,
#                 pose_batch_edge_index, pose_batch_vector,
#                 batch_edge_index, batch_edge_types)

# print(op[0].shape, op[1].shape)

        
#%%
