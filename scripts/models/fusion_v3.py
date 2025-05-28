import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, TopKPooling, LayerNorm, RGCNConv, GATConv
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

# class Fusion(torch.nn.Module):
#     def __init__(self, in_channels, heads, num_relations, pose_fusion_dropout, 
#                  mod_fusion_dropout, num_sub_classes, num_super_classes, type='graph'):
#         super(Fusion, self).__init__()
#         self.type = type
#         # asset self type to be one of
#         assert self.type in ['cat', 'trans', 'graph', 'cmft', 'mfi'], 'Invalid fusion type!'

#         if self.type == 'cat': 
#             self.cat_fusion = CatFusion(num_sub_classes, num_super_classes, mod_fusion_dropout)
#         elif self.type == 'cmft':
#             self.cmft = CMFT(backbone_dim=in_channels, c_dim=in_channels, num_c=num_super_classes)
#         elif self.type == 'mfi':
#             self.mfi = MFI(in_channels, num_super_classes)
#         elif self.type == 'graph':
#             self.pose_fusion = PoseFusion(in_channels, heads, num_super_classes, pose_fusion_dropout)
#             self.mod_fusion = ModFusion(in_channels, num_relations,
#                                         num_sub_classes, num_super_classes,
#                                         mod_fusion_dropout)
        
#     def forward(self, body, face, r_hand, l_hand, ecg, flow,
#                     pose_batch_edge_index, pose_batch_vector,
#                     batch_edge_index, batch_edge_types):
        
#         if self.type == 'graph':
#             pose = [body, face, r_hand, l_hand]
#             pose = [p.unsqueeze(1) for p in pose]
#             pose = torch.cat(pose, dim=1)
            
#             pose_op, fused_pose = self.pose_fusion(pose, pose_batch_edge_index, pose_batch_vector)
            
#             ecg = ecg.unsqueeze(1)
#             flow = flow.unsqueeze(1)
#             fused_pose = fused_pose.unsqueeze(1)
            
#             all_mod = torch.cat([ecg, flow, fused_pose], dim=1)
            
#             logits = self.mod_fusion(all_mod, batch_edge_index, batch_edge_types)
        
#         elif self.type == 'cmft':
#             pose_op, logits = self.cmft(body, face, r_hand, l_hand, ecg, flow)
#         elif self.type == 'cat':
#             pose_op, logits = self.cat_fusion(body, face, r_hand, l_hand, ecg, flow)
#         elif self.type == 'mfi':
#             pose_op, logits = self.mfi(body, face, r_hand, l_hand, ecg, flow)
        
#         return logits, pose_op

class CrossModalAttention(nn.Module):
    """Cross-modal attention module that helps validate features across modalities"""
    def __init__(self, in_channels):
        super(CrossModalAttention, self).__init__()
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.gate = nn.Sequential(
            nn.Linear(in_channels*2, in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: [B, N, C] where N is number of modalities
        residual = x
        q = self.query(x)  # [B, N, C]
        k = self.key(x)    # [B, N, C]
        v = self.value(x)  # [B, N, C]
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)  # [B, N, N]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)  # [B, N, C]
        
        # Gating mechanism to control information flow
        gate_input = torch.cat([out, residual], dim=-1)
        gate = self.gate(gate_input)
        out = gate * out + (1 - gate) * residual
        
        return self.norm(out), attn


class ConfidenceWeighting(nn.Module):
    """Generates confidence scores for each modality based on their features"""
    def __init__(self, in_channels, num_modalities):
        super(ConfidenceWeighting, self).__init__()
        self.confidence_net = nn.Sequential(
            nn.Linear(in_channels, in_channels//2),
            nn.ReLU(),
            nn.Linear(in_channels//2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, N, C]
        # Generate confidence scores for each modality
        confidence = self.confidence_net(x)  # [B, N, 1]
        return confidence


class AdaptiveModFusion(nn.Module):
    def __init__(self, in_channels, heads, num_relations, num_sub_classes, 
                 num_super_classes, dropout_ratio=0.0):
        super(AdaptiveModFusion, self).__init__()
        
        # Cross-modal attention for feature validation
        self.cross_attn = CrossModalAttention(in_channels)
        
        # Confidence weighting for adaptive modality importance
        self.confidence = ConfidenceWeighting(in_channels, 3)  # 3 nodes: ecg, flow, fused_pose
        
        # Dynamic graph convolution that adapts to input
        self.dynamic_conv = GATConv(in_channels, in_channels, heads=heads, dropout=dropout_ratio)
        
        # Original RGCN for relation modeling
        self.norm_before_rgcn = LayerNorm(in_channels)
        self.relation_gcn = RGCNConv(in_channels=in_channels,
                                    out_channels=in_channels,
                                    num_relations=num_relations)
        self.norm_after_rgcn = LayerNorm(in_channels)
        
        # Feature mixing
        self.feature_mixer = nn.Sequential(
            nn.Linear(in_channels*2, in_channels),
            nn.ReLU(),
            nn.Dropout(dropout_ratio)
        )
        
        # Classification head
        self.super_class_head = nn.Linear(in_channels, num_super_classes)
        
        # Store attention weights for visualization/analysis
        self.attention_weights = None
        self.confidence_scores = None
        
    def forward(self, x, batch_edge_index, batch_edge_types):
        B, N, C = x.shape  # (Batch, Nodes, Channels)
        
        # 1. Apply cross-modal attention for feature validation
        validated_x, self.attention_weights = self.cross_attn(x)
        
        # 2. Generate confidence scores for modality weighting
        self.confidence_scores = self.confidence(validated_x)  # [B, N, 1]
        
        # 3. Apply confidence weighting
        weighted_x = validated_x * self.confidence_scores
        
        # 4. Combine original and validated features
        x = self.feature_mixer(torch.cat([x, weighted_x], dim=-1))
        
        # 5. Apply original RGCN with normalized features
        x_norm = self.norm_before_rgcn(x)
        x_rgcn = self.relation_gcn(x_norm.view(-1, C), batch_edge_index, batch_edge_types)
        x_rgcn = F.relu(self.norm_after_rgcn(x_rgcn.view(B, N, -1)))
        
        # 6. Global pooling with confidence weighting
        confidence_sum = self.confidence_scores.sum(dim=1)
        # Clamp the denominator to prevent division by a very small number
        denominator = torch.clamp(confidence_sum, min=1e-8) 
        x_pooled = (x_rgcn * self.confidence_scores).sum(dim=1) / denominator
        
        # 7. Classification
        super_class_logits = self.super_class_head(x_pooled)
        
        self.fusion_feats = x_pooled
        return super_class_logits


class EnhancedPoseFusion(nn.Module):
    def __init__(self, in_channels, heads, num_super_classes, dropout):
        super(EnhancedPoseFusion, self).__init__()
        
        # Cross-modal attention for pose components
        self.cross_attn = CrossModalAttention(in_channels)
        
        # Original transformer GCN
        self.tgcn = TransformerConv(in_channels=in_channels,
                                   out_channels=in_channels,
                                   heads=heads,
                                   dropout=dropout)
        
        # Additional components
        self.norm1 = LayerNorm(in_channels * heads)
        self.confidence = ConfidenceWeighting(in_channels, 4)  # 4 nodes: body, face, r_hand, l_hand
        
        # Feature aggregation with learnable weights
        self.adaptive_pool = nn.Sequential(
            nn.Linear(in_channels * heads * 4, in_channels * heads),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.norm2 = LayerNorm(in_channels * heads)
        self.mlp = nn.Linear(in_features=in_channels * heads, out_features=in_channels)
        self.cls_head = nn.Linear(in_channels * heads, num_super_classes)
        
        # Store attention for analysis
        self.attention_weights = None
        self.confidence_scores = None

    def forward(self, pose, batch_edge_index, batch_vector):
        B, N, C = pose.shape  # (Batch, Nodes, Channels)
        
        # 1. Cross-modal attention between pose components
        validated_pose, self.attention_weights = self.cross_attn(pose)
        
        # 2. Original GCN operation
        pose_feat = self.tgcn(pose.view(-1, C), batch_edge_index)
        pose_feat = pose_feat.view(B, N, -1)  # Reshape to (Batch, Nodes, Channels)
        pose_feat = F.relu(self.norm1(pose_feat))
        
        # 3. Generate confidence for each pose component
        self.confidence_scores = self.confidence(pose)  # [B, N, 1]
        
        # 4. Weight features by confidence and combine with cross-attention features
        weighted_pose = pose_feat * self.confidence_scores
        
        # 5. Adaptive pooling instead of fixed topk pooling
        flattened = weighted_pose.view(B, -1)  # Flatten all pose features
        pooled_feat = self.adaptive_pool(flattened)
        pooled_feat = F.relu(self.norm2(pooled_feat))
        
        # 6. Generate outputs
        op = self.mlp(pooled_feat)
        cls_out = self.cls_head(pooled_feat)
        
        self.pose_feats = op
        return cls_out, op


class Fusion(torch.nn.Module):
    # Update the graph fusion type components
    def __init__(self, in_channels, heads, num_relations, pose_fusion_dropout, 
                 mod_fusion_dropout, num_sub_classes, num_super_classes, type='graph'):
        super(Fusion, self).__init__()
        self.type = type
        assert self.type in ['cat', 'trans', 'graph', 'cmft', 'mfi'], 'Invalid fusion type!'

        if self.type == 'cat': 
            self.cat_fusion = CatFusion(num_sub_classes, num_super_classes, mod_fusion_dropout)
        elif self.type == 'cmft':
            print('Using CMFT')
            self.cmft = CMFT(backbone_dim=in_channels, c_dim=in_channels, num_c=num_super_classes)
            
        elif self.type == 'mfi':
            self.mfi = MFI(in_channels, num_super_classes)
        elif self.type == 'graph':
            # Replace with enhanced versions
            print(40*'=')
            print("Using GRACE")
            print(40*'=')
            self.pose_fusion = EnhancedPoseFusion(in_channels, heads, num_super_classes, pose_fusion_dropout)
            self.mod_fusion = AdaptiveModFusion(in_channels, heads, num_relations,
                                                num_sub_classes, num_super_classes,
                                                mod_fusion_dropout)
        
    def forward(self, body, face, r_hand, l_hand, ecg, flow,
                pose_batch_edge_index, pose_batch_vector,
                batch_edge_index, batch_edge_types):
        # The forward implementation remains the same
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