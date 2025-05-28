# import resnet
import torch.nn as nn
# from ops.basic_ops import ConsensusModule, Identity
# from transforms import *
from torch.nn.init import normal_ as normal
from torch.nn.init import constant_ as constant
import torch, math
from typing import Optional
from torch import Tensor
import random


class BEF(nn.Module):
    def __init__(self, channel, reduction=8):
        super(BEF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputt):
        x = inputt.permute(0, 2, 1).contiguous()
        b, c, f = x.size()
        gap = self.avg_pool(x).view(b, c)
        y = self.fc(gap).view(b, c, 1)
        out = x * y.expand_as(x)

        return out.permute(0, 2, 1).contiguous()


class SA(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(SA, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # MLP, used for FFN
        # self.activation = nn.ReLU(inplace=True)
        # self.linear_in = nn.Linear(d_model, dim_feedforward)
        # self.dropout_mlp = nn.Dropout(dropout)
        # self.linear_out = nn.Linear(dim_feedforward, d_model)
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.se = BEF(channel=d_model, reduction=8)

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                val=None):

        src_self = self.self_attention(src, src, value=val if val is not None else src,
                                       attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]

        src = src + self.drop1(src_self)
        src = self.norm1(src)
        # tmp = self.linear_out(self.dropout_mlp(self.activation(self.linear_in(src))))  # FFN

        tmp = self.se(src)
        src = self.norm2(src + self.drop2(tmp))

        return src


class CA(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CA, self).__init__()
        self.crs_attention1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.crs_attention2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # MLP, used for FF
        # self.activation = nn.ReLU(inplace=True)
        # self.linear_in_1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout_mlp_1 = nn.Dropout(dropout)
        # self.linear_out_1 = nn.Linear(dim_feedforward, d_model)
        self.drop_1 = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model)

        # self.linear_in_2 = nn.Linear(d_model, dim_feedforward)
        # self.dropout_mlp_2 = nn.Dropout(dropout)
        # self.linear_out_2 = nn.Linear(dim_feedforward, d_model)
        self.drop_2 = nn.Dropout(dropout)
        self.norm_2 = nn.LayerNorm(d_model)

        self.se1 = BEF(channel=d_model, reduction=8)    # 替换mlp可行
        self.se2 = BEF(channel=d_model, reduction=8)

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                ):

        src1_cross = self.crs_attention1(query=src1,
                                         key=src2,
                                         value=src2, attn_mask=src2_mask,
                                         key_padding_mask=src2_key_padding_mask)[0]

        src2_cross = self.crs_attention2(query=src2,
                                         key=src1,
                                         value=src1, attn_mask=src1_mask,
                                         key_padding_mask=src1_key_padding_mask)[0]

        src1 = src1 + self.drop1(src1_cross)
        src1 = self.norm1(src1)
        # tmp = self.linear_out_1(self.dropout_mlp_1(self.activation(self.linear_in_1(src1))))  # FFN

        tmp = self.se1(src1)
        src1 = self.norm_1(src1 + self.drop_1(tmp))

        src2 = src2 + self.drop2(src2_cross)
        src2 = self.norm2(src2)
        # tmp = self.linear_out_2(self.dropout_mlp_2(self.activation(self.linear_in_2(src2))))  # FFN

        tmp = self.se2(src2)
        src2 = self.norm_2(src2 + self.drop_2(tmp))

        return src1, src2


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        # x = tensor_list.tensors
        nt, f, c = tensor_list.size()
        tensor_list = tensor_list.permute(0, 2, 1).contiguous().view(nt, c, int(f ** 0.5), int(f ** 0.5))
        mask = (tensor_list[:, 0, :, :] != 0).int()
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # 沿y方向累加，(1，1，1)--(1，2，3)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # 沿x方向累加，(1，1，1).T--(1，2，3).T
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # 第三个维度是num_pos_feats的2倍
        # return pos
        return pos.flatten(2).permute(0, 2, 1).contiguous()


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list):
        nt, f, c = tensor_list.size()
        x = tensor_list.permute(0, 2, 1).contiguous().view(nt, c, int(f ** 0.5), int(f ** 0.5))
        h, w = x.shape[-2:]
        # x = tensor_list
        # h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos.flatten(2).permute(0, 2, 1).contiguous()
        # return pos


# -------------------------
# ------ CMFT Model -------
# Cross Modality Fusion Transformer
# -------------------------
class FusionNet(nn.Module):
    def __init__(self, backbone_dim=2048, c_dim=512, num_c=60):
        super(FusionNet, self).__init__()

        self.c_dim = c_dim  # 降维后的通道数
        self.backbone_dim = backbone_dim
        self.droprate = 0.3  # transformer的droprate
        self.nheads = 8
        self.dim_feedforward = 1024  # transformer中MLP的隐层节点数
        self.layers = 4
        self.pos_embedding = PositionEmbeddingSine(c_dim // 2)

        self.reduce_channel1 = nn.Conv2d(self.backbone_dim, c_dim, kernel_size=1, bias=False)
        self.reduce_channel2 = nn.Conv2d(self.backbone_dim, c_dim, kernel_size=1, bias=False)
        self.reduce_channel3 = nn.Conv2d(self.backbone_dim, c_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_dim)
        self.bn2 = nn.BatchNorm2d(c_dim)
        self.bn3 = nn.BatchNorm2d(c_dim)

        self.sa1 = SA(c_dim, self.nheads, self.dim_feedforward, self.droprate)
        self.sa2 = SA(c_dim, self.nheads, self.dim_feedforward, self.droprate)
        self.sa3 = SA(c_dim, self.nheads, self.dim_feedforward, self.droprate)
        self.ca_list = nn.ModuleList([CA(c_dim, self.nheads, self.dim_feedforward, self.droprate)
                                      for _ in range(self.layers)])

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # self.drop1 = nn.Dropout(0.5)
        # self.drop2 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.3)

        # self.fc_out1 = nn.Linear(c_dim, num_c)
        # self.fc_out2 = nn.Linear(c_dim, num_c)
        self.fc_out3 = nn.Linear(c_dim, num_c)

        std = 0.001

        normal(self.fc_out3.weight, 0, std)
        constant(self.fc_out3.bias, 0)

    def forward(self, img_feat1, img_feat2, img_feat3):
        img_feat1 = img_feat1.unsqueeze(-1).unsqueeze(-1)
        img_feat2 = img_feat2.unsqueeze(-1).unsqueeze(-1)
        img_feat3 = img_feat3.unsqueeze(-1).unsqueeze(-1)
        # 对channel做attention
        img_feat1 = self.reduce_channel1(img_feat1)  # con1x1 减少通道数
        img_feat2 = self.reduce_channel2(img_feat2)
        img_feat3 = self.reduce_channel3(img_feat3)

        img_feat1 = self.bn1(img_feat1)
        img_feat2 = self.bn2(img_feat2)
        img_feat3 = self.bn3(img_feat3)

        # (N, L, E),where L is the target sequence length, N is the batch size, E is the embedding dimension.
        img_feat1 = img_feat1.flatten(2).permute(0, 2, 1).contiguous()  # b f c
        img_feat2 = img_feat2.flatten(2).permute(0, 2, 1).contiguous()
        img_feat3 = img_feat3.flatten(2).permute(0, 2, 1).contiguous()

        feat1 = img_feat1 + self.pos_embedding(img_feat1)
        feat2 = img_feat2 + self.pos_embedding(img_feat2)
        feat3 = img_feat3 + self.pos_embedding(img_feat3)

        for ca in self.ca_list:
            feat1 = self.sa1(feat1)
            feat2 = self.sa2(feat2)
            feat3 = self.sa3(feat3)
            feat1, feat2 = ca(feat1, feat2)
            _, feat3 = ca((feat1+feat2), feat3)

        feat_fus = feat1 + feat2 + feat3

        feat_fus = feat_fus.permute(0, 2, 1).contiguous()
        
        feat_fus = self.avgpool(feat_fus).squeeze(2)
       
        feat_fus = self.drop3(feat_fus)

        feat_fus = self.fc_out3(feat_fus)

        return feat_fus


class CMFT(nn.Module):
    def __init__(self, backbone_dim=2048, c_dim=512, num_c=60):
        super(CMFT, self).__init__()

        self.fc_pose = nn.Linear(256*4, 256) 
        self.fusion = FusionNet(backbone_dim=backbone_dim, c_dim=c_dim, num_c=num_c)
        self.cls_out = nn.Linear(256, num_c)

    def forward(self, flow, ecg, body, face, r_hand, l_hand):

        pose = torch.cat([body, face, r_hand, l_hand], dim=1)
        pose_feat = self.fc_pose(pose)
        fusion = self.fusion(flow, ecg, pose_feat)
        pose_cls_scores = self.cls_out(pose_feat)
        
        self.pose_feats = pose_feat # B*256
        self.fusion_feats = fusion # B*256
        
        return pose_cls_scores, fusion
#%%

# model = CMFT(backbone_dim=256, c_dim=256, num_c=5)

# x = torch.randn((2,256))

# y  = model(x,x,x,x,x,x)