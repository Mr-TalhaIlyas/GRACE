#%%
import os, psutil
# os.chdir(os.path.dirname(__file__))
os.chdir('/home/user01/Data/npj/scripts/')
from models.utils import graph
import torch
import math
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
# from mmcv.utils import _BatchNorm
from torch.nn.modules.batchnorm import _BatchNorm
from typing import Dict, List, Optional, Union, Tuple

EPS = 1e-4


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)
    

class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.init_weights()

    def forward(self, x, A=None, alpha=1):
        # Input: N, C, T, V
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        # X1, X2: N, R, V
        # N, R, V, 1 - N, R, 1, V
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        # N, R, V, V
        x1 = self.conv4(x1) * alpha + (A[None, None] if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctu->nctv', x1, x3)
        return x1

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)


class unit_ctrgcn(nn.Module):
    def __init__(self, in_channels, out_channels, A):

        super(unit_ctrgcn, self).__init__()
        inter_channels = out_channels // 4
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels

        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()

        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.A = nn.Parameter(A.clone())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = None

        for i in range(self.num_subset):
            z = self.convs[i](x, self.A[i], self.alpha)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        
class unit_tcn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, norm='BN', dropout=0):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1] if norm is not None else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))

    def init_weights(self):
        conv_init(self.conv)
        bn_init(self.bn, 1)

class MSTCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 act_cfg=dict(type='ReLU'),
                 tcn_dropout=0):

        super().__init__()
        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        branch_channels_rem = out_channels - branch_channels * (self.num_branches - 1)

        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                build_activation_layer(act_cfg),
                unit_tcn(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            build_activation_layer(act_cfg),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels_rem, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels_rem)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

        self.act = build_activation_layer(act_cfg)
        self.drop = nn.Dropout(tcn_dropout)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        out = self.act(out)
        out = self.drop(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, _BatchNorm):
                bn_init(m, 1)
                
class CTRGCNBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 kernel_size=5,
                 dilations=[1, 2],
                 tcn_dropout=0):
        super(CTRGCNBlock, self).__init__()
        self.gcn1 = unit_ctrgcn(in_channels, out_channels, A)
        self.tcn1 = MSTCN(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilations=dilations,
            residual=False,
            tcn_dropout=tcn_dropout)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y

    def init_weights(self):
        self.tcn1.init_weights()
        self.gcn1.init_weights()

class CTRGCN(nn.Module):
    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 num_person=2,
                 num_nodes: int = 25,
                 inward_edges: List[Tuple[int, int]] = [(i, i) for i in range(25)],
                 **kwargs):
        super(CTRGCN, self).__init__()

        # self.graph = Graph(**graph_cfg)
        # A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        # self.register_buffer('A', A)
        
        self.num_nodes = num_nodes
        self.inward_edges = inward_edges
        
        A = graph.Graph(self.num_nodes, self.inward_edges,
                    labeling_mode='spatial').get_adjacency_matrix()
        A = torch.from_numpy(A).float()
        self.register_buffer('A', A)
        
        self.num_person = num_person
        self.base_channels = base_channels

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'tcn_dropout'}
        modules = [CTRGCNBlock(in_channels, base_channels, A.clone(), residual=False, **kwargs0)]
        for i in range(2, num_stages + 1):
            in_channels = base_channels
            out_channels = base_channels * (1 + (i in inflate_stages))
            stride = 1 + (i in down_stages)
            modules.append(CTRGCNBlock(base_channels, out_channels, A.clone(), stride=stride, **kwargs))
            base_channels = out_channels
        self.net = nn.ModuleList(modules)

    def init_weights(self):
        for module in self.net:
            module.init_weights()

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N, M * V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for gcn in self.net:
            x = gcn(x)

        x = x.reshape((N, M) + x.shape[1:])
        return x

class GCNHead(nn.Module):
    """The classification head for GCN.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to ``dict(type='CrossEntropyLoss')``.
        dropout (float): Probability of dropout layer. Defaults to 0.
        init_cfg (dict or list[dict]): Config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(self, num_classes, sup_classes, in_channels, fusion_in_channels, dropout = 0.):
        super().__init__()
        self.in_channels = in_channels
        self.sup_classes = sup_classes
        self.num_classes = num_classes
        self.fusion_in_channels = fusion_in_channels
        self.dropout = dropout
        self.dropout_ratio = dropout
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc_seizure = nn.Linear(self.in_channels, self.sup_classes) # seizure output
        # self.fc_aux = nn.Linear(self.in_channels, self.num_classes) # auxilary output 
        self.fc_feat = nn.Linear(self.in_channels, self.fusion_in_channels) # feature output

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward features from the upstream network.

        Args:
            x (torch.Tensor): Features from the upstream network.

        Returns:
            torch.Tensor: Classification scores with shape (B, num_classes).
        """

        N, M, C, T, V = x.shape
        x = x.view(N * M, C, T, V)
        x = self.pool(x)
        x = x.view(N, M, C)
        x = x.mean(dim=1)
        assert x.shape[1] == self.in_channels

        if self.dropout is not None:
            x = self.dropout(x)

        seizure_conf = self.fc_seizure(x)
        # cls_score = self.fc_aux(x)
        modality_feats = self.fc_feat(x)

        return seizure_conf, modality_feats

class PoseCTGCN(nn.Module):
    def __init__(self, num_classes, sup_classes, num_persons,
                 backbone_in_channels, head_in_channels,
                 num_nodes, inward_edges, dropout,
                 fusion_in_channels,
                 checkpoint_path=None):
        super(PoseCTGCN, self).__init__()
        # Input [N, M, T, V, C]
        self.backbone = CTRGCN(in_channels=backbone_in_channels,
                                num_person=num_persons,
                                num_nodes=num_nodes,
                                inward_edges=inward_edges)
        # Input [N, M, C, T, V]
        self.cls_head = GCNHead(num_classes=num_classes,
                                sup_classes=sup_classes,
                                in_channels=head_in_channels,
                                fusion_in_channels=fusion_in_channels,
                                dropout=dropout)
        if checkpoint_path is not None:
            self.load_pretrained(checkpoint_path)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cls_head(x)
        return x

    def load_pretrained(self, checkpoint_path):
        print('Loading Pose-CTRGCN pretrianed chkpt...')
        chkpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        try:
            # load pretrained
            # del chkpt['state_dict']['backbone.A'] # delete the saved adjacency matrix 
            pretrained_dict = chkpt['state_dict']
        except KeyError:
            # load pretrained
            # del chkpt['model_state_dict']['backbone.A'] # delete the saved adjacency matrix 
            pretrained_dict = chkpt['model_state_dict']
        # load model state dict
        state = self.backbone.state_dict()
        # loop over both dicts and make a new dict where name and the shape of new state match
        # with the pretrained state dict.
        matched, unmatched = [], []
        new_dict = {}
        for i, j in zip(pretrained_dict.items(), state.items()):
            pk, pv = i # pretrained state dictionary
            nk, nv = j # new state dictionary
            # if name and weight shape are same
            if pk == nk or pv.shape == nv.shape: #.strip('backbone.')
                new_dict[nk] = pv
                matched.append(pk)
            else:
                unmatched.append(pk)

        state.update(new_dict)
        self.backbone.load_state_dict(state, strict=False)
        print('Pre-trained state loaded successfully...')
        print(f'Mathed kyes: {len(matched)}, Unmatched Keys: {len(unmatched)}')
        print(40*'=')
#%%

# model = CTRGCN(in_channels=3,
#                 base_channels=64,
#                 num_stages=10,
#                 inflate_stages=[5, 8],
#                 down_stages=[5, 8],
#                 pretrained=None,
#                 num_person=1,
#                 num_nodes=graph.coco_num_node,
#                 inward_edges=graph.coco_inward_edges)

# inputs = torch.randn(5, 1, 150, 17, 3)
# output = model(inputs)
# print(output.shape)

#%%
# model = PoseCTGCN(num_classes=60, num_persons=1, sup_classes=2,
                
#                 backbone_in_channels=3, head_in_channels=256,
#                 num_nodes=graph.coco_num_node, inward_edges=graph.coco_inward_edges,
#                 dropout=0.2,
#                  fusion_in_channels=512,
#                 checkpoint_path='/home/user01/Data/npj/scripts/models/pretrained/ctrgcn_body_17kpts.pth')

# inputs = torch.randn(5, 1, 100, 17, 3)
# output = model(inputs)
# print(output[0].shape, output[1].shape)
#%%
