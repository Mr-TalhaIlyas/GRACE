#%%
import copy as cp
from typing import Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
import graph

'''
For Paper:
https://doi.org/10.1109/AVSS52988.2021.9663770.
'''

# from models.utils.tools import load_pretrained_chkpt
class unit_aagcn(nn.Module):
    """The graph convolution unit of AAGCN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        A (torch.Tensor): The adjacency matrix defined in the graph
            with shape of `(num_subsets, num_joints, num_joints)`.
        coff_embedding (int): The coefficient for downscaling the embedding
            dimension. Defaults to 4.
        adaptive (bool): Whether to use adaptive graph convolutional layer.
            Defaults to True.
        attention (bool): Whether to use the STC-attention module.
            Defaults to True.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
                dict(type='Constant', layer='BatchNorm2d', val=1,
                     override=dict(type='Constant', name='bn', val=1e-6)),
                dict(type='Kaiming', layer='Conv2d', mode='fan_out'),
                dict(type='ConvBranch', name='conv_d')
            ]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        coff_embedding: int = 4,
        adaptive: bool = True,
        attention: bool = False,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(
                type='Constant',
                layer='BatchNorm2d',
                val=1,
                override=dict(type='Constant', name='bn', val=1e-6)),
            dict(type='Kaiming', layer='Conv2d', mode='fan_out'),
            dict(type='ConvBranch', name='conv_d')
        ]
    ) -> None:

        if attention:
            attention_init_cfg = [
                dict(
                    type='Constant',
                    layer='Conv1d',
                    val=0,
                    override=dict(type='Xavier', name='conv_sa')),
                dict(
                    type='Kaiming',
                    layer='Linear',
                    mode='fan_in',
                    override=dict(type='Constant', val=0, name='fc2c'))
            ]
            init_cfg = cp.copy(init_cfg)
            init_cfg.extend(attention_init_cfg)

        super(unit_aagcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        self.attention = attention

        num_joints = A.shape[-1]

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if self.adaptive:
            self.A = nn.Parameter(A)

            self.alpha = nn.Parameter(torch.zeros(1))
            self.conv_a = nn.ModuleList()
            self.conv_b = nn.ModuleList()
            for i in range(self.num_subset):
                self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
                self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
        else:
            self.register_buffer('A', A)

        if self.attention:
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            # s attention
            ker_joint = num_joints if num_joints % 2 else num_joints - 1
            pad = (ker_joint - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_joint, padding=pad)
            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)

        self.down = lambda x: x
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels))

        self.bn = nn.BatchNorm2d(out_channels)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            for i in range(self.num_subset):
                A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(
                    N, V, self.inter_c * T)
                A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
                A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
                A1 = self.A[i] + A1 * self.alpha
                A2 = x.view(N, C * T, V)
                # print(A1.dtype, A2.dtype)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z
        else:
            for i in range(self.num_subset):
                A1 = self.A[i]
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z

        y = self.relu(self.bn(y) + self.down(x))

        if self.attention:
            # spatial attention first
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))  # N 1 V
            y = y * se1.unsqueeze(-2) + y
            # then temporal attention
            se = y.mean(-1)  # N C T
            se1 = self.sigmoid(self.conv_ta(se))  # N 1 T
            y = y * se1.unsqueeze(-1) + y
            # then spatial temporal attention ??
            se = y.mean(-1).mean(-1)  # N C
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))  # N C
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # A little bit weird
        return y


class unit_tcn(nn.Module):
    """The basic unit of temporal convolutional network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the temporal convolution kernel.
            Defaults to 9.
        stride (int): Stride of the temporal convolution. Defaults to 1.
        dilation (int): Spacing between temporal kernel elements.
            Defaults to 1.
        norm (str): The name of norm layer. Defaults to ``'BN'``.
        dropout (float): Dropout probability. Defaults to 0.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
                dict(type='Constant', layer='BatchNorm2d', val=1),
                dict(type='Kaiming', layer='Conv2d', mode='fan_out')
            ]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dilation: int = 1,
        norm: str = 'BN',
        dropout: float = 0,
        init_cfg: Union[Dict, List[Dict]] = [
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Kaiming', layer='Conv2d', mode='fan_out')
        ]
    ) -> None:
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
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1] \
            if norm is not None else nn.Identity()

        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.drop(self.bn(self.conv(x)))



class AAGCNBlock(nn.Module):
    """The basic block of AAGCN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        A (torch.Tensor): The adjacency matrix defined in the graph
            with shape of `(num_subsets, num_nodes, num_nodes)`.
        stride (int): Stride of the temporal convolution. Defaults to 1.
        residual (bool): Whether to use residual connection. Defaults to True.
        init_cfg (dict or list[dict], optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A: torch.Tensor,
                 stride: int = 1,
                 residual: bool = True,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__()

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {
            k: v
            for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']
        }
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_aagcn')
        assert gcn_type in ['unit_aagcn']

        self.gcn = unit_aagcn(in_channels, out_channels, A)#, **gcn_kwargs)

        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(
                out_channels, out_channels, 9, stride=stride)#, **tcn_kwargs)

        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(
                in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


# @MODELS.register_module()
class AAGCN(nn.Module):
    def __init__(self,
                 # graph_cfg: Dict,
                 in_channels: int = 3,
                 base_channels: int = 64,
                 data_bn_type: str = 'MVC',
                 num_person: int = 2,
                 num_stages: int = 10,
                 inflate_stages: List[int] = [5, 8],
                 down_stages: List[int] = [5, 8],
                 num_nodes: int = 25,
                 inward_edges: List[Tuple[int, int]] = [(i, i) for i in range(25)],
                 **kwargs) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.inward_edges = inward_edges

        A = graph.Graph(self.num_nodes, self.inward_edges,
                    labeling_mode='spatial').get_adjacency_matrix()
        A = torch.from_numpy(A).float()
        # print(A)
        assert data_bn_type in ['MVC', 'VC', None]
        self.data_bn_type = data_bn_type
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_person = num_person
        self.num_stages = num_stages
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        if self.data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif self.data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        modules = []
        if self.in_channels != self.base_channels:
            modules = [
                AAGCNBlock(
                    in_channels,
                    base_channels,
                    A.clone(),
                    1,
                    residual=False,
                    **lw_kwargs[0])
            ]

        for i in range(2, num_stages + 1):
            in_channels = base_channels
            out_channels = base_channels * (1 + (i in inflate_stages))
            stride = 1 + (i in down_stages)
            modules.append(
                AAGCNBlock(
                    base_channels,
                    out_channels,
                    A.clone(),
                    stride=stride,
                    **lw_kwargs[i - 1]))
            base_channels = out_channels

        if self.in_channels == self.base_channels:
            self.num_stages -= 1

        self.gcn = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))

        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4,
                                          2).contiguous().view(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)

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

    def __init__(self, num_classes, sup_classes, in_channels, dropout = 0.):
        super().__init__()
        self.in_channels = in_channels
        self.sup_classes = sup_classes
        self.num_classes = num_classes
        self.dropout = dropout
        self.dropout_ratio = dropout
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc_seizure = nn.Linear(self.in_channels, self.sup_classes) # seizure output
        # self.fc_aux = nn.Linear(self.in_channels, self.num_classes) # auxilary output 
        self.fc_feat = nn.Linear(self.in_channels, 256) # feature output

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
    
class PoseGCN(nn.Module):
    def __init__(self, num_classes, sup_classes, num_persons,
                 backbone_in_channels, head_in_channels,
                 num_nodes, inward_edges,
                 checkpoint_path=None):
        super(PoseGCN, self).__init__()
        # Input [N, M, T, V, C]
        self.backbone = AAGCN(in_channels=backbone_in_channels, num_person=num_persons,
                              num_nodes=num_nodes, inward_edges=inward_edges)
        # Input [N, M, C, T, V]
        self.cls_head = GCNHead(num_classes=num_classes, sup_classes=sup_classes,
                                in_channels=head_in_channels)
        if checkpoint_path is not None:
            self.load_pretrained(checkpoint_path)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cls_head(x)
        return x

    def load_pretrained(self, checkpoint_path):
        print('Loading PoseGCN pretrianed chkpt...')
        chkpt = torch.load(checkpoint_path)
        try:
            # load pretrained
            del chkpt['state_dict']['backbone.A'] # delete the saved adjacency matrix 
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
            if pk.strip('backbone.') == nk and pv.shape == nv.shape: #.strip('backbone.')
                new_dict[nk] = pv
                matched.append(pk)
            else:
                unmatched.append(pk)

        state.update(new_dict)
        self.backbone.load_state_dict(state, strict=False)
        print('Pre-trained state loaded successfully...')
        print(f'Mathed kyes: {len(matched)}, Unmatched Keys: {len(unmatched)}')
        print(40*'=')
        # print(f'Done loading {checkpoint_path}.')
        # print(unmatched)
#%%
# model = PoseGCN(num_classes=60, num_persons=1,
#                 backbone_in_channels=3, head_in_channels=256,
#                 num_nodes=graph.coco_num_node, inward_edges=graph.coco_inward_edges,
#                 checkpoint_path=None, sup_classes=2
#                 )


# inputs = torch.randn(5, 1, 150, 17, 3)
# output = model(inputs)


