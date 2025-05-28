import copy as cp
from typing import Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from models.utils import graph
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
            self.bn_a = nn.ModuleList()
            self.bn_b = nn.ModuleList()   
            for i in range(self.num_subset):
                self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
                self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
                self.bn_a.append(nn.BatchNorm2d(inter_channels))
                self.bn_b.append(nn.BatchNorm2d(inter_channels))
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

        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"!!! unit_aagcn: NaN/Inf in input x. Shape: {x.shape}")
            # return x # Or handle as error

        y = None
        if self.adaptive:
            for i in range(self.num_subset):
                # --- Step 1: conv_a and conv_b outputs ---
                # A1_conv = self.conv_a[i](x)
                # A2_conv = self.conv_b[i](x)
                A1_conv_raw = self.conv_a[i](x)
                A2_conv_raw = self.conv_b[i](x)

                # Option 1: Tanh to squash values
                A1_conv_bn = self.bn_a[i](A1_conv_raw) # If BatchNorm is added
                A2_conv_bn = self.bn_b[i](A2_conv_raw) # If BatchNorm is added
                A1_conv = torch.tanh(A1_conv_bn)
                A2_conv = torch.tanh(A2_conv_bn)
                
                if torch.isnan(A1_conv).any() or torch.isinf(A1_conv).any():
                    print(f"!!! unit_aagcn (subset {i}): NaN/Inf in A1_conv output. Min: {A1_conv.min():.2e}, Max: {A1_conv.max():.2e}")
                if torch.isnan(A2_conv).any() or torch.isinf(A2_conv).any():
                    print(f"!!! unit_aagcn (subset {i}): NaN/Inf in A2_conv output. Min: {A2_conv.min():.2e}, Max: {A2_conv.max():.2e}")

                A1_permuted = A1_conv.permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
                A2_viewed = A2_conv.view(N, self.inter_c * T, V)
                
                # --- Step 2: Matmul and Normalization ---
                matmul_out = torch.matmul(A1_permuted, A2_viewed)
                if torch.isnan(matmul_out).any() or torch.isinf(matmul_out).any():
                    print(f"!!! unit_aagcn (subset {i}): NaN/Inf after matmul. Min: {matmul_out.min():.2e}, Max: {matmul_out.max():.2e}")
                    # torch.save({'A1_permuted': A1_permuted.detach().cpu(), 'A2_viewed': A2_viewed.detach().cpu()}, f'debug_matmul_s{i}.pt')


                # Denominator for normalization: self.inter_c * T
                # A1_permuted.size(-1) is self.inter_c * T
                denominator = A1_permuted.size(-1) 
                if denominator == 0: # Should not happen with valid T and inter_c
                    print(f"!!! unit_aagcn (subset {i}): Denominator for normalization is zero!")
                    normalized_adaptive_val = matmul_out # Avoid division by zero, but this is likely an error state
                else:
                    normalized_adaptive_val = matmul_out / denominator

                if torch.isnan(normalized_adaptive_val).any() or torch.isinf(normalized_adaptive_val).any():
                    print(f"!!! unit_aagcn (subset {i}): NaN/Inf after normalization. Min: {normalized_adaptive_val.min():.2e}, Max: {normalized_adaptive_val.max():.2e}")

                # --- Step 3: Tanh ---
                A1_adaptive_part = self.tan(normalized_adaptive_val)  # N V V
                if torch.isnan(A1_adaptive_part).any(): # isinf is not expected after tanh unless input was NaN
                    print(f"!!! unit_aagcn (subset {i}): NaN in A1_adaptive_part (output of tanh). This means normalized_adaptive_val was NaN.")
                
                # --- Step 4: Multiplication with self.alpha (HIGHLY SUSPECT for MulBackward0) ---
                current_alpha_val = self.alpha
                if torch.isnan(current_alpha_val).any() or torch.isinf(current_alpha_val).any():
                    print(f"!!! unit_aagcn (subset {i}): self.alpha is NaN/Inf! Value: {current_alpha_val.item()}")
                
                # This is the multiplication: A1_adaptive_part * current_alpha_val
                A1_update_term = A1_adaptive_part * current_alpha_val
                if torch.isnan(A1_update_term).any() or torch.isinf(A1_update_term).any():
                    print(f"!!! unit_aagcn (subset {i}): NaN/Inf in A1_update_term (A1_adaptive_part * self.alpha).")
                    print(f"    A1_adaptive_part has NaN: {torch.isnan(A1_adaptive_part).any()}, Min: {A1_adaptive_part.min():.2e}, Max: {A1_adaptive_part.max():.2e}")
                    print(f"    self.alpha: {current_alpha_val.item():.4f}")
                    # torch.save({'A1_adaptive_part': A1_adaptive_part.detach().cpu(), 'alpha': current_alpha_val.detach().cpu()}, f'debug_alpha_mult_s{i}.pt')

                # --- Step 5: Adding base adjacency ---
                current_A_i = self.A[i]
                if torch.isnan(current_A_i).any() or torch.isinf(current_A_i).any():
                     print(f"!!! unit_aagcn (subset {i}): self.A[{i}] (base adjacency) is NaN/Inf!")

                A1 = current_A_i + A1_update_term # This is the final adaptive adjacency for this path
                if torch.isnan(A1).any() or torch.isinf(A1).any():
                    print(f"!!! unit_aagcn (subset {i}): NaN/Inf in final A1 (adjacency matrix).")

                # --- Step 6: Prepare for graph convolution ---
                A2 = x.view(N, C * T, V) # A2 is reassigned to reshaped input features
                
                # --- Step 7: Main graph convolution matmul ---
                if torch.isnan(A2).any() or torch.isinf(A2).any(): # Should be clean if input x was clean
                     print(f"!!! unit_aagcn (subset {i}): NaN/Inf in A2 (reshaped x) before main matmul.")
                if torch.isnan(A1).any() or torch.isinf(A1).any(): # A1 is the learned adj
                     print(f"!!! unit_aagcn (subset {i}): NaN/Inf in A1 (learned adj) before main matmul.")

                main_conv_matmul_out = torch.matmul(A2, A1)
                if torch.isnan(main_conv_matmul_out).any() or torch.isinf(main_conv_matmul_out).any():
                    print(f"!!! unit_aagcn (subset {i}): NaN/Inf in main_conv_matmul_out (A2 @ A1).")

                z = self.conv_d[i](main_conv_matmul_out.view(N, C, T, V))
                if torch.isnan(z).any() or torch.isinf(z).any():
                    print(f"!!! unit_aagcn (subset {i}): NaN/Inf in z (output of self.conv_d).")
                
                y = z + y if y is not None else z
        else:
            # ... (non-adaptive path, ensure it's also stable) ...
            for i in range(self.num_subset):
                A1 = self.A[i] # Non-adaptive, uses fixed A
                A2 = x.view(N, C * T, V)
                # Check A1 and A2 if non-adaptive path is ever taken and causes issues
                if torch.isnan(A1).any() or torch.isinf(A1).any():
                    print(f"!!! unit_aagcn (subset {i}, non-adaptive): NaN/Inf in self.A[{i}].")
                if torch.isnan(A2).any() or torch.isinf(A2).any():
                    print(f"!!! unit_aagcn (subset {i}, non-adaptive): NaN/Inf in A2 (reshaped x).")

                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                if torch.isnan(z).any() or torch.isinf(z).any():
                    print(f"!!! unit_aagcn (subset {i}, non-adaptive): NaN/Inf in z.")
                y = z + y if y is not None else z
        
        # --- After loop, before residual and attention ---
        if y is not None and (torch.isnan(y).any() or torch.isinf(y).any()):
            print(f"!!! unit_aagcn: NaN/Inf in y before residual connection.")
        
        res_x = self.down(x)
        if torch.isnan(res_x).any() or torch.isinf(res_x).any():
            print(f"!!! unit_aagcn: NaN/Inf in self.down(x) (residual path).")

        y = self.relu(self.bn(y) + res_x) # self.down(x) is the residual

        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"!!! unit_aagcn: NaN/Inf in y after residual and bn/relu.")

        # ... (attention block, add checks here too if attention=True) ...
        if self.attention:
            # spatial attention first
            se = y.mean(-2)  # N C V
            if torch.isnan(se).any() or torch.isinf(se).any(): print(f"NaN/Inf in se for spatial attention")
            se1 = self.sigmoid(self.conv_sa(se))  # N 1 V
            if torch.isnan(se1).any() or torch.isinf(se1).any(): print(f"NaN/Inf in se1 for spatial attention")
            y = y * se1.unsqueeze(-2) + y # Additive attention here, could also be y = y * se1.unsqueeze(-2)
            if torch.isnan(y).any() or torch.isinf(y).any(): print(f"NaN/Inf after spatial attention")
            
            # then temporal attention
            se = y.mean(-1)  # N C T
            if torch.isnan(se).any() or torch.isinf(se).any(): print(f"NaN/Inf in se for temporal attention")
            se1 = self.sigmoid(self.conv_ta(se))  # N 1 T
            if torch.isnan(se1).any() or torch.isinf(se1).any(): print(f"NaN/Inf in se1 for temporal attention")
            y = y * se1.unsqueeze(-1) + y # Additive attention
            if torch.isnan(y).any() or torch.isinf(y).any(): print(f"NaN/Inf after temporal attention")
            
            # then channel attention (original code has spatial temporal attention ??)
            se = y.mean(-1).mean(-1)  # N C
            if torch.isnan(se).any() or torch.isinf(se).any(): print(f"NaN/Inf in se for channel attention")
            se1 = self.relu(self.fc1c(se))
            if torch.isnan(se1).any() or torch.isinf(se1).any(): print(f"NaN/Inf in se1 for channel attention (after fc1c)")
            se2 = self.sigmoid(self.fc2c(se1))  # N C
            if torch.isnan(se2).any() or torch.isinf(se2).any(): print(f"NaN/Inf in se2 for channel attention (after fc2c)")
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y # Additive attention
            if torch.isnan(y).any() or torch.isinf(y).any(): print(f"NaN/Inf after channel attention")

        return y
# ... rest of your gcn.py ...
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """Defines the computation performed at every call."""
    #     N, C, T, V = x.size()

    #     y = None
    #     if self.adaptive:
    #         for i in range(self.num_subset):
    #             A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(
    #                 N, V, self.inter_c * T)
    #             A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
    #             A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
    #             A1 = self.A[i] + A1 * self.alpha
    #             A2 = x.view(N, C * T, V)
    #             # print(A1.dtype, A2.dtype)
    #             z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
    #             y = z + y if y is not None else z
    #     else:
    #         for i in range(self.num_subset):
    #             A1 = self.A[i]
    #             A2 = x.view(N, C * T, V)
    #             z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
    #             y = z + y if y is not None else z

    #     y = self.relu(self.bn(y) + self.down(x))

    #     if self.attention:
    #         # spatial attention first
    #         se = y.mean(-2)  # N C V
    #         se1 = self.sigmoid(self.conv_sa(se))  # N 1 V
    #         y = y * se1.unsqueeze(-2) + y
    #         # then temporal attention
    #         se = y.mean(-1)  # N C T
    #         se1 = self.sigmoid(self.conv_ta(se))  # N 1 T
    #         y = y * se1.unsqueeze(-1) + y
    #         # then spatial temporal attention ??
    #         se = y.mean(-1).mean(-1)  # N C
    #         se1 = self.relu(self.fc1c(se))
    #         se2 = self.sigmoid(self.fc2c(se1))  # N C
    #         y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
    #         # A little bit weird
    #     return y


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
    
class PoseGCN(nn.Module):
    def __init__(self, num_classes, sup_classes, num_persons,
                 backbone_in_channels, head_in_channels,
                 num_nodes, inward_edges, dropout,
                 fusion_in_channels,
                 checkpoint_path=None):
        super(PoseGCN, self).__init__()
        # Input [N, M, T, V, C]
        self.backbone = AAGCN(in_channels=backbone_in_channels, num_person=num_persons,
                              num_nodes=num_nodes, inward_edges=inward_edges)
        # Input [N, M, C, T, V]
        self.cls_head = GCNHead(num_classes=num_classes, sup_classes=sup_classes,
                                in_channels=head_in_channels, fusion_in_channels=fusion_in_channels,
                                dropout=dropout)
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
#                 checkpoint_path="/home/talha/Data/mme/scripts/models/pretrained/gcn_body_17kpts_kinetic400.pth"
#                 )

# chkpt = torch.load("/home/talha/Data/mme/scripts/models/pretrained/gcn_body_17kpts_kinetic400.pth")

# # load pretrained
# del chkpt['state_dict']['backbone.A'] # delete the saved adjacency matrix 
# pretrained_dict = chkpt['state_dict']
# # load model state dict
# state = model.state_dict()
# # loop over both dicts and make a new dict where name and the shape of new state match
# # with the pretrained state dict.
# matched, unmatched = [], []
# new_dict = {}
# for i, j in zip(pretrained_dict.items(), state.items()):
#     pk, pv = i # pretrained state dictionary
#     nk, nv = j # new state dictionary
#     # if name and weight shape are same
#     if pk.strip('backbone.') == nk: #.strip('backbone.')
#         new_dict[nk] = pv
#         matched.append(pk)
#     elif pv.shape == nv.shape:
#         new_dict[nk] = pv
#         matched.append(pk)
#     else:
#         unmatched.append(pk)

# state.update(new_dict)
# model.load_state_dict(state)
# print('Pre-trained state loaded successfully...')
# print(f'Mathed kyes: {len(matched)}, Unmatched Keys: {len(unmatched)}')
# #%%
# model.backbone.load_state_dict(chkpt['state_dict'])
# # Example input (replace with actual input)
# inputs = torch.randn(5, 2, 150, 17, 3)
# output = model(inputs)
# # print(output.shape)

