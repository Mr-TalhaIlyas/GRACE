import torch
import torch.nn as nn
from tsai.models.InceptionTime import InceptionBlock
from tsai.models.layers import GAP1d
from fastai.learner import Module, ifnone

class FeatureProjection(nn.Module):
    def __init__(self, dim, op_dim=None, dropout=0.3):
        super().__init__()
        if op_dim is None:
            op_dim = dim

        # MLP that maps dim→op_dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, op_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(op_dim, op_dim),
        )
        # If dims differ, learn a skip; otherwise just identity
        self.skip = nn.Identity() if dim == op_dim else nn.Linear(dim, op_dim, bias=False)
        # self.norm = nn.LayerNorm(op_dim)

    def forward(self, x):
        # Project the raw features if needed
        residual = self.skip(x)   # either x itself or a learned resize to op_dim
        out      = self.mlp(x)    # x → op_dim
        # return self.norm(residual + out)
        return residual + out  # skip connection

    
class InceptionTime(Module):
    """
    InceptionTime backbone with separate projection and classification branches from GAP features.

    Args:
        c_in (int): number of input channels
        c_out (int): number of output classes
        seq_len (int, optional): input sequence length (not used internally)
        nf (int): number of filters per Inception branch
        proj_dropout (float): dropout rate for projection MLP
        fc_dropout (float): dropout rate before the final classifier
        nb_filters (int, optional): legacy argument alias for nf
        pretrained_path (str, optional): path to pretrained weights
    """
    def __init__(
        self,
        c_in: int,
        c_out: int,
        seq_len: int = None,
        nf: int = 32,
        proj_dropout: float = 0.2,
        fc_dropout: float = 0.1,
        nb_filters: int = None,
        fusion_in_channels: int = None,
        pretrained_path: str = None,
        **kwargs
    ):
        super().__init__()
        nf = ifnone(nf, nb_filters)

        # Inception block produces 4*nf output channels
        self.inceptionblock = InceptionBlock(c_in, nf, **kwargs)
        # Global average pooling over time
        self.gap = GAP1d(1)
        # Projection MLP: Linear → ReLU → Dropout (for fusion features)
        # self.proj = nn.Sequential(
        #     nn.Linear(nf * 4, nf * 4),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(proj_dropout)
        # )
        self.proj = FeatureProjection(nf * 4, op_dim=fusion_in_channels)
        # Dropout before final classifier (on raw GAP features)
        self.dropout = nn.Dropout(fc_dropout)
        # Final linear classifier
        self.fc = nn.Linear(nf * 4, c_out)

        print(f"InceptionTime v2: [in={c_in}, out={c_out}, filters={nf}, proj_dropout={proj_dropout}, fc_dropout={fc_dropout}]")
        print("=" * 40)

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

    def forward(self, x: torch.Tensor):
        """
        Forward pass returns:
            logits: raw class scores (B, c_out)
            features: projected features (B, 4*nf) for fusion
        """
        # 1) Inception feature extraction: (B, 4*nf, L)
        x = self.inceptionblock(x)
        # 2) Global average pooling -> (B, 4*nf, 1)
        f = self.gap(x)
        # # 3) Remove last dimension -> (B, 4*nf)
        # f = f.squeeze(-1)
        # 4a) Classification branch: dropout on raw GAP features
        logits = self.fc(self.dropout(f))
        # 4b) Projection branch: MLP for downstream fusion
        f_proj = self.proj(f)
        return logits, f_proj

    def _load_pretrained(self, pretrained_path=None):
        
        if pretrained_path is not None:
            print(f'Loading pretrained weights from {pretrained_path}')
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.load_state_dict(state_dict, strict=True)
            print('Pretrained weights loaded successfully.')
        else:
            pass
        print(40*'=')