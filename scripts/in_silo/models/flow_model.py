from mmaction.models.backbones import ResNet3dSlowFast
import torch
import torch.nn as nn
from models.utils.tools import load_pretrained_chkpt

# pretrained_path ='C:/Users/talha/Desktop/MICCAI/models/pretrained/slowfast_r50_4x16x1_kinetics400-rgb.pth'


class SlowFast(nn.Module):
    def __init__(self,
                 num_classes: int,
                 sup_classes: int,
                 in_channels: int=2048+256, # ResNet3dSlowFast input channels FIXED
                 fusion_in_channels: int = 256, # as all other modalities output same feature size
                 dropout_ratio: float = 0.3,
                 pretrained_path: str = None):
        
        super(SlowFast, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.num_classes = num_classes
        self.sup_classes = sup_classes
        self.mod_feats = fusion_in_channels

        self.backbone = ResNet3dSlowFast()

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc_seizure = nn.Linear(in_channels, self.sup_classes)# softmax
        # self.fc_aux = nn.Linear(in_channels, self.num_classes)# softmax
        self.fc_feat = nn.Linear(in_channels, self.mod_feats)

        if pretrained_path is not None:
            self.init_weights(pretrained_path)

    def forward(self, x):
        # [B, C, T, H, W]
        x = self.backbone(x)
        # ([N, channel_slow, T1, H, W], [(N, channel_fast, T2, H, W)])
        x_slow, x_fast = x
        # ([N, channel_slow, 1, 1, 1], [N, channel_fast, 1, 1, 1])
        x_slow = self.avg_pool(x_slow)
        x_fast = self.avg_pool(x_fast)
        # [N, channel_fast + channel_slow, 1, 1, 1]
        x = torch.cat((x_fast, x_slow), dim=1)

        if self.dropout is not None:
            x = self.dropout(x)
        # [N, mod_feats]
        x = x.view(x.size(0), -1)
        # [N x num_classes]
        seizure_conf = self.fc_seizure(x)
        # cls_score = self.fc_aux(x)
        # modality_feats = self.fc_feat(x)

        return {'flow_outputs': seizure_conf}

    def init_weights(self, pretrained_path=None):
        print('Loading SlowFast pretrained chkpts...')
        if pretrained_path is not None:
            chkpt = torch.load(pretrained_path)

            pretrained_dict = chkpt['state_dict']
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
                if pk.strip('backbone.') == nk:# and pv.shape == nv.shape: #.strip('backbone.')
                    new_dict[nk] = pv
                    matched.append(pk)
                elif pv.shape == nv.shape:
                    new_dict[nk] = pv
                    matched.append(pk)
                else:
                    unmatched.append(pk)

            state.update(new_dict)
            self.backbone.load_state_dict(state, strict=False)
            print('Pre-trained SlowFast state loaded successfully...')
            print(f'Mathed kyes: {len(matched)}, Unmatched Keys: {len(unmatched)}')
            print(40*'=')
            print(unmatched)