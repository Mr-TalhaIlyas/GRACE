import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from transformers import ASTModel, ASTConfig
# from ddf.ddf import DDFPack

class ResBlock(nn.Module):
    def __init__(self, channel_size: int, negative_slope: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel_size)
        )

    def forward(self, x):
        return x + self.block(x)

class LearnableResizer(nn.Module):
    def __init__(self,):
        super().__init__()
        # Update for fixed output size (128x1024)
        self.output_size = (1024, 128)

        n = 16
        r = 2
        slope = 0.2

        # Change input and output channels to 1 for grayscale images
        self.module1 = nn.Sequential(
            nn.Conv2d(1, n, kernel_size=7, padding=3),  # Updated input channels
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(n, n, kernel_size=1),
            nn.LeakyReLU(slope, inplace=True),
            nn.BatchNorm2d(n)
        )

        resblocks = []
        for i in range(r):
            resblocks.append(ResBlock(n, slope))
        self.resblocks = nn.Sequential(*resblocks)

        self.module3 = nn.Sequential(
            nn.Conv2d(n, n, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n)
        )

        self.module4 = nn.Conv2d(n, 1, kernel_size=7, padding=3)  # Updated output channels

        # Removed scale factor and updated interpolate
        self.interpolate = partial(F.interpolate, size=self.output_size,
                                   mode='bilinear', align_corners=False)

    def forward(self, x):
        residual = self.interpolate(x)

        out = self.module1(x)
        out_residual = self.interpolate(out)

        out = self.resblocks(out_residual)
        out = self.module3(out)
        out = out + out_residual

        out = self.module4(out)
        out = out + residual

        return out

class EWT(nn.Module):
    def __init__(self, num_classes: int,
                 sup_classes: int,
                 in_channels: int = 768,
                 mod_feats: int = 256, # as all other modalities output same feature size
                 dropout_ratio: float = 0.3,
                 pretrained_path: str = None):
        super(EWT, self).__init__()
        
        self.num_classes = num_classes
        self.sup_classes = sup_classes
        self.mod_feats = mod_feats
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.resizer = LearnableResizer()
        # configuration = ASTConfig()
        # self.ast = ASTModel(configuration)
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        
        self.fc_seizure = nn.Linear(self.in_channels, self.sup_classes)# sigmoid
        # self.fc_aux = nn.Linear(self.in_channels, self.num_classes)# softmax
        self.fc_feat = nn.Linear(self.in_channels, self.mod_feats)

        if pretrained_path is not None:
            self.init_weights(pretrained_path)

    def forward(self, x):
        # B*C*H*W
        x = self.resizer(x).squeeze(1) # B*TIME*SCALE
        x = self.ast(x)
        x = x.pooler_output # B*TIME*SCALE
        
        if self.dropout_ratio != 0:
            x = self.dropout(x)

        # [N, mod_feats]
        x = x.view(x.size(0), -1)
        # [N x num_classes]
        seizure_conf = self.fc_seizure(x)
        # cls_score = self.fc_aux(x)
        modality_feats = self.fc_feat(x)

        return seizure_conf, modality_feats
    
    def init_weights(self, checkpoint_path):
        print('Loading EWT pretrianed chkpt...')
        checkpoint = torch.load(checkpoint_path)
        matched, unmatched = self.ast.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f'Done loading {checkpoint_path}.')
        print(f'Mathed kyes: {len(matched)}, Unmatched Keys: {len(unmatched)}')
        print(40*'=')


# model = EWT(5,
# pretrained_path='/home/talha/Data/mme/scripts/models/pretrained/ast-finetuned-audioset-10-10-0.4593.pth')

# x = torch.randn([2,1,2500,128])

# y = model(x)