#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

'''
For Paper:
https://doi.org/10.1109/AVSS52988.2021.9663770.
'''

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# function to get the vgg19 features only
def Face_CNN():
    model = VGG('VGG19')
    checkpoint = torch.load('/home/user01/Data/mme/scripts_miccai/baselines/hou/PrivateTest_model.t7')
    model.load_state_dict(checkpoint['net'])
    model.classifier = nn.Identity()
    return model
# example usage for vgg19
# model = Face_CNN()
# x = torch.randn((4,3,48,48))
# y = model(x)
# print(y.shape)
#%%

class TCNBlock(nn.Module):
    def __init__(self, input_size=512, kernel_size=3, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.dropout_rate = dropout

        # Temporal Convolutional Layers
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2  # To maintain sequence length
        )
        self.bn1 = nn.BatchNorm1d(input_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        )
        self.bn2 = nn.BatchNorm1d(input_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the TCN block.
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, 512]
        Returns:
            torch.Tensor: Output tensor of shape [B, T, 512]
        """
        residual = x  # Save input for the residual connection

        # Permute to [B, C, T] for Conv1d
        x = x.permute(0, 2, 1)

        # First temporal convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Second temporal convolutional layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # Permute back to [B, T, C]
        x = x.permute(0, 2, 1)

        # Residual connection and final activation
        out = self.relu2(x + residual) # BxTxC
        
        out = x.mean(dim=1)
        
        return out
    
# combine Face_CNN and TCNBlock in Face_TCN
class Face_TCN(nn.Module):
    def __init__(self):
        super(Face_TCN, self).__init__()
        self.face_cnn = Face_CNN()
        self.tcn_block = TCNBlock()
        for param in self.face_cnn.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        """
        Forward pass for the Face TCN model.
        Args:
            x (torch.Tensor): Input tensor of shape [B, 3, T, H, W]
        Returns:
            torch.Tensor: Output tensor of shape [B, T, 512]
        """
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)
        x = self.face_cnn(x)
        x = x.view(B, T, -1)
        x = self.tcn_block(x)
        return x
#%%

def Pose_CNN(frames_per_clip=32, remove_fc=True):
    '''
    Takes input tensor of shape [B, C, T, H, W]
    '''
    model_name = f'r2plus1d_34_{frames_per_clip}_kinetics'
    model = torch.hub.load(
        'moabitcoin/ig65m-pytorch',
        model_name,
        num_classes=400,
        pretrained=True,
    )
    if remove_fc:
        model.fc = nn.Identity()
    return model

# USAGE
# model = Pose_CNN(32)
# x = torch.randn((4,3,48,112,112))
# y = model(x)

#%%
# combine Pose_CNN and TCNBlock in Pose_TCN
class Pose_TCN(nn.Module):
    def __init__(self, frames_per_clip):
        super(Pose_TCN, self).__init__()
        self.pose_cnn = Pose_CNN(frames_per_clip)
        self.tcn_block = TCNBlock()
        for param in self.pose_cnn.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        """
        Forward pass for the Pose TCN model.
        Args:
            x (torch.Tensor): Input tensor of shape [B, 3, T, 112, 112]
        Returns:
            torch.Tensor: Output tensor of shape [B, T, 512]
        """
        B, C, T, H, W = x.shape
        x = self.pose_cnn(x)
        # the output of the pose_cnn is [B, 512] so no need for TCN
        # but staying consistent with paper
        x = x.unsqueeze(1)
        x = self.tcn_block(x)
        return x

def check_trainable_layers(model):
    for name, param in model.named_parameters():
        status = "Trainable" if param.requires_grad else "Frozen"
        print(f"Layer: {name}, Status: {status}")
#%%