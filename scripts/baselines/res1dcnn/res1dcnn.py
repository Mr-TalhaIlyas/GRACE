#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # First conv
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        # Second conv
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection to match dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Res1DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(Res1DCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # as in Paper: Personalized Real-Time Federated Learning for Epileptic Seizure Detection
        #
        # Each block consists of two conv layers. The first conv in each block may downsample the feature dimension by stride=2.
        
        # Block 1: 64 -> 64 filters
        self.layer1 = ResidualBlock(in_channels=64, out_channels=64, stride=2)
        
        # Block 2: 64 -> 128 filters
        self.layer2 = ResidualBlock(in_channels=64, out_channels=128, stride=2)
        
        # Block 3: 128 -> 256 filters
        self.layer3 = ResidualBlock(in_channels=128, out_channels=256, stride=2)
        
        # Block 4: 256 -> 512 filters
        self.layer4 = ResidualBlock(in_channels=256, out_channels=512, stride=2)
        # add dropout layer
        self.dropout = nn.Dropout(0.5)
        
        self.fc = nn.Linear(512 * 40, num_classes)  # 7 * 512 → num_classes
        
    def forward(self, x):
        # x: (B, 1, 2500)
        out = self.conv1(x)  # (B, 64, L') where L' ~ 1250
        out = self.bn1(out)
        out = self.relu(out)
        
        # maxpool reduces dimension further
        out = self.maxpool(out)  # after pooling: ~ (B, 64, ~625)
        
        out = self.layer1(out)   # (B, 64, ~312)
        out = self.layer2(out)   # (B, 128, ~156)
        out = self.layer3(out)   # (B, 256, ~78)
        out = self.layer4(out)   # (B, 512, ~40) 
        
        
        # Global flatten
        out = out.view(out.size(0), -1)  # (B, 512*40)
        out = self.dropout(out)
        out = self.fc(out)               # (B, 2)
        return out
#%%
# Example usage:
if __name__ == "__main__":
    
    x = torch.randn(4, 1, 2500)
    model = Res1DCNN(num_classes=2)
    y = model(x)
    print(y.shape)  #  (4, 2)
