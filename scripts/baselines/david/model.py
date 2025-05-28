#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
from lstm_pm import LSTM_PM
'''
Paper:
https://ieeexplore.ieee.org/document/8629065.
'''
class LandMarkBased_SeizureDet(nn.Module):
    def __init__(self, num_classes, time_steps=48):
        super(LandMarkBased_SeizureDet, self).__init__()
        self.pose_machine = LSTM_PM(outclass=num_classes, T=time_steps)
        # Auxiliary dense layers
        self.pose_dense = nn.Linear(200, 8)
        self.flow_dense = nn.Linear(4096, 8)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=16, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, pose_features, optical_flow, center_map, input_flow_features=None):
        """
        pose_features: [batch_size, 1, 200]
        flow_features: [batch_size, 4, 4096] will extract these online
        optical_flow: [batch_size, C*T, H, W]
        Returns:
        output: [batch_size, num_classes]
        """
        batch_size = pose_features.size(0)
        
        if input_flow_features is None:
            # computation heavy
            flow_features, _ = self.pose_machine(optical_flow, center_map)  # [batch_size, 4, 4096]
        else:
            flow_features = input_flow_features
        
        # Process pose features
        pose_out = self.pose_dense(pose_features.squeeze(1))  # [batch_size, 200] -> [batch_size, 8]
        # Expand pose_out to match time steps
        pose_out = pose_out.unsqueeze(1).expand(-1, flow_features.size(1), -1)  # [batch_size, 4, 8]
        
        # Process optical flow features
        flow_out = self.flow_dense(flow_features)  # [batch_size, 4, 4096] -> [batch_size, 4, 8]
        
        # Concatenate pose and flow features
        combined = torch.cat([pose_out, flow_out], dim=-1)  # [batch_size, 4, 16]
        
        # First LSTM layer
        lstm_out, _ = self.lstm1(combined)  # [batch_size, 4, 128]
        # Second LSTM layer
        lstm_out, _ = self.lstm2(lstm_out)  # [batch_size, 4, 64]
        
        # Take the last output
        last_output = lstm_out[:, -1, :]  # [batch_size, 64]
        
        # Output layer
        logits = self.fc(last_output)  # [batch_size, num_classes]
        output = F.log_softmax(logits, dim=-1)
        
        return output


# batch_size = 2
# num_classes = 3
# pose_features = torch.randn(batch_size, 1, 200)
# flow_features = torch.randn(batch_size, 4, 4096)

# model = LandMarkBased_SeizureDet(num_classes)
# output = model(pose_features, flow_features)
# print("Output shape:", output.shape)
# print("Output:", output)
#%%


class RegionBased_SeizureDet(nn.Module):
    def __init__(self, num_classes=2):
        super(RegionBased_SeizureDet, self).__init__()

        # CNN layers
        self.conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU()
        self.batch_norm = nn.BatchNorm2d(8)
        self.max_pool = nn.MaxPool2d(kernel_size=12, stride=12)
        
        # Calculate the output dimensions after max pooling
        # Input dimensions are 155 x 200
        self.pool_out_height = (155 + 2 * 0 - 1 * (12 - 1) - 1) // 12 + 1  # Adjust for kernel size and stride
        self.pool_out_width = (200 + 2 * 0 - 1 * (12 - 1) - 1) // 12 + 1

        # Fully connected layer
        self.fc = nn.Linear(8 * self.pool_out_height * self.pool_out_width, 64)

        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)

        # Output layer
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, seq_len, channels, height, width)
        """
        batch_size, channels, seq_len, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4)  # Shape: (batch_size, seq_len, channels, height, width)
        
        # Initialize a list to hold CNN outputs
        cnn_features = []

        for t in range(seq_len):
            # Get the t-th frame
            frame = x[:, t, :, :, :]  # Shape: (batch_size, channels, height, width)

            # CNN forward pass
            out = self.conv(frame)  # Output shape: (batch_size, 8, height, width)
            out = self.leaky_relu(out)
            out = self.batch_norm(out)
            out = self.max_pool(out)  # Output shape: (batch_size, 8, pool_out_height, pool_out_width)

            # Flatten
            out = out.view(batch_size, -1)  # Shape: (batch_size, feature_size)

            # Fully connected layer
            out = F.relu(self.fc(out))  # Shape: (batch_size, 64)

            cnn_features.append(out.unsqueeze(1))  # Shape: (batch_size, 1, 64)

        # Concatenate features along time dimension
        cnn_features = torch.cat(cnn_features, dim=1)  # Shape: (batch_size, seq_len, 64)

        # LSTM layers
        lstm_out, _ = self.lstm1(cnn_features)  # Output shape: (batch_size, seq_len, 128)
        lstm_out, _ = self.lstm2(lstm_out)      # Output shape: (batch_size, seq_len, 64)

        # Take the output from the last time step
        final_out = lstm_out[:, -1, :]  # Shape: (batch_size, 64)

        # Output layer
        logits = self.output_layer(final_out)  # Shape: (batch_size, num_classes)
        output = F.log_softmax(logits, dim=-1)

        return output

# x = torch.randn(4, 3, 50, 155, 200) # 50 frames @ 5 fps ~ 10 seconds

# model = RegionBased_SeizureDet(num_classes=2)
# output = model(x)
# print("Output shape:", output.shape)
#%%