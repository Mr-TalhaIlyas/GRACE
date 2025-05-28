import torch
from torch import nn 
import torch.nn.functional as F 
import copy

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(21, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(64, 72, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(72, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Flatten(start_dim=1),
            nn.Linear(128*3, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.encode(x).squeeze(dim=-1)


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, hidden_dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, hidden_dim)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, 
                                                                    num_f_maps, hidden_dim, 
                                                                    num_classes if s == num_stages-2 else hidden_dim
                                                                    )) for s in range(num_stages-1)])

    def forward(self, x):
        out = self.stage1(x)
        for s in self.stages:
            out = s(F.leaky_relu(out, negative_slope=0.4))
        return out
    
class MultiStageModelPred(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, hidden_dim, num_classes):
        super(MultiStageModelPred, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, hidden_dim)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, 
                                                                    num_f_maps, hidden_dim, 
                                                                    hidden_dim
                                                                    )) for s in range(num_stages-1)])

    def forward(self, x):
        out = self.stage1(x)
        output = out.unsqueeze(dim=0)
        for s in self.stages:
            out = s(F.leaky_relu(out, negative_slope=0.4))
            output = torch.cat([out.unsqueeze(dim=0), output], axis=0)
        return output

class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)

class MSTCN_(nn.Module):
    def __init__(self, 
                num_layers=8,
                num_f_maps=32,
                num_stages=4,
                dim=760,
                num_classes=1
        ) -> None:
        super().__init__()
        self.tcn = MultiStageModel(
                    num_layers=num_layers,
                    num_f_maps=num_f_maps,
                    num_stages=num_stages,
                    dim=dim,
                    num_classes=num_classes
        )
        self.predict = nn.Sequential(
            nn.Conv2d(num_stages, 64, kernel_size=(3, 4), stride=(1, 2), padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(64, 1, kernel_size=(3, 4), stride=(1, 2), padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Flatten(start_dim=1),
            nn.Linear(144, 4), #174
            nn.Softmax(dim=-1)
        )  

    def forward(self, x):
        x = self.tcn(x.permute(0, 2, 1))
        x = self.predict(x.permute(1, 0, 2, 3))
        return x

class MSTCN_SQS(nn.Module):
    def __init__(self, 
                num_layers=8,
                num_f_maps=32,
                num_stages=4,
                dim=760,
                hidden_dim=32,
                num_classes=1
        ) -> None:
        super().__init__()
        self.tcn = MultiStageModel(
                    num_layers=num_layers,
                    num_f_maps=num_f_maps,
                    num_stages=num_stages,
                    dim=dim,
                    num_classes=num_classes,
                    hidden_dim=hidden_dim
        )

    def forward(self, x):
        x = self.tcn(x.permute(0, 2, 1))
        return torch.softmax(x, dim=1)

class MSTCN_C(nn.Module):
    def __init__(self, 
                num_layers=8,
                num_f_maps=32,
                num_stages=4,
                dim=760,
                hidden_dim=32,
                num_classes=1
        ) -> None:
        super().__init__()
        self.tcn = MultiStageModelPred(
                    num_layers=num_layers,
                    num_f_maps=num_f_maps,
                    num_stages=num_stages,
                    dim=dim,
                    num_classes=num_classes,
                    hidden_dim=hidden_dim
        )
        self.predict = nn.Sequential(
            nn.Conv2d(num_stages, 64, kernel_size=(3, 4), stride=(1, 2), padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(64, 1, kernel_size=(3, 4), stride=(1, 2), padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Flatten(start_dim=1),
            #nn.Linear(232, 4), #174
            nn.Linear(248, 4),
            nn.Softmax(dim=-1)
        ) 

    def forward(self, x):
        x = self.tcn(x.permute(0, 2, 1))
        x = self.predict(x.permute(1, 0, 2, 3))
        return torch.softmax(x, dim=1)


class CNN1DEncode(nn.Module):
    def __init__(self, feature_dim, out_dim) -> None:
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv1d(feature_dim, out_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.4),
            nn.Dropout1d(p=0.4),
            nn.Conv1d(out_dim, out_dim*2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.4),
            nn.Dropout1d(p=0.4),
            nn.Conv1d(out_dim*2, out_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.4),
            nn.Dropout1d(p=0.4),
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.4),
            nn.Dropout1d(p=0.4),
        )
    def forward(self, x):
        return self.encode(x.permute(0, 2, 1)).permute(0, 2, 1)
