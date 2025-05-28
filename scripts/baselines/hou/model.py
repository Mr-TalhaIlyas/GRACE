import torch, graph
import torch.nn as nn
import torch.nn.functional as F
from agcn import PoseGCN
from cnn import Face_TCN, Pose_TCN

# Placeholder for the teacher models (pretrained appearance models)
class TCNp(nn.Module):
    def __init__(self, num_classes):
        super(TCNp, self).__init__()
        self.backbone = Pose_TCN(32)
        self.fc = nn.Linear(512, num_classes)
        # for param in self.backbone.pose_cnn.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        # xx shape: [B, 3, T, H, W]
        x = self.backbone(x)
        logits = self.fc(x)  # Output shape: [B, num_classes]
        return logits

class TCNf(nn.Module):
    def __init__(self, num_classes):
        super(TCNf, self).__init__()
        self.backbone = Face_TCN()
        self.fc = nn.Linear(512, num_classes)
        # for param in self.backbone.face_cnn.parameters():
        #     param.requires_grad = False
            
    def forward(self, x):
        x = self.backbone(x)
        logits = self.fc(x)
        return logits

# Placeholder for the student models (AGCN models)
class AGCNp(nn.Module):
    def __init__(self, num_classes):
        super(AGCNp, self).__init__()
        self.gcn = PoseGCN(num_classes=num_classes, num_persons=1,
                            backbone_in_channels=3, head_in_channels=256,
                            num_nodes=graph.num_nodes_body_hou,
                            inward_edges=graph.body_inward_edges_hou,
                            checkpoint_path=None, sup_classes=2)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: [B, 256]
        _, x = self.gcn(x)
        logits = self.fc(x)
        return logits

class AGCNf(nn.Module):
    def __init__(self, num_classes):
        super(AGCNf, self).__init__()
        self.gcn = PoseGCN(num_classes=num_classes, num_persons=1,
                            backbone_in_channels=3, head_in_channels=256,
                            num_nodes=graph.num_nodes_face_hou,
                            inward_edges=graph.face_inward_edges_hou,
                            checkpoint_path=None, sup_classes=2)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        _, x = self.gcn(x)
        logits = self.fc(x)
        return logits

def check_trainable_layers(model):
    for name, param in model.named_parameters():
        status = "Trainable" if param.requires_grad else "Frozen"
        print(f"Layer: {name}, Status: {status}")