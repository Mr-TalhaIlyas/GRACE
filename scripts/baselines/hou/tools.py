import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import video_transform
# import sklearn.metrics as skm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cross_entropy_loss(outputs, labels):
    
    return nn.CrossEntropyLoss()(outputs, labels)

def knowledge_distillation_loss(student_logits, teacher_logits, temperature=1.0):
    """
    Knowledge distillation loss (KL divergence) between teacher and student outputs.
    Args:
        student_logits: Logits from the student model, shape [B, num_classes]
        teacher_logits: Logits from the teacher model, shape [B, num_classes]
        temperature: Temperature parameter for softening the probabilities
    """
    # Soften the probabilities
    student_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

    # KL divergence loss multiplied by temperature squared
    kd_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return kd_loss


def train_agcnp(agcnp_model, tcnp_model, train_loader, optimizer, lambda_kd=0.5, temperature=1.0):
    agcnp_model.train()
    tcnp_model.eval()  # Teacher model is in evaluation mode

    for data in train_loader:
        # shape from N*C*T*V*M -> N*M*T*V*C
        keypoint_inputs = data['body'].permute(0,4,2,3,1).to(torch.float).to(device)
        # shape from BTHWC -> BCTHW
        appearance_features = video_transform(data['frames']).to(device) # [B, C, T, H, W]
        labels = data['super_lbls'].argmax(1)  # [B*C] -> [B]
        labels = labels.type(torch.LongTensor).to(device)

        # Forward pass through AGCNp (student model)
        agcnp_outputs = agcnp_model(keypoint_inputs)  # [B, num_classes]

        # Forward pass through TCNp (teacher model)
        with torch.no_grad():
            tcnp_outputs = tcnp_model(appearance_features)  # [B, num_classes]

        # Compute losses
        ce_loss = cross_entropy_loss(agcnp_outputs, labels)
        kd_loss = knowledge_distillation_loss(agcnp_outputs, tcnp_outputs, temperature)

        total_loss = ce_loss + lambda_kd * kd_loss

        # Backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item(), ce_loss.item(), kd_loss.item()

def train_agcnf(agcnf_model, tcnf_model, train_loader, optimizer, lambda_kd=0.5, temperature=1.0):
    agcnf_model.train()
    tcnf_model.eval()  # Teacher model is in evaluation mode

    for data in train_loader:
        # shape from N*C*T*V*M -> N*M*T*V*C
        keypoint_inputs = data['face'].permute(0,4,2,3,1).to(torch.float).to(device)
        # shape from BTHWC -> BCTHW
        appearance_features = video_transform(data['face_frames']).to(device) # [B, C, T, H, W]
        labels = data['super_lbls'].argmax(1)  # [B*C] -> [B]
        labels = labels.type(torch.LongTensor).to(device)

        agcnf_outputs = agcnf_model(keypoint_inputs)  # [B, num_classes]

        with torch.no_grad():
            tcnf_outputs = tcnf_model(appearance_features)  # [B, num_classes]

        ce_loss = cross_entropy_loss(agcnf_outputs, labels)
        kd_loss = knowledge_distillation_loss(agcnf_outputs, tcnf_outputs, temperature)

        total_loss = ce_loss + lambda_kd * kd_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item(), ce_loss.item(), kd_loss.item()

def inference(agcnp_model, agcnf_model, test_loader):
    agcnp_model.eval()
    agcnf_model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            keypoint_inputs_pose = data['body'].permute(0,4,2,3,1).to(torch.float).to(device)
            keypoint_inputs_face = data['face'].permute(0,4,2,3,1).to(torch.float).to(device)
            labels = data['super_lbls'].argmax(1)  # [B*C] -> [B]
            labels = labels.type(torch.LongTensor).to(device)

            agcnp_outputs = agcnp_model(keypoint_inputs_pose)  # [B, num_classes]
            agcnf_outputs = agcnf_model(keypoint_inputs_face)  # [B, num_classes]
            agcnp_probs = F.softmax(agcnp_outputs, dim=1)
            agcnf_probs = F.softmax(agcnf_outputs, dim=1)

            combined_probs = (agcnp_probs + agcnf_probs) / 2  # Ensambling

            _, preds = torch.max(combined_probs, 1)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_predictions, all_labels

'''
PRETRAINING
'''

def pretrain_tcnp(tcnp_model, train_loader, optimizer):
    tcnp_model.train()

    for data in train_loader:
        appearance_features = video_transform(data['frames']).to(device) # [B, C, T, H, W]
        labels = data['super_lbls'].argmax(1)  # [B*C] -> [B]
        labels = labels.type(torch.LongTensor).to(device)

        tcnp_outputs = tcnp_model(appearance_features)  # [B, num_classes]

        ce_loss = cross_entropy_loss(tcnp_outputs, labels)

        optimizer.zero_grad()
        ce_loss.backward()
        optimizer.step()
        
        return ce_loss.item()

def pretrain_tcnf(tcnf_model, train_loader, optimizer):
    tcnf_model.train()

    for data in train_loader:
        appearance_features = video_transform(data['face_frames']).to(device) # [B, C, T, H, W]
        labels = data['super_lbls'].argmax(1)  # [B*C] -> [B]
        labels = labels.type(torch.LongTensor).to(device)

        tcnf_outputs = tcnf_model(appearance_features)  # [B, num_classes]

        ce_loss = cross_entropy_loss(tcnf_outputs, labels)

        optimizer.zero_grad()
        ce_loss.backward()
        optimizer.step()
        
        return ce_loss.item()

def tcn_inference(tcnp_model, tcnf_model, test_loader):
    tcnp_model.eval()
    tcnf_model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            body_features = video_transform(data['frames']).to(device) # [B, C, T, H, W]
            face_features = video_transform(data['face_frames']).to(device) # [B, C, T, H, W]
            labels = data['super_lbls'].argmax(1)  # [B*C] -> [B]
            labels = labels.type(torch.LongTensor).to(device)

            tcnp_outputs = tcnp_model(body_features)  # [B, num_classes]
            tcnf_outputs = tcnf_model(face_features)  # [B, num_classes]
            tcnp_probs = F.softmax(tcnp_outputs, dim=1)
            tcnf_probs = F.softmax(tcnf_outputs, dim=1)

            combined_probs = (tcnp_probs + tcnf_probs) / 2  # Ensambling

            _, preds = torch.max(combined_probs, 1)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_predictions, all_labels