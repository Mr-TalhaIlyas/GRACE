#%%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
sys.path.append('/home/user01/Data/npj/scripts/')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from buffer_utils.feature_loader import SequenceFeatureDataset # Your updated dataset
from buffer_utils.buffer_model import EmformerClassifier # Your EmformerClassifier wrapper
import torch.nn.functional as F # Added
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
from tqdm import tqdm

# --- Configuration ---
BASE_FEATURE_DIR = '/home/user01/Data/npj/datasets/alfred/features/'
FEATURE_TYPE = "ecg_feats" # Single feature type as per new Dataset

SEQUENCE_LENGTH = 6  
STRIDE = 1           
BATCH_SIZE = 16      
NUM_EPOCHS = 100
LEARNING_RATE = 1e-6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Emformer parameters
# INPUT_DIM will be set dynamically
NUM_HEADS = 8
FFN_DIM = 1024 
NUM_LAYERS = 6 
DROPOUT = 0.3
EMFORMER_SEGMENT_LENGTH = SEQUENCE_LENGTH 
RIGHT_CONTEXT_LENGTH = 0 
LEFT_CONTEXT_LENGTH = 0  

NUM_CLASSES = 3 # Non-seizure, TCS, PNES (for probabilistic labels)

# --- Data Loading ---
# Using the updated SequenceFeatureDataset (which should be SequenceFeatureDatasetV2)
train_dataset = SequenceFeatureDataset(
    base_feature_dir=BASE_FEATURE_DIR,
    data_split="train",
    feature_type=FEATURE_TYPE,
    sequence_length=SEQUENCE_LENGTH,
    stride=STRIDE,
    augmentation_pad_prob=0.3, # Example, adjust as needed
    min_valid_on_pad=3          # Example, ensure min_valid_on_pad < sequence_length
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True) # Shuffle True for training

val_dataset = SequenceFeatureDataset(
    base_feature_dir=BASE_FEATURE_DIR,
    data_split="val",
    feature_type=FEATURE_TYPE,
    sequence_length=SEQUENCE_LENGTH,
    stride=STRIDE, # For validation, often stride = sequence_length for non-overlapping
    augmentation_pad_prob=0.0, # Typically no augmentation for validation
    min_valid_on_pad=5
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4, pin_memory=True)

#%%
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

INPUT_DIM = 0
if len(train_dataset) > 0:
    sample_seq, sample_lbl, sample_len = train_dataset[0]
    print(f"Sample sequence shape: {sample_seq.shape}") 
    print(f"Sample label shape: {sample_lbl.shape}, Label: {sample_lbl}")
    print(f"Sample valid length: {sample_len}")
    INPUT_DIM = sample_seq.shape[1] 
    print(f"Actual INPUT_DIM from data: {INPUT_DIM}")
else:
    print("Training dataset is empty. Cannot determine INPUT_DIM automatically.")
    # Fallback or error
    INPUT_DIM = 512 # Provide a default or raise an error
    # exit() 

if INPUT_DIM == 0:
    raise ValueError("INPUT_DIM could not be determined from the dataset.")
#%%
# --- Model Definition ---
# Use the EmformerClassifier wrapper
model = EmformerClassifier(
    input_dim=INPUT_DIM,
    num_heads=NUM_HEADS,
    ffn_dim=FFN_DIM,
    num_layers=NUM_LAYERS,
    segment_length=EMFORMER_SEGMENT_LENGTH,
    num_classes=NUM_CLASSES, # Pass num_classes to the wrapper
    dropout=DROPOUT,
    left_context_length=LEFT_CONTEXT_LENGTH,
    right_context_length=RIGHT_CONTEXT_LENGTH
).to(DEVICE)

# --- Loss and Optimizer ---
# For probabilistic targets, KLDivLoss is appropriate.
# Model output should be log-probabilities. Target should be probabilities.
# criterion = nn.KLDivLoss(reduction='batchmean') 
criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.57007489, 2.54781421, 1.17185046], device=DEVICE)) # CrossEntropyLoss for hard labels
optimizer = optim.AdamW(model.parameters(),
                        lr=LEARNING_RATE,
                        betas=(0.9, 0.99),
                        weight_decay=0.05) # Optimizer for the wrapped model

# --- Training and Validation Loop ---
best_val_f1 = 0.0
model_save_path = "/home/user01/Data/npj/scripts/buffer_utils/chkpt/best_emformer_classifier.pth"
#%%
for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    total_train_loss = 0
    train_pred_hard_labels = []
    train_target_hard_labels = []

    for sequences, labels, lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
        sequences, labels, lengths = sequences.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)
        
        optimizer.zero_grad()
        
        log_probs_output = model(sequences, lengths) # Model returns log-probabilities
        
        # KLDivLoss expects input (log_probs_output) and target (labels are already probs)
        loss = criterion(log_probs_output, labels)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        # For metrics, convert predictions and targets to hard labels
        train_pred_hard_labels.extend(torch.argmax(log_probs_output, dim=1).cpu().numpy())
        train_target_hard_labels.extend(torch.argmax(labels, dim=1).cpu().numpy()) # labels are [B, C]
        
    avg_train_loss = total_train_loss / len(train_loader)
    train_f1_macro = f1_score(train_target_hard_labels, train_pred_hard_labels, average='macro', zero_division=0)
    print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}, Train F1-Macro: {train_f1_macro:.4f}")
    # print("Train Classification Report (on hard labels):")
    # print(classification_report(train_target_hard_labels, train_pred_hard_labels, zero_division=0, target_names=["Non-S", "TCS", "PNES"]))

    # Validation
    model.eval()
    total_val_loss = 0
    val_pred_hard_labels = []
    val_target_hard_labels = []
    
    with torch.no_grad():
        for sequences, labels, lengths in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
            sequences, labels, lengths = sequences.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)
            
            log_probs_output = model(sequences, lengths)
            
            loss = criterion(log_probs_output, labels)
            total_val_loss += loss.item()
            
            val_pred_hard_labels.extend(torch.argmax(log_probs_output, dim=1).cpu().numpy())
            val_target_hard_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())
            
    avg_val_loss = total_val_loss / len(val_loader)
    val_f1_macro = f1_score(val_target_hard_labels, val_pred_hard_labels, average='macro', zero_division=0)
    print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}, Val F1-Macro: {val_f1_macro:.4f}")
    print("Validation Classification Report (on hard labels):")
    # Ensure target_names match your class order (0, 1, 2)
    print(classification_report(val_target_hard_labels, val_pred_hard_labels, zero_division=0, target_names=["Non-S", "TCS", "PNES"]))
    print("Validation Confusion Matrix (on hard labels):")
    print(confusion_matrix(val_target_hard_labels, val_pred_hard_labels, labels=[0,1,2]))

    if val_f1_macro > best_val_f1:
        best_val_f1 = val_f1_macro
        torch.save(model.state_dict(), model_save_path)
        print(f"Epoch {epoch+1}: New best model saved to {model_save_path} with Val F1-Macro: {best_val_f1:.4f}")

print("Training finished.")
print(f"Best Validation F1-Macro: {best_val_f1:.4f}")
print(f"Best model saved at: {model_save_path}")
# %%
