# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 22:18:07 2024

@author: talha
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming binary cross-entropy loss for seizure confidence
# and cross-entropy loss for classification scores
bce_loss = nn.BCEWithLogitsLoss()
ce_loss = nn.CrossEntropyLoss()

def get_loss(outputs, seizure_conf_label, cls_score_label, aux_weight=0.4, main_weight=0.6):
    total_loss = 0.0

    # Calculate and weight loss for each auxiliary output
    for key in outputs:
        if key != 'fusion_outputs':  # Handle auxiliary outputs
            seizure_conf_loss = bce_loss(outputs[key][0], seizure_conf_label)
            cls_score_loss = ce_loss(outputs[key][1].squeeze(), cls_score_label)
            aux_loss = seizure_conf_loss + cls_score_loss  # Combine losses for this output
            total_loss += aux_weight * aux_loss  # Apply auxiliary weight directly

    # Calculate loss for main (fusion) output
    fusion_seizure_conf_loss = bce_loss(outputs['fusion_outputs'][0], seizure_conf_label)
    fusion_cls_score_loss = ce_loss(outputs['fusion_outputs'][1].squeeze(), cls_score_label)
    fusion_loss = fusion_seizure_conf_loss + fusion_cls_score_loss

    # Add weighted main loss to total
    total_loss += main_weight * fusion_loss

    # No need to normalize auxiliary losses as each is weighted individually
    return total_loss

# # Example usage
# # Assuming dummy labels for demonstration
# batch_size = 4
# num_classes = 5
# seizure_conf_label = torch.rand((batch_size, 1))  #
# cls_score_label = torch.rand((batch_size, 5))#torch.empty(batch_size, dtype=torch.long).random_(num_classes)  # Example multi-class labels

# # Assuming 'x' is the output from your model
# loss = calculate_loss(x, seizure_conf_label, cls_score_label)

# print(loss)
