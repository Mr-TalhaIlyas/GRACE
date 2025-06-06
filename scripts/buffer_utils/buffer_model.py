#%%
from torchaudio.models import Emformer # Make sure torchaudio is installed
import torch.nn.functional as F
import torch
import torch.nn as nn

class EmformerClassifier(nn.Module):
    def __init__(self, input_dim, num_heads, ffn_dim, num_layers, segment_length, num_classes,
                 dropout=0.1, activation="silu", left_context_length=0, right_context_length=0):
        super().__init__()
        self.emformer = Emformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            segment_length=segment_length,
            dropout=dropout,
            activation=activation,
            left_context_length=left_context_length,
            right_context_length=right_context_length
        )
        self.dropout = nn.Dropout(0.5)
        # The Emformer output dimension is the same as input_dim
        self.classification_head = nn.Linear(input_dim, num_classes)

    def forward(self, sequences, lengths):
        """
        Args:
            sequences (Tensor): Input tensor of shape (batch, num_frames, input_dim).
            lengths (Tensor): Lengths tensor of shape (batch,) representing actual sequence lengths.
        
        Returns:
            Tensor: Log-probabilities over classes, shape (batch, num_classes).
        """
        # emformer_output shape: (batch, num_frames, input_dim)
        emformer_output, _ = self.emformer(sequences, lengths) 
        
        # Pool features: Average pooling over the valid sequence length
        # We need to be careful with padding if lengths vary.
        # A robust way is to gather the last valid output or mask before pooling.
        # For simplicity with Emformer's segment processing, if all input `lengths` are equal to `segment_length`
        # (or if Emformer handles padding internally for its output based on input lengths),
        # mean pooling over dim=1 might be acceptable.
        # However, a more accurate pooling for variable lengths:
        pooled_output_list = []
        for i in range(emformer_output.size(0)): # Iterate over batch
            valid_length = lengths[i].item()
            # Take the mean of the outputs corresponding to the valid length
            pooled_output_list.append(torch.mean(emformer_output[i, :valid_length, :], dim=0))
        pooled_output = torch.stack(pooled_output_list) # Shape: (batch_size, input_dim)

        # If all lengths are guaranteed to be SEQUENCE_LENGTH (no effective padding for the model)
        # pooled_output = torch.mean(emformer_output, dim=1) 
        pooled_output = self.dropout(pooled_output)
        logits = self.classification_head(pooled_output)
        # Return log-softmax for KLDivLoss
        return F.log_softmax(logits, dim=-1)
