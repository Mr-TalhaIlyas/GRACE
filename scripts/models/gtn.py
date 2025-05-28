#%%

import torch
from torch.nn import Module, ModuleList, MultiheadAttention
import math
import torch.nn.functional as F

class Encoder(Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 n_heads: int, # Changed from q, v, h to n_heads
                 apply_attn_mask: bool = False, # General purpose, e.g. causal for step-wise if needed
                 dropout: float = 0.1):
        super(Encoder, self).__init__()
        
        # Using PyTorch's MultiheadAttention
        # embed_dim is d_model, num_heads is n_heads
        # kdim and vdim default to embed_dim if not specified
        self.mha = MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        
        self.feedforward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layerNormal_1 = torch.nn.LayerNorm(d_model)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model)
        self.apply_attn_mask = apply_attn_mask # Store if this encoder needs a mask

    def forward(self, x, src_mask=None): # x shape: (B, S, d_model)
                                        # src_mask for nn.MultiheadAttention should be (S, S) or (B*num_heads, S, S)
        residual = x
        
        # For nn.MultiheadAttention, query, key, value are all x
        # attn_mask should be (L,S) or (N*num_heads, L,S) where L=target_len, S=source_len
        # If self.apply_attn_mask is True, we expect a causal mask of shape (S,S)
        # to be passed or generated.
        # For classification, typically no mask or only padding mask.
        # If a causal mask is needed for this encoder:
        current_attn_mask = None
        if self.apply_attn_mask and self.training: # e.g., for causal step-wise
            S = x.size(1)
            current_attn_mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
            # nn.MultiheadAttention expects True where attention is NOT allowed.

        x_mha, attn_weights = self.mha(query=x, key=x, value=x, attn_mask=current_attn_mask, need_weights=True)
        # attn_weights shape is (B, S, S) if need_weights=True and average_attn_weights=True (default)
        
        x_mha = self.dropout(x_mha)
        x = self.layerNormal_1(x_mha + residual)

        residual = x
        x_ff = self.feedforward(x)
        x_ff = self.dropout(x_ff)
        x = self.layerNormal_2(x_ff + residual)
        
        return x, attn_weights # Return attention weights for inspection

class FeedForward(Module): # Remains the same
    def __init__(self,
                 d_model: int,
                 d_hidden: int = 512):
        super(FeedForward, self).__init__()
        self.linear_1 = torch.nn.Linear(d_model, d_hidden)
        self.linear_2 = torch.nn.Linear(d_hidden, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        return x

# Custom MultiHeadAttention class is no longer needed if using nn.MultiheadAttention

class GTN(Module):
    def __init__(self,
                 d_model: int,
                 d_input: int,      # Sequence length (T)
                 d_channel: int,    # Number of channels (C)
                 d_output: int,     # Number of output classes
                 d_hidden: int,     # Hidden dim for FFN in Encoders
                 fusion_dim: int,   # Optional: intermediate dim after gating before output
                 n_heads: int,      # Number of attention heads
                 N: int,            # Number of encoder layers
                 dropout: float = 0.1,
                 pe_step_wise: bool = True,
                 # For classification, causal mask is usually False for both.
                 # Set to True if you specifically want causal attention for the time dimension.
                 apply_causal_mask_step_wise: bool = False, 
                 apply_causal_mask_channel_wise: bool = False): # Typically False
        super(GTN, self).__init__()

        self.d_model = d_model
        self._d_input = d_input # T
        self._d_channel = d_channel # C

        # Step-wise branch (operates on time steps)
        self.embedding_step_wise = torch.nn.Linear(d_channel, d_model)
        self.encoder_list_1 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden, n_heads=n_heads,
                                                  apply_attn_mask=apply_causal_mask_step_wise,
                                                  dropout=dropout)
                                           for _ in range(N)])
        self.pe_step_wise = pe_step_wise

        # Channel-wise branch (operates on channels)
        self.embedding_channel_wise = torch.nn.Linear(d_input, d_model)
        self.encoder_list_2 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden, n_heads=n_heads,
                                                  apply_attn_mask=apply_causal_mask_channel_wise, # Usually False
                                                  dropout=dropout)
                                           for _ in range(N)])

        gate_input_dim = d_model * 2
        self.gate_fc = torch.nn.Linear(gate_input_dim, 2)

        self.use_fusion_projection = fusion_dim > 0
        if self.use_fusion_projection:
            self.feature_gather = torch.nn.Linear(gate_input_dim, fusion_dim)
            self.output_linear = torch.nn.Linear(fusion_dim, d_output)
        else:
            self.output_linear = torch.nn.Linear(gate_input_dim, d_output)


    def _generate_positional_encoding(self, seq_len: int, d_model: int, device: torch.device):
        # Standard sinusoidal PE
        pe = torch.zeros(seq_len, d_model, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * \
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0: # Handle odd d_model if necessary, though usually d_model is even
             pe[:, 1::2] = torch.cos(position * div_term)[:,:d_model//2]
        else:
             pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):         
        x = x.permute(0, 2, 1).contiguous() # (B, C, T) -> (B, T, C)
        B, T, C = x.shape
        current_device = x.device

        # 1. Step-wise Encoder Branch (Temporal Attention)
        encoding_1 = self.embedding_step_wise(x) # (B, T, d_model)
        
        if self.pe_step_wise:
            pe = self._generate_positional_encoding(T, self.d_model, current_device)
            encoding_1 = encoding_1 + pe.unsqueeze(0)

        attn_weights_step_wise = None
        for encoder in self.encoder_list_1:
            encoding_1, attn_weights_step_wise = encoder(encoding_1) # No src_mask needed if not causal

        pooled_encoding_1 = encoding_1.mean(dim=1) # (B, d_model)

        # 2. Channel-wise Encoder Branch (Feature/Channel Attention)
        x_transposed = x.transpose(1, 2) # (B, C, T)
        encoding_2 = self.embedding_channel_wise(x_transposed) # (B, C, d_model)
        
        attn_weights_channel_wise = None
        for encoder in self.encoder_list_2:
            encoding_2, attn_weights_channel_wise = encoder(encoding_2) # No src_mask needed

        pooled_encoding_2 = encoding_2.mean(dim=1) # (B, d_model)

        # 3. Gating and Fusion
        concatenated_features = torch.cat([pooled_encoding_1, pooled_encoding_2], dim=-1)
        gate_weights = F.softmax(self.gate_fc(concatenated_features), dim=-1)
        
        gated_encoding_1 = pooled_encoding_1 * gate_weights[:, 0:1]
        gated_encoding_2 = pooled_encoding_2 * gate_weights[:, 1:2]
        final_encoding_representation = torch.cat([gated_encoding_1, gated_encoding_2], dim=-1)

        if self.use_fusion_projection:
            final_encoding_representation = self.feature_gather(final_encoding_representation)
        
        output = self.output_linear(final_encoding_representation)

        # Return attention weights for potential analysis/visualization
        return output, final_encoding_representation #, attn_weights_step_wise, attn_weights_channel_wise
    
#%%
# Example Usage (ensure parameters match your data)
# if __name__ == '__main__':
# batch_size = 16
# d_input_T = 2500
# d_channel_C = 19
# d_model_emb = 256
# d_hidden_ffn = 512
# d_output_classes = 2
# fusion_intermediate_dim = 512 # 0 to disable

# num_encoder_layers_N = 8
# num_heads_h = 8 # Make sure d_model_emb is divisible by num_heads_h

# if d_model_emb % num_heads_h != 0:
#     raise ValueError(f"d_model ({d_model_emb}) must be divisible by n_heads ({num_heads_h})")

# device_to_use = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = GTN(d_model=d_model_emb,
#             d_input=d_input_T,
#             d_channel=d_channel_C,
#             d_output=d_output_classes,
#             d_hidden=d_hidden_ffn,
#             fusion_dim=fusion_intermediate_dim,
#             n_heads=num_heads_h,
#             N=num_encoder_layers_N,
#             dropout=0.1,
#             pe_step_wise=True,
#             apply_causal_mask_step_wise=False, # Usually False for classification
#             apply_causal_mask_channel_wise=False # Almost always False
#             )#.to(device_to_use)

# dummy_x = torch.randn(batch_size, d_input_T, d_channel_C).to(device_to_use)
# predictions, fusion_features = model(dummy_x)

# print("Input x shape:", dummy_x.shape)
# print("Predictions shape:", predictions.shape)
# print("Fusion features shape:", fusion_features.shape)

# from torchinfo import summary
# summary(model, input_size=(batch_size, d_channel_C, d_input_T),
#          depth=2)


# %%
# from torchview import draw_graph
# model = GTN(d_model=d_model_emb,
#             d_input=d_input_T,
#             d_channel=d_channel_C,
#             d_output=d_output_classes,
#             d_hidden=d_hidden_ffn,
#             fusion_dim=fusion_intermediate_dim,
#             n_heads=num_heads_h,
#             N=num_encoder_layers_N,
#             dropout=0.1,
#             pe_step_wise=True,
#             apply_causal_mask_step_wise=False, # Usually False for classification
#             apply_causal_mask_channel_wise=False # Almost always False
#             )

# model_graph = draw_graph(model, input_data=dummy_x, device='meta', depth=1)
# model_graph.visual_graph