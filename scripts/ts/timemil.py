import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import random


from math import ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    """Convolutional module: Conv1D + BatchNorm + (optional) ReLU."""

    def __init__(
        self,
        n_in_channels: int,
        n_out_channels: int,
        kernel_size: int,
        padding_mode: str = "replicate",
        include_relu: bool = True,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv1d(
                in_channels=n_in_channels,
                out_channels=n_out_channels,
                kernel_size=kernel_size,
                padding="same",
                padding_mode=padding_mode,
            ),
            nn.BatchNorm1d(num_features=n_out_channels),
        ]
        if include_relu:
            layers.append(nn.ReLU())
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        return out


def manual_pad(x: torch.Tensor, min_length: int) -> torch.Tensor:
    """
    Manual padding function that pads x to a minimum length with replicate padding.
    PyTorch padding complains if x is too short relative to the desired pad size, hence this function.

    :param x: Input tensor to be padded.
    :param min_length: Length to which the tensor will be padded.
    :return: Padded tensor of length min_length.
    """
    # Calculate amount of padding required
    pad_amount = min_length - x.shape[-1]
    # Split either side
    pad_left = pad_amount // 2
    pad_right = pad_amount - pad_left
    # Pad left (replicate first value)
    # print(x.shape)
    # print(pad_left)
    # pad_x = F.pad(x, [pad_left, 0], mode="constant", value=x[:, :, 0].item())
    pad_x = F.pad(x, [pad_left, 0], mode="constant", value=0.)
    # Pad right (replicate last value)
    # pad_x = F.pad(pad_x, [0, pad_right], mode="constant", value=x[:, :, -1].item())
    pad_x = F.pad(pad_x, [0, pad_right], mode="constant", value=0.)
    # print(pad_x.shape)
    return pad_x


class InceptionTimeFeatureExtractor(nn.Module):

    def __init__(
        self,
        n_in_channels: int,
        out_channels: int = 32,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.instance_encoder = nn.Sequential(
            InceptionBlock(n_in_channels, out_channels=out_channels, padding_mode=padding_mode),
            InceptionBlock(out_channels * 4, out_channels, padding_mode=padding_mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch doesn't like replicate padding if the input tensor is too small, so pad manually to min length
        min_len = 21
        if x.shape[-1] >= min_len:
            return self.instance_encoder(x)
        else:
            padded_x = manual_pad(x, min_len)
            return self.instance_encoder(padded_x)


class InceptionBlock(nn.Module):
    """Inception block of three Inception modules, where each module has a residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        bottleneck_channels: int = 32,
        padding_mode: str = "replicate",
        n_modules: int = 3,
    ) -> None:
        super().__init__()
        # Create Inception modules that are run sequentially
        inception_modules = []
        for i in range(n_modules):
            inception_modules.append(
                InceptionModule(
                    in_channels=in_channels if i == 0 else out_channels * 4,
                    out_channels=out_channels,
                    bottleneck_channels=bottleneck_channels,
                    padding_mode=padding_mode,
                ),
            )
        self.inception_modules = nn.Sequential(*inception_modules)

        # Create residual that is run in parallel to the Inception modules
        self.residual = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=4 * out_channels,
                kernel_size=1,
                padding="same",
                padding_mode=padding_mode,
            ),
            nn.BatchNorm1d(num_features=4 * out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_modules = self.inception_modules(x)
        x_residual = self.residual(x)
        return F.relu(x_modules + x_residual)


class InceptionModule(nn.Module):
    """Inception module with bottleneck, conv layers, and max pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        bottleneck_channels: int = 32,
        padding_mode: str = "replicate",
    ) -> None:
        super().__init__()

        # Setup bottleneck
        self.bottleneck: nn.Module
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                padding="same",
                padding_mode=padding_mode,
            )
        else:
            self.bottleneck = nn.Identity()
            bottleneck_channels = 1

        # Set up conv layers but don't stack sequentially as these will be run in parallel
        self.conv_layers = nn.ModuleList()
        for kernel_size in [10, 20, 40]:
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=bottleneck_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                    padding_mode=padding_mode,
                )
            )

        # Set up max pooling with bottleneck
        self.max_pooling_w_bottleneck = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, padding=1, stride=1),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
                padding_mode=padding_mode,
            ),
        )

        self.activation = nn.Sequential(nn.BatchNorm1d(num_features=4 * out_channels), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply bottleneck
        x_bottleneck = self.bottleneck(x)
        # Pass through conv layers and max pooling in parallel
        z0 = self.conv_layers[0](x_bottleneck)
        z1 = self.conv_layers[1](x_bottleneck)
        z2 = self.conv_layers[2](x_bottleneck)
        z3 = self.max_pooling_w_bottleneck(x)
        # Stack and pass through activation
        z = torch.cat([z0, z1, z2, z3], dim=1)
        z = self.activation(z)
        return z


def exists(val):
    return val is not None

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

# main attention class

class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask = None, return_attn = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks
        
        # print(x.shape)
        
        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)
            
            # print(x[0,1:20,:])
            if exists(mask):
                mask = F.pad(mask, (padding, 0), value = False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        
        
        # print(n)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out

# transformer

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Nystromformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        attn_values_residual = True,
        attn_values_residual_conv_kernel = 33,
        attn_dropout = 0.,
        ff_dropout = 0.   
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, NystromAttention(dim = dim, dim_head = dim_head, heads = heads, num_landmarks = num_landmarks, pinv_iterations = pinv_iterations, residual = attn_values_residual, residual_conv_kernel = attn_values_residual_conv_kernel, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, dropout = ff_dropout))
            ]))

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x
        return x
    

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dropout=0.2,dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=dropout
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x
    

### Define Wavelet Kernel
def mexican_hat_wavelet(size, scale, shift): #size :d*kernelsize  scale:d*1 shift:d*1
    """
    Generate a Mexican Hat wavelet kernel.

    Parameters:
    size (int): Size of the kernel.
    scale (float): Scale of the wavelet.
    shift (float): Shift of the wavelet.

    Returns:
    torch.Tensor: Mexican Hat wavelet kernel.
    """
  
    x = torch.linspace(-( size[1]-1)//2, ( size[1]-1)//2, size[1]).cuda()
    # print(x.shape)
    x = x.reshape(1,-1).repeat(size[0],1)
    # print(x.shape)
    # print(shift.shape)
    x = x - shift  # Apply the shift

    # Mexican Hat wavelet formula
    C = 2 / ( 3**0.5 * torch.pi**0.25)
    wavelet = C * (1 - (x/scale)**2) * torch.exp(-(x/scale)**2 / 2)*1  /(torch.abs(scale)**0.5)

    return wavelet #d*L

class WaveletEncoding(nn.Module):
    def __init__(self, dim=512, max_len = 256,hidden_len = 512,dropout=0.0):
        super().__init__()
       
        #n_w =3
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.proj_3 = nn.Linear(dim, dim)
        
         
        
    def forward(self,x,wave1,wave2,wave3):
        cls_token, feat_token = x[:, 0], x[:, 1:]
        
        x = feat_token.transpose(1, 2)
        
        
        D = x.shape[1]
        scale1, shift1 =wave1[0,:],wave1[1,:]
        wavelet_kernel1 = mexican_hat_wavelet(size=(D,19), scale=scale1, shift=shift1)
        scale2, shift2 =wave2[0,:],wave2[1,:]
        wavelet_kernel2 = mexican_hat_wavelet(size=(D,19), scale=scale2, shift=shift2)
        scale3, shift3 =wave3[0,:],wave3[1,:]
        wavelet_kernel3 = mexican_hat_wavelet(size=(D,19), scale=scale3, shift=shift3)
        
         #Eq. 11
        pos1= torch.nn.functional.conv1d(x,wavelet_kernel1.unsqueeze(1),groups=D,padding ='same')
        pos2= torch.nn.functional.conv1d(x,wavelet_kernel2.unsqueeze(1),groups=D,padding ='same')
        pos3= torch.nn.functional.conv1d(x,wavelet_kernel3.unsqueeze(1),groups=D,padding ='same')
        x = x.transpose(1, 2)   #B*N*D
        # print(x.shape)
        
        

        #Eq. 10
        x = x + self.proj_1(pos1.transpose(1, 2)+pos2.transpose(1, 2)+pos3.transpose(1, 2))# + mixup_encording
        
        # mixup token information
        
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TimeMIL(nn.Module):
    def __init__(self, in_features, n_classes=2, mDim=64, max_seq_len=400,dropout=0.):
        super().__init__()
     

        # define backbone Can be replace here
        self.feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=in_features )
        
       

            
        # define WPE    
        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
        self.wave1 = torch.randn(2, mDim,1 )
        self.wave1[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 )  #make sure scale >0
        self.wave1 = nn.Parameter(self.wave1)
        
        self.wave2 = torch.zeros(2, mDim,1 )
        self.wave2[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave2 = nn.Parameter(self.wave2)
        
        self.wave3 = torch.zeros(2, mDim,1 )
        self.wave3[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave3 = nn.Parameter(self.wave3)    
            
        self.wave1_ = torch.randn(2, mDim,1 )
        self.wave1_[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave1_ = nn.Parameter(self.wave1)
        
        self.wave2_ = torch.zeros(2, mDim,1 )
        self.wave2_[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave2_ = nn.Parameter(self.wave2)
        
        self.wave3_ = torch.zeros(2, mDim,1 )
        self.wave3_[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave3_ = nn.Parameter(self.wave3)        
            
            
            
            
        hidden_len = 2* max_seq_len
            
        # define class token      
        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
        self.pos_layer =  WaveletEncoding(mDim,max_seq_len,hidden_len) 
        self.pos_layer2 =  WaveletEncoding(mDim,max_seq_len,hidden_len) 
        # self.pos_layer = ConvPosEncoding1D(mDim)
        self.layer1 = TransLayer(dim=mDim,dropout=dropout)
        self.layer2 = TransLayer(dim=mDim,dropout=dropout)
        self.norm = nn.LayerNorm(mDim)
        # self._fc2 = nn.Linear(mDim, n_classes)
        self._fc2 = nn.Sequential(
            nn.Linear(mDim,mDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mDim, n_classes)
        ) 
        
        
        self.alpha = nn.Parameter(torch.ones(1))
        
        

        
        initialize_weights(self)
        
        
    def forward(self, x,warmup=False):
        

        x1 = self.feature_extractor(x)
        x1 = x1.transpose(1, 2)
        x= x1

        

        B, seq_len, D = x.shape

        
        view_x = x.clone()
        
      
        global_token = x.mean(dim=1)#[0]
      
        
        
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)    #B * 1 * d
      
        x = torch.cat((cls_tokens, x), dim=1)
        # WPE1
        x = self.pos_layer(x,self.wave1,self.wave2,self.wave3)
      
        # TransLayer x1
        x = self.layer1(x)
        # WPE2
        x = self.pos_layer2(x,self.wave1_,self.wave2_,self.wave3_)
        
       
        # TransLayer x2
        
       
        x = self.layer2(x)
       
        # only cls_token is used for cls
        x = x[:,0]
   
        
        # stablity of training random initialized global token
        if warmup:
            x = 0.1*x+0.99*global_token   
 
        logits = self._fc2(x)
            
        return logits

if __name__ == "__main__":
    x = torch.randn(3, 19, 400).cuda()
    model = TimeMIL(in_features=19,n_classes=2,mDim=128).cuda()
    ylogits =model(x)
    print(ylogits.shape)

