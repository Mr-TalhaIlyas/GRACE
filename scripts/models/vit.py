#%%
# import os, psutil
# # os.chdir(os.path.dirname(__file__))
# os.chdir('/home/talha/Data/mme/ecg_test/')

# from config import config

# # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# # The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"] = "0";
import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=5000):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]
        #print ('X in pos embded',x.size())
        position_embeddings = self.pe(position_ids)
        #print ('Position emded',position_embeddings.size())
        position_embeddings = position_embeddings[:,:x.shape[1]]
        #print ('New pos emd', position_embeddings.size())
        return x + position_embeddings

class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(
                                dim, heads=heads, dropout_rate=attn_dropout_rate
                            ),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

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

class DilatedPropB(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilations, num_feat_maps=12):
        super(DilatedPropB, self).__init__()

        self.dilation_fac = dilations

        self.dil_cnv_block = torch.nn.ModuleList(
            nn.Conv1d(in_channels, num_feat_maps, kernel_size=3, padding=self.p(d, 3), dilation=d) for d in
            self.dilation_fac
        )

        self.gate = torch.nn.Conv1d(num_feat_maps*len(self.dilation_fac) + in_channels, in_channels, kernel_size=1)
        self.cnv1d_transform = nn.Conv1d(in_channels, out_channels, kernel_size=3, dilation=1, padding=1)

        self.bn_input = nn.BatchNorm1d(in_channels)
        self.bn_output = nn.BatchNorm1d(out_channels)

    def p(self, d, k):
        return int((d * (k - 1)) / 2)

    def forward(self, x):
        x = self.bn_input(x)
        residual = x
        # dilated propagation
        x_dil = []
        for layer in self.dil_cnv_block:
            x_dil.append(layer(x))
        x_dil = torch.cat(x_dil, dim=1)

        x_dil = F.relu(x_dil)
        x_dil = F.dropout(x_dil, 0.4)

        x = torch.cat([residual, x_dil], dim=1)        
        x = self.gate(x).relu()
        
        x = self.cnv1d_transform(x).relu()
        x = F.dropout(x, 0.4)
        x = self.bn_output(x)

        return x
    
class DILATED_Encode(torch.nn.Module):
    def __init__(self):
        super(DILATED_Encode, self).__init__()

        self.dl1 = DilatedPropB(19, 24, dilations=[1, 2, 3, 4, 5, 7], num_feat_maps=6)
        self.dl2 = DilatedPropB(24, 24, dilations=[1, 2, 3, 4, 5], num_feat_maps=8)
        self.dl3 = DilatedPropB(24, 32, dilations=[1, 2, 3, 4, 5], num_feat_maps=12)
        self.dl4 = DilatedPropB(32, 32, dilations=[1, 2, 3, 4, 5], num_feat_maps=12)
        self.dl5 = DilatedPropB(32, 48, dilations=[1, 2, 3, 4], num_feat_maps=16)
        self.dl6 = DilatedPropB(48, 48, dilations=[1, 2, 3, 4], num_feat_maps=16)
        self.dl7 = DilatedPropB(48, 64, dilations=[1, 2, 3], num_feat_maps=36)
        self.dl8 = DilatedPropB(64, 72, dilations=[1, 2, 3], num_feat_maps=42)
        self.dl9 = DilatedPropB(72, 72, dilations=[1, 2], num_feat_maps=42)
        self.dl10 = DilatedPropB(72, 72, dilations=[1], num_feat_maps=48)

        ## weigths initialization wih xavier method
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                torch.init.xavier_normal_(m.weight)
                torch.init.constant_(m.bias, 0)

    def forward(self, x):
        if len(x.shape) != 3:
            x = x.unsqueeze(dim=1)

        x = self.dl1(x)
        x = self.dl2(x)
        x = F.avg_pool1d(x, kernel_size=3)

        x = self.dl3(x)
        x = self.dl4(x)

        x = self.dl5(x)
        x = self.dl6(x)
        x = F.avg_pool1d(x, kernel_size=3)

        x = self.dl7(x)
        x = self.dl8(x)

        x = self.dl9(x)
        x = self.dl10(x)

        return x
    
class VIT(nn.Module):
    def __init__(
        self,
        input_dim,
        time_dim,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        class_dim=4,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        positional_encoding_type="learned",
        return_embedding = False,
        mode = 'lin'
    ):
        super(VIT, self).__init__()

        assert embedding_dim % num_heads == 0
        
        # transformer parameters
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.time_dim = time_dim
        self.num_layers = num_layers
        
        # dropouts
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate

        # class-token
        self.seq_length = time_dim + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))        
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)
        self.encoder = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        
        self.linear_encoding = nn.Linear(self.input_dim, embedding_dim)
        self.pre_head_ln = nn.LayerNorm(embedding_dim)
        self.data_ln = nn.LayerNorm(self.input_dim)
        

        self.classifier = nn.Linear(embedding_dim, class_dim)
        self.mod_feats = nn.Linear(embedding_dim, 256)  #config['mod_feats']

        self.return_embedding = return_embedding
        if mode =='conv': 
            self.ecgdim = CNN1DEncode(input_dim, embedding_dim)
        else:
            print('default, a linear transform')
            self.ecgdim = nn.Linear(input_dim, embedding_dim)

    def forward(self, xt):
        
        x = self.data_ln(xt)
        x = self.ecgdim(x)

        x = self.pre_head_ln(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        # add the class-tokens
        x = torch.cat((x, cls_tokens), dim=1)       
        x = self.position_encoding(x)
        x = self.pe_dropout(x) 

        # apply transformer
        z = self.encoder(x)
        z = self.pre_head_ln(z)

        # compute the output from the class-token
        y = self.classifier(z[:, -1, :].flatten(start_dim=1))
        f = self.mod_feats(z[:, -1, :].flatten(start_dim=1))
        
        # If return_embedding is True, return the feature embedding before the classifier
        if self.return_embedding:
            return z[:, -1, :]
        else:
            return y, f  # torch.softmax(y, dim=-1)
    
class DILVIT(nn.Module):
    def __init__(self, embedding_dim = 128, 
                       num_heads = 8, 
                       num_layers = 4,#8
                       hidden_dim = 64,
                       num_classes=3,
                       return_embedding=False,
                       pretrained_path=None):
        super().__init__()

        self.encode = DILATED_Encode()
        self.vit = VIT(time_dim=277, 
                       input_dim = 72, 
                       embedding_dim = embedding_dim, 
                       num_heads = num_heads, 
                       num_layers = num_layers,
                       hidden_dim = hidden_dim, 
                       dropout_rate=0.5, 
                       attn_dropout_rate=0.2,
                       class_dim=num_classes,
                       return_embedding=return_embedding)
        if pretrained_path is not None:
            self.init_weights(pretrained_path)
        

    def forward(self, x):
        x = self.encode(x)
        x, f = self.vit(x.permute(0, 2, 1))
        
        return x, f
    
    def init_weights(self, pretrained_path=None):
        print('Loading ECG-ViT pretrained chkpts...')
        if pretrained_path is not None:
            chkpt = torch.load(pretrained_path)

            pretrained_dict = chkpt['model_state_dict']
            # load model state dict
            state = self.state_dict()
            # loop over both dicts and make a new dict where name and the shape of new state match
            # with the pretrained state dict.
            matched, unmatched = [], []
            new_dict = {}
            for i, j in zip(pretrained_dict.items(), state.items()):
                pk, pv = i # pretrained state dictionary
                nk, nv = j # new state dictionary
                # if name and weight shape are same
                if pk == nk:# and pv.shape == nv.shape: #.strip('backbone.')
                    new_dict[nk] = pv
                    matched.append(pk)
                elif pv.shape == nv.shape:
                    new_dict[nk] = pv
                    matched.append(pk)
                else:
                    unmatched.append(pk)

            state.update(new_dict)
            self.load_state_dict(state, strict=False)
            print('Pre-trained ECG-ViT state loaded successfully...')
            print(f'Mathed kyes: {len(matched)}, Unmatched Keys: {len(unmatched)}')
            print(40*'=')
            print(unmatched)
    

#%%
# model = DILVIT()
# # # encode = DILATED_Encode()
# x = torch.randn((5,19,2500))

# y, f = model(x)
# print('Y shape:', y.shape)
# #%
# encode = DILATED_Encode()
# z = encode(x)
# print('Z shape:', z.shape)

# # %%
