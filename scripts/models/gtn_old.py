#%%
#  https://github.com/ZZUFaceBookDL/GTN/
import torch
from torch.nn import Module
from torch.nn import ModuleList
import math
import torch.nn.functional as F

class Encoder(Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool = False,
                 dropout: float = 0.1):
        super(Encoder, self).__init__()

        self.MHA = MultiHeadAttention(d_model=d_model, q=q, v=v, h=h, mask=mask, device=device, dropout=dropout)
        self.feedforward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layerNormal_1 = torch.nn.LayerNorm(d_model)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model)

    def forward(self, x, stage):

        residual = x
        x, score = self.MHA(x, stage)
        x = self.dropout(x)
        x = self.layerNormal_1(x + residual)

        residual = x
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.layerNormal_2(x + residual)

        return x, score

class FeedForward(Module):
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
    

class MultiHeadAttention(Module):
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool=False,
                 dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.W_q = torch.nn.Linear(d_model, q * h)
        self.W_k = torch.nn.Linear(d_model, q * h)
        self.W_v = torch.nn.Linear(d_model, v * h)

        self.W_o = torch.nn.Linear(v * h, d_model)

        self.device = device
        self._h = h
        self._q = q

        self.mask = mask
        self.dropout = torch.nn.Dropout(p=dropout)
        self.score = None

    def forward(self, x, stage):
        Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0)
        K = torch.cat(self.W_k(x).chunk(self._h, dim=-1), dim=0)
        V = torch.cat(self.W_v(x).chunk(self._h, dim=-1), dim=0)

        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        self.score = score

        if self.mask and stage == 'train':
            mask = torch.ones_like(score[0])
            mask = torch.tril(mask, diagonal=0)
            score = torch.where(mask > 0, score, torch.Tensor([-2**32+1]).expand_as(score[0]).to(self.device))

        score = F.softmax(score, dim=-1)

        attention = torch.matmul(score, V)

        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        self_attention = self.W_o(attention_heads)

        return self_attention, self.score
    


class GTN(Module):
    def __init__(self,
                 d_model: int,
                 d_input: int,
                 d_channel: int,
                 d_output: int,
                 d_hidden: int,
                 fusion_dim: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 device: str,
                 dropout: float = 0.1,
                 pe: bool = False,
                 mask: bool = False):
        super(GTN, self).__init__()

        self.encoder_list_1 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  mask=mask,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.encoder_list_2 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.embedding_channel = torch.nn.Linear(d_channel, d_model)
        self.embedding_input = torch.nn.Linear(d_input, d_model)

        self.gate = torch.nn.Linear(d_model * d_input + d_model * d_channel, 2)
        # self.feature_gather = torch.nn.Linear(d_model * d_input + d_model * d_channel, fusion_dim)
        self.output_linear = torch.nn.Linear(d_model * d_input + d_model * d_channel, d_output)

        self.pe = pe
        self._d_input = d_input
        self._d_model = d_model

    def forward(self, x, stage='train'):
        """
        Forward Propagation
        :param x: Input
        :param stage: Used to describe whether this is the training process of the training set or the testing process of the test set (mask mechanism is not applied during testing)
        :return: Output, 2D vector after gate, score matrix in step-wise encoder, score matrix in channel-wise encoder, 3D matrix after step-wise embedding, 3D matrix after channel-wise embedding, gate
        """
        # step-wise
        # The score matrix is input, with mask and positional encoding (pe) added by default
        encoding_1 = self.embedding_channel(x)
        input_to_gather = encoding_1

        if self.pe:
            pe = torch.ones_like(encoding_1[0])
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            encoding_1 = encoding_1 + pe

        for encoder in self.encoder_list_1:
            encoding_1, score_input = encoder(encoding_1, stage)

        # channel-wise
        # score矩阵为channel 默认不加mask和pe
        encoding_2 = self.embedding_input(x.transpose(-1, -2))
        channel_to_gather = encoding_2

        for encoder in self.encoder_list_2:
            encoding_2, score_channel = encoder(encoding_2, stage)

        # 三维变二维
        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)

        # gate
        gate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2], dim=-1)), dim=-1)
        encoding = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1)

        # fusion features
        # op_feats = self.feature_gather(encoding)
        # class scores
        output = self.output_linear(encoding)

        # return output, encoding, score_input, score_channel, input_to_gather, channel_to_gather, gate
        return output#, op_feats
    
    
#%%
    
# model = GTN(d_model=512, d_hidden=256,
#             d_input=2500, d_channel=19,
#             fusion_dim=512,
#             d_output=2, # no. of classes
#             q=8, v=8, h=8, N=8,
#             device='cuda', dropout=0.2,
#             pe=True, mask=True)

# model = model.to('cuda')
# #%%
# x = torch.randn(8, 2500, 19).to('cuda')

# y = model(x, 'train')

# # for i in y:
# #     print(i.shape)
# #%%
# from torchinfo import summary
# summary(model, input_size=(8, 2500, 19), depth=5)


# #%%
# from torchview import draw_graph
# model = GTN(d_model=512, d_hidden=256,
#             d_input=2500, d_channel=19,
#             fusion_dim=512,
#             d_output=2, # no. of classes
#             q=8, v=8, h=8, N=2,
#             device='meta', dropout=0.2,
#             pe=True, mask=True)
# x = torch.randn(3, 2500, 19)#.to('cuda')
# model_graph = draw_graph(model, input_data=x, device='meta', depth=1)
# model_graph.visual_graph



#%%




# encoding, score_input, score_channel, input_to_gather, channel_to_gather, gate
    
# torch.Size([5, 2])
# torch.Size([5, 1289728])
# torch.Size([40, 2500, 2500])
# torch.Size([40, 19, 19])
# torch.Size([5, 2500, 512])
# torch.Size([5, 19, 512])
# torch.Size([5, 2])
#%%