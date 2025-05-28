import torch
import torch.nn as nn
import torch.nn.functional as F
from attn import FullAttention, ProbAttention, AttentionLayer
from Transformer import TransformerModel
from PositionalEncoding import FixedPositionalEncoding, LearnedPositionalEncoding
from cnn import CNN1DEncode

__all__ = ['ViT_B16', 'ViT_B32', 'ViT_L16', 'ViT_L32', 'ViT_H14']

class VisionTransformerSeqtoSeq(nn.Module):
    def __init__(
        self,
        input_dim,
        time_dim,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        positional_encoding_type="learned",
        return_embedding = False
    ):
        super(VisionTransformerSeqtoSeq, self).__init__()

        assert embedding_dim % num_heads == 0
        
        # transformer parameters
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
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
        self.classifier = nn.Linear(embedding_dim, 4)

        self.return_embedding = return_embedding

    def forward(self, x):
        
        x = self.data_ln(x)
        x = self.linear_encoding(x)

        x = self.pre_head_ln(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        # add the class-tokens
        x = torch.cat((x, cls_tokens), dim=1)       
        x = self.position_encoding(x)
        x = self.pe_dropout(x) 

        # apply transformer
        x = self.encoder(x)
        x = self.pre_head_ln(x)

        return self.classifier(x[:, :-1, ]).softmax(dim=-1)

class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        time_dim,
        embedding_dim,
        num_heads,
        num_layers,
        dec_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        positional_encoding_type="learned",
        return_embedding = False
    ):
        super(VisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        
        # transformer parameters
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.time_dim = time_dim
        
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
        self.classifier = nn.Linear(embedding_dim, 4)

        self.return_embedding = return_embedding
        self.ecgdim = CNN_DIM(1, 112, 128)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8, batch_first=True),
            num_layers=dec_layers
        )

    def forward(self, xt, xtp1):
        
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

        # decode
        with torch.no_grad():
            xtp1 = self.ecgdim(xtp1)

        #xtp1_hat = self.decoder(
        #    tgt=xtp1, memory=z[:, :self.time_dim, :], tgt_mask=mask
        #)

        return torch.softmax(y, dim=-1)

class CNN_DIM(nn.Module):
    def __init__(self, inc, embdim, outemb_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(inc, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.project = nn.Linear(embdim, outemb_dim)
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=1)
        x =  self.encoder(x).permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.project(x)

class VisionTransformerVanilla(nn.Module):
    def __init__(
        self,
        input_dim,
        time_dim,
        embedding_dim,
        num_heads,
        num_layers,
        dec_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        positional_encoding_type="learned",
        return_embedding = False
    ):
        super(VisionTransformerVanilla, self).__init__()

        assert embedding_dim % num_heads == 0
        
        # transformer parameters
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.time_dim = time_dim
        
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
        self.classifier = nn.Linear(embedding_dim, 4)

        self.return_embedding = return_embedding
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8, batch_first=True),
            num_layers=dec_layers
        )

    def forward(self, xt, xtp1, mask):
        
        x = self.data_ln(xt)
        x = self.linear_encoding(x)

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

        # decode
        xtp1 = self.linear_encoding(xtp1)
        
        xtp1_hat = self.decoder(
            tgt=xtp1, memory=z[:, :self.time_dim, :], tgt_mask=mask
        )

        return torch.softmax(y, dim=-1), xtp1_hat

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

        return torch.softmax(y, dim=-1)

class VIT_SEQ(nn.Module):
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
        super(VIT_SEQ, self).__init__()

        assert embedding_dim % num_heads == 0
        
        # transformer parameters
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
        
        self.pre_head_ln = nn.LayerNorm(embedding_dim)
        self.data_ln = nn.LayerNorm(input_dim)
        self.project = nn.Linear(input_dim, embedding_dim)
        
        self.classifier = nn.Linear(embedding_dim, class_dim)

        self.return_embedding = return_embedding

    def forward(self, xt):
        
        x = self.data_ln(xt)
        x = self.project(x)

        x = self.pre_head_ln(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        # add the class-tokens
        x = torch.cat((x, cls_tokens), dim=1)       
        x = self.position_encoding(x)
        x = self.pe_dropout(x) 

        # apply transformer
        z = self.encoder(x)
        z = self.pre_head_ln(z)

        return z

class VIT_TC(nn.Module):
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
        positional_encoding_type="fixed",
        return_embedding = False,
        mode = 'lin'
    ):
        super(VIT_TC, self).__init__()

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
        self.seq_length = time_dim + 1 + 1
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))        
        self.trans_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))        
        
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
        self.transition = nn.Linear(embedding_dim, 2)

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
        trans_token = self.trans_token.expand(x.shape[0], -1, -1)


        # add the class-tokens
        x = torch.cat((x, cls_tokens, trans_token), dim=1)       
        x = self.position_encoding(x)
        x = self.pe_dropout(x) 

        # apply transformer
        z = self.encoder(x)
        z = self.pre_head_ln(z)

        # compute the output from the class-token
        y = self.classifier(z[:, -2, :].flatten(start_dim=1)) 
        t = self.transition(z[:, -1, :].flatten(start_dim=1))

        if self.return_embedding: 
            return torch.softmax(y, dim=-1), torch.softmax(t, dim=-1), z[:, :-2, :]

        return torch.softmax(y, dim=-1), torch.softmax(t, dim=-1)

class VIT_M(nn.Module):
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
        super(VIT_M, self).__init__()

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
        self.seq_length = time_dim + 1 + 1
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))        
        self.metaclass_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))        
        
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
        
        self.classifier = nn.Linear(embedding_dim, 2)
        self.metaclassifier = nn.Linear(embedding_dim, class_dim)

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
        metacls_tokens = self.metaclass_token.expand(x.shape[0], -1, -1)

        # add the class-tokens
        x = torch.cat((x, metacls_tokens, cls_tokens), dim=1)       
        x = self.position_encoding(x)
        x = self.pe_dropout(x) 

        # apply transformer
        z = self.encoder(x)
        z = self.pre_head_ln(z)

        # compute the output from the class-token
        y = self.classifier(z[:, -1, :].flatten(start_dim=1))
        ymeta = self.metaclassifier(z[:, -2, :].flatten(start_dim=1))  

        return torch.softmax(y, dim=-1), torch.softmax(ymeta, dim=-1)
    
class VIT_MT(nn.Module):
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
        super(VIT_MT, self).__init__()

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
        self.seq_length = time_dim + 1 + 1 + 1
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))        
        self.metaclass_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))  
        self.action_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))        
        
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
        
        self.classifier = nn.Linear(embedding_dim, 2)
        self.metaclassifier = nn.Linear(embedding_dim, class_dim)
        self.anticipation = nn.Linear(embedding_dim, 2)

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
        metacls_tokens = self.metaclass_token.expand(x.shape[0], -1, -1)
        action_tokens = self.action_token.expand(x.shape[0], -1, -1)

        # add the class-tokens
        x = torch.cat((x, action_tokens, metacls_tokens, cls_tokens), dim=1)       
        x = self.position_encoding(x)
        x = self.pe_dropout(x) 

        # apply transformer
        z = self.encoder(x)
        z = self.pre_head_ln(z)

        # compute the output from the class-token
        y = self.classifier(z[:, -1, :].flatten(start_dim=1))
        ymeta = self.metaclassifier(z[:, -2, :].flatten(start_dim=1))  
        transition = self.anticipation(z[:, -3, :].flatten(start_dim=1))  

        return torch.softmax(y, dim=-1), torch.softmax(ymeta, dim=-1), torch.softmax(transition, dim=-1)

class Transformer(nn.Module):
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
        super(Transformer, self).__init__()

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

        self.return_embedding = return_embedding
        if mode =='conv': 
            self.ecgdim = CNN1DEncode(input_dim, embedding_dim)
        else:
            print('default, a linear transform')
            self.ecgdim = nn.Linear(input_dim, embedding_dim)

        # transformer-decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

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

        return torch.softmax(y, dim=-1)

#%% 
# Usage
# model = Transformer(input_dim=1024,
#                     time_dim=8, #<- # of segments
#                     embedding_dim=128,
#                     num_heads=8,
#                     num_layers=3,
#                     hidden_dim=64)

# x = torch.randn((7,8,1024))

# y = model(x)