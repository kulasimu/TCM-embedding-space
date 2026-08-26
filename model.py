import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


os.environ['KMP_DUPLICATE_LIB_OK']='True'

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward,bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model,bias=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src2, weights = self.self_attn(src, src, src, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weights


class TransformerCrossEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, activation="relu"):
        super(TransformerCrossEncoderLayer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward,bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model,bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerCrossEncoderLayer, self).__setstate__(state)

    def forward(self, src_x, src_y, src_mask=None, src_key_padding_mask=None): # src_x: herb input, src_y: symptom input
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src_x2, weights = self.cross_attn(src_x, src_y, src_y, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)
        src_x = src_x + self.dropout1(src_x2)
        src_x = self.norm1(src_x)
        src_x2 = self.linear2(self.dropout(self.activation(self.linear1(src_x))))
        src_x = src_x + self.dropout2(src_x2)
        src_x = self.norm2(src_x)
        return src_x, weights


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    __constants__ = ['norm']
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        output = src
        weights = []
        for mod in self.layers:
            output, weight = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            weights.append(weight)

        if self.norm is not None:
            output = self.norm(output)
        return output, weights


class TransformerCrossEncoder(nn.Module):
    __constants__ = ['norm']
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerCrossEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src_x, src_y, mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        output = src_x
        weights = []
        for mod in self.layers:
            output, weight = mod(output,src_y, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            weights.append(weight)

        if self.norm is not None:
            output = self.norm(output)
        return output, weights


def generate_square_mask(s):
    mask = (torch.triu(torch.ones(s,s))==1).transpose(0,1)
    mask = mask.float().masked_fill(mask==0,float('-inf')).masked_fill(mask==1,float(0.0))
    return mask


class TransformerContra(nn.Module):
    def __init__(self, sym_input_dim,herb_vocab_size,embedding_dim, hidden_size, nheads, n_layers, num_output1,num_output2,
                 dropout,sym_pad=None,herb_pad_idx=None,device='cpu'):
        super(TransformerContra, self).__init__()

        self.sym_pad = sym_pad
        self.herb_pad_idx = herb_pad_idx
        self.device= device
        # Embedding layer (for herb and symptom)
        self.herb_embedding = nn.Embedding(herb_vocab_size, embedding_dim)
        self.sym_embedding = nn.Linear(sym_input_dim, embedding_dim,bias=False)

        # Self-attention encoder layers
        self_att_layer = TransformerEncoderLayer(embedding_dim, nheads, hidden_size, dropout)
        self.sym_encoder_layers = TransformerEncoder(self_att_layer, num_layers=n_layers)
        self.herb_encoder_layers = TransformerEncoder(self_att_layer, num_layers=n_layers)


        # Cross-attention encoder layers
        cross_att_layer = TransformerCrossEncoderLayer(embedding_dim, nheads, hidden_size, dropout)
        self.cross_encoder_layers = TransformerCrossEncoder(cross_att_layer, num_layers=n_layers)

        # MLP decoder
        self.sym_decoder_layers = nn.Linear(embedding_dim, num_output1)
        self.herb_decoder_layers = nn.Linear(embedding_dim, num_output2)

    def trainableEmbedding(self,x1,x2): # only embed the herb input
        return self.sym_embedding(x1), self.herb_embedding(x2)

    def decoding(self,x12,x21):
        y1 = self.sym_decoder_layers(x12)
        y2 = self.herb_decoder_layers(x21)
        return y1,y2

    def get_last_attention_output(self):
        '''
        :return: output of last layer of cross_attention block
        '''
        return torch.mean(self.x21, dim=0), torch.mean(self.x12, dim=0)

    def forward(self, x1,x2,mask_input=None):
        # key_pad_x1 = (x1 == self.sym_pad_idx) # Symptom padding mask
        key_pad_x1 = (torch.sum(x1 == self.sym_pad, 2) > 0) # Symptom padding mask
        key_pad_x2 = (x2 == self.herb_pad_idx) # Herb padding mask
        x1,x2 = self.trainableEmbedding(x1,x2)

        x1 = x1.permute(1, 0, 2)
        x2 = x2.permute(1, 0, 2)

        if mask_input == 'herb':
            x1, w1 = self.sym_encoder_layers(x1,src_key_padding_mask=key_pad_x1)
            herb_mask = generate_square_mask(x2.shape[0]).to(self.device)
            x2, w2 = self.herb_encoder_layers(x2,mask=herb_mask,src_key_padding_mask=key_pad_x2)
        elif mask_input == 'symptom':
            sym_mask = generate_square_mask(x1.shape[0]).to(self.device)
            x1, w1 = self.sym_encoder_layers(x1, mask=sym_mask, src_key_padding_mask=key_pad_x1)
            x2, w2 = self.herb_encoder_layers(x2, src_key_padding_mask=key_pad_x2)
        else:
            x1, w1 = self.sym_encoder_layers(x1, src_key_padding_mask=key_pad_x1)
            x2, w2 = self.herb_encoder_layers(x2, src_key_padding_mask=key_pad_x2)


        z1 = torch.mean(x1,dim=0) # Embedding representation of x1 (symptoms)
        z2 = torch.mean(x2,dim=0) # Embedding representation of x2 (herbs)

        x21,w21 = self.cross_encoder_layers(x2,x1,src_key_padding_mask=key_pad_x1) # symptom to herb attention
        x12, w12 = self.cross_encoder_layers(x1, x2, src_key_padding_mask=key_pad_x2) # herb to symptom attention

        self.x21 = x21
        self.x12 = x12

        y1_,y2_ = self.decoding(x12,x21)

        return y1_.permute(1, 0, 2),y2_.permute(1, 0, 2),z1,z2,[w1,w2,w12,w21]

    def forward_independent(self, x1, x2):
        """
        Independent reconstruction mode.

        Symptom branch:
            symptoms -> symptom self-encoder -> symptom decoder

        Herb branch:
            herbs -> herb self-encoder -> herb decoder

        No cross-attention and no symptom-herb alignment are used.
        This is intended for the independent reconstruction null model.
        """

        # Padding masks
        key_pad_x1 = (torch.sum(x1 == self.sym_pad, 2) > 0)
        key_pad_x2 = (x2 == self.herb_pad_idx)

        # Token embeddings
        x1, x2 = self.trainableEmbedding(x1, x2)

        # Transformer expects [seq_len, batch, dim]
        x1 = x1.permute(1, 0, 2)
        x2 = x2.permute(1, 0, 2)

        # Use causal masks for autoregressive reconstruction
        sym_mask = generate_square_mask(x1.shape[0]).to(self.device)
        herb_mask = generate_square_mask(x2.shape[0]).to(self.device)

        # Self-encoding only
        x1, w1 = self.sym_encoder_layers(
            x1,
            mask=sym_mask,
            src_key_padding_mask=key_pad_x1,
        )

        x2, w2 = self.herb_encoder_layers(
            x2,
            mask=herb_mask,
            src_key_padding_mask=key_pad_x2,
        )

        # Independent embeddings
        z1 = torch.mean(x1, dim=0)
        z2 = torch.mean(x2, dim=0)

        # Decode each modality from itself
        y1_ = self.sym_decoder_layers(x1)
        y2_ = self.herb_decoder_layers(x2)

        return y1_.permute(1, 0, 2), y2_.permute(1, 0, 2), z1, z2, [w1, w2]


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive



class Multimodal_AE(torch.nn.Module):

    def __init__(self, input_dim, en_dims, de_dims,dropout=0.1, out_sig=True):
        super(Multimodal_AE, self).__init__()
        self.outact = out_sig
        self.dropout = nn.Dropout(dropout)
        # self.input_layer = nn.Sequential(nn.Linear(input_dim, en_dims[0], bias=True), nn.BatchNorm1d(en_dims[0]), nn.ReLU(True))
        self.input_layer = nn.Sequential(nn.Linear(input_dim, en_dims[0], bias=True), nn.ReLU(True))

        encoder_layers = []
        for l in range(len(en_dims)-1):
            encoder_layers.append(nn.Linear(en_dims[l], en_dims[l+1], bias=True))
            if l < len(en_dims)-2:
                encoder_layers.append(nn.BatchNorm1d(en_dims[l + 1]))
                encoder_layers.append(nn.ReLU(True))
        self.encoder_module = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for l in range(len(de_dims)-1):
            decoder_layers.append(nn.Linear(de_dims[l], de_dims[l+1], bias=True))
            decoder_layers.append(nn.BatchNorm1d(de_dims[l+1]))
            decoder_layers.append(nn.ReLU(True))
        self.decoder_module = nn.Sequential(*decoder_layers)

        self.output_layer = nn.Linear(de_dims[-1], input_dim, bias=True)

    def encoder(self, x):
        x = self.input_layer(x)
        x = self.dropout(x)
        return self.encoder_module(x)

    def decoder(self, z):
        x = self.decoder_module(z)
        x = self.dropout(x)
        return self.output_layer(x)

    def forward(self, x):
        z = self.encoder(x)
        x_ = self.decoder(z)

        # if self.outact:
        #     return torch.sigmoid(x_), z
        # else:
        #     return x_, z
        return x_, z



class SimpleEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        hidden_dims=(512,),      # 例如 (512,) 或 (1024, 512, 256)
        dropout: float = 0.2,
        act: str = "gelu",       # "relu" | "gelu"
        norm: str | None = 'ln'  # None | "bn" | "ln"
    ):
        super().__init__()

        dims = [input_dim] + list(hidden_dims) + [out_dim]
        layers = []

        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_d, out_d))
            is_last = (i == len(dims) - 2)
            if is_last:
                # 最后一层：直接输出，不加激活/Dropout
                break

            if norm == "bn":
                layers.append(nn.BatchNorm1d(out_d))
            elif norm == "ln":
                layers.append(nn.LayerNorm(out_d))

            if act == "relu":
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.GELU())

            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Transformer 编码器（很小就够）
# 不用位置编码（集合无序），或者每次随机 shuffle tokens。
class SetTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask=None): # input x: 形状 [B, K, d]; pad_mask：形状 [B, K] 的 bool
        y = self.encoder(x, src_key_padding_mask=pad_mask)  # [B,K,d]
        if pad_mask is None:
            pooled = y.mean(dim=1)
        else:
            mask = (~pad_mask).float()  # 1 for real
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (y * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.out_ln(pooled)      # 形状 [B, d] 的集合 embedding（组合 embedding） 每个样本一个向量 z_S，仍然是 256 维（对齐 herb 空间）。


