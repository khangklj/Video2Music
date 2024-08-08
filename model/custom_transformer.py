import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_kan import KANLinear
from rotary_embedding_torch import RotaryEmbedding
from torch.nn import Module
from torch.nn.modules.transformer import _get_clones
from torch.nn.init import *
from copy import deepcopy

from torch.nn.functional import linear, softmax, dropout

# https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
class RMSNorm(Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

class MyRMSNorm(Module):
    def __init__(self, dim, eps=1e-6, batch_first=False): # dim = d_model
        super(MyRMSNorm, self).__init__()
        self.norm = RMSNorm(dim)
        self.batch_first = batch_first

    def forward(self, x):
        if self.batch_first: # x.shape = (batch, seq_len, d_model)
            return self.norm(x)
        else: # x.shape = (seq_len, batch, d_model)
            x = x.permute(1,0,2)
            x = self.norm(x)
            x = x.permute(1,0,2)
            return x
        pass

class MyMultiheadAttention(Module):
    def __init__(self, d_model, num_head, dropout=0.0, batch_first=False, use_KAN=False, RoPE=False):
        super(MyMultiheadAttention, self).__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.head_dim = d_model // num_head
        self.dropout = nn.Dropout(dropout)

        self.att = nn.MultiheadAttention(d_model, num_head, dropout)

        if RoPE:
            self.rope = RoPE
    
    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None, **kwargs):
        if self.rope is not None:
            pass

class TransformerEncoderLayer(Module):
    def __init__(self, self_att_layer, ff_layer, norm=None, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = deepcopy(self_att_layer)
        self.ff = deepcopy(ff_layer)

        self.norm1, self.norm2 = _get_clones(norm, 2)
        # self.dropout1 = Dropout(dropout)
        # self.dropout2 = Dropout(dropout)
    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + src2
        src = self.norm1(src)
        src2 = self.ff(src)
        src = src + src2
        src = self.norm2(src)
        return src

class TransformerDecoderLayer(Module):
    def __init__(self, self_att_layer, cross_att_layer, ff_layer, norm=None, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        self.self_attn = deepcopy(self_att_layer)
        self.cross_attn = deepcopy(cross_att_layer)
        self.ff = deepcopy(ff_layer)
        
        self.norm1, self.norm2, self.norm3 = _get_clones(norm, 3)
        # self.dropout1 = Dropout(dropout)
        # self.dropout2 = Dropout(dropout)
        # self.dropout3 = Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)

        tgt2 = self.ff(tgt)
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt

class TransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = deepcopy(norm)
    def forward(self, src, mask=None, src_key_padding_mask=None, **kwargs):
        output = src
        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)
        if self.norm:
            output = self.norm(output)
        return output

class TransformerDecoder(Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = deepcopy(norm)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, **kwargs):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output