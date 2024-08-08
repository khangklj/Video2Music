import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_kan import KANLinear
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

# https://www.facebook.com/photo/?fbid=122146964042123211&set=pcb.122146964084123211
def get_rotation_matrix(d_model, max_seq_len, period):
    freqs = 1.0 / (period ** (torch.arange(0, d_model, 2) / d_model))
    token_indexes = torch.arange(max_seq_len)
    thetas = torch.outer(token_indexes, freqs).float()
    return torch.polar(torch.ones_like(thetas), thetas)

class RoPE(Module):
    def __init__(self, d_model, max_seq_len, period=10000.0, dropout=0.0):
        super(RoPE, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)
        self.period = period

        self.rotation_matrix = get_rotation_matrix(d_model, max_seq_len, period)

    def forward(self, queries, keys):
        batch_size, num_heads, seq_length, head_dim = queries.shape

        queries = queries.reshape(batch_size, num_heads, seq_length, head_dim // 2, 2)
        keys = keys.reshape(batch_size, num_heads, seq_length, head_dim // 2, 2)

        queries_complex = torch.view_as_complex(queries)
        keys_complex = torch.view_as_complex(keys)

        rotation_matrix = self.rotation_maxtrix[:seq_length]

        queries_rotated = queries_complex * rotation_matrix
        keys_rotated = keys_complex * rotation_matrix

        new_queries = torch.view_as_real(queries_rotated)
        new_keys = torch.view_as_real(keys_rotated)

        new_queries = new_queries.reshape(batch_size, num_heads, seq_length, head_dim)
        new_keys = new_keys.reshape(batch_size, num_heads, seq_length, head_dim)

        return new_queries, new_keys
    
class MyRoPE(Module):
    def __init__(self, d_model, max_seq_len, period=10000.0, dropout=0.0, batch_first=False):
        super(MyRoPE, self).__init__()
        self.batch_first = batch_first
        self.rope = RoPE(d_model, max_seq_len, period, dropout)

    def forward(self, queries, keys):
        if self.batch_first:
            new_queries, new_keys = self.rope(queries, keys)
        else:
            queries = queries.permute(1, 0, 2, 3)
            keys = keys.permute(1, 0, 2, 3)

            new_queries, new_keys = self.rope(queries, keys)

            new_queries = new_queries.permute(1, 0, 2, 3)
            new_keys = new_keys.permute(1, 0, 2, 3)

        return new_queries, new_keys
        

class MyMultiheadAttention(Module):
    def __init__(self, d_model, num_head, dropout=0.0, batch_first=False, use_KAN=False, RoPE=False):
        super(MyMultiheadAttention, self).__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.head_dim = d_model // num_head
        self.dropout = nn.Dropout(dropout)

        if not use_KAN:
            self.W_q, self.W_k, self.W_v, self.out = _get_clones(nn.Linear(d_model, d_model), 4)
        else:
            self.W_q, self.W_k, self.W_v, self.out = _get_clones(KANLinear(d_model, d_model), 4)

        if RoPE:
            self.rope = RoPE
    
    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None, **kwargs):
        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v)

        # Reshape Q, K, V for multi-head attention # (batch_size, num_head, seq_len, head_dim)
        q = q.view(q.size(0), q.size(1), self.num_head, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_head, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_head, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            q, k = self.rope.rotate_queries_and_keys(q, k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply attention mask, if provided
        if attn_mask is not None:
            attn_scores += attn_mask

        # Apply key padding mask, if provided
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_head, seq_len, head_dim)

        # Combine heads and apply the final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(q.size(0), -1, self.d_model)  # (batch_size, seq_len, d_model)
        attn_output = self.out(attn_output)

        return attn_output

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