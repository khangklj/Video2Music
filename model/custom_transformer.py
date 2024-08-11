import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_kan import KANLinear
from torch.nn import Module
from torch.nn.modules.transformer import _get_clones
from torch.nn.init import *
from copy import deepcopy
from utilities.device import get_device

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

# By our and need batch_first option
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

# By ChatGPT and some modify
class RotaryPositionalEmbedding(Module):
    def __init__(self, dim):
        super(RotaryPositionalEmbedding, self).__init__()
        self.dim = dim
        self.inv_freq = (1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))).to(get_device())
        self.inv_freq.requires_grad_(False)

    def get_angles(self, pos_seq):
        angles = pos_seq[:, None] * self.inv_freq[None, :]
        return torch.cat([angles, angles], dim=-1)

    def forward(self, q, k): # q.shape = (seq_len, batch_size, d_model)
        seq_len, batch_size, d_model = q.shape

        q = q.permute(1, 0, 2) # q.shape = (batch_size, seq_len, d_model)
        k = k.permute(1, 0, 2)

        pos_seq = torch.arange(seq_len, dtype=torch.float32, device=q.device, requires_grad=False)
        angles = self.get_angles(pos_seq)

        cos = angles.cos()
        sin = angles.sin()

        # The query and key are rotated using the angles
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)

        # del cos, sin, pos_seq
        # torch.cuda.empty_cache()

        q_rot = q_rot.permute(1, 0, 2)
        k_rot = k_rot.permute(1, 0, 2)

        return q_rot, k_rot

    def rotate_half(self, x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = torch.cat((-x2, x1), dim=-1)
        return x_rotated
    
# By our and need batch_first option
class MyRoPE(Module):
    def __init__(self, d_model, dropout=0.0, batch_first=False):
        super(MyRoPE, self).__init__()
        self.batch_first = batch_first
        self.rope = RotaryPositionalEmbedding(d_model).to(get_device())

    def forward(self, queries, keys):
        if self.batch_first:
            new_queries, new_keys = self.rope(queries, keys)
        else:
            queries = queries.permute(1, 0, 2)
            keys = keys.permute(1, 0, 2)

            new_queries, new_keys = self.rope(queries, keys)

            new_queries = new_queries.permute(1, 0, 2)
            new_keys = new_keys.permute(1, 0, 2)

        return new_queries, new_keys

# By our and need batch_first, use_KAN, RoPE option       
class MyMultiheadAttention(Module):
    def __init__(self, d_model, num_head, dropout=0.0, batch_first=False, use_KAN=False, RoPE=None):
        super(MyMultiheadAttention, self).__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.head_dim = d_model // num_head
        # self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.use_KAN = use_KAN

        if not use_KAN:
            self.W_q, self.W_k, self.W_v, self.W_out = _get_clones(nn.Linear(d_model, d_model), 4)
        else:
            self.W_q, self.W_k, self.W_v, self.W_out = _get_clones(KANLinear(d_model, d_model), 4)

        self.rope = deepcopy(RoPE) if RoPE is not None else None
    
        self._reset_parameters(use_KAN)

    def _reset_parameters(self, use_KAN):
        if not use_KAN:
            xavier_uniform_(self.W_q.weight)
            xavier_uniform_(self.W_k.weight)
            xavier_uniform_(self.W_v.weight)
            xavier_uniform_(self.W_out.weight)
            self.W_q.bias.data.fill_(0.0)
            self.W_k.bias.data.fill_(0.0)
            self.W_v.bias.data.fill_(0.0)
            self.W_out.bias.data.fill_(0.0)

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None, **kwargs):
        if not self.batch_first: # q.shape = (seq_len, batch_size, d_model)
            return self._calcuate_attn(q, k, v, key_padding_mask, attn_mask)
        else:
            q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
            attn_out = self._calcuate_attn(q, k, v, key_padding_mask, attn_mask)
            return attn_out.permute(1, 0, 2)
    
    def _calcuate_attn(self, q, k, v, key_padding_mask=None, attn_mask=None):
        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v) # q.shape = (seq_len, batch_size, d_model)

        if self.rope is not None:
            q, k = self.rope(q, k)

        # Reshape Q, K, V for multi-head attention # (seq_len, batch_size, num_head, head_dim)
        q = q.view(q.size(0), q.size(1), self.num_head, self.head_dim)
        k = k.view(k.size(0), k.size(1), self.num_head, self.head_dim)
        v = v.view(v.size(0), v.size(1), self.num_head, self.head_dim)

        q = torch.permute(q, (2, 1, 0, 3)) # q.shape = (num_head, batch_size, seq_len, head_dim)
        k = torch.permute(k, (2, 1, 0, 3))
        v = torch.permute(v, (2, 1, 0, 3))

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply attention mask, if provided
        if attn_mask is not None:
            attn_scores += attn_mask

        # Apply key padding mask, if provided
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        # attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)  # attn_output.shape = (num_head, batch_size, seq_len, head_dim)

        # Combine heads and apply the final linear layer
        attn_output = attn_output.contiguous().view(q.shape[2], q.shape[1], self.d_model)  # attn_output.shape = (seq_len, batch_size, d_model)
        attn_output = self.W_out(attn_output)

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