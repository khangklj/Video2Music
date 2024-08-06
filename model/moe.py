import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn.modules.transformer import _get_clones
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import *

from torch.nn.functional import linear, softmax, dropout
from torch.nn import MultiheadAttention
from .rope import *
from efficient_kan import KANLinear

class KANExpert(Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(KANExpert, self).__init__()
        self.linear1 = KANLinear(d_model, d_ff + 1)
        self.linear2 = KANLinear(d_ff + 1, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class GLUExpert(Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(GLUExpert, self).__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.gate = Linear(d_model, d_ff)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        x_ff = self.linear1(x)
        x_gated = self.gate(x)
        x_ff = self.dropout(x_ff * F.silu(x_gated))
        x_ff = self.linear2(x_ff)
        return x_ff

# Source: https://www.facebook.com/photo?fbid=122146963988123211&set=pcb.122146964084123211
class MoELayer(Module):
    def __init__(self, expert, d_model, d_ff=2048, n_experts=8, n_experts_per_token=2, dropout=0.1):
        super(MoELayer, self).__init__()
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.experts = _get_clones(expert, n_experts)
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        gate_logits = self.gate(x)
        weights, selected_experts = torch.topk(gate_logits, self.n_experts_per_token)
        weights = softmax(weights, dim=1)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            batch_idx, token_idx, topk_idx = torch.where(selected_experts == i)
            weight = weights[batch_idx, token_idx, topk_idx, None]
            out[batch_idx, token_idx] += weight * self.dropout(expert(x[batch_idx, token_idx]))
        return out
    
class SharedMoELayer(Module):
    def __init__(self, expert, d_model, d_ff=2048, n_experts=8, n_experts_per_token=2, dropout=0.1):
        super(SharedMoELayer, self).__init__()
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.experts = _get_clones(expert, n_experts)
        self.shared_expert = _get_clones(expert, 1)[0]
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        gate_logits = self.gate(x)
        weights, selected_experts = torch.topk(gate_logits, self.n_experts_per_token)
        weights = softmax(weights, dim=1)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            batch_idx, token_idx, topk_idx = torch.where(selected_experts == i)
            weight = weights[batch_idx, token_idx, topk_idx, None]
            out[batch_idx, token_idx] += weight * self.dropout(expert(x[batch_idx, token_idx]))

        out += self.shared_expert(x)
        return out

class TransformerEncoderLayerMoE_RoPE(Module):
    def __init__(self, d_model, nhead, moelayer, rotation_matrix, dropout=0.1):
        super(TransformerEncoderLayerMoE_RoPE, self).__init__()
        self.self_attn = MultiheadAttention_RoPE(d_model, nhead, rotation_matrix)
        self.moe = moelayer

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        # self.dropout1 = Dropout(dropout)
        # self.dropout2 = Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)
        
        src = src + src2
        src = self.norm1(src)
        src2 = self.moe(src)
        src = src + src2
        src = self.norm2(src)
        return src
    
class TransformerDecoderLayerMoE_RoPE(Module):
    def __init__(self, d_model, nhead, moelayer, rotation_matrix, dropout=0.1):
        super(TransformerDecoderLayerMoE_RoPE, self).__init__()
        
        self.self_attn = MultiheadAttention_RoPE(d_model, nhead, rotation_matrix)
        self.multihead_attn = MultiheadAttention_RoPE(d_model, nhead, rotation_matrix)
        # Implementation of Feedforward model
        self.moe = moelayer

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        # self.dropout1 = Dropout(dropout)
        # self.dropout2 = Dropout(dropout)
        # self.dropout3 = Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)
        
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)

        tgt2 = self.moe(tgt)
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt

class TransformerEncoderLayerMoE(Module):
    def __init__(self, d_model, nhead, moelayer, dropout=0.1):
        super(TransformerEncoderLayerMoE, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.moe = moelayer

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        # self.dropout1 = Dropout(dropout)
        # self.dropout2 = Dropout(dropout)
    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + src2
        src = self.norm1(src)
        src2 = self.moe(src)
        src = src + src2
        src = self.norm2(src)
        return src

class TransformerDecoderLayerMoE(Module):
    def __init__(self, d_model, nhead, moelayer, dropout=0.1):
        super(TransformerDecoderLayerMoE, self).__init__()
        
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.moe = moelayer

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        # self.dropout1 = Dropout(dropout)
        # self.dropout2 = Dropout(dropout)
        # self.dropout3 = Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)

        tgt2 = self.moe(tgt)
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt
