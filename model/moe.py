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
from typing import Optional

import sys

if '/kaggle/working/efficient-kan/src' not in sys.path:
    sys.path.append('/kaggle/working/efficient-kan/src')

from efficient_kan import KANLinear

class KAN_GLUExpert(Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(KANExpert, self).__init__()
        self.w1 = KANLinear(d_model, d_ff)
        self.w2 = KANLinear(d_model, d_ff)
        self.w3 = KANLinear(d_ff, d_model)

    def forward(self, x):
        return self.w3(self.w1(x) * self.w2(x))

class GLUExpert(Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(GLUExpert, self).__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.gate = Linear(d_model, d_ff)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x):
        x_ff = self.dropout1(self.linear1(x))
        x_gated = self.dropout2(self.gate(x))
        x_ff = x_ff * F.silu(x_gated)
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
        self.dropout = Dropout(dropout)
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
            out[batch_idx, token_idx] += weight * expert(x[batch_idx, token_idx])
        return out

class TransformerEncoderLayerMoE(Module):
    def __init__(self, d_model, nhead, moelayer, dropout=0.1):
        super(TransformerEncoderLayerMoE, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.moe = moelayer

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.moe(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerDecoderLayerMoE(Module):
    def __init__(self, d_model, nhead, moelayer, dropout=0.1):
        super(TransformerDecoderLayerMoE, self).__init__()
        
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.moe = moelayer

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.moe(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
