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
from utilities.device import get_device
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
        # self.dropout = Dropout(dropout)

    def forward(self, x):
        x_ff = self.linear1(x)
        x_gated = self.gate(x)
        x_ff = x_ff * F.sigmoid(x_gated)
        x_ff = self.linear2(x_ff)
        return x_ff

# Source: https://www.facebook.com/photo?fbid=122146963988123211&set=pcb.122146964084123211
class MoELayer(Module):
    def __init__(self, expert, d_model, n_experts=8, n_experts_per_token=2, dropout=0.1):
        super(MoELayer, self).__init__()
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.experts = _get_clones(expert, n_experts)
        self.gate = nn.Linear(d_model, n_experts)

    def forward(self, x):
        gate_logits = self.gate(x)
        weights, selected_experts = torch.topk(gate_logits, self.n_experts_per_token)
        weights = softmax(weights, dim=-1, dtype=torch.float).to(get_device())
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            token_idx, batch_idx, topk_idx = torch.where(selected_experts == i)
            weight = weights[token_idx, batch_idx, topk_idx]
            out[token_idx, batch_idx] += weight.unsqueeze(1) * self.dropout(expert(x[token_idx, batch_idx]))
        return out
    
class SharedMoELayer(Module):
    def __init__(self, expert, d_model, n_experts=8, n_experts_per_token=2, dropout=0.1, use_KAN=False):
        super(SharedMoELayer, self).__init__()
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.experts = _get_clones(expert, n_experts)
        self.shared_expert = _get_clones(expert, 1)[0]

        if not use_KAN:
            self.gate = nn.Linear(d_model, n_experts)
        else:
            self.gate = KANLinear(d_model, n_experts)

        self.temperature = torch.tensor([10000]).requires_grad_(False)
        self.decay_rate = torch.tensor([0.9]).requires_grad_(False)

    def forward(self, x):
        gate_logits = self.gate(x)

        # Balancing the experts
        if self.training and self.temperature.item() > 1:
            gate_logits /= self.temperature
            self.temperature *= self.decay_rate

        weights, selected_experts = torch.topk(gate_logits, self.n_experts_per_token)
        weights = softmax(weights, dim=-1, dtype=torch.float).to(get_device())
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            token_idx, batch_idx, topk_idx = torch.where(selected_experts == i)
            
            if token_idx.shape[0] == 0:
                continue

            weight = weights[token_idx, batch_idx, topk_idx]
            out[token_idx, batch_idx] += weight.unsqueeze(1) * self.dropout(expert(x[token_idx, batch_idx]))

        # Sharing
        out += self.shared_expert(x)
        return out
