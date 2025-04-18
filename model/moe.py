import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from lion_pytorch import Lion

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
import copy
from utilities.argument_funcs import parse_train_args
from third_party.log_experts import update_expert_counts
from third_party.log_maxvio import update_maxvio

import os

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
        x_ff = x_ff * F.silu(x_gated)
        x_ff = self.linear2(self.dropout(x_ff))
        return x_ff
    
class AngleGLUExpert(Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(AngleGLUExpert, self).__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model // 2) # Modified from GLUExpert
        self.gate = Linear(d_model, d_ff)
        # self.dropout = Dropout(dropout)

    def forward(self, x):
        x_ff = self.linear1(x)
        x_gated = self.gate(x)
        x_ff = x_ff * F.silu(x_gated)
        x_ff = self.linear2(x_ff)
        return x_ff

class TopKScheduler(Module):
    def __init__(self, n_experts=8, min_n_experts_per_token=2, update_step=16):
        super(TopKScheduler, self).__init__()
        self.n_experts = n_experts
        self.min_n_experts_per_token = min_n_experts_per_token
        self.k = n_experts

        self.update_step = update_step
        self.counting_step = 0

    def step(self):
        self.counting_step += 1
        if self.counting_step % self.update_step == 0:
            self.k = max(self.min_n_experts_per_token, self.k - 1)

    def getK(self):
        return self.k
    
class TemperatureScheduler(Module):
    def __init__(self, temperature_min=0.8, temperature_max=1.1, temperature_step=0.0005):
        super(TemperatureScheduler, self).__init__()
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.temperature_step = temperature_step
        self.t = self.temperature_min

    def step(self):
        self.t += self.temperature_step
        self.t = min(self.t, self.temperature_max)

    def getT(self):
        return self.t

class ShannonEntropy(Module):
    def __init__(self, eps=1e-10):
        super(ShannonEntropy, self).__init__()
        self.eps = eps

    def forward(self, x: Tensor): # x.shape = (batch_size, n)
        x /= x.sum(dim=1) # get probability
        x += self.eps # prevent log(0)
        entropy = -torch.sum(x * (torch.log(x) / 1.79175946923), dim=1) # 1.79175946923 = ln(6)
        return entropy.mean()

# Self-Balance Routing Network
# class SBRN(Module):
#     def __init__(self, router, n_experts=8, n_experts_per_token=2):
#         super(SBRN, self).__init__()
#         self.n_experts = n_experts
#         self.n_experts_per_token = n_experts_per_token
#         self.router = copy.deepcopy(router)
#         self.optim = AdamW(self.router.parameters(), lr=0.005, weight_decay=0.01, maximize=True)
#         self.loss_func = ShannonEntropy()
#         self.count = torch.zeros((1, self.n_experts), requires_grad=False).to(get_device())

#     def _routing(self, x, k=2):
#         gate_logits = self.router(x)
#         weights, selected_experts = torch.topk(gate_logits, k)
#         return weights, selected_experts

#     def forward(self, x, k=2, t=1.0):
#         weights, selected_experts = self._routing(x, k)
#         weights = F.softmax(weights / t, dim=1, dtype=torch.float).to(get_device())
#         return weights, selected_experts
    
#     def reset_count(self):
#         self.count = torch.zeros((1, self.n_experts), requires_grad=False).to(get_device())

#     def count_experts(self, x, k=2):
#         _, selected_experts = self._routing(x, k)
        
#         for i in range(self.n_experts):
#             self.count[0, i] += (selected_experts == i).sum().item()
    
#     def step(self, x, k=2):
#         self.count_experts(x, k)

#         loss = torch.autograd.Variable(self.loss_func(self.count), requires_grad=True)
#         loss.backward()
#         self.optim.step()

#         self.optim.zero_grad()

# Source: https://www.facebook.com/photo?fbid=122146963988123211&set=pcb.122146964084123211
class MoELayer(Module):
    def __init__(self, expert, d_model, n_experts=8, n_experts_per_token=2, dropout=0.1, topk_scheduler=None, temperature_scheduler=None):
        super(MoELayer, self).__init__()
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.experts = _get_clones(expert, n_experts)
        self.gate = nn.Linear(d_model, n_experts)

        # If has topk scheduler then no need n_experts and n_experts_per_token
        if topk_scheduler is not None:
            self.topk_scheduler = topk_scheduler
        
        if temperature_scheduler is not None:
            self.temperature_scheduler = temperature_scheduler

    def forward(self, x):
        if hasattr(self, 'topk_scheduler') and self.training:
            self.topk_scheduler.step()
            k = self.topk_scheduler.getK()
        else:
            k = self.n_experts_per_token

        if hasattr(self, 'temperature_scheduler') and self.training:
            self.temperature_scheduler.step()
            t = self.temperature_scheduler.getT()
        else:
            t = 1.0
            
        gate_logits = self.gate(x) / t
        
        # Logging
        update_expert_counts(selected_experts)

        weights, selected_experts = torch.topk(gate_logits, k)
        weights = softmax(weights, dim=-1, dtype=torch.float).to(get_device())
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            token_idx, batch_idx, topk_idx = torch.where(selected_experts == i)

            if token_idx.shape[0] == 0:
                continue

            weight = weights[token_idx, batch_idx, topk_idx]
            out[token_idx, batch_idx] += weight.unsqueeze(1) * self.dropout(expert(x[token_idx, batch_idx]))
        return out
    
class SharedMoELayer(Module):
    def __init__(self, expert, d_model, n_experts=8, n_experts_per_token=2, dropout=0.1, balancing=False, topk_scheduler=None, temperature_scheduler=None, use_KAN=False):
        super(SharedMoELayer, self).__init__()
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.experts = _get_clones(expert, n_experts)
        self.balancing = balancing

        # If has topk scheduler then no need n_experts and n_experts_per_token
        if topk_scheduler is not None:
            self.topk_scheduler = topk_scheduler
        
        if temperature_scheduler is not None:
            self.temperature_scheduler = temperature_scheduler

        if not use_KAN:
            self.gate = nn.Linear(d_model, n_experts)
        else:
            self.gate = KANLinear(d_model, n_experts)

        if self.balancing:
            self.register_buffer('bias', torch.zeros((n_experts, 1)))
            self.bias = self.bias.to(get_device())
            self.update_rate = 0.01

        self.shared_expert = _get_clones(expert, 1)[0]

    def forward(self, x):
        if hasattr(self, 'topk_scheduler') and self.training:
            self.topk_scheduler.step()
            k = self.topk_scheduler.getK()
        else:
            k = self.n_experts_per_token

        if hasattr(self, 'temperature_scheduler'):
            self.temperature_scheduler.step()
            t = self.temperature_scheduler.getT()
        else:
            t = 1.0
            
        gate_logits = self.gate(x)

        if not self.balancing:
            weights, selected_experts = torch.topk(gate_logits, k)

            c = torch.bincount(selected_experts.flatten(), minlength=6).to(torch.float)
            # c = torch.cat((torch.tensor([0]).to(get_device()), c))
            # c = c[1:]

            if not self.training:
                # Logging
                update_maxvio(c)
        else:
            b = self.bias.T.unsqueeze(0)
            if self.training:
                tmp = gate_logits + b
                # tmp = gate_logits * b
            else:
                tmp = gate_logits

            weights, selected_experts = torch.topk(tmp, k, dim=-1)

            if self.training:
                # Only get gate_logits
                weights = torch.gather(gate_logits, dim=-1, index=selected_experts)

            c = torch.bincount(selected_experts.flatten(), minlength=6).to(self.bias.dtype)
            # c = torch.cat((torch.tensor([0]).to(get_device()), c))
            # c = c[1:]

            if self.training: 
                c_mean = torch.mean(c)
                e = c_mean - c

                e = e.unsqueeze(1)
                self.bias += self.update_rate * e
                # self.bias += self.update_rate * torch.sign(e)
            else:
                # Logging
                update_maxvio(c)
        
        # Logging
        update_expert_counts(selected_experts, self.training)

        weights = softmax(weights / t, dim=-1, dtype=torch.float).to(get_device())
        # weights = F.sigmoid(weights / t).to(get_device())
        out = torch.zeros((*x.shape[:-1], self.d_model), device=get_device())
        for i, expert in enumerate(self.experts):
            token_idx, batch_idx, topk_idx = torch.where(selected_experts == i)
            
            if token_idx.shape[0] == 0:
                continue

            weight = weights[token_idx, batch_idx, topk_idx]
            out[token_idx, batch_idx] += weight.unsqueeze(1) * self.dropout(expert(x[token_idx, batch_idx]))

        # Sharing
        out += 1.0 / k * self.shared_expert(x)
        return out

# class SelfBalanceSharedMoELayer(Module):
#     def __init__(self, expert, d_model, n_experts=8, n_experts_per_token=2, dropout=0.1, topk_scheduler=None, temperature_scheduler=None, use_KAN=False):
#         super(SelfBalanceSharedMoELayer, self).__init__()
#         self.n_experts = n_experts
#         self.n_experts_per_token = n_experts_per_token
#         self.d_model = d_model
#         self.dropout = nn.Dropout(dropout)
#         self.experts = _get_clones(expert, n_experts)

#         # If has topk scheduler then no need n_experts and n_experts_per_token
#         if topk_scheduler is not None:
#             self.topk_scheduler = topk_scheduler
        
#         if temperature_scheduler is not None:
#             self.temperature_scheduler = temperature_scheduler

#         if not use_KAN:
#             router = nn.Linear(d_model, n_experts)
#         else:
#             router = KANLinear(d_model, n_experts)

#         self.shared_expert = _get_clones(expert, 1)[0]
#         self.gate = SBRN(router, n_experts, n_experts_per_token)
#         self.state = 'training'

#     def forward(self, x):
#         if hasattr(self, 'topk_scheduler') and self.training:
#             self.topk_scheduler.step()
#             k = self.topk_scheduler.getK()
#         else:
#             k = self.n_experts_per_token

#         if hasattr(self, 'temperature_scheduler'):
#             self.temperature_scheduler.step()
#             t = self.temperature_scheduler.getT()
#         else:
#             t = 1.0

#         if self.training:
#             # self.gate.reset_count()
#             self.gate.step(x, k)
#         else:
#             self.gate.count_experts(x, k)

#         if self.training and self.state == 'evaluating':
#             self.state = 'training'
#             print(round(self.gate.count.std().item()), end='\t')
#             print(self.gate.count.min().item(), self.gate.count.max().item(), sep='\t', end='\t')
#             # print(self.gate.count[0])
#             self.gate.reset_count()
            
#         if not self.training and self.state == 'training':
#             self.state = 'evaluating'
#             self.gate.reset_count()
        
#         weights, selected_experts = self.gate(x, k, t)

#         # Logging
#         update_expert_counts(selected_experts)

#         out = torch.zeros_like(x)
#         for i, expert in enumerate(self.experts):
#             token_idx, batch_idx, topk_idx = torch.where(selected_experts == i)
            
#             if token_idx.shape[0] == 0:
#                 continue

#             weight = weights[token_idx, batch_idx, topk_idx]
#             out[token_idx, batch_idx] += weight.unsqueeze(1) * self.dropout(expert(x[token_idx, batch_idx]))

#         # Sharing
#         out += 1.0 / k * self.shared_expert(x)
#         return out
