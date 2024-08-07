import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_kan import KANLinear
from rotary_embedding_torch import RotaryEmbedding

# # https://www.facebook.com/photo?fbid=122146964042123211&set=pcb.122146964084123211
# def get_rotation_matrix(dim, context_size, period):
#     freqs = 1.0 / (period ** (torch.arange(0, dim, 2) / dim))
#     token_indexes = torch.arange(context_size)
#     thetas = torch.outer(token_indexes, freqs).float()
#     return torch.polar(torch.ones_like(thetas), thetas)

# class RoPE(nn.Module):
#     def __init__(self, rotation_matrix):
#         super(RoPE, self).__init__()
#         self.rotation_matrix = rotation_matrix

#     def forward(self, queries, keys):
#         batch_size, num_heads, seq_length, head_dim = queries.size()

#         queries = queries.reshape(batch_size, num_heads, seq_length, head_dim // 2, 2)
#         keys = keys.reshape(batch_size, num_heads, seq_length, head_dim // 2, 2)

#         queries_complex = torch.view_as_complex(queries)
#         keys_complex = torch.view_as_complex(keys)

#         rotation_matrix = self.rotation_matrix[:seq_length]

#         print(queries_complex.shape, rotation_matrix.shape)
#         queries_ratated = queries_complex * rotation_matrix
#         keys_rotated = keys_complex * rotation_matrix

#         new_queries = torch.view_as_real(queries_ratated)
#         new_keys = torch.view_as_real(keys_rotated)

#         new_queries = new_queries.reshape(batch_size, num_heads, seq_length, head_dim)
#         new_keys = new_keys.reshape(batch_size, num_heads, seq_length, head_dim)

#         return new_queries, new_keys
    
class MultiheadAttention_RoPE(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiheadAttention_RoPE, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv_linear = KANLinear(hidden_size * 3, hidden_size * 3)
        self.out = KANLinear(hidden_size, hidden_size)
        self.position_emb = RotaryEmbedding(hidden_size) # input must (batch, heads, seq len, dimension of head)

    def forward(self, q, k, v, attn_mask=None):
        x = torch.cat((q, k, v), dim=2)
        batch_size, seg_length, hidden_size = x.size()

        qkv = self.qkv_linear(x)
        qkv = qkv.reshape(batch_size, seg_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.transpose(1, 2)
        queries, keys, values = qkv.chunk(3, dim=-1)
        queries, keys = self.position_emb.rotate_queries_and_keys(queries, keys)

        scores = torch.matmul(queries, keys.transpose(2, 3))
        scores = scores / (self.head_dim ** 0.5) + attn_mask

        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, values)
        context = context.transpose(1, 2)
        context = context.reshape(batch_size, seg_length, hidden_size)
        output = self.out(context)
        return output