import torch
import torch.nn as nn

# https://www.facebook.com/photo?fbid=122146964042123211&set=pcb.122146964084123211
def get_rotation_matrix(dim, context_size, period):
    freqs = 1.0 / (period ** (torch.arange(0, dim, 2) / dim))
    token_indexes = torch.arange(context_size)
    thetas = torch.outer(token_indexes, freqs).float()
    return torch.polar(torch.ones_like(thetas), thetas)

class RoPE(nn.Module):
    def __init__(self, rotation_matrix):
        super(RoPE, self).__init__()
        self.rotation_matrix = rotation_matrix

    def forward(self, queries, keys):
        batch_size, num_heads, seq_length, head_dim = queries.size()

        queries = queries.reshape(batch_size, num_heads, seq_length, head_dim // 2, 2)
        keys = keys.reshape(batch_size, num_heads, seq_length, head_dim // 2, 2)

        queries_complex = torch.view_as_complex(queries)
        keys_complex = torch.view_as_complex(keys)

        rotation_matrix = self.rotation_matrix[:seq_length]

        queries_ratated = queries_complex * rotation_matrix
        keys_rotated = keys_complex * rotation_matrix

        new_queries = torch.view_as_real(queries_ratated)
        new_keys = torch.view_as_real(keys_rotated)

        new_queries = new_queries.reshape(batch_size, num_heads, seq_length, head_dim)
        new_keys = new_keys.reshape(batch_size, num_heads, seq_length, head_dim)

        return new_queries, new_keys