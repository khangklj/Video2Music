# https://blog.eleuther.ai/rotary-embeddings/
import torch

# Rotary Skip Connection
class RoSC(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, angle, seq_dim=1):
        emb = torch.cat((angle, angle), dim=-1).to(x.device)
        emb_cos = emb.cos()[:, None, :]
        emb_sin = emb.sin()[:, None, :]

        return apply_rotary_pos_emb(x, emb_cos, emb_sin)

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, :]
            self.sin_cached = emb.sin()[:, None, :]

        return apply_rotary_pos_emb(x, self.cos_cached, self.sin_cached)

# rotary pos emb helpers:
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in torch < 1.8.0

def apply_rotary_pos_emb(x, cos, sin):
    return ((x * cos) + (rotate_half(x) * sin))