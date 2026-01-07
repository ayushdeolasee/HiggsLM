import torch
import torch.nn as nn

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_length=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_length).float()
        freqs = torch.einsum('i,j->ij', position, inv_freq)
        self.register_buffer('cos_cached', torch.cos(freqs), persistent=False)
        self.register_buffer('sin_cached', torch.sin(freqs), persistent=False)
    
    def forward(self, x):
        seq_len = x.size(-2)
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        x_rotated = self.apply_rotary_pos_emb(x, cos, sin)
        return x_rotated
    
    def apply_rotary_pos_emb(self, x, cos, sin):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x1 * sin + x2 * cos
        x_rotated = torch.stack([x1_rotated, x2_rotated], dim=-1)
        x_rotated = x_rotated.flatten(start_dim=-2)
        return x_rotated
