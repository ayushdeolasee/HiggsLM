import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, input_shape, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(input_shape))
        self.b = nn.Parameter(torch.ones(input_shape))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        output = x / rms 
        output = (output * self.g) + self.b
        return output 

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

class GroupedQueryAttention(nn.Module):
    def __init__(self, n_embed, n_head, kv_per_head, eps=1e-6):
        super().__init__()
        self.n_embd = n_embed
        self.n_head = n_head
        self.kv_per_head = kv_per_head
        self.n_groups = self.n_head // self.kv_per_head
        self.eps = eps

        self.keys = nn.Linear(self.n_embd, (self.n_embd // self.n_head) * self.n_groups)
        self.values = nn.Linear(self.n_embd, (self.n_embd // self.n_head) * self.n_groups) 
        self.quries = nn.Linear(self.n_embd, self.n_embd) 

        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.rope = RotaryPositionEmbedding(self.n_embd // self.n_head, self.seq_length)

    def forward(self, x):
        B, T, C = x.size() 
        keys = self.keys(x)
        values = self.values(x)
        quries = self.quries(x)

        k = keys.view(B, T, self.n_groups, C // self.n_head).transpose(1, 2).repeat_interleave(self.kv_per_head, dim=1)# (B, nh, T, hs)
        v = values.view(B, T, self.n_groups, C // self.n_head).transpose(1, 2).repeat_interleave(self.kv_per_head, dim=1) # (B, nh, T, hs)
        q = quries.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q_norm = torch.norm(q, dim=-1, keepdim=True)  
        k_norm = torch.norm(k, dim=-1, keepdim=True)
        q_hat = q / (q_norm + self.eps)
        k_hat = k / (k_norm + self.eps)

        factor = self.alpha * math.sqrt(C // self.n_head)
        factor = factor.view(1, self.n_head, 1, 1)
        q_scaled = q_hat * factor
        q = self.rope(q_scaled)
        k = self.rope(k_hat)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) 
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, embed_dim):
        super(MLP, self).__init__()
        
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim), 
        )
        self.scale_init = 1
    
    def forward(self, x):
        return self.MLP(x)

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, kv_per_head):
        super(Block, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.rmsnorm1 = RMSNorm(self.embed_dim) 
        self.MultiheadAttention = GroupedQueryAttention(self.embed_dim, self.num_heads, kv_per_head)
        self.rmsnrom2 = RMSNorm(self.embed_dim)
        self.MLP = MLP(self.embed_dim)
        
    def forward(self, x):
        x = x + self.MultiheadAttention(self.rmsnorm1(x))
        x = x + self.MLP(self.rmsnrom2(x))
        return x

class Model(nn.Module):
    def __init__(self, block_size, vocab_size, embed_dim, num_heads, num_blocks, kv_per_head):
        super(Model, self).__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim  
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.kv_per_head = kv_per_head

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        self.blocks = nn.ModuleList([Block(self.embed_dim, self.num_heads, self.kv_per_head) for _ in range(self.num_blocks)])

        self.rmsnrom3 = RMSNorm(self.embed_dim)
        self.lm_linear = nn.Linear(self.embed_dim, self.vocab_size)
        
        self.lm_linear.weight = self.embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, "NANOGPT_SCALE_INIT"):
            std *= (2 * self.num_blocks) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        output = self.rmsnrom3(x)
        output = self.lm_linear(output)
        return output
