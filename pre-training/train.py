import inspect
import math
import os

import numpy as np
import torch
import torch.backends
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu")
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderLite:
    def __init__(self, B: int, T: int, split: str, data_root: str):
        self.B = B
        self.T = T
        assert split in {"train", "val"}

        shards = [s for s in os.listdir(data_root) if split in s]
        shards = sorted(os.path.join(data_root, s) for s in shards)
        assert shards, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")

        self.shards = shards
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T
        return x, y

class RMSNorm(nn.Module):
    def __init__(self, input_shape: int, eps: float = 1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(input_shape))
        self.b = nn.Parameter(torch.ones(input_shape))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        output = x / rms 
        output = (output * self.g) + self.b
        return output 

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head):
        self.n_embd = n_embed
        self.n_head = n_head
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, embed_dim):
        super(MLP, self).__init__()
        
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, 4 *  embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim), 
        )
        self.scale_init = 1
    def forward(self, x):
        return self.MLP(x)

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(Block, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.rmsnorm1 = RMSNorm(embed_dim) 
        self.MultiheadAttention = CausalSelfAttention(self.embed_dim, self.num_heads)
        self.rmsnorm2 = RMSNorm(embed_dim)
        self.MLP = MLP(embed_dim)
        
    def forward(self, x):
        x = x + self.MultiheadAttention(self.rmsnorm1(x))
        x = x + self.MLP(self.rmsnorm2(x))
        return x

class Model1(nn.Module):
    def __init__(self, block_size, vocab_size, embed_dim, num_heads, num_blocks, dropout):
        super(Model1, self).__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim  
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.positional_embedding = nn.Embedding(self.block_size, self.embed_dim)

        self.blocks = nn.ModuleList([Block(self.embed_dim, self.num_heads, self.dropout) for _ in range(self.num_blocks)])

        self.rmsnorm3 = RMSNorm(self.embed_dim)
        self.lm_linear = nn.Linear(self.embed_dim, self.vocab_size)
        
        #weight sharing scheme
        self.lm_linear.weight = self.embedding.weight

        # Apply parameter initialization as per GPT2 model
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
        B, T = x.shape
        x = self.embedding(x)
        x = x + self.positional_embedding(torch.arange(T, device=device))
        for block in self.blocks:
            x = block(x)
        output = self.rmsnorm3(x)
        output = self.lm_linear(output)
        return output


def get_lr(step: int, *, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def train(
    *,
    dataset_location: str = "data",
    vocab_size: int = 50304,
    weight_decay: float = 0.1,
    block_size: int = 1024,
    batch_size: int = 4,
    epochs: int = 19073,
    total_batch_size: int = 524288,
    lr: float = 3e-4,
    warmup_steps: int = 150,
    embed_dim: int = 768,
    num_heads: int = 16,
    num_blocks: int = 8,
    dropout: float = 0.2,
):
    max_steps = epochs
    max_lr = lr
    min_lr = max_lr * 0.1

    assert total_batch_size % (batch_size * block_size) == 0, "total_batch_size must be divisble by B * T"
    grad_accum_steps = total_batch_size // (batch_size * block_size)

    train_dataloader = DataLoaderLite(batch_size, block_size, "train", dataset_location)
    val_dataloader = DataLoaderLite(batch_size, block_size, "val", dataset_location)

    model = Model1(block_size, vocab_size, embed_dim, num_heads, num_blocks, dropout).to(device)
    if device != "mps":
        model = torch.compile(model)

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device == "cuda"

    loss_fn = nn.CrossEntropyLoss()

    param_dict = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in param_dict if p.dim() >= 2]
    nodecay_params = [p for p in param_dict if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=lr, fused=use_fused)
    print("Parameter weight decay set")

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_acum = 0.0

        for _ in range(grad_accum_steps):
            x, y = train_dataloader.next_batch()
            x, y = x.to(device), y.to("cpu")
            output = model(x).to("cpu")
            train_loss = loss_fn(output.view(-1, output.size(-1)), y.view(-1))
            train_loss = train_loss.to(device)
            output = output.to(device)

            train_loss = train_loss / grad_accum_steps
            loss_acum += train_loss.detach()
            train_loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        lr_step = get_lr(epoch, warmup_steps=warmup_steps, max_steps=max_steps, max_lr=max_lr, min_lr=min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_step

        with torch.no_grad():
            x, y = val_dataloader.next_batch()
            x, y = x.to(device), y.to("cpu")
            output = model(x).to("cpu")
            val_loss = loss_fn(output.view(-1, output.size(-1)), y.view(-1))

        print_statement = (
            f"Epoch: {epoch}| Train Loss: {loss_acum.item()} | Val Loss: {val_loss.item()} | Norm: {norm} | lr: {lr_step}"
        )
        with open("log.txt", "a") as fd:
            fd.write(f"{print_statement}\n")
        print(print_statement)

    torch.save(model.state_dict(), "model_weights.pth")
    torch.save(optimizer.state_dict(), "optimizer_weights.pth")


if __name__ == "__main__":
    train()
