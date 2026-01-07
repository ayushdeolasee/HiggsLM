import torch.backends
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import os
import logging
import inspect
from rich import print
from rich.logging import RichHandler
import wandb

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")

device = torch.device("cpu")
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps" 

print(f"[green]Using device: {device}[/green] :smile: ")

# Hyperparameters
BATCH_SIZE = 8
BLOCK_SIZE = 1024
EPOCHS = 19073
LEARNING_RATE = 3e-4
MAX_STEPS = 19073
WARMUP_STEPS = 750
MAX_LR = 3e-4
MIN_LR = 3e-5
DATA_ROOT = "data"
GRAD_ACCUM_STEPS = 8
VOCAB_SIZE = 50304
WEIGHT_DECAY = 0.01

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, split, data_root):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        data_root = data_root
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        if (len(shards) > 0) == False:
            log.error(f"no shards found for split {split}")
        # assert len(shards) > 0, f"no shards found for split {split}"
        print(f"[green]found {len(shards)} shards for split {split}[/green]")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T
        return x, y

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
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        keys = self.keys(x)
        values = self.values(x)
        quries = self.quries(x)

        # Batch size, number of heads, sequence lenght, head size
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

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
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
    def __init__(self, embed_dim, num_heads):
        super(Block, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.rmsnorm1 = RMSNorm(embed_dim) 
        self.MultiheadAttention = CausalSelfAttention(self.embed_dim, self.num_heads)
        self.rmsnrom2 = RMSNorm(embed_dim)
        self.MLP = MLP(embed_dim)
        
    def forward(self, x):
        x = x + self.MultiheadAttention(self.rmsnorm1(x))
        x = x + self.MLP(self.rmsnrom2(x))
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

        self.blocks = nn.ModuleList([Block(self.embed_dim, self.num_heads, self.dropout) for _ in range(self.num_blocks)])

        self.rmsnrom3 = RMSNorm(self.embed_dim)
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
        for block in self.blocks:
            x = block(x)
        output = self.rmsnrom3(x)
        output = self.lm_linear(output)
        return output


def get_lr(it, warmup_steps, max_lr, max_steps, min_lr):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def train(batch_size, block_size, epochs, lr, max_steps, warmup_steps, max_lr, data_root, grad_accum_steps, vocab_size, weight_decay, min_lr):
    # Initialize wandb
    config = {
        "batch_size": batch_size,
        "block_size": block_size,
        "epochs": epochs,
        "learning_rate": lr,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "max_lr": max_lr,
        "min_lr": min_lr,
        "data_root": data_root,
        "grad_accum_steps": grad_accum_steps,
        "vocab_size": vocab_size,
        "weight_decay": weight_decay,
        "embed_dim": 1024,
        "num_heads": 16,
        "num_blocks": 8,
        "dropout": 0.2
    }
    
    wandb.init(
        project="fineweb-gpt2-training",
        name="gpt2-standard-training",
        config=config,
        tags=["gpt2", "transformer", "standard"]
    )
    
    train_dataloader = DataLoaderLite(B=batch_size, T=block_size, split="train", data_root=data_root)
    val_dataloader = DataLoaderLite(B=batch_size, T=block_size, split="val", data_root=data_root)

    model = Model1(block_size=block_size, vocab_size=vocab_size, embed_dim=1024, num_heads=16, num_blocks=8, dropout=0.2).to(device)
    
    # Log model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    })
    
    if device == "mps":
        pass
    else:
        model = torch.compile(model)
        print("[green]Using compiled model[/green]")

    # Watch model for gradient tracking
    wandb.watch(model, log="all", log_freq=100)

    param_dict = [p for p in model.parameters()]
    param_dict = [p for p in param_dict if p.requires_grad]
    decay_params = [p for p in param_dict if p.dim() >= 2]
    nodecay_params = [p for p in param_dict if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    hidden_weights = [p for p in model.blocks.parameters() if p.ndim >= 2]
    hidden_gains_biases = [p for p in model.blocks.parameters() if p.ndim < 2]
    nonhidden_params = [*model.lm_linear.parameters(), *model.embedding.parameters()]
    param_groups = [
        dict(params=hidden_weights, use_muon=True,
         lr=0.02, weight_decay=0.01),
        dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
         lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
    ]
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups) 

    loss = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_acum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_dataloader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                output = model(x)
                train_loss = loss(output.view(-1, output.size(-1)), y.view(-1))
            train_loss = train_loss / grad_accum_steps
            loss_acum += train_loss.detach()
            train_loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    
        lr = get_lr(epoch, warmup_steps, max_lr, max_steps, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with torch.no_grad():
            x, y = val_dataloader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                output = model(x)
                val_loss = loss(output.view(-1, output.size(-1)), y.view(-1))

        # Log to wandb
        wandb.log({
            "train/loss": loss_acum.item(),
            "val/loss": val_loss.item(),
            "train/perplexity": torch.exp(loss_acum).item(),
            "val/perplexity": torch.exp(val_loss).item(),
            "training/epoch": epoch,
            "training/learning_rate": lr,
            "training/gradient_norm": norm.item(),
        })

        print(f"[purple]Epoch[/purple]: {epoch}| [blue]Train Loss[/blue]: {loss_acum.item()} | [magenta]Val Loss[/magenta]: {val_loss.item()} | [bold cyan]Norm[/bold cyan]: {norm} | [bold turquoise4]lr[/bold turquoise4]: {lr}")

    if os.path.exists("weights"):
        pass
    else:
        print("[yellow]creating a weights directory[/yellow]") 
        os.mkdir("weights")
        
    torch.save(model.state_dict(), "./weights/model1-edu_weights.pth")
    torch.save(optimizer.state_dict(), "./weights/optimizer1-edu_weights.pth")
    
    wandb.finish()

# Main training call
train(
    batch_size=BATCH_SIZE,
    block_size=BLOCK_SIZE,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    max_steps=MAX_STEPS,
    warmup_steps=WARMUP_STEPS,
    max_lr=MAX_LR,
    data_root=DATA_ROOT,
    grad_accum_steps=GRAD_ACCUM_STEPS,
    vocab_size=VOCAB_SIZE,
    weight_decay=WEIGHT_DECAY,
    min_lr=MIN_LR
) 
