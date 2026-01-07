import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# Device selection
device = torch.device("cpu")
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

print(f"Using device: {device}")

# Model hyperparameters (must match training configuration)
BLOCK_SIZE = 1024
VOCAB_SIZE = 50304
EMBED_DIM = 1024
NUM_HEADS = 16
NUM_BLOCKS = 8
DROPOUT = 0.2


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


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.n_embd = n_embed
        self.n_head = n_head
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
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

    def forward(self, x):
        return self.MLP(x)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(Block, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.rmsnorm1 = RMSNorm(embed_dim)
        self.MultiheadAttention = CausalSelfAttention(self.embed_dim, self.num_heads)
        self.rmsnrom2 = RMSNorm(embed_dim)  # Keep typo to match saved weights
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
        self.blocks = nn.ModuleList([
            Block(self.embed_dim, self.num_heads, self.dropout) 
            for _ in range(self.num_blocks)
        ])
        self.rmsnrom3 = RMSNorm(self.embed_dim)  # Keep typo to match saved weights
        self.lm_linear = nn.Linear(self.embed_dim, self.vocab_size)

        # Weight sharing scheme
        self.lm_linear.weight = self.embedding.weight

    def forward(self, x):
        B, T = x.shape
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        output = self.rmsnrom3(x)
        output = self.lm_linear(output)
        return output


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=40):
    """Generate text from a prompt using the model."""
    model.eval()
    
    # Encode the prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate tokens one at a time
    for _ in range(max_new_tokens):
        # Crop tokens to block_size if needed
        idx_cond = tokens if tokens.size(1) <= BLOCK_SIZE else tokens[:, -BLOCK_SIZE:]
        
        # Get model predictions
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        # Apply top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        tokens = torch.cat((tokens, next_token), dim=1)
        
        # Stop if we hit end of text token
        if next_token.item() == tokenizer.eot_token:
            break
    
    # Decode and return
    generated_tokens = tokens[0].tolist()
    return tokenizer.decode(generated_tokens)


def main():
    # Load tokenizer (GPT-2 tokenizer via tiktoken)
    print("Loading tokenizer...")
    tokenizer = tiktoken.encoding_for_model("gpt2")
    
    # Initialize model
    print("Initializing model...")
    model = Model1(
        block_size=BLOCK_SIZE,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT
    )
    
    # Load weights
    weights_path = "./models/model1-edu_weights.pth"
    print(f"Loading weights from {weights_path}...")
    
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully! Total parameters: {total_params:,}")
    
    # Generate text
    prompt = "photosynthesis is "
    print(f"\nPrompt: {prompt}")
    print("-" * 50)
    
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=100,
        temperature=0.8,
        top_k=40
    )
    
    print(f"Generated:\n{generated_text}")


if __name__ == "__main__":
    main()
