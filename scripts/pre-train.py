import torch
import os
import wandb
import torch.nn as nn

from llm.dataloader import DataLoaderLite
from llm.gpt import Model
from llm.optimizer import SingleDeviceMuonWithAuxAdam

batch_size = 8
block_size = 1024
epochs = 19073
learning_rate = 3e-4
max_steps = 19073
warmup_steps = 750 
max_lr = 3e-4
min_lr = 3e-5
data_root = "data"
grad_accum_steps = 8
vocab_size = 50304
weight_decay = 0.01
embed_dim = 1024
num_heads = 16
num_blocks = 8
kv_per_head = embed_dim // num_heads


device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"

if torch.cuda.is_available():
    device = "cuda"

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
    "embed_dim": embed_dim,
    "num_heads": num_heads,
    "num_blocks": num_blocks,
}

wandb.init(
    project="fineweb-gpt2-training",
    name="gpt2-standard-training",
    config=config,
    tags=["gpt2", "transformer", "standard"]
)
 
train_dataloader = DataLoaderLite(B=batch_size, T=block_size, split="train", data_root=data_root)
val_dataloader = DataLoaderLite(B=batch_size, T=block_size, split="val", data_root=data_root)

model = Model(block_size=block_size, vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, num_blocks=num_blocks, kv_per_head=kv_per_head).to(device)

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

    # Save model checkpoint every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        if not os.path.exists("weights"):
            os.mkdir("weights")
        mid_save_path = "./weights/model-mid-train-save.pth"
        # Delete old save if it exists
        if os.path.exists(mid_save_path):
            os.remove(mid_save_path)
            print(f"[yellow]Deleted old checkpoint[/yellow]")
        torch.save(model.state_dict(), mid_save_path)
        print(f"[green]Saved checkpoint at epoch {epoch + 1}[/green]")

        # Generate text from prompts
        print("\n[bold cyan]Running inference on test prompts...[/bold cyan]\n")
        enc = tiktoken.get_encoding("gpt2")
        model.eval()
        
        prompts = ["Photosynthesis is ", "3 + 7(5-7) is equal to "]
        
        for prompt in prompts:
            print(f"[yellow]Prompt:[/yellow] {prompt}")
            tokens = enc.encode(prompt)
            tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            
            with torch.no_grad():
                for _ in range(50):  # Generate 50 tokens
                    logits = model(tokens)
                    logits = logits[:, -1, :]  # Get last token logits
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    tokens = torch.cat([tokens, next_token], dim=1)
                    if tokens.size(1) >= block_size:
                        break
            
            generated_text = enc.decode(tokens[0].tolist())
            print(f"[green]Output:[/green] {generated_text}\n")
        
        model.train()  # Set back to training mode

if not os.path.exists("weights"):
    print("[yellow]creating a weights directory[/yellow]") 
    os.mkdir("weights")
    
# Save final model
torch.save(model.state_dict(), "./weights/model-final-save.pth")
torch.save(optimizer.state_dict(), "./weights/optimizer-final-save.pth")
print("[green]Saved final model weights[/green]")

wandb.finish()

