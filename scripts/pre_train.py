import torch
import os
import wandb
import torch.nn as nn
import argparse
import tiktoken
import torch.nn.functional as F
import time

from llm.dataloader import DataLoaderLite
from llm.gpt import Model
from llm.optimizer import MuonAdamW 
from llm.lr import get_lr
from llm.checkpoint_manager import save_checkpoint

parser = argparse.ArgumentParser(description="Run the pre-training script")
# TODO: Inconsitency in naming for num_heads and n_heads
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for dataset")
parser.add_argument(
    "--seq_length", type=int, default=1024, help="Seq length for the model"
)
parser.add_argument(
    "--epochs", type=int, default=19083, help="Number of epochs for the training loop"
)
parser.add_argument(
    "--learning_rate", type=int, default=3e-4, help="Base learning rate"
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=750,
    help="Number of epochs for learning rate scheduler",
)
parser.add_argument("--lr", type=int, default=3e-4, help="Base learning rate")
parser.add_argument(
    "--max_lr",
    type=int,
    default=3e-4,
    help="Maximum learning rate for learning rate scheduler",
)
parser.add_argument(
    "--min_lr",
    type=int,
    default=3e-5,
    help="Minimum learning rate for learning rate scheduler",
)
parser.add_argument(
    "--data_root",
    type=str,
    default="./data",
    help="Folder for the data used for training",
)
parser.add_argument(
    "--grad_accum_steps",
    type=int,
    default=8,
    help="Number of steps for which the gradients are accumulated",
)
parser.add_argument(
    "--vocab_size", type=int, default=50304, help="Vocab size from the tokenizer"
)
parser.add_argument(
    "--weight_decay",
    type=int,
    default=0.01,
    help="Weight decay constant used for the optimizer",
)
parser.add_argument(
    "--embed_dim", type=int, default=1024, help="Hidden dimension of the MLP layers"
)
parser.add_argument(
    "--num_heads",
    type=int,
    default=16,
    help="Number of heads for the multi-head attention",
)
parser.add_argument(
    "--num_blocks",
    type=int,
    default=8,
    help="Number of transformer blocks in the model",
)
parser.add_argument(
    "--query_heads_per_kv",
    type=int,
    default=2,
    help="Number of query heads that share each key/value head in grouped-query attention",
)
parser.add_argument(
    "--wandb", action=argparse.BooleanOptionalAction, help="Whether to use wandb for logging"
)
parser.add_argument("--use_checkpointing", action=argparse.BooleanOptionalAction, help="Use checkpointing to trade vram usage for compute")

args = parser.parse_args()
total_time = 0.0

assert args.max_lr > args.min_lr, "max_lr should be greater than min_lr"
assert args.warmup_steps < args.epochs, "warmup_steps should be less than epochs"
assert os.path.exists(args.data_root), f"data_root folder {args.data_root} does not exist"

device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
if torch.cuda.is_available():
    device = "cuda"

print("[green]Using device:[green]", device)

if (args.wandb == True):
    config = {
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "warmup_steps": args.warmup_steps,
        "max_lr": args.max_lr,
        "min_lr": args.min_lr,
        "data_root": args.data_root,
        "grad_accum_steps": args.grad_accum_steps,
        "vocab_size": args.vocab_size,
        "weight_decay": args.weight_decay,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_blocks": args.num_blocks,
    }

    wandb.init(
        project="HiggsLM",
        name="higgslm-pre-training",
        config=config,
        tags=["higgslm", "transformer", "pre-training"],
    )

train_dataloader = DataLoaderLite(
    B=args.batch_size, T=args.seq_length, split="train", data_root=args.data_root
)

val_dataloader = DataLoaderLite(
    B=args.batch_size, T=args.seq_length, split="val", data_root=args.data_root
)

model = Model(
    seq_length=args.seq_length,
    vocab_size=args.vocab_size,
    embed_dim=args.embed_dim,
    num_heads=args.num_heads,
    num_blocks=args.num_blocks,
    query_heads_per_kv=args.query_heads_per_kv,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

if (args.wandb == True):
    wandb.config.update(
        {"total_parameters": total_params, "trainable_parameters": trainable_params}
    )
    wandb.watch(model, log="all", log_freq=100)
if device == "mps":
    pass
else:
    if args.use_checkpointing: 
        torch._functorch.config.activation_memory_budget = 0.5
        model = torch.compile(model)
        print("[green]Using compiled model with activation_memory_budget of 0.5[/green]")
    else:
        model = torch.compile(model)
        print("[green]Using compiled model[/green]")

print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")

embedding_param = [p for p in model.embedding.parameters()]
projection_params = [p for p in model.lm_linear.parameters()]
block_params = [p for p in model.blocks.parameters() if p.ndim >= 2]
hidden_gains_biases = [p for p in model.blocks.parameters() if p.ndim < 2]

# TODO: add this as passable parameters
matrix_lr = 1e-4
weight_decay = 0.01 

"""
param_dict = [p for p in model.parameters()]
param_dict = [p for p in param_dict if p.requires_grad]
decay_params = [p for p in param_dict if p.dim() >= 2]
nodecay_params = [p for p in param_dict if p.dim() < 2]

optim_groups = [
    {"params": decay_params, "weight_decay": args.weight_decay},
    {"params": nodecay_params, "weight_decay": 0.0},
]

hidden_weights = [p for p in model.blocks.parameters() if p.ndim >= 2]
hidden_gains_biases = [p for p in model.blocks.parameters() if p.ndim < 2]
nonhidden_params = [*model.lm_linear.parameters(), *model.embedding.parameters()]
"""

param_groups = [
    dict(kind="adamw", params=projection_params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01, eps=1e-10), 
    dict(kind="adamw", params=embedding_param, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01, eps=1e-10), 
    dict(kind="adamw", params=hidden_gains_biases, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01, eps=1e-10) 
]

for shape in sorted({p.shape for p in block_params}):
    group_params = [p for p in block_params if p.shape == shape]
    param_groups.append(dict(kind='muon', params=group_params, lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,))

optimizer = MuonAdamW(param_groups)
loss = nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    optimizer.zero_grad()
    loss_acum = 0.0
    
    start_time = time.perf_counter() 
    for micro_step in range(args.grad_accum_steps):
        x, y = train_dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            output = model(x)
            train_loss = loss(output.view(-1, output.size(-1)), y.view(-1))
        
        train_loss = train_loss / args.grad_accum_steps
        loss_acum += train_loss.detach()
        train_loss.backward()
    end_time = time.perf_counter()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    elapsed_time = end_time - start_time
    total_time += elapsed_time 
    
    lr = get_lr(epoch, args.warmup_steps, args.max_lr, args.epochs, args.min_lr)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    if epoch % 100 == 0:
        with torch.no_grad():
            x, y = val_dataloader.next_batch()
            x, y = x.to(device), y.to(device)
       
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                output = model(x)
                val_loss = loss(output.view(-1, output.size(-1)), y.view(-1))
        
    
    if (args.wandb == True): 
        wandb.log(
            {
                "train/loss": loss_acum.item(),
                "val/loss": val_loss.item(),
                "train/perplexity": torch.exp(loss_acum).item(),
                "val/perplexity": torch.exp(val_loss).item(),
                "train/elapsed_time": elapsed_time, 
                "train/epoch": epoch,
                "train/learning_rate": lr,
                "train/gradient_norm": norm.item(),
            }
        )

    print(
            f"[purple]Epoch[/purple]: {epoch}| [blue]Train Loss[/blue]: {loss_acum.item()} | [magenta]Val Loss[/magenta]: {val_loss.item()} | [bold cyan]Norm[/bold cyan]: {norm} | [bold turquoise4]lr[/bold turquoise4]: {lr} | [bold yellow]Elapsed Time[/bold yellow]: {elapsed_time:.2f} seconds"
    )

    if (epoch + 1) % 1000 == 0:
        save_checkpoint(model, optimizer, epoch + 1, path="./weights", filename = "model_mid_pre_train.pth") 

        print("\n[bold cyan]Running inference on test prompts...[/bold cyan]\n")
        enc = tiktoken.get_encoding("gpt2")
        model.eval()

        # TODO: Use an actual eval test instead of these two hard-coded prompts  
        prompts = ["Photosynthesis is ", "3 + 7(5-7) is equal to "]

        for prompt in prompts:
            print(f"[yellow]Prompt:[/yellow] {prompt}")
            tokens = enc.encode(prompt)
            tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

            with torch.no_grad():
                for _ in range(50): 
                    logits = model(tokens)
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    tokens = torch.cat([tokens, next_token], dim=1)
                    if tokens.size(1) >= args.seq_length:
                        break

            generated_text = enc.decode(tokens[0].tolist())
            print(f"[green]Output:[/green] {generated_text}\n")

        model.train() 
save_checkpoint(model, optimizer, args.epochs, path="./weights", filename="pre_train.pth")
print(f"[green]Training completed in {total_time:.2f} seconds. Average time per epoch: {(total_time / args.epochs):.2f}. Final checkpoint saved.[/green]")

if (args.wandb == True):
    wandb.finish()
