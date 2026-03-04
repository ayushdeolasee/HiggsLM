import torch
import os

def save_checkpoint(model, optimizer, epoch, path):
    if not os.path.exists(path):
        os.mkdir(path)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, path)
    print(f"[green]Checkpoint saved at epoch {epoch} to {path}[/green]")

def load_checkpoint(model, optimizer, path):
    pass
