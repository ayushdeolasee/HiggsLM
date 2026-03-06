import torch
import os

def save_checkpoint(model, optimizer, epoch, path, filename):
    save_path = os.path.join(path, filename)
    
    if not os.path.exists(path):
        os.mkdir(path)
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"[yellow]Deleted old checkpoint[/yellow]")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, save_path)
    print(f"[green]Checkpoint saved at epoch {epoch} to {path}[/green]")

def load_checkpoint(model, optimizer, path):
    pass
