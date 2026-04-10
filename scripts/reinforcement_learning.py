import torch


device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using deivce: {device}")
