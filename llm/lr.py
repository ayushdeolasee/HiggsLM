import math

def get_lr(epoch, warmup_steps, max_lr, epochs, min_lr):
    if epoch < warmup_steps:
        return max_lr * (epoch+1) / warmup_steps
    if epoch > epochs:
        return min_lr
    decay_ratio = (epoch - warmup_steps) / (epochs - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# Weight decay scheduler for Muon optimizer (linearly decays to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)
