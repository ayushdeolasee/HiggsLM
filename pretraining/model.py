import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    BATCH_SIZE: int = 8
    BLOCK_SIZE: int = 1024
    EPOCHS: int = 19073
    LEARNING_RATE: float = 3e-4
    MAX_STEPS: int = 19073
    WARMUP_STEPS: int = 750
    MAX_LR: float = 3e-4
    MIN_LR: float = 3e-5
    GRAD_ACCUM_STEPS: int = 8
    VOCAB_SIZE: int = 50304
    WEIGHT_DECAY: float = 0.01


