"""Fixed training recipe, shared by both generator initializations."""
import math

import torch


def learning_rate(step):
    """Linear warmup to step 35; cosine decay to 75; constant thereafter."""
    if step < 1:
        raise ValueError("optimizer steps are one-based")
    if step <= 35:
        return 1.5e-6 * step / 50
    if step >= 75:
        return 2.1e-7
    phase = math.pi * (step - 35) / 40
    return 2.1e-7 + 0.5 * (1.05e-6 - 2.1e-7) * (1 + math.cos(phase))


def make_optimizer(parameters, device):
    return torch.optim.AdamW(
        parameters, lr=learning_rate(1), betas=(0.9, 0.95),
        eps=1e-6, weight_decay=1e-6, fused=torch.device(device).type == 'cuda',
    )
