import torch
from torch.optim import lr_scheduler, optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def cosine_with_linear_wamrup(optimizer: optimizer, num_epochs: int, warmup_epochs: int, steps_per_epoch: int):
    cosine_scheduler = CosineAnnealingLR(optimizer, (num_epochs - warmup_epochs) * steps_per_epoch)
    linear_scheduler = LinearLR(optimizer, total_iters=warmup_epochs * steps_per_epoch)
    lr_scheduler = SequentialLR(optimizer, [linear_scheduler, cosine_scheduler], milestones=[warmup_epochs * steps_per_epoch])
    return lr_scheduler