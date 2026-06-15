"""Shared optimizer / LR-schedule helpers for the feed-forward trainers.

Both `scripts/train.py` (HAT) and `scripts/train-vit.py` (ViT-S/2) import these so
the two models are trained under the *identical* recipe — Muon for 2D weight
matrices, AdamW for 1D params / biases / norms, with a selectable cosine or WSD
schedule. Keeping them in one place guarantees the baseline comparison is fair.
"""

import math

import torch


def _build_cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    init_lr: float,
    final_lr: float,
    warmup_steps: int = 0,
):
    def lr_lambda(step_idx: int) -> float:
        if step_idx < 0:
            step_idx = 0

        if warmup_steps > 0 and step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)

        progress = max(
            0.0,
            min(
                1.0,
                (step_idx - warmup_steps) / float(max(1, total_steps - warmup_steps)),
            ),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return (final_lr / init_lr) + (1.0 - final_lr / init_lr) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _build_wsd_with_warmup(optimizer, total_steps, warmup_steps=0, decay_frac=0.2):
    """Warmup -> stable (constant at peak) -> linear decay to 0 (WSD / trapezoidal).

    Linear warmup over ``warmup_steps``, constant at the peak LR until the final
    ``decay_frac`` of training (default 20%), then linear decay to 0.
    """
    decay_start = int(total_steps * (1.0 - decay_frac))

    def lr_lambda(step_idx: int) -> float:
        if step_idx < 0:
            step_idx = 0
        if warmup_steps > 0 and step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)
        if step_idx < decay_start:
            return 1.0
        decay_steps = max(1, total_steps - decay_start)
        return max(0.0, 1.0 - (step_idx - decay_start) / float(decay_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_lr_scheduler(cfg, optimizer, total_steps, base_lr, min_lr, warmup_steps):
    """Dispatch on ``cfg.train.lr_schedule``: ``cosine`` (default) or ``wsd``
    (constant 80% + linear decay to 0)."""
    if cfg.train.get("lr_schedule", "cosine") == "wsd":
        return _build_wsd_with_warmup(optimizer, total_steps, warmup_steps)
    return _build_cosine_with_warmup(optimizer, total_steps, base_lr, min_lr, warmup_steps)


def build_param_groups(model, lr_adamw, lr_muon, wd):
    muon_params = []  # 2D weights -> Muon, with decay
    adamw_decay = []  # non-2D that still get decay
    adamw_no_decay = []  # biases, norms, embeddings -> AdamW, no decay

    no_decay_keywords = {"bias", "norm", "ln_", "layernorm", "embedding"}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        skip_decay = any(kw in name.lower() for kw in no_decay_keywords)
        if p.ndim == 2 and not skip_decay:
            muon_params.append(p)
        elif skip_decay:
            adamw_no_decay.append(p)
        else:
            adamw_decay.append(p)

    return {
        "muon": [{"params": muon_params, "lr": lr_muon}],
        "adamw": [
            {"params": adamw_decay, "lr": lr_adamw, "weight_decay": wd},
            {"params": adamw_no_decay, "lr": lr_adamw, "weight_decay": 0.0},
        ],
    }
