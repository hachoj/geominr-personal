from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..inr_common import make_pixel_luts, normalize_global_theta

from .model import ImplicitVolSiren


def _ssim_index(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 7,
    data_range: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3.")
    if pred.shape != target.shape:
        raise ValueError("pred and target must have matching shapes for SSIM.")
    if pred.ndim != 4:
        raise ValueError("Expected pred/target shape [B,1,H,W] for SSIM.")

    pad = window_size // 2
    pred_pad = F.pad(pred, (pad, pad, pad, pad), mode="reflect")
    target_pad = F.pad(target, (pad, pad, pad, pad), mode="reflect")

    mu_x = F.avg_pool2d(pred_pad, kernel_size=window_size, stride=1)
    mu_y = F.avg_pool2d(target_pad, kernel_size=window_size, stride=1)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = (
        F.avg_pool2d(pred_pad * pred_pad, kernel_size=window_size, stride=1) - mu_x2
    )
    sigma_y2 = (
        F.avg_pool2d(target_pad * target_pad, kernel_size=window_size, stride=1)
        - mu_y2
    )
    sigma_xy = (
        F.avg_pool2d(pred_pad * target_pad, kernel_size=window_size, stride=1) - mu_xy
    )

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / (denominator + eps)
    return ssim_map.mean()


def fit_patient_implicitvol(
    *,
    slices_u8: np.ndarray,
    angles_rad: np.ndarray,
    steps: int,
    batch_patches: int,
    lr: float,
    device: str | torch.device,
    hidden_dim: int,
    num_layers: int,
    w0_first: float,
    w0_hidden: float,
    patch_size: int = 32,
    ssim_window_size: int = 7,
    loss_mode: str = "ssim",
    num_frequencies: int = 10,
    lr_step_interval: int = 10,
    lr_gamma: float = 0.9954,
    log_every: int = 100,
) -> tuple[ImplicitVolSiren, dict[str, Any]]:
    if slices_u8.ndim != 3:
        raise ValueError(
            f"Expected slices_u8 shape [S,H,W], got {tuple(slices_u8.shape)}"
        )
    if angles_rad.ndim != 1:
        raise ValueError(
            f"Expected angles_rad shape [S], got {tuple(angles_rad.shape)}"
        )
    if slices_u8.shape[0] != angles_rad.shape[0]:
        raise ValueError("Number of slices and angle entries must match.")

    dev = torch.device(device)
    model = ImplicitVolSiren(
        input_dim=3,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        w0_first=w0_first,
        w0_hidden=w0_hidden,
        architecture="official",
        num_frequencies=num_frequencies,
        include_input=True,
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if lr_step_interval > 0 and 0.0 < lr_gamma < 1.0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_step_interval,
            gamma=lr_gamma,
        )

    slices = torch.from_numpy(slices_u8)  # CPU uint8 [S,H,W]
    angles = torch.from_numpy(angles_rad.astype(np.float32))  # CPU float32 [S]
    num_slices, height, width = slices.shape
    if patch_size > height or patch_size > width:
        raise ValueError(
            f"patch_size={patch_size} is too large for HxW={height}x{width}."
        )
    if ssim_window_size > patch_size:
        raise ValueError(
            f"ssim_window_size={ssim_window_size} cannot exceed patch_size={patch_size}."
        )
    if loss_mode not in {"mse", "ssim"}:
        raise ValueError(
            f"Unsupported loss_mode={loss_mode}. Expected one of: mse, ssim."
        )
    angle_min = float(angles.min().item())
    angle_max = float(angles.max().item())

    r_lut, x_lut = make_pixel_luts(height, width)  # CPU float32
    patch_area = patch_size * patch_size
    num_patches = max(1, int(batch_patches))
    points_per_step = int(num_patches * patch_area)
    row_offsets = torch.arange(patch_size, dtype=torch.int64).view(1, patch_size, 1)
    col_offsets = torch.arange(patch_size, dtype=torch.int64).view(1, 1, patch_size)

    loss_value = float("nan")
    ssim_value = float("nan")

    model.train()
    for step in range(1, steps + 1):
        slice_idx = torch.randint(0, num_slices, (num_patches,), dtype=torch.int64)
        row0 = torch.randint(0, height - patch_size + 1, (num_patches,), dtype=torch.int64)
        col0 = torch.randint(0, width - patch_size + 1, (num_patches,), dtype=torch.int64)

        rows = row0.view(-1, 1, 1) + row_offsets
        cols = col0.view(-1, 1, 1) + col_offsets
        slice_ids = slice_idx.view(-1, 1, 1).expand(-1, patch_size, patch_size)

        targets = slices[slice_ids, rows, cols].float() / 127.5 - 1.0
        targets = targets.unsqueeze(1).to(dev, non_blocking=True)

        x = x_lut[cols].expand(-1, patch_size, patch_size)
        r = r_lut[rows].expand(-1, patch_size, patch_size)
        theta_per_patch = normalize_global_theta(angles[slice_idx], angle_min, angle_max)
        theta = theta_per_patch.view(-1, 1, 1).expand(-1, patch_size, patch_size)

        coords = torch.stack([x, r, theta], dim=-1).reshape(-1, 3)
        coords = coords.to(dev, non_blocking=True)

        preds = model(coords).view(num_patches, 1, patch_size, patch_size)
        if loss_mode == "ssim":
            ssim_idx = _ssim_index(
                preds,
                targets,
                window_size=ssim_window_size,
                data_range=2.0,
            )
            loss = 1.0 - ssim_idx
            ssim_value = float(ssim_idx.item())
        else:
            loss = F.mse_loss(preds, targets)
            ssim_value = float("nan")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_value = float(loss.item())
        if log_every > 0 and (step == 1 or step % log_every == 0 or step == steps):
            if loss_mode == "ssim":
                print(
                    f"[implicitvol] step {step:5d}/{steps} | loss={loss_value:.6f} | ssim={ssim_value:.6f}",
                    flush=True,
                )
            else:
                print(
                    f"[implicitvol] step {step:5d}/{steps} | loss={loss_value:.6f} | mse={loss_value:.6f}",
                    flush=True,
                )

    stats: dict[str, Any] = {
        "model_kwargs": {
            "input_dim": 3,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "w0_first": w0_first,
            "w0_hidden": w0_hidden,
            "architecture": "official",
            "num_frequencies": int(num_frequencies),
            "include_input": True,
        },
        "num_slices": int(num_slices),
        "height": int(height),
        "width": int(width),
        "angle_min_rad": angle_min,
        "angle_max_rad": angle_max,
        "last_loss": loss_value,
        "last_ssim_index": ssim_value,
        "steps": int(steps),
        "batch_patches": int(num_patches),
        "batch_points": int(points_per_step),
        "num_patches": int(num_patches),
        "patch_size": int(patch_size),
        "ssim_window_size": int(ssim_window_size),
        "loss_mode": loss_mode,
        "lr": float(lr),
        "lr_step_interval": int(lr_step_interval),
        "lr_gamma": float(lr_gamma),
        "last_lr": float(optimizer.param_groups[0]["lr"]),
    }
    return model, stats
