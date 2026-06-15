from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..inr_common import make_pixel_luts, normalize_global_theta

from .model import UltraNerfIntensityMLP
from .render import gaussian_kernel, render_patch_intensity

MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _ssim_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int,
    data_range: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have matching shapes for SSIM.")
    if pred.ndim != 4:
        raise ValueError("Expected pred/target shape [B,1,H,W] for SSIM.")

    ws = int(window_size)
    ws = min(ws, pred.shape[-2], pred.shape[-1])
    if ws % 2 == 0:
        ws -= 1
    if ws < 1:
        ws = 1

    pad = ws // 2
    if ws == 1:
        pred_pad = pred
        target_pad = target
    else:
        pred_pad = F.pad(pred, (pad, pad, pad, pad), mode="reflect")
        target_pad = F.pad(target, (pad, pad, pad, pad), mode="reflect")

    mu_x = F.avg_pool2d(pred_pad, kernel_size=ws, stride=1)
    mu_y = F.avg_pool2d(target_pad, kernel_size=ws, stride=1)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = (
        F.avg_pool2d(pred_pad * pred_pad, kernel_size=ws, stride=1) - mu_x2
    ).clamp_min(0.0)
    sigma_y2 = (
        F.avg_pool2d(target_pad * target_pad, kernel_size=ws, stride=1) - mu_y2
    ).clamp_min(0.0)
    sigma_xy = F.avg_pool2d(pred_pad * target_pad, kernel_size=ws, stride=1) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    l_map = (2.0 * mu_xy + c1) / (mu_x2 + mu_y2 + c1 + eps)
    cs_map = (2.0 * sigma_xy + c2) / (sigma_x2 + sigma_y2 + c2 + eps)
    return l_map, cs_map


def _ms_ssim_index(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 7,
    data_range: float = 1.0,
    weights: tuple[float, ...] = MS_SSIM_WEIGHTS,
    eps: float = 1e-8,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have matching shapes for MS-SSIM.")
    if pred.ndim != 4 or pred.shape[1] != 1:
        raise ValueError("Expected pred/target shape [B,1,H,W] for MS-SSIM.")

    x = pred
    y = target

    levels = len(weights)
    max_levels = 1
    h, w = int(x.shape[-2]), int(x.shape[-1])
    while max_levels < levels and h >= 2 and w >= 2:
        max_levels += 1
        h //= 2
        w //= 2
    use_weights = weights[:max_levels]

    mcs: list[torch.Tensor] = []
    ssim_l = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)

    for level in range(max_levels):
        l_map, cs_map = _ssim_components(
            x,
            y,
            window_size=window_size,
            data_range=data_range,
            eps=eps,
        )
        l_val = l_map.mean(dim=(1, 2, 3))
        cs_val = cs_map.mean(dim=(1, 2, 3))
        ssim_l = l_val

        if level < max_levels - 1:
            mcs.append(cs_val.clamp_min(0.0))
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            y = F.avg_pool2d(y, kernel_size=2, stride=2)

    out = torch.ones_like(ssim_l)
    for i, cs in enumerate(mcs):
        out = out * torch.pow(cs.clamp_min(1e-8), use_weights[i])
    out = out * torch.pow(ssim_l.clamp_min(1e-8), use_weights[-1])
    return out.mean()


def fit_patient_ultra_nerf(
    *,
    slices_u8: np.ndarray,
    angles_rad: np.ndarray,
    steps: int,
    batch_patches: int,
    lr: float,
    device: str | torch.device,
    hidden_dim: int,
    num_layers: int,
    num_frequencies: int,
    encoder_use_pi: bool = False,
    loss_mode: str = "ssim_mse",
    ssim_weight: float = 0.9,
    mse_weight: float = 0.1,
    patch_size: int = 32,
    ssim_window_size: int = 7,
    lr_decay_steps: int = 250000,
    lr_decay_rate: float = 0.1,
    psf_kernel_size: int = 3,
    stochastic_render: bool = True,
    log_every: int = 100,
) -> tuple[UltraNerfIntensityMLP, dict[str, Any]]:
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
    model = UltraNerfIntensityMLP(
        input_dim=3,
        num_frequencies=num_frequencies,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        encoder_use_pi=encoder_use_pi,
        architecture="official",
        output_mode="raw",
        intensity_range="zero_one",
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if lr_decay_steps > 0 and 0.0 < lr_decay_rate < 1.0:
        gamma = float(lr_decay_rate) ** (1.0 / float(lr_decay_steps))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

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
    if loss_mode not in {"mse", "ssim_mse"}:
        raise ValueError(
            f"Unsupported loss_mode={loss_mode}. Expected one of: mse, ssim_mse."
        )

    angle_min = float(angles.min().item())
    angle_max = float(angles.max().item())
    r_lut, x_lut = make_pixel_luts(height, width)  # CPU float32
    psf_kernel = gaussian_kernel(size=psf_kernel_size, device=dev)

    patch_area = patch_size * patch_size
    num_patches = max(1, int(batch_patches))
    points_per_step = int(num_patches * patch_area)
    row_offsets = torch.arange(patch_size, dtype=torch.int64).view(1, patch_size, 1)
    col_offsets = torch.arange(patch_size, dtype=torch.int64).view(1, 1, patch_size)

    loss_value = float("nan")
    mse_value = float("nan")
    ssim_value = float("nan")
    model.train()
    for step in range(1, steps + 1):
        slice_idx = torch.randint(0, num_slices, (num_patches,), dtype=torch.int64)
        row0 = torch.randint(0, height - patch_size + 1, (num_patches,), dtype=torch.int64)
        col0 = torch.randint(0, width - patch_size + 1, (num_patches,), dtype=torch.int64)

        rows = row0.view(-1, 1, 1) + row_offsets
        cols = col0.view(-1, 1, 1) + col_offsets
        slice_ids = slice_idx.view(-1, 1, 1).expand(-1, patch_size, patch_size)

        targets = slices[slice_ids, rows, cols].float() / 255.0
        targets = targets.unsqueeze(1).to(dev, non_blocking=True)

        x = x_lut[cols].expand(-1, patch_size, patch_size)
        r = r_lut[rows].expand(-1, patch_size, patch_size)
        theta_per_patch = normalize_global_theta(angles[slice_idx], angle_min, angle_max)
        theta = theta_per_patch.view(-1, 1, 1).expand(-1, patch_size, patch_size)
        coords_patch = torch.stack([x, r, theta], dim=-1).to(dev, non_blocking=True)

        preds = render_patch_intensity(
            model=model,
            coords_patch=coords_patch,
            psf_kernel=psf_kernel,
            stochastic=stochastic_render,
        )
        mse_loss = F.mse_loss(preds, targets)

        if loss_mode == "ssim_mse":
            ms_ssim_idx = _ms_ssim_index(
                preds,
                targets,
                window_size=ssim_window_size,
                data_range=1.0,
            )
            ssim_loss = 1.0 - ms_ssim_idx
            loss = ssim_weight * ssim_loss + mse_weight * mse_loss
            ssim_value = float(ms_ssim_idx.item())
        else:
            loss = mse_loss
            ssim_value = float("nan")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_value = float(loss.item())
        mse_value = float(mse_loss.item())
        if log_every > 0 and (step == 1 or step % log_every == 0 or step == steps):
            if loss_mode == "ssim_mse":
                print(
                    f"[ultra_nerf] step {step:5d}/{steps} | "
                    f"loss={loss_value:.6f} | mse={mse_value:.6f} | ms_ssim={ssim_value:.6f}",
                    flush=True,
                )
            else:
                print(
                    f"[ultra_nerf] step {step:5d}/{steps} | loss={loss_value:.6f} | mse={mse_value:.6f}",
                    flush=True,
                )

    stats: dict[str, Any] = {
        "model_kwargs": {
            "input_dim": 3,
            "num_frequencies": num_frequencies,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "encoder_use_pi": bool(encoder_use_pi),
            "architecture": "official",
            "output_mode": "raw",
            "intensity_range": "zero_one",
        },
        "num_slices": int(num_slices),
        "height": int(height),
        "width": int(width),
        "angle_min_rad": angle_min,
        "angle_max_rad": angle_max,
        "normalization": "zero_one",
        "renderer": "faithful_ultra_nerf",
        "stochastic_render": bool(stochastic_render),
        "psf_kernel_size": int(psf_kernel_size),
        "ssim_metric": "ms_ssim",
        "last_loss": loss_value,
        "last_mse_loss": mse_value,
        "last_ssim_index": ssim_value,
        "steps": int(steps),
        "batch_patches": int(num_patches),
        "batch_points": int(points_per_step),
        "num_patches": int(num_patches),
        "patch_size": int(patch_size),
        "loss_mode": loss_mode,
        "ssim_weight": float(ssim_weight),
        "mse_weight": float(mse_weight),
        "ssim_window_size": int(ssim_window_size),
        "lr": float(lr),
        "lr_decay_steps": int(lr_decay_steps),
        "lr_decay_rate": float(lr_decay_rate),
        "last_lr": float(optimizer.param_groups[0]["lr"]),
    }
    return model, stats
