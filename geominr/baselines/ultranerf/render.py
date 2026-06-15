from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import UltraNerfIntensityMLP


def gaussian_kernel(
    *,
    size: int = 3,
    mean: float = 0.0,
    std: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if size < 0:
        raise ValueError("size must be >= 0")
    dev = torch.device(device) if device is not None else None
    grid = torch.arange(-size, size + 1, device=dev, dtype=dtype)
    # Matches the official TensorFlow code: wider spread in x than y.
    vals_x = torch.exp(-0.5 * ((grid - mean) / (std * 2.0)) ** 2) / (std * 2.0)
    vals_y = torch.exp(-0.5 * ((grid - mean) / std) ** 2) / std
    kernel = torch.einsum("i,j->ij", vals_x, vals_y)
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel.view(1, 1, kernel.shape[0], kernel.shape[1])


def _exclusive_cumprod(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.numel() == 0:
        return x
    ones_shape = list(x.shape)
    ones_shape[dim] = 1
    prefix = torch.ones(ones_shape, device=x.device, dtype=x.dtype)
    shifted = torch.cat(
        [prefix, x.narrow(dim, 0, x.shape[dim] - 1)],
        dim=dim,
    )
    return torch.cumprod(shifted, dim=dim)


def render_ultra_nerf_from_raw(
    *,
    raw: torch.Tensor,
    r_positions: torch.Tensor,
    psf_kernel: torch.Tensor,
    stochastic: bool,
) -> torch.Tensor:
    # raw: [B, W_rays, H_samples, 5], r_positions: [B, W_rays, H_samples]
    if raw.ndim != 4 or raw.shape[-1] != 5:
        raise ValueError(f"Expected raw shape [B,W,H,5], got {tuple(raw.shape)}")
    if r_positions.shape != raw.shape[:-1]:
        raise ValueError(
            f"r_positions must match raw[...,0] shape. "
            f"Got {tuple(r_positions.shape)} vs {tuple(raw.shape[:-1])}"
        )

    dists = torch.abs(r_positions[..., 1:] - r_positions[..., :-1])
    dists = torch.cat([dists, dists[..., -1:]], dim=-1).clamp_min(1e-8)

    attenuation_coeff = torch.abs(raw[..., 0])
    attenuation = torch.exp(-attenuation_coeff * dists)
    attenuation_transmission = _exclusive_cumprod(attenuation, dim=-1)

    reflection_coeff = torch.sigmoid(raw[..., 1])
    border_prob = torch.sigmoid(raw[..., 2])
    density_prob = torch.sigmoid(raw[..., 3])
    scatter_amp = torch.sigmoid(raw[..., 4])

    if stochastic:
        border_indicator = torch.bernoulli(border_prob)
        scatter_density = torch.bernoulli(density_prob)
    else:
        border_indicator = border_prob
        scatter_density = density_prob

    reflection_transmission = _exclusive_cumprod(
        1.0 - reflection_coeff * border_indicator,
        dim=-1,
    )
    scatter_map = scatter_density * scatter_amp

    # Conv over (ray, sample) map, matching official 2D PSF application.
    pad = psf_kernel.shape[-1] // 2
    border_conv = F.conv2d(
        border_indicator.unsqueeze(1),
        psf_kernel,
        padding=pad,
    ).squeeze(1)
    psf_scatter = F.conv2d(
        scatter_map.unsqueeze(1),
        psf_kernel,
        padding=pad,
    ).squeeze(1)

    transmission = attenuation_transmission * reflection_transmission
    b = transmission * psf_scatter
    r = transmission * reflection_coeff * border_conv
    intensity = torch.clamp(b + r, 0.0, 1.0)
    return intensity


def render_patch_intensity(
    *,
    model: UltraNerfIntensityMLP,
    coords_patch: torch.Tensor,
    psf_kernel: torch.Tensor,
    stochastic: bool,
    query_chunk_points: int | None = None,
) -> torch.Tensor:
    # coords_patch: [B, H, W, 3] in normalized (x, r, theta).
    if coords_patch.ndim != 4 or coords_patch.shape[-1] != 3:
        raise ValueError(
            f"Expected coords_patch shape [B,H,W,3], got {tuple(coords_patch.shape)}"
        )

    bsz, h, w, _ = coords_patch.shape
    coords_flat = coords_patch.reshape(-1, 3)
    if query_chunk_points is None or int(query_chunk_points) <= 0:
        raw_flat = model.forward_raw(coords_flat)
    else:
        q = int(query_chunk_points)
        raw_chunks: list[torch.Tensor] = []
        for start in range(0, coords_flat.shape[0], q):
            end = min(start + q, coords_flat.shape[0])
            raw_chunks.append(model.forward_raw(coords_flat[start:end]))
        raw_flat = torch.cat(raw_chunks, dim=0)
    raw_hw = raw_flat.view(bsz, h, w, 5)

    # Renderer expects [B, W_rays, H_samples, 5]
    raw_wh = raw_hw.permute(0, 2, 1, 3).contiguous()
    r_positions = coords_patch[..., 1].permute(0, 2, 1).contiguous()

    intensity_wh = render_ultra_nerf_from_raw(
        raw=raw_wh,
        r_positions=r_positions,
        psf_kernel=psf_kernel,
        stochastic=stochastic,
    )

    intensity_hw = intensity_wh.permute(0, 2, 1).unsqueeze(1).contiguous()
    return intensity_hw
