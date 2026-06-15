import math
import os

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm


def trilinear_sample_old(slices, xyz):
    B, _, H, W = slices.shape
    _, N, _ = xyz.shape
    slices = slices.view(B, 2, 1, H, W).permute(0, 2, 1, 3, 4)  # [B,1,2,H,W]
    out_lin = F.grid_sample(
        slices,
        xyz.view(B, N, 1, 1, 3),
        align_corners=False,
    )  # [B,1,N,1,1]
    out_lin = out_lin.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # [B,N,1]
    return out_lin

def trilinear_sample(slices: Float[Tensor, "B C H W"], times: Float[Tensor, "B"]):
    B, C, _, _ = slices.shape
    if C != 2:
        raise ValueError(f"Expected `slices` shape [B,2,H,W], got C={C}.")

    t = times.to(device=slices.device, dtype=slices.dtype).view(B, 1, 1, 1)
    t = t.clamp(0.0, 1.0)
    return torch.lerp(slices[:, 0:1], slices[:, 1:2], t)


def create_xrtheta(
    B, H, W, target_angles, conditioning_angles, device, align_corners=False
):
    """
    Args:
        B: Batch size
        H: Height
        W: Width
        target_angles: [B] angle of the GT slice
        conditioning_angles: [B, 2] the two conditioning slice angles (unordered)
        device: Device to create tensors on
        align_corners: Whether to use align_corners mode
    Returns:
        grid_xrtheta: [B,H*W,3] in (x,r,theta) order, normalized for grid_sample
    """
    h = torch.arange(H, device=device, dtype=torch.float32)  # 0..H-1
    w = torch.arange(W, device=device, dtype=torch.float32)  # 0..W-1
    r, x = torch.meshgrid(h, w, indexing="ij")  # 2 x [H,W]

    if align_corners:
        r_norm = 2.0 * r / (H - 1) - 1.0
        x_norm = 2.0 * x / (W - 1) - 1.0
    else:
        r_norm = (r + 0.5) / H * 2 - 1
        x_norm = (x + 0.5) / W * 2 - 1

    r_norm = r_norm.reshape(1, -1).expand(B, -1)  # [B,H*W]
    x_norm = x_norm.reshape(1, -1).expand(B, -1)  # [B,H*W]

    mins = conditioning_angles.min(dim=1, keepdim=True).values  # [B,1]
    maxs = conditioning_angles.max(dim=1, keepdim=True).values  # [B,1]
    den = (maxs - mins).clamp_min(1e-8)

    rel_theta = ((target_angles.unsqueeze(1) - mins) / den).clamp(
        0, 1
    )  # [B,1] -> [0,1]

    if align_corners:
        theta_norm = 2.0 * rel_theta - 1.0  # [-1,1]
    else:
        theta_norm = rel_theta - 0.5  # [-0.5,0.5]

    theta_norm = theta_norm.expand(B, H * W)  # [B,H*W]

    grid_xrtheta = torch.stack([x_norm, r_norm, theta_norm], dim=-1)  # (x,r,theta)
    return grid_xrtheta


def strip_orig_mod(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    def clean_key(key: str) -> str:
        cleaned = key.replace("_orig_mod", "")
        return cleaned.lstrip(".")

    return {clean_key(k): v for k, v in state_dict.items()}
