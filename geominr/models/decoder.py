import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import math


class Decoder(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: int) -> None:
        super().__init__()

        self.spatial_embedding = nn.Sequential(
            nn.Linear(3, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

        self.temp = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, xrtheta):
        """
        Args:
            x: Tensor, [B,K,H,W]
            xrtheta: Tensor, [B,N,3]
        Returns:
            out: Tensor, [B,N,1]
        """
        _, K, H, W = x.shape
        dtype = x.dtype

        xy_normalized = xrtheta[..., :2]  # [B,N,2] in normalized coords [-1,1]
        grid = xy_normalized.unsqueeze(2).to(dtype)  # [B,N,1,2]

        sampled_features: Tensor = F.grid_sample(
            x,
            grid,
            mode="nearest",
            padding_mode="border",
            align_corners=False,
        )  # [B,K,N,1]
        sampled_features: Tensor = sampled_features.squeeze(-1).permute(
            0, 2, 1
        )  # [B,N,K]

        dx, dy = self._compute_spatial_offsets(
            xy_normalized, H, W, align_corners=False
        )  # [B,N] each

        z_coord = xrtheta[..., 2:3].to(dtype)  # [B,N,1]

        dx_expanded = dx.unsqueeze(-1)  # [B,N,1]
        dy_expanded = dy.unsqueeze(-1)  # [B,N,1]
        pos_input: Tensor = torch.cat(
            [dx_expanded, dy_expanded, z_coord], dim=-1
        )  # [B,N,3]
        pos_embed = self.spatial_embedding(pos_input)  # [B,N,K]

        scale = 1 / ((K**0.5) * torch.exp(self.temp))

        x = F.sigmoid(
            torch.sum(sampled_features * pos_embed, dim=-1, keepdim=True) * scale
        )  # [B,N,1]

        return x  # [B,N,1]

    def _compute_spatial_offsets(self, xy_normalized, H, W, align_corners=False):
        """
        Args:
            xy_normalized: [B,N,2] normalized coordinates in [-1,1]
            H, W: height and width of feature map
            align_corners: whether to use align_corners mode

        Returns:
            dx, dy: each [B,N] in pixel units (sub-pixel offsets from nearest pixel)
        """
        if align_corners:
            width_denom = max(W - 1, 1)
            height_denom = max(H - 1, 1)
            x_pix = (xy_normalized[..., 0] + 1.0) * 0.5 * width_denom
            y_pix = (xy_normalized[..., 1] + 1.0) * 0.5 * height_denom
        else:
            x_pix = (xy_normalized[..., 0] + 1.0) * 0.5 * W - 0.5
            y_pix = (xy_normalized[..., 1] + 1.0) * 0.5 * H - 0.5

        x_nearest: Tensor = torch.round(x_pix)
        y_nearest: Tensor = torch.round(y_pix)

        dx = x_pix - x_nearest
        dy = y_pix - y_nearest

        return dx, dy
