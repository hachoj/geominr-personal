import torch
from torch import Tensor
import torch.nn as nn


class CAB(nn.Module):
    def __init__(
        self,
        channels: int,
        init_cab_weight: float,
        channel_reduction: int,
        squeeze_factor: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels // squeeze_factor, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // squeeze_factor, channels, kernel_size=3, padding=1),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, channels // channel_reduction),
            nn.GELU(),
            nn.Linear(channels // channel_reduction, channels),
            nn.Sigmoid(),
        )

        self.weight = nn.Parameter(torch.tensor(init_cab_weight))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B,C,H,W]
        Returns:
            out: [B,C,H,W]
        """
        B, C, H, W = x.shape

        x_proj = self.proj(x)  # [B,C,H,W]
        x_pooled = self.gap(x_proj).reshape(B, C)  # [B,C]

        w = self.channel_attention(x_pooled)  # [B,C]
        w = w[:, :, None, None]  # [B,C,1,1]
        out = self.weight * (x_proj * w)

        return out
