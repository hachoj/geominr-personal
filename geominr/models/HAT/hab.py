import torch
from torch import Tensor
import torch.nn as nn

from .sw_msa import SWMSA
from .cab import CAB


class HAB(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        init_cab_weight: float,
        cab_channel_reduction: int,
        squeeze_factor: int,
        H: int,
        W: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: int,
        global_index: int,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.cab = CAB(
            embed_dim, init_cab_weight, cab_channel_reduction, squeeze_factor
        )
        shift = 0 if global_index % 2 == 0 else window_size // 2
        self.swmsa = SWMSA(H, W, embed_dim, num_heads, window_size, shift)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor, [B,C,H,W]
        Returns:
            x: Tensor, [B,C,H,W]
        """
        resid = x

        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)

        x = self.cab(x) + self.swmsa(x) + resid

        resid = x

        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)
        x = x + resid

        return x
