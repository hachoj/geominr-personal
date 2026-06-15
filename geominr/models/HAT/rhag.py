import torch
from torch import Tensor
import torch.nn as nn

from .hab import HAB
from .ocab import OCAB


class RHAG(nn.Module):
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
        overlap_ratio: float,
        num_hab_blocks: int,
        global_index: int,
    ) -> None:
        super().__init__()

        blocks = []
        for _ in range(num_hab_blocks):
            blocks.append(
                HAB(
                    embed_dim=embed_dim,
                    init_cab_weight=init_cab_weight,
                    cab_channel_reduction=cab_channel_reduction,
                    squeeze_factor=squeeze_factor,
                    H=H,
                    W=W,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    global_index=global_index,
                )  # pyrefly:ignore
            )
            global_index += 1
        blocks.append(
            OCAB(
                embed_dim=embed_dim,
                window_size=window_size,
                overlap_ratio=overlap_ratio,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
        )
        self.blocks = nn.Sequential(*blocks)

        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Args:
            x: [B,C,H,W]
        Returns:
            x: [B,C,H,W]
        """
        x = x + self.out_proj(self.blocks(x))
        return x
