from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Bool, Float
from torch import Tensor


class Patchify(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_kernel: Tuple[int, int] = (7, 7),
        patch_stride: Tuple[int, int] = (4, 4),
        patch_padding: Tuple[int, int] = (3, 3),
        use_bias=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_kernel,
            padding=patch_padding,
            stride=patch_stride,
            bias=use_bias,
        )

        self.apply(self._init_weights)

    def forward(
        self,
        x: Float[Tensor, "B 3 image_height image_width"],
        mask: Bool[Tensor, "B 3 image_height image_width"] | None = None,
    ) -> Float[Tensor, "B N C'"]:
        if mask is not None:
            x[mask] = 0
            x: Float[Tensor, "B C H W"] = self.conv(x)
            B, C, H, W = x.shape
            x: Float[Tensor, "B N C"] = rearrange(x, "B C H W -> B (H W) C")
        else:
            x: Float[Tensor, "B C H W"] = self.conv(x)
            B, C, H, W = x.shape
            x: Float[Tensor, "B N C"] = rearrange(x, "B C H W -> B (H W) C")
        return x

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
