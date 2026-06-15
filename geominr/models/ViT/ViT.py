import math
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from .patchify import Patchify
from .vit_block import VitBlock


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

        half: int = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0)) * torch.arange(half) / (half - 1)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, times: Float[Tensor, "b"]) -> torch.Tensor:
        b: int = times.shape[0]
        device, dtype = times.device, times.dtype

        angles = times[:, None] * self.freqs[None, :]  # pyrefly:ignore
        emb = torch.empty((b, self.dim), device=device, dtype=dtype)
        emb[:, 0::2] = torch.sin(angles)
        emb[:, 1::2] = torch.cos(angles)
        return emb


class ViT(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_blocks: int,
        mlp_ratio: float,
        input_size: Tuple[int, int] = (224, 224),
        patch_kernel: Tuple[int, int] = (7, 7),
        patch_stride: Tuple[int, int] = (4, 4),
        patch_padding: Tuple[int, int] = (3, 3),
        drop_path_p: float = 0.0,
        norm: str = "layernorm",
        use_swiglu: bool = False,
        use_bias=True,
    ):
        super().__init__()
        assert norm in ["layernorm", "rmsnorm"]
        self.patch_stride = patch_stride
        self.input_size = input_size
        self.num_patches = (input_size[0] // patch_stride[0]) * (
            input_size[1] // patch_stride[1]
        )
        self.patchify = Patchify(
            in_channels=2,
            out_channels=dim,
            patch_kernel=patch_kernel,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            use_bias=use_bias,
        )

        self.positional_embedding = nn.Parameter(
            torch.randn((1, self.num_patches, dim))
        )
        self.time_embedding = SinusoidalTimeEmbedding(dim=dim)

        self.blocks = nn.ModuleList(
            [
                VitBlock(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    num_heads=num_heads,
                    drop_path_p=drop_path_p,
                    norm=norm,
                    use_swiglu=use_swiglu,
                    use_bias=use_bias,
                )
                for _ in range(num_blocks)
            ]
        )

        self.linear_out = nn.Linear(dim, math.prod(patch_stride), bias=use_bias)

        self.apply(self._init_weights)

    def forward(
        self, x: Float[Tensor, "B C H W"], time: Float[Tensor, "B"]
    ) -> Float[Tensor, "B 1 H W"]:
        x: Float[Tensor, "B N C"] = self.patchify(x)
        x: Float[Tensor, "B N C"] = x + self.positional_embedding
        x: Float[Tensor, "B N C"] = x + self.time_embedding(time)[:, None, :]

        for block in self.blocks:
            x = block(x)

        p1, p2 = self.patch_stride
        h, w = self.input_size[0] // p1, self.input_size[1] // p2

        x: Float[Tensor, "B N p1*p2"] = self.linear_out(x)
        x: Float[Tensor, "B 1 H W"] = rearrange(
            x,
            "B (h w) (p1 p2 C) -> B C (h p1) (w p2)",
            h=h,
            w=w,
            p1=p1,
            p2=p2,
            C=1,
        )

        return x


    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
