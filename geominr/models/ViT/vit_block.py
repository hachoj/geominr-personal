import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .attention import AttentionBlock
from .drop_path import DropPath
from .ffn import FFN


class VitBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        num_heads: int,
        drop_path_p: float,
        norm: str = "layernorm",
        use_swiglu: bool = False,
        use_bias=True,
    ):
        super().__init__()
        assert norm in ["layernorm", "rmsnorm"]
        self.dim = dim
        self.drop_path = DropPath(drop_path_p)

        self.norm1 = nn.LayerNorm(dim) if norm == "layernorm" else nn.RMSNorm(dim)
        self.attention = AttentionBlock(
            dim=dim,
            num_heads=num_heads,
            use_bias=use_bias,
        )
        self.norm2 = nn.LayerNorm(dim) if norm == "layernorm" else nn.RMSNorm(dim)
        self.ffn = FFN(
            dim=dim,
            mlp_ratio=mlp_ratio,
            use_swiglu=use_swiglu,
            use_bias=use_bias,
        )

    def forward(self, x: Float[Tensor, "B N C"]) -> Float[Tensor, "B N C"]:
        x = x + self.drop_path(self.attention(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
