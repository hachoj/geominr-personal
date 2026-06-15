import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        use_bias=True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        head_dim = dim // num_heads
        num_heads = num_heads

        self.project_kqv = nn.Linear(dim, 3 * dim, bias=use_bias)

        self.proj = nn.Linear(dim, dim, bias=use_bias)

        self.head_dim = head_dim
        self.num_heads = num_heads

        self.apply(self._init_weights)

    def forward(self, x: Float[Tensor, "B N C"]) -> Float[Tensor, "B N C"]:
        B, N, _ = x.shape

        x: Float[Tensor, "B N 3C"] = self.project_kqv(x)
        x: Float[Tensor, "3 B H N C"] = rearrange(
            x,
            "B N (three head_dim num_heads) -> three B num_heads N head_dim",
            three=3,
            head_dim=self.head_dim,
            num_heads=self.num_heads,
        )
        q, k, v = x[0], x[1], x[2]

        x: Float[Tensor, "B num_heads N head_dim"] = F.scaled_dot_product_attention(
            q, k, v
        )
        x: Float[Tensor, "B N C"] = rearrange(
            x, "B num_heads N head_dim -> B N (num_heads head_dim)"
        )

        x: Float[Tensor, "B N C"] = self.proj(x)

        return x

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
