import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class FFN(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,   
        use_swiglu: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()
        self.use_swiglu = use_swiglu

        if self.use_swiglu:
            hidden_dim: int = int(dim * mlp_ratio * (2/3))
            self.in_proj = nn.Linear(
                in_features=dim, out_features=hidden_dim, bias=use_bias
            )
            self.gate = nn.Linear(
                in_features=dim, out_features=hidden_dim, bias=use_bias
            )
            self.act = nn.SiLU()
            self.out_proj = nn.Linear(
                in_features=hidden_dim, out_features=dim, bias=use_bias
            )
        else:
            hidden_dim: int = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(in_features=dim, out_features=hidden_dim, bias=use_bias),
                nn.GELU(),
                nn.Linear(in_features=hidden_dim, out_features=dim, bias=use_bias),
            )

        self.apply(self._init_weights)

    def forward(self, x: Float[Tensor, "B N C"]) -> Float[Tensor, "B N C"]:
        if self.use_swiglu:
            x = self.act(self.in_proj(x)) * self.gate(x)
            x = self.out_proj(x)
        elif not self.use_swiglu:
            x = self.mlp(x)
        return x

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
