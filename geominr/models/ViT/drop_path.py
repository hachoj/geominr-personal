import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        assert p >= 0.0 and p <= 1.0, "invalid probability"
        self.p = float(p)

    def forward(self, x: Float[Tensor, "B ..."]) -> Float[Tensor, "B ..."]:
        """
        Args:
            x: Any tensor with a batch dimension

        Returns:
            Roughly p percent of the items in a batch will be masked to zeros
        """
        if self.p == 0.0 or not self.training:
            return x

        B = x.shape[0]

        keep_p = 1 - self.p
        shape = (B, ) + (1, ) * (x.ndim - 1)
        mask = (torch.rand(B) <= keep_p).float().to(device=x.device)
        mask = mask.view(shape)
        mask = mask * (1/(1-self.p))
        x = x * mask
        return x 