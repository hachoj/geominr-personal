import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


from .HAT.hat import HAT
from .decoder import Decoder


class model(nn.Module):
    def __init__(
        self,
        in_dim: int,
        K: int,
        embed_dim: int,
        init_cab_weight: float,
        cab_channel_reduction: int,
        squeeze_factor: int,
        H: int,
        W: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: int,
        num_rhag_blocks: int,
        overlap_ratio: float,
        num_hab_blocks: int,
        use_arc_embed: bool = True,
        use_swmca: bool = True,
    ) -> None:
        super().__init__()

        self.encoder = HAT(
            in_dim=in_dim,
            K=K,
            embed_dim=embed_dim,
            init_cab_weight=init_cab_weight,
            cab_channel_reduction=cab_channel_reduction,
            squeeze_factor=squeeze_factor,
            H=H,
            W=W,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            num_rhag_blocks=num_rhag_blocks,
            overlap_ratio=overlap_ratio,
            num_hab_blocks=num_hab_blocks,
            use_arc_embed=use_arc_embed,
            use_swmca=use_swmca,
        )
        self.decoder = Decoder(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
        )

    def _encode(self, x, y, arcs):
        return self.encoder(x, y, arcs)  # pyrefly:ignore

    def _decode(self, x, xrtheta):
        return self.decoder(x, xrtheta)  # pyrefly:ignore

    def forward(
        self,
        left_slice,
        right_slice,
        arcs,
        xrtheta,
        xrtheta_left=None,
        xrtheta_right=None,
    ):
        """
        Args:
            left_slice: Bx1xHxW
            right_slice: Bx1xHxW
            arcs: BxH
            xrtheta: BxNx3
            xrtheta_left: BxNx3
            xrtheta_right: BxNx3
        Returns:
            pred: BxNx1
            Optionally:
                left_pred: BxNx1
                right_pred: BxNx1
        """
        latent_space = self._encode(left_slice, right_slice, arcs)

        pred = self._decode(latent_space, xrtheta)

        if xrtheta_left is not None and xrtheta_right is not None:
            left_pred = self._decode(latent_space, xrtheta_left)
            right_pred = self._decode(latent_space, xrtheta_right)
            return pred, left_pred, right_pred

        return pred
