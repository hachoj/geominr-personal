import torch
from torch import Tensor
import torch.nn as nn

from .sw_mca import SWMCA
from .sw_msa import SWMSA
from .rhag import RHAG


class HAT(nn.Module):
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

        self.use_arc_embed = use_arc_embed
        self.use_swmca = use_swmca

        self.in_proj = nn.Conv2d(in_dim, embed_dim, kernel_size=3, padding=1)

        global_index = 0

        rhag_blocks = []
        for _ in range(num_rhag_blocks):
            rhag_blocks.append(
                RHAG(
                    embed_dim=embed_dim,
                    init_cab_weight=init_cab_weight,
                    cab_channel_reduction=cab_channel_reduction,
                    squeeze_factor=squeeze_factor,
                    H=H,
                    W=W,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    overlap_ratio=overlap_ratio,
                    num_hab_blocks=num_hab_blocks,
                    global_index=global_index,
                )
            )
            global_index += num_hab_blocks

        # Cross-stream mixing between the two slice paths. SW-MCA exchanges
        # information via shifted-window cross-attention; the ablation
        # (use_swmca=False) swaps in a per-stream SW-MSA self-attention block
        # to keep the parameter count comparable.
        cross_cls = SWMCA if use_swmca else SWMSA
        cross_blocks = []
        for i in range(num_rhag_blocks):
            shift = 0 if i % 2 == 0 else window_size // 2
            cross_blocks.append(
                cross_cls(
                    H=H,
                    W=W,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                )
            )

        self.rhag_blocks = nn.ModuleList(rhag_blocks)
        self.swmca_blocks = nn.ModuleList(cross_blocks)
        self.dual_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        if self.use_arc_embed:
            self.arc_embed = nn.Conv1d(1, embed_dim, kernel_size=1)

        merged_rhag_blocks = []
        for _ in range(num_rhag_blocks):
            merged_rhag_blocks.append(
                RHAG(
                    embed_dim=2 * embed_dim,
                    init_cab_weight=init_cab_weight,
                    cab_channel_reduction=cab_channel_reduction,
                    squeeze_factor=squeeze_factor,
                    H=H,
                    W=W,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    overlap_ratio=overlap_ratio,
                    num_hab_blocks=num_hab_blocks,
                    global_index=global_index,
                )
            )
            global_index += num_hab_blocks
        self.merged_rhag_blocks = nn.ModuleList(merged_rhag_blocks)

        self.out_proj = nn.Conv2d(2 * embed_dim, K, kernel_size=3, padding=1)

    def forward(self, x, y, arcs):
        """
        Args:
            x: [B,C,H,W]
            y: [B,C,H,W]
            arcs: [B,H]
        Returns:
            out: [B,K,H,W]
        """
        W = x.shape[-1]

        x = self.in_proj(x)
        y = self.in_proj(y)
        resid_x = x
        resid_y = y
        if self.use_arc_embed:
            arc_embed = self.arc_embed(arcs.unsqueeze(1))  # [B,C,H]
            arc_embed = arc_embed.unsqueeze(-1).expand(
                -1, -1, -1, W
            )  # [B,C,H,W]
            x = x + arc_embed
            y = y + arc_embed

        for rhag, cross in zip(self.rhag_blocks, self.swmca_blocks):
            x = rhag(x)
            y = rhag(y)
            if self.use_swmca:
                x, y = cross(x, y)
            else:
                x = cross(x)
                y = cross(y)

        x = x + resid_x
        y = y + resid_y
        x = self.dual_proj(x)
        y = self.dual_proj(y)

        merged = torch.cat((x, y), dim=1)  # [B,2*C,H,W]

        merged_resid = merged

        for rhag in self.merged_rhag_blocks:
            merged = rhag(merged)

        out = merged_resid + merged

        out = self.out_proj(out)

        return out
