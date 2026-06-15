import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math

from .hat_utils import create_local_window, reverse_local_window


class SWMSA(nn.Module):
    def __init__(
        self,
        H: int,
        W: int,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.kqv = nn.Linear(embed_dim, 3 * embed_dim)
        self.num_heads = num_heads

        self.shift = (-shift_size, -shift_size)
        self.reverse_shift = (shift_size, shift_size)

        assert (
            embed_dim % num_heads == 0
        ), "num heads must evenly divide the embedding dim"

        self.attn_bias = nn.Embedding((2 * window_size - 1) ** 2, num_heads)
        self.register_buffer(
            "rel_pos", self.create_rel_pos(window_size), persistent=True
        )

        region_ids = self.generate_region_id_map(
            H, W, self.window_size, self.reverse_shift[0]
        )  # [H,W]
        region_ids = region_ids.unsqueeze(0).unsqueeze(-1)  # [1,H,W,1]
        region_ids = create_local_window(region_ids, self.window_size).squeeze(
            -1
        )  # [1,num_windows,M*M]

        attn_mask = self.generate_attn_mask(region_ids)  # [1,num_windows,M*M,M*M]
        attn_mask = attn_mask.reshape(
            -1, window_size**2, window_size**2
        )  # [num_windows,M*M,M*M]
        attn_mask = attn_mask.unsqueeze(1).expand(
            -1, self.num_heads, window_size**2, window_size**2
        )  # [num_windows,NH,M*M,M*M]

        self.mix = nn.Linear(embed_dim, embed_dim)

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B,C,H,W]
        Retruns:
            out: [B,C,H,W]
        """
        B, C, H, W = x.shape
        M = self.window_size

        assert H % M == 0, f"Height {H} must be divisible by window_size {M}"
        assert W % M == 0, f"Width {W} must be divisible by window_size {M}"

        x = x.permute(0, 2, 3, 1)

        x = torch.roll(x, self.shift, dims=(1, 2))
        x = create_local_window(x, self.window_size)  # [B,num_windows,M*M,C]
        _, num_windows, M2, _ = x.shape
        x = x.view(B * num_windows, M2, C)  # [B*num_windows,M*M,C]
        k, q, v = torch.chunk(self.kqv(x), 3, dim=-1)  # 3 x [B*num_windows,M*M,C]
        k = k.reshape(B * num_windows, M2, self.num_heads, C // self.num_heads).permute(
            0, 2, 1, 3
        )  # [B*num_windows,NH,M*M,D]
        q = q.reshape(B * num_windows, M2, self.num_heads, C // self.num_heads).permute(
            0, 2, 1, 3
        )  # [B*num_windows,NH,M*M,D]
        v = v.reshape(B * num_windows, M2, self.num_heads, C // self.num_heads).permute(
            0, 2, 1, 3
        )  # [B*num_windows,NH,M*M,D]

        b = self.attn_bias(self.rel_pos).reshape(M2, M2, self.num_heads)  # [L,L,NH]
        b = b.unsqueeze(0).permute(0, 3, 1, 2)  # [1,NH,L,L]

        attn_mask = self.attn_mask.unsqueeze(0).expand(  # pyright:ignore
            B, -1, -1, -1, -1
        )
        attn_mask = attn_mask.reshape(-1, self.num_heads, M2, M2)

        x = self.scaled_dot_product_attention(
            q, k, v, b, attn_mask
        )  # [B*num_windows,M*M,C]
        x = self.mix(x)
        x = x.reshape(B, num_windows, M2, C)  # [B,num_windows,M*M,C]
        x = reverse_local_window(x, self.window_size, H, W)  # [B,H,W,C]

        x = torch.roll(x, self.reverse_shift, dims=(1, 2))

        out = x.permute(0, 3, 1, 2)

        return out

    def scaled_dot_product_attention(
        self, q: Tensor, k: Tensor, v: Tensor, b: Tensor, attn_mask: Tensor
    ) -> Tensor:
        """
        Args:
            q: [B*Nw,NH,L,D]
            k: [B*Nw,NH,L,D]
            v: [B*Nw,NH,L,D]
            b: [1,NH,L,L]
            attn_mask: [B*Nw,NH,L,L]
        Returns:
            out: [B*Nw,L,C]
        """
        N, H, L, D = q.shape
        scale_factor = 1 / math.sqrt(D)

        similarity_scores = F.softmax(
            q @ k.transpose(-2, -1) * scale_factor + b + attn_mask, dim=3
        )  # [B*Nw,NH,L,L]
        out = similarity_scores @ v

        out = out.permute(0, 2, 1, 3).reshape(N, L, -1)

        return out

    def create_rel_pos(self, M: int) -> Tensor:
        """
        Args:
            M: int, window size
        Retruns:
            relative pos: Tensor, [L*L]
        """
        # coords_h, coords_w: [M]
        coords_h = torch.arange(M)
        coords_w = torch.arange(M)

        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )  # [2,M,M]

        coords_flat = coords.reshape(2, -1)  # [2,L]

        rel_diff_matrix = coords_flat.unsqueeze(2) - coords_flat.unsqueeze(1)  # [2,L,L]
        rel_diff_matrix = rel_diff_matrix.permute(1, 2, 0)  # [L,L,2]
        rel_diff_matrix = rel_diff_matrix + (M - 1)  # now indexed [0, 2M-2]
        rel_diff_matrix = (
            rel_diff_matrix[:, :, 0] * (2 * M - 1) + rel_diff_matrix[:, :, 1]
        )  # [L,L]
        return rel_diff_matrix.reshape(-1).to(dtype=torch.long)

    def generate_region_id_map(self, H: int, W: int, window_size: int, shift_size: int):
        """
        Args:
            H: height
            W: width
            window_size: window size
            shift_size: shift size
        Returns:
            region_id_map: [H,W]
        """
        id_map = torch.zeros((H, W), dtype=torch.long)
        h_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        w_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        current_id = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                id_map[h_slice, w_slice] = current_id
                current_id += 1
        return id_map

    def generate_attn_mask(self, region_id_maps: Tensor) -> Tensor:
        """
        Args:
            region_id_maps: [B,num_windows,M*M]
            M:
        Retruns:
        """
        diffs = region_id_maps.unsqueeze(-1) - region_id_maps.unsqueeze(
            -2
        )  # [B,num_windows,M*M,M*M]
        attn_mask = torch.zeros_like(diffs, dtype=torch.float32)
        attn_mask = attn_mask.masked_fill(diffs != 0, float("-inf"))
        return attn_mask
