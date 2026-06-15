import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OCAB(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        window_size: int,
        overlap_ratio: float,
        num_heads: int,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            OCA(embed_dim, window_size, overlap_ratio, num_heads),
        )

        self.block_2 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B,C,H,W]
        Returns:
            x: [B,C,H,W]
        """
        x = x.permute(0, 2, 3, 1)  # [B,H,W,C]

        x = x + self.block_1(x)
        x = x + self.block_2(x)

        x = x.permute(0, 3, 1, 2)  # [B,C,H,W]
        return x


class OCA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        window_size: int,
        overlap_ratio: float,
        num_heads: int,
    ) -> None:
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "num heads must evenly divide the embedding dim"

        self.embed_dim = embed_dim
        self.window_size = window_size
        self.overlap = int(round(window_size * overlap_ratio))
        self.Mo = self.window_size + 2 * self.overlap
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.Mo - 1) * (2 * self.Mo - 1), num_heads)
        )
        self.register_buffer(
            "relative_position_index",
            self.create_asymmetric_rel_pos(self.window_size, self.Mo, self.overlap),
            persistent=False,
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B,H,W,C]
        Returns:
            x: [B,H,W,C]
        """
        B, H, W, C = x.shape
        M = self.window_size
        Mo = self.Mo
        ov = self.overlap

        x = x.permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]

        # Padding
        pad_h = (M - H % M) % M
        pad_w = (M - W % M) % M
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        _, _, Hp, Wp = x.shape

        nW = (Hp // M) * (Wp // M)
        q_tokens = x.view(B, C, Hp // M, M, Wp // M, M)
        q_tokens = q_tokens.permute(0, 2, 4, 3, 5, 1).contiguous()
        q_tokens = q_tokens.view(-1, M * M, C)  # [B*Nw,Lq,C]

        BnW = q_tokens.shape[0]
        Lq = M * M

        kv_cols = F.unfold(x, kernel_size=Mo, stride=M, padding=ov)  # [B,C*Lk,Nw]
        Lk = Mo * Mo

        kv_cols = kv_cols.view(B, C * Lk, nW).transpose(1, 2).contiguous()
        kv_cols = kv_cols.view(BnW, Lk, C)  # [B*Nw,Lk,C]

        q = self.q_proj(q_tokens)  # [B*Nw,Lq,C]
        q = q.reshape(BnW, Lq, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # [B*Nw,NH,Lq,D]

        kv = self.kv_proj(kv_cols)  # [B*Nw,Lk,2*C]
        kv = kv.reshape(BnW, Lk, 2, self.num_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        )
        k, v = kv[0], kv[1]  # [B*Nw,NH,Lk,D]

        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)  # pyrefly:ignore
        ].view(
            Lq, Lk, self.num_heads
        )  # [Lq,Lk,NH]
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # [1,NH,Lq,Lk]

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)

        y = y.transpose(1, 2).contiguous().view(BnW, Lq, C)
        y = self.proj(y)

        y = y.view(B, Hp // M, Wp // M, M, M, C)
        y = y.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B,C,Hp,Wp]
        y_bchw = y.view(B, C, Hp, Wp)

        if pad_h > 0 or pad_w > 0:
            y_bchw = y_bchw[:, :, :H, :W]

        return y_bchw.permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]

    def create_asymmetric_rel_pos(self, M: int, Mo: int, ov: int) -> Tensor:
        """
        Args:
            M: Query window size
            Mo: Key window size
            ov: Overlap
        Returns:
            rel_index: [Lq,Lk]
        """
        qy, qx = torch.arange(ov, ov + M), torch.arange(ov, ov + M)
        ky, kx = torch.arange(0, Mo), torch.arange(0, Mo)

        qyy, qxx = torch.meshgrid(qy, qx, indexing="ij")
        kyy, kxx = torch.meshgrid(ky, kx, indexing="ij")

        q_flat = torch.stack([qyy, qxx], dim=0).flatten(1)  # [2,Lq]
        k_flat = torch.stack([kyy, kxx], dim=0).flatten(1)  # [2,Lk]

        rel = q_flat[:, :, None] - k_flat[:, None, :]  # [2,Lq,Lk]
        rel = rel.permute(1, 2, 0).contiguous()  # [Lq,Lk,2]

        rel[..., 0] += Mo - 1
        rel[..., 1] += Mo - 1
        rel_index = rel[..., 0] * (2 * Mo - 1) + rel[..., 1]  # [Lq,Lk]

        return rel_index.to(dtype=torch.long)
