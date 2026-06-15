from __future__ import annotations

import math

import torch
import torch.nn as nn


class Sine(nn.Module):
    def __init__(self, w0: float = 30.0) -> None:
        super().__init__()
        self.w0 = float(w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        w0: float,
        is_first: bool = False,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.activation = Sine(w0=w0)
        self.in_dim = int(in_dim)
        self.w0 = float(w0)
        self.is_first = bool(is_first)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / max(1, self.in_dim)
            else:
                bound = math.sqrt(6.0 / max(1, self.in_dim)) / self.w0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


def encode_position(
    coords: torch.Tensor,
    *,
    levels: int = 10,
    include_input: bool = True,
) -> torch.Tensor:
    out = [coords] if include_input else []
    for i in range(levels):
        freq = 2.0**i
        out.append(torch.sin(freq * coords))
        out.append(torch.cos(freq * coords))
    return torch.cat(out, dim=-1)


class ImplicitVolSiren(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 8,
        w0_first: float = 30.0,
        w0_hidden: float = 1.0,
        architecture: str = "legacy",
        num_frequencies: int = 10,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        if architecture not in {"legacy", "official"}:
            raise ValueError("architecture must be one of: legacy, official")
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.architecture = architecture
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)

        if self.architecture == "legacy":
            layers: list[nn.Module] = []
            layers.append(
                SirenLayer(
                    self.input_dim,
                    self.hidden_dim,
                    w0=w0_first,
                    is_first=True,
                )
            )
            for _ in range(self.num_layers - 2):
                layers.append(
                    SirenLayer(
                        self.hidden_dim,
                        self.hidden_dim,
                        w0=w0_hidden,
                        is_first=False,
                    )
                )
            self.features = nn.Sequential(*layers)
            self.head = nn.Linear(self.hidden_dim, 1)
            nn.init.uniform_(self.head.weight, -1e-4, 1e-4)
            nn.init.uniform_(self.head.bias, -1e-4, 1e-4)
            return

        # Official ImplicitVol architecture: PE(3 -> 63), split 4+4 SIREN blocks with skip cat.
        pe_dim = self.input_dim * (2 * self.num_frequencies + (1 if self.include_input else 0))
        n0 = self.num_layers // 2
        n1 = self.num_layers - n0
        if n0 < 1 or n1 < 1:
            raise ValueError("num_layers must provide at least one layer per stage.")

        layers0: list[nn.Module] = [
            SirenLayer(pe_dim, self.hidden_dim, w0=w0_first, is_first=True)
        ]
        for _ in range(n0 - 1):
            layers0.append(
                SirenLayer(self.hidden_dim, self.hidden_dim, w0=w0_hidden, is_first=False)
            )
        self.layers0 = nn.Sequential(*layers0)

        layers1: list[nn.Module] = [
            SirenLayer(pe_dim + self.hidden_dim, self.hidden_dim, w0=w0_hidden, is_first=False)
        ]
        for _ in range(n1 - 1):
            layers1.append(
                SirenLayer(self.hidden_dim, self.hidden_dim, w0=w0_hidden, is_first=False)
            )
        self.layers1 = nn.Sequential(*layers1)

        self.fc_feature = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.img_layers = SirenLayer(
            self.hidden_dim, self.hidden_dim // 2, w0=w0_hidden, is_first=False
        )
        self.fc_img = nn.Linear(self.hidden_dim // 2, 1)
        nn.init.constant_(self.fc_img.bias, 0.02)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if self.architecture == "legacy":
            # Legacy behavior kept for backward-compatibility with older checkpoints.
            x = self.features(coords)
            return torch.tanh(self.head(x))

        pos_enc = encode_position(
            coords,
            levels=self.num_frequencies,
            include_input=self.include_input,
        )
        x = self.layers0(pos_enc)
        x = torch.cat([x, pos_enc], dim=-1)
        x = self.layers1(x)
        feat = self.fc_feature(x)
        x = self.img_layers(feat)
        return self.fc_img(x)
