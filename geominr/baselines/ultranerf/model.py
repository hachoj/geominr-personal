from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 3,
        num_frequencies: int = 10,
        include_input: bool = True,
        use_pi: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        self.use_pi = bool(use_pi)
        freq_bands = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

        base = 2 * num_frequencies
        if include_input:
            base += 1
        self.out_dim = self.input_dim * base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = x.unsqueeze(-1) * self.freq_bands
        phase = math.pi * xb if self.use_pi else xb
        sin_feats = torch.sin(phase).reshape(x.shape[0], -1)
        cos_feats = torch.cos(phase).reshape(x.shape[0], -1)
        if self.include_input:
            return torch.cat([x, sin_feats, cos_feats], dim=-1)
        return torch.cat([sin_feats, cos_feats], dim=-1)


class UltraNerfIntensityMLP(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 3,
        num_frequencies: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 8,
        encoder_use_pi: bool = True,
        architecture: Literal["legacy", "official"] = "legacy",
        output_mode: Literal["intensity", "raw"] = "intensity",
        intensity_range: Literal["minus_one_one", "zero_one"] = "minus_one_one",
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        if architecture not in {"legacy", "official"}:
            raise ValueError("architecture must be one of: legacy, official")
        if output_mode not in {"intensity", "raw"}:
            raise ValueError("output_mode must be one of: intensity, raw")
        if intensity_range not in {"minus_one_one", "zero_one"}:
            raise ValueError("intensity_range must be one of: minus_one_one, zero_one")

        self.architecture = architecture
        self.output_mode = output_mode
        self.intensity_range = intensity_range
        self.encoder = FourierEncoder(
            input_dim=input_dim,
            num_frequencies=num_frequencies,
            include_input=True,
            use_pi=encoder_use_pi,
        )
        self.skips = {4}

        layers: list[nn.Module] = []
        for i in range(num_layers):
            if i == 0:
                in_dim = self.encoder.out_dim
            else:
                in_dim = hidden_dim
                if self.architecture == "legacy" and i in self.skips:
                    in_dim += self.encoder.out_dim
                if self.architecture == "official" and (i - 1) in self.skips:
                    in_dim += self.encoder.out_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.raw_head = nn.Linear(hidden_dim, 5)

    @staticmethod
    def raw_to_point_intensity(raw: torch.Tensor) -> torch.Tensor:
        attenuation_coeff = F.softplus(raw[..., 0])
        reflection_coeff = torch.sigmoid(raw[..., 1])
        border_prob = torch.sigmoid(raw[..., 2])
        scatter_density = torch.sigmoid(raw[..., 3])
        scatter_amp = torch.sigmoid(raw[..., 4])

        attenuation = torch.exp(-attenuation_coeff)
        reflection = reflection_coeff * border_prob
        scatter = scatter_density * scatter_amp
        return torch.clamp(attenuation * (reflection + scatter), 0.0, 1.0)

    def forward_raw(self, coords: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(coords)
        x = encoded
        for i, layer in enumerate(self.layers):
            if self.architecture == "legacy" and i in self.skips:
                x = torch.cat([x, encoded], dim=-1)
            if self.architecture == "official" and (i - 1) in self.skips:
                x = torch.cat([x, encoded], dim=-1)
            x = F.relu(layer(x), inplace=False)
        return self.raw_head(x)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        raw = self.forward_raw(coords)
        if self.output_mode == "raw":
            return raw

        intensity01 = self.raw_to_point_intensity(raw)
        if self.intensity_range == "minus_one_one":
            intensity = intensity01 * 2.0 - 1.0
        else:
            intensity = intensity01
        return intensity.unsqueeze(-1)
