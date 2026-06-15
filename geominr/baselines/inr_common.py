from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch


UF_ID_PATTERN = re.compile(r"(UF\d{3})")


def load_test_ids(path: str | Path) -> list[str]:
    ids: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value:
                ids.append(value)
    return ids


def extract_patient_id(path_or_name: str | Path) -> str | None:
    match = UF_ID_PATTERN.search(str(path_or_name))
    if match is None:
        return None
    return match.group(1)


def resolve_patient_dir(dicom_root: str | Path, patient_id: str) -> Path:
    root = Path(dicom_root)
    for label in ("positive", "negative"):
        candidate = root / label / patient_id
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve patient {patient_id} under {root}/positive or {root}/negative."
    )


def load_patient_volume_from_dicom(patient_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    # Local import keeps this helper lightweight in scripts that do not touch DICOM.
    from geominr.models.utils.reconstruct import extract_slices

    extracted = extract_slices(str(patient_dir))
    items = sorted(extracted.items(), key=lambda kv: kv[1]["angle"])
    if not items:
        raise ValueError(f"No slices found in {patient_dir}")

    slices = np.stack([v["slice_volume"] for _, v in items], axis=0).astype(np.uint8)
    angles_deg = np.asarray([v["angle"] for _, v in items], dtype=np.float32)
    angles_rad = np.deg2rad(angles_deg).astype(np.float32)
    return slices, angles_rad


def normalize_global_theta(
    angles_rad: torch.Tensor, angle_min_rad: float, angle_max_rad: float
) -> torch.Tensor:
    denom = max(float(angle_max_rad - angle_min_rad), 1e-8)
    rel = (angles_rad - angle_min_rad) / denom
    return rel.clamp(0.0, 1.0) - 0.5


def pair_coords_to_global_coords(
    xrtheta_pair: torch.Tensor,
    conditioning_angles_rad: torch.Tensor,
    angle_min_rad: float,
    angle_max_rad: float,
    *,
    crop_row_offset: int,
    crop_col_offset: int,
    patch_height: int,
    patch_width: int,
    full_height: int,
    full_width: int,
) -> torch.Tensor:
    if xrtheta_pair.ndim != 2 or xrtheta_pair.shape[-1] != 3:
        raise ValueError(
            f"Expected xrtheta_pair shape [N,3], got {tuple(xrtheta_pair.shape)}."
        )
    if conditioning_angles_rad.numel() != 2:
        raise ValueError(
            f"Expected conditioning angles with 2 values, got shape {tuple(conditioning_angles_rad.shape)}."
        )

    mins = conditioning_angles_rad.min()
    maxs = conditioning_angles_rad.max()
    rel_theta_pair = (xrtheta_pair[:, 2] + 0.5).clamp(0.0, 1.0)
    abs_theta = mins + rel_theta_pair * (maxs - mins)
    theta_global = normalize_global_theta(abs_theta, angle_min_rad, angle_max_rad)

    x_patch = xrtheta_pair[:, 0]
    r_patch = xrtheta_pair[:, 1]

    col_local = ((x_patch + 1.0) * 0.5 * float(patch_width)) - 0.5
    row_local = ((r_patch + 1.0) * 0.5 * float(patch_height)) - 0.5

    col_global = col_local + float(crop_col_offset)
    row_global = row_local + float(crop_row_offset)

    x_global = ((col_global + 0.5) / float(full_width)) * 2.0 - 1.0
    r_global = ((row_global + 0.5) / float(full_height)) * 2.0 - 1.0

    out = xrtheta_pair.clone()
    out[:, 0] = x_global.clamp(-1.0, 1.0)
    out[:, 1] = r_global.clamp(-1.0, 1.0)
    out[:, 2] = theta_global
    return out


def make_pixel_luts(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(height, dtype=torch.float32)
    cols = torch.arange(width, dtype=torch.float32)
    r_lut = (rows + 0.5) / float(height) * 2.0 - 1.0
    x_lut = (cols + 0.5) / float(width) * 2.0 - 1.0
    return r_lut, x_lut
