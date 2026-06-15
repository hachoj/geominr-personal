from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from geominr.baselines.implicitvol.model import ImplicitVolSiren
from geominr.baselines.ultranerf.model import UltraNerfIntensityMLP
from geominr.baselines.ultranerf.render import gaussian_kernel, render_patch_intensity
from geominr.models.utils.reconstruct import extract_slices, save_volume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct a 3D volume from a fitted INR checkpoint "
            "using configurable AP voxel count and depth."
        )
    )
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to patient DICOM directory (e.g., .../positive/UF022).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fitted INR checkpoint (e.g., output/baselines/fits/implicitvol/UF022.pt).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="output/baselines/inference",
        help="Output directory root for saved NIfTI volumes.",
    )
    parser.add_argument(
        "--volume-depth",
        type=int,
        default=48,
        help="Number of SI voxels in reconstructed volume.",
    )
    parser.add_argument(
        "--ap-voxel-count",
        type=int,
        default=1536,
        help="Number of AP voxels (LR is 2x this value).",
    )
    parser.add_argument(
        "--probe-radius",
        type=float,
        default=12.5,
        help="Probe radius in mm used in geometry reconstruction.",
    )
    parser.add_argument(
        "--chunk-points",
        type=int,
        default=262144,
        help="Number of INR query points per forward chunk.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Torch device.",
    )
    return parser.parse_args()


def _resolve_model_label(model_path: str) -> str:
    parent = Path(model_path).parent.name
    stem = Path(model_path).stem
    if parent:
        return parent.replace(" ", "_")
    return stem.replace(" ", "_")


def _resolve_loss_label(method: str, loss_mode: str) -> str:
    if not loss_mode:
        return ""
    mode = str(loss_mode).strip().lower()
    if method == "ultra_nerf":
        if mode == "ssim_mse":
            return "mixed"
        if mode == "mse":
            return "mse"
    if method == "implicitvol":
        if mode == "ssim":
            return "ssim"
        if mode == "mse":
            return "mse"
    return mode.replace(" ", "_")


def load_inr_model(
    model_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, float | int | str]]:
    ckpt = torch.load(model_path, map_location="cpu")
    method = ckpt.get("method")
    if method is None:
        model_path_l = model_path.lower()
        if "implicitvol" in model_path_l:
            method = "implicitvol"
        elif "ultra_nerf" in model_path_l or "ultra-nerf" in model_path_l:
            method = "ultra_nerf"
        else:
            raise ValueError(
                "Could not infer checkpoint method. Expected `method` in checkpoint "
                "or a model path containing `implicitvol` / `ultra_nerf`."
            )

    kwargs = ckpt["model_kwargs"]
    fit_stats = ckpt.get("fit_stats", {})
    if method == "implicitvol":
        model = ImplicitVolSiren(**kwargs)
        output_mode = "intensity"
        intensity_range = "minus_one_one"
    elif method == "ultra_nerf":
        model = UltraNerfIntensityMLP(**kwargs)
        output_mode = str(kwargs.get("output_mode", "intensity"))
        if "intensity_range" in kwargs:
            intensity_range = str(kwargs["intensity_range"])
        elif fit_stats.get("normalization") == "zero_one":
            intensity_range = "zero_one"
        else:
            intensity_range = "minus_one_one"
    else:
        raise ValueError(f"Unsupported INR method: {method}")

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    meta: dict[str, float | int | str] = {
        "method": str(method),
        "height": int(ckpt["height"]),
        "width": int(ckpt["width"]),
        "angle_min_rad": float(ckpt["angle_min_rad"]),
        "angle_max_rad": float(ckpt["angle_max_rad"]),
        "output_mode": output_mode,
        "intensity_range": intensity_range,
        "psf_kernel_size": int(fit_stats.get("psf_kernel_size", 3)),
        "loss_mode": str(fit_stats.get("loss_mode", "")),
    }
    return model, meta


def reconstruct_volume_inr(
    *,
    model: torch.nn.Module,
    extracted_slices: dict,
    meta: dict[str, float | int | str],
    depth: int,
    ap_voxel_count: int,
    probe_radius: float,
    chunk_points: int,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    if depth <= 0:
        raise ValueError("depth must be a positive integer")
    if ap_voxel_count <= 0:
        raise ValueError("ap_voxel_count must be a positive integer")
    if chunk_points <= 0:
        raise ValueError("chunk_points must be positive")

    items = sorted(extracted_slices.items(), key=lambda kv: kv[0])
    if not items:
        raise ValueError("No extracted slices found.")

    spacing_xy = float(items[0][1]["spacing"][0])
    H = int(items[0][1]["slice_volume"].shape[0])
    W = int(items[0][1]["slice_volume"].shape[1])

    fit_h = int(meta["height"])
    fit_w = int(meta["width"])
    if H != fit_h or W != fit_w:
        raise ValueError(
            f"Checkpoint expects HxW={fit_h}x{fit_w} but DICOM volume is {H}x{W}."
        )

    angle_min_rad = float(meta["angle_min_rad"])
    angle_max_rad = float(meta["angle_max_rad"])
    angle_den = max(angle_max_rad - angle_min_rad, 1e-8)

    height_mm = spacing_xy * H
    width_mm = spacing_xy * W

    ap_mm = height_mm + probe_radius
    lr_mm = 2.0 * ap_mm
    si_mm = width_mm

    ap_num = int(ap_voxel_count)
    lr_num = int(2 * ap_voxel_count)
    si_num = int(depth)

    spacing_ap = ap_mm / float(ap_num)
    spacing_lr = lr_mm / float(lr_num)
    spacing_si = si_mm / float(si_num)

    height_plus_r = float(height_mm + probe_radius)
    inv_spacing_xy = float(1.0 / spacing_xy)

    xs = torch.arange(lr_num, device=device, dtype=torch.float32) * spacing_lr
    ys = torch.arange(ap_num, device=device, dtype=torch.float32) * spacing_ap
    dist_mid = xs[:, None] - height_plus_r
    dist_post = height_plus_r - ys[None, :]

    theta_rad_xy = -torch.atan(dist_mid / torch.clamp(dist_post, min=1e-8))
    d_xy = torch.sqrt(dist_mid**2 + dist_post**2)
    j_float_xy = (height_plus_r - d_xy) * inv_spacing_xy

    r_norm_xy = ((j_float_xy + 0.5) / float(H)) * 2.0 - 1.0
    theta_norm_xy = ((theta_rad_xy - angle_min_rad) / angle_den).clamp(0.0, 1.0) - 0.5

    valid_xy = (
        (j_float_xy >= 0.0)
        & (j_float_xy <= float(H - 1))
        & (theta_rad_xy >= angle_min_rad)
        & (theta_rad_xy <= angle_max_rad)
    )
    valid_flat = valid_xy.reshape(-1)
    valid_idx = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)

    r_valid = r_norm_xy.reshape(-1)[valid_flat]
    theta_valid = theta_norm_xy.reshape(-1)[valid_flat]
    num_valid = int(valid_idx.numel())
    valid_hw = valid_xy.transpose(0, 1).contiguous()  # [AP, LR]
    r_hw = r_norm_xy.transpose(0, 1).contiguous()  # [AP, LR]
    theta_hw = theta_norm_xy.transpose(0, 1).contiguous()  # [AP, LR]

    volume = np.zeros((lr_num, ap_num, si_num), dtype=np.uint8)
    if num_valid == 0:
        return {
            "volume": volume,
            "spacing_LR": spacing_lr,
            "spacing_AP": spacing_ap,
            "spacing_SI": spacing_si,
        }

    use_faithful_un = (
        str(meta["method"]) == "ultra_nerf"
        and str(meta["output_mode"]) == "raw"
    )
    psf_kernel = None
    if use_faithful_un:
        psf_kernel = gaussian_kernel(
            size=int(meta["psf_kernel_size"]),
            device=device,
            dtype=torch.float32,
        )

    with torch.no_grad():
        for z in tqdm(range(si_num), desc="INR inference (z)", unit="slice"):
            i_float = float(z * spacing_si * inv_spacing_xy)
            if i_float < 0.0 or i_float > float(W - 1):
                continue

            x_norm_value = ((i_float + 0.5) / float(W)) * 2.0 - 1.0

            if use_faithful_un:
                x_hw = torch.full_like(r_hw, x_norm_value)
                coords_patch = torch.stack([x_hw, r_hw, theta_hw], dim=-1).unsqueeze(0)
                pred_hw01 = render_patch_intensity(
                    model=model,
                    coords_patch=coords_patch,
                    psf_kernel=psf_kernel,
                    stochastic=False,
                    query_chunk_points=chunk_points,
                ).squeeze(0).squeeze(0)
                pred_hw01 = pred_hw01 * valid_hw.to(pred_hw01.dtype)
                pred_hw_u8 = (pred_hw01.clamp(0.0, 1.0) * 255.0).round().clamp(0, 255)
                volume[:, :, z] = pred_hw_u8.transpose(0, 1).to(torch.uint8).cpu().numpy()
                continue

            x_valid = torch.full(
                (num_valid,),
                x_norm_value,
                dtype=torch.float32,
                device=device,
            )
            coords = torch.stack([x_valid, r_valid, theta_valid], dim=1)

            preds: list[torch.Tensor] = []
            for start in range(0, num_valid, chunk_points):
                end = min(start + chunk_points, num_valid)
                pred_raw = model(coords[start:end])
                if (
                    str(meta["method"]) == "ultra_nerf"
                    and str(meta["output_mode"]) == "raw"
                    and pred_raw.ndim == 2
                    and pred_raw.shape[1] == 5
                ):
                    pred = UltraNerfIntensityMLP.raw_to_point_intensity(pred_raw)
                elif pred_raw.ndim == 2 and pred_raw.shape[1] == 5:
                    pred = UltraNerfIntensityMLP.raw_to_point_intensity(pred_raw)
                else:
                    pred = pred_raw.squeeze(-1)
                preds.append(pred)

            pred_full = torch.cat(preds, dim=0)
            if str(meta["intensity_range"]) == "zero_one":
                pred_u8 = (pred_full.clamp(0.0, 1.0) * 255.0).round().clamp(0, 255).to(
                    torch.uint8
                )
            else:
                pred_u8 = (
                    ((pred_full.clamp(-1.0, 1.0) + 1.0) * 127.5)
                    .round()
                    .clamp(0, 255)
                    .to(torch.uint8)
                )

            flat = torch.zeros(lr_num * ap_num, dtype=torch.uint8, device=device)
            flat[valid_idx] = pred_u8
            volume[:, :, z] = flat.view(lr_num, ap_num).cpu().numpy()

    return {
        "volume": volume,
        "spacing_LR": spacing_lr,
        "spacing_AP": spacing_ap,
        "spacing_SI": spacing_si,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    image_path = args.image_path
    model_path = args.model_path
    save_path = args.save_path

    print(f"Using device: {device}")
    print(f"Loading INR checkpoint: {model_path}")
    model, meta = load_inr_model(model_path, device)
    print(
        "Loaded model "
        f"(method={meta['method']}, H={meta['height']}, W={meta['width']}, "
        f"theta=[{meta['angle_min_rad']:.5f}, {meta['angle_max_rad']:.5f}] rad)"
    )

    print(f"Extracting DICOM slices from: {image_path}")
    extracted_slices = extract_slices(image_path)
    print(f"Extracted {len(extracted_slices)} slices")

    reconstructed = reconstruct_volume_inr(
        model=model,
        extracted_slices=extracted_slices,
        meta=meta,
        depth=args.volume_depth,
        ap_voxel_count=args.ap_voxel_count,
        probe_radius=args.probe_radius,
        chunk_points=args.chunk_points,
        device=device,
    )

    patient_name = os.path.basename(image_path.rstrip("/"))
    model_label = _resolve_model_label(model_path)
    loss_label = _resolve_loss_label(str(meta["method"]), str(meta.get("loss_mode", "")))
    if loss_label:
        model_label = f"{model_label}_{loss_label}"
    out_prefix = os.path.join(save_path, f"{patient_name}_{model_label}_inr")
    save_volume(extracted_slices, reconstructed, out_prefix)
    print(f"Saved INR reconstruction to: {out_prefix}.nii.gz")


if __name__ == "__main__":
    main()
