from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path
from typing import Any

import lpips
import numpy as np
import torch
from tqdm import tqdm

from geominr.baselines.implicitvol.model import ImplicitVolSiren
from geominr.baselines.inr_common import extract_patient_id, load_test_ids, pair_coords_to_global_coords
from geominr.baselines.ultranerf.model import UltraNerfIntensityMLP
from geominr.baselines.ultranerf.render import gaussian_kernel, render_patch_intensity
from data.data import _build_item_index, _rand_crop2d_content_aware
from geominr.models.utils.metrics import EPI, PSNR, SSIM_slicewise, SSI
from geominr.models.utils.utils import create_xrtheta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fitted INR baselines by querying at the same coordinates as eval.py."
        )
    )
    parser.add_argument("--test-ids", type=str, default="data/test_ids.txt")
    parser.add_argument("--test-data-dir", type=str, default="data/dataset/test")
    parser.add_argument("--fits-root", type=str, default="output/baselines/fits")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["implicitvol", "ultra_nerf"],
        choices=["implicitvol", "ultra_nerf"],
    )
    parser.add_argument(
        "--patch-size",
        nargs=2,
        type=int,
        default=[64, 64],
        metavar=("H", "W"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--output-dir", type=str, default="output/baselines/eval")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If > 0, evaluate only this many test windows.",
    )
    parser.add_argument(
        "--index-workers",
        type=int,
        default=8,
        help="Workers for indexing .npz paths/angles.",
    )
    parser.add_argument("--lpips-net", type=str, default="alex")
    return parser.parse_args()


def compute_stats(values: list[torch.Tensor]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    all_values = torch.cat(values).float()
    return float(all_values.mean().item()), float(all_values.std().item())


def summarize_storage(
    *,
    storage: dict[str, dict[str, list[torch.Tensor]]],
    num_samples: int,
) -> dict[str, Any]:
    metric_names = ["psnr", "ssim", "epi", "ssi", "lpips"]
    results: dict[str, Any] = {"num_samples": int(num_samples)}

    for metric in metric_names:
        key = metric.upper()
        mean, std = compute_stats(storage[metric]["all"])
        results[f"avg_{key}"] = mean
        results[f"std_{key}"] = std

        mean_q, std_q = compute_stats(storage[metric]["query"])
        results[f"avg_{key}_query"] = mean_q
        results[f"std_{key}_query"] = std_q

        mean_l, std_l = compute_stats(storage[metric]["linear"])
        results[f"avg_{key}_linear"] = mean_l
        results[f"std_{key}_linear"] = std_l
    return results


def render_results_text(method: str, results: dict[str, Any]) -> str:
    metric_groups = [("Total", ""), ("Query Only", "_query"), ("Linear", "_linear")]
    metrics_list = ["PSNR", "SSIM", "EPI", "SSI", "LPIPS"]

    lines: list[str] = []
    lines.append("Eval completed")
    lines.append(f"Method: {method}")
    lines.append(f"Samples: {results['num_samples']}")
    lines.append("-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#")
    for group_name, suffix in metric_groups:
        if group_name != "Total":
            lines.append(f" --------- {group_name} ----------")
            lines.append("-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#")
        for metric in metrics_list:
            key_mean = f"avg_{metric}{suffix}"
            key_std = f"std_{metric}{suffix}"
            lines.append(
                f"Average {metric}{' ('+group_name+')' if group_name == 'Linear' else ''}: "
                f"{results[key_mean]:.5f} +- {results[key_std]:.5f}"
            )
        lines.append("-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#")
    return "\n".join(lines)


def load_fitted_model(
    *,
    method: str,
    patient_id: str,
    fits_root: Path,
    device: torch.device,
    cache: dict[str, tuple[torch.nn.Module, dict[str, Any]]],
) -> tuple[torch.nn.Module, dict[str, Any]]:
    cache_key = f"{method}:{patient_id}"
    if cache_key in cache:
        return cache[cache_key]

    ckpt_path = fits_root / method / f"{patient_id}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing fitted checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    kwargs = ckpt["model_kwargs"]

    if method == "implicitvol":
        model = ImplicitVolSiren(**kwargs)
    elif method == "ultra_nerf":
        model = UltraNerfIntensityMLP(**kwargs)
    else:
        raise ValueError(f"Unsupported method: {method}")

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    meta = {
        "angle_min_rad": float(ckpt["angle_min_rad"]),
        "angle_max_rad": float(ckpt["angle_max_rad"]),
        "height": int(ckpt["height"]),
        "width": int(ckpt["width"]),
        "fit_stats": ckpt.get("fit_stats", {}),
    }
    cache[cache_key] = (model, meta)
    return model, meta


def predict_slice(
    *,
    method: str,
    model: torch.nn.Module,
    meta: dict[str, Any],
    xrtheta_pair: torch.Tensor,
    conditioning_angles_rad: torch.Tensor,
    patch_h: int,
    patch_w: int,
    crop_row_offset: int,
    crop_col_offset: int,
    ultra_kernel_cache: dict[tuple[torch.device, int], torch.Tensor],
) -> torch.Tensor:
    coords_pair = xrtheta_pair[0]
    coords_global = pair_coords_to_global_coords(
        coords_pair,
        conditioning_angles_rad[0],
        meta["angle_min_rad"],
        meta["angle_max_rad"],
        crop_row_offset=crop_row_offset,
        crop_col_offset=crop_col_offset,
        patch_height=patch_h,
        patch_width=patch_w,
        full_height=int(meta["height"]),
        full_width=int(meta["width"]),
    )

    if (
        method == "ultra_nerf"
        and isinstance(model, UltraNerfIntensityMLP)
        and getattr(model, "output_mode", "intensity") == "raw"
    ):
        coords_patch = coords_global.view(1, patch_h, patch_w, 3)
        fit_stats = meta.get("fit_stats", {})
        kernel_size = int(fit_stats.get("psf_kernel_size", 3))
        cache_key = (coords_patch.device, kernel_size)
        if cache_key not in ultra_kernel_cache:
            ultra_kernel_cache[cache_key] = gaussian_kernel(
                size=kernel_size,
                device=coords_patch.device,
                dtype=coords_patch.dtype,
            )
        pred01 = render_patch_intensity(
            model=model,
            coords_patch=coords_patch,
            psf_kernel=ultra_kernel_cache[cache_key],
            stochastic=False,
        )
        # Keep eval metrics directly comparable with legacy [-1, 1] code path.
        return pred01 * 2.0 - 1.0

    pred = model(coords_global)
    if pred.ndim == 2 and pred.shape[1] > 1:
        pred = UltraNerfIntensityMLP.raw_to_point_intensity(pred).unsqueeze(-1)
    pred = pred.squeeze(-1).view(1, 1, patch_h, patch_w)

    if (
        method == "ultra_nerf"
        and isinstance(model, UltraNerfIntensityMLP)
        and getattr(model, "intensity_range", "minus_one_one") == "zero_one"
    ):
        pred = pred * 2.0 - 1.0
    return pred


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    test_ids = set(load_test_ids(args.test_ids))
    if not test_ids:
        raise ValueError(f"No IDs found in {args.test_ids}")

    items = _build_item_index(
        args.test_data_dir,
        desc="indexing eval set",
        max_workers=args.index_workers,
    )

    selected: list[tuple[int, dict[str, Any]]] = []
    for idx, rec in enumerate(items):
        patient_id = extract_patient_id(rec["path"])
        if patient_id is not None and patient_id in test_ids:
            selected.append((idx, rec))

    if args.max_samples > 0:
        selected = selected[: args.max_samples]
    if not selected:
        raise ValueError("No matching test samples after filtering by test_ids.")

    patch_h, patch_w = int(args.patch_size[0]), int(args.patch_size[1])
    fits_root = Path(args.fits_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lpips_loss = lpips.LPIPS(net=args.lpips_net).to(device)
    lpips_loss.eval()
    for p in lpips_loss.parameters():
        p.requires_grad = False

    metric_fns = {
        "psnr": lambda pred, tgt: PSNR(pred, tgt, reduction="none"),
        "ssim": lambda pred, tgt: SSIM_slicewise(pred, tgt, reduction="none"),
        "epi": lambda pred, tgt: EPI(pred, tgt, reduction="none"),
        "ssi": lambda pred, tgt: SSI(pred, tgt, reduction="none"),
        "lpips": lambda pred, tgt: lpips_loss(
            pred.repeat(1, 3, 1, 1), tgt.repeat(1, 3, 1, 1)
        ),
    }
    metric_names = ["psnr", "ssim", "epi", "ssi", "lpips"]

    storage: dict[str, dict[str, dict[str, list[torch.Tensor]]]] = {}
    for method in args.methods:
        storage[method] = {m: {"all": [], "query": [], "linear": []} for m in metric_names}

    model_cache: dict[str, tuple[torch.nn.Module, dict[str, Any]]] = {}
    ultra_kernel_cache: dict[tuple[torch.device, int], torch.Tensor] = {}
    num_samples = 0

    with torch.no_grad():
        for global_idx, rec in tqdm(selected, desc="Evaluating INR fits", unit="sample"):
            sample_path = rec["path"]
            patient_id = extract_patient_id(sample_path)
            if patient_id is None:
                continue

            with np.load(sample_path, allow_pickle=False) as f:
                slices = torch.from_numpy(f["slices"]).float() / 127.5 - 1.0
                angles = torch.from_numpy(np.deg2rad(f["angles"]).astype(np.float32))

            h0, w0 = _rand_crop2d_content_aware(
                slices, patch_h, patch_w, mode="test", seed=args.seed + global_idx
            )
            slices = slices[:, h0 : h0 + patch_h, w0 : w0 + patch_w]

            target_left = slices[0].view(1, 1, patch_h, patch_w).to(device)
            target_query = slices[1].view(1, 1, patch_h, patch_w).to(device)
            target_right = slices[2].view(1, 1, patch_h, patch_w).to(device)

            angles = angles.to(device)
            conditioning_angles = angles[[0, 2]].unsqueeze(0)
            xyz = create_xrtheta(
                1, patch_h, patch_w, angles[1].unsqueeze(0), conditioning_angles, device
            )
            xyz_left = create_xrtheta(
                1, patch_h, patch_w, angles[0].unsqueeze(0), conditioning_angles, device
            )
            xyz_right = create_xrtheta(
                1, patch_h, patch_w, angles[2].unsqueeze(0), conditioning_angles, device
            )

            t = (xyz[0, :, 2] + 0.5).clamp(0.0, 1.0).view(1, 1, patch_h, patch_w)
            out_linear = torch.lerp(target_left, target_right, t)

            for method in args.methods:
                model, meta = load_fitted_model(
                    method=method,
                    patient_id=patient_id,
                    fits_root=fits_root,
                    device=device,
                    cache=model_cache,
                )

                out = predict_slice(
                    method=method,
                    model=model,
                    meta=meta,
                    xrtheta_pair=xyz,
                    conditioning_angles_rad=conditioning_angles,
                    patch_h=patch_h,
                    patch_w=patch_w,
                    crop_row_offset=h0,
                    crop_col_offset=w0,
                    ultra_kernel_cache=ultra_kernel_cache,
                )
                out_left = predict_slice(
                    method=method,
                    model=model,
                    meta=meta,
                    xrtheta_pair=xyz_left,
                    conditioning_angles_rad=conditioning_angles,
                    patch_h=patch_h,
                    patch_w=patch_w,
                    crop_row_offset=h0,
                    crop_col_offset=w0,
                    ultra_kernel_cache=ultra_kernel_cache,
                )
                out_right = predict_slice(
                    method=method,
                    model=model,
                    meta=meta,
                    xrtheta_pair=xyz_right,
                    conditioning_angles_rad=conditioning_angles,
                    patch_h=patch_h,
                    patch_w=patch_w,
                    crop_row_offset=h0,
                    crop_col_offset=w0,
                    ultra_kernel_cache=ultra_kernel_cache,
                )

                for metric in metric_names:
                    fn = metric_fns[metric]
                    val_query = fn(out, target_query).detach().flatten().cpu()
                    val_left = fn(out_left, target_left).detach().flatten().cpu()
                    val_right = fn(out_right, target_right).detach().flatten().cpu()
                    val_linear = fn(out_linear, target_query).detach().flatten().cpu()

                    storage[method][metric]["query"].append(val_query)
                    storage[method][metric]["all"].append(val_left)
                    storage[method][metric]["all"].append(val_right)
                    storage[method][metric]["all"].append(val_query)
                    storage[method][metric]["linear"].append(val_linear)

            num_samples += 1

    all_results: dict[str, Any] = {}
    for method in args.methods:
        method_results = summarize_storage(storage=storage[method], num_samples=num_samples)
        all_results[method] = method_results

        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)

        text = render_results_text(method, method_results)
        print(text, flush=True)

        with open(method_dir / "results.txt", "w", encoding="utf-8") as f:
            f.write(text + "\n")
        with open(method_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(method_results, f, indent=2)

    summary_payload = {
        "test_ids": sorted(test_ids),
        "num_samples": num_samples,
        "methods": args.methods,
        "results": all_results,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"Saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
