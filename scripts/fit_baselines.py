from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from geominr.baselines.implicitvol.fit import fit_patient_implicitvol
from geominr.baselines.inr_common import (
    load_patient_volume_from_dicom,
    load_test_ids,
    resolve_patient_dir,
)
from geominr.baselines.ultranerf.fit import fit_patient_ultra_nerf


B200_DEFAULTS = {
    "implicitvol_steps": 10000,
    "implicitvol_batch_patches": 16,
    "implicitvol_lr": 1e-3,
    "implicitvol_lr_step_interval": 10,
    "implicitvol_lr_gamma": 0.9954,
    "implicitvol_hidden_dim": 384,
    "implicitvol_num_layers": 8,
    "implicitvol_num_frequencies": 10,
    "implicitvol_w0_first": 30.0,
    "implicitvol_w0_hidden": 1.0,
    "implicitvol_patch_size": 256,
    "implicitvol_ssim_window_size": 7,
    "implicitvol_loss_mode": "ssim",
    "ultra_nerf_steps": 10000,
    "ultra_nerf_batch_patches": 16,
    "ultra_nerf_lr": 1e-4,
    "ultra_nerf_lr_decay_steps": 250000,
    "ultra_nerf_lr_decay_rate": 0.1,
    "ultra_nerf_hidden_dim": 384,
    "ultra_nerf_num_layers": 8,
    "ultra_nerf_num_frequencies": 10,
    "ultra_nerf_encoder_use_pi": False,
    "ultra_nerf_patch_size": 256,
    "ultra_nerf_ssim_window_size": 7,
    "ultra_nerf_loss_mode": "ssim_mse",
    "ultra_nerf_ssim_weight": 0.9,
    "ultra_nerf_mse_weight": 0.1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit ImplicitVol-style and Ultra-NeRF-style INRs on full DICOM sweeps."
    )
    parser.add_argument(
        "--test-ids",
        type=str,
        default="data/test_ids.txt",
        help="Path to text file with UF IDs.",
    )
    parser.add_argument(
        "--dicom-root",
        type=str,
        default="data/Original_Data_in_DICOM_Format",
        help="Root folder with positive/negative DICOM subject dirs.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="output/baselines/fits",
        help="Where fitted checkpoints will be written.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["implicitvol", "ultra_nerf"],
        choices=["implicitvol", "ultra_nerf"],
        help="Which INR methods to fit.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Torch device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=0,
        help="If > 0, only fit this many patients from test_ids.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing checkpoints.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=250,
        help="Logging frequency in optimizer steps.",
    )

    # B200-tuned ImplicitVol-inspired defaults
    parser.add_argument(
        "--implicitvol-steps",
        type=int,
        default=B200_DEFAULTS["implicitvol_steps"],
    )
    parser.add_argument(
        "--implicitvol-batch-patches",
        type=int,
        default=B200_DEFAULTS["implicitvol_batch_patches"],
        help="Number of random patches per optimizer step.",
    )
    parser.add_argument(
        "--implicitvol-lr",
        type=float,
        default=B200_DEFAULTS["implicitvol_lr"],
    )
    parser.add_argument(
        "--implicitvol-lr-step-interval",
        type=int,
        default=B200_DEFAULTS["implicitvol_lr_step_interval"],
        help="StepLR interval for ImplicitVol; set <=0 to disable scheduler.",
    )
    parser.add_argument(
        "--implicitvol-lr-gamma",
        type=float,
        default=B200_DEFAULTS["implicitvol_lr_gamma"],
        help="StepLR gamma for ImplicitVol; valid range (0,1).",
    )
    parser.add_argument(
        "--implicitvol-hidden-dim",
        type=int,
        default=B200_DEFAULTS["implicitvol_hidden_dim"],
    )
    parser.add_argument(
        "--implicitvol-num-layers",
        type=int,
        default=B200_DEFAULTS["implicitvol_num_layers"],
    )
    parser.add_argument(
        "--implicitvol-num-frequencies",
        type=int,
        default=B200_DEFAULTS["implicitvol_num_frequencies"],
    )
    parser.add_argument(
        "--implicitvol-w0-first",
        type=float,
        default=B200_DEFAULTS["implicitvol_w0_first"],
    )
    parser.add_argument(
        "--implicitvol-w0-hidden",
        type=float,
        default=B200_DEFAULTS["implicitvol_w0_hidden"],
    )
    parser.add_argument(
        "--implicitvol-patch-size",
        type=int,
        default=B200_DEFAULTS["implicitvol_patch_size"],
    )
    parser.add_argument(
        "--implicitvol-ssim-window-size",
        type=int,
        default=B200_DEFAULTS["implicitvol_ssim_window_size"],
    )
    parser.add_argument(
        "--implicitvol-loss-mode",
        type=str,
        choices=["ssim", "mse"],
        default=B200_DEFAULTS["implicitvol_loss_mode"],
    )

    # B200-tuned Ultra-NeRF-inspired defaults
    parser.add_argument(
        "--ultra-nerf-steps",
        type=int,
        default=B200_DEFAULTS["ultra_nerf_steps"],
    )
    parser.add_argument(
        "--ultra-nerf-batch-patches",
        type=int,
        default=B200_DEFAULTS["ultra_nerf_batch_patches"],
        help="Number of random patches per optimizer step.",
    )
    parser.add_argument(
        "--ultra-nerf-lr",
        type=float,
        default=B200_DEFAULTS["ultra_nerf_lr"],
    )
    parser.add_argument(
        "--ultra-nerf-lr-decay-steps",
        type=int,
        default=B200_DEFAULTS["ultra_nerf_lr_decay_steps"],
        help="Exponential LR decay horizon in optimizer steps for Ultra-NeRF; set <=0 to disable scheduler.",
    )
    parser.add_argument(
        "--ultra-nerf-lr-decay-rate",
        type=float,
        default=B200_DEFAULTS["ultra_nerf_lr_decay_rate"],
        help="Target LR multiplier reached at ultra-nerf-lr-decay-steps.",
    )
    parser.add_argument(
        "--ultra-nerf-hidden-dim",
        type=int,
        default=B200_DEFAULTS["ultra_nerf_hidden_dim"],
    )
    parser.add_argument(
        "--ultra-nerf-num-layers",
        type=int,
        default=B200_DEFAULTS["ultra_nerf_num_layers"],
    )
    parser.add_argument(
        "--ultra-nerf-num-frequencies",
        type=int,
        default=B200_DEFAULTS["ultra_nerf_num_frequencies"],
    )
    parser.add_argument(
        "--ultra-nerf-encoder-use-pi",
        action="store_true",
        default=B200_DEFAULTS["ultra_nerf_encoder_use_pi"],
        help="Use sin/cos(pi*x) encoding. Default False matches Ultra-NeRF repo embedding.",
    )
    parser.add_argument(
        "--ultra-nerf-patch-size",
        type=int,
        default=B200_DEFAULTS["ultra_nerf_patch_size"],
    )
    parser.add_argument(
        "--ultra-nerf-ssim-window-size",
        type=int,
        default=B200_DEFAULTS["ultra_nerf_ssim_window_size"],
    )
    parser.add_argument(
        "--ultra-nerf-loss-mode",
        type=str,
        choices=["ssim_mse", "mse"],
        default=B200_DEFAULTS["ultra_nerf_loss_mode"],
    )
    parser.add_argument(
        "--ultra-nerf-ssim-weight",
        type=float,
        default=B200_DEFAULTS["ultra_nerf_ssim_weight"],
    )
    parser.add_argument(
        "--ultra-nerf-mse-weight",
        type=float,
        default=B200_DEFAULTS["ultra_nerf_mse_weight"],
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    *,
    out_path: Path,
    model: torch.nn.Module,
    method: str,
    patient_id: str,
    patient_dir: Path,
    fit_stats: dict[str, Any],
) -> None:
    state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {
        "method": method,
        "patient_id": patient_id,
        "patient_dir": str(patient_dir),
        "model_kwargs": fit_stats["model_kwargs"],
        "angle_min_rad": fit_stats["angle_min_rad"],
        "angle_max_rad": fit_stats["angle_max_rad"],
        "height": fit_stats["height"],
        "width": fit_stats["width"],
        "state_dict": state_dict_cpu,
        "fit_stats": fit_stats,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def main() -> None:
    run_start = time.perf_counter()
    run_start_utc = datetime.now(timezone.utc).isoformat()

    args = parse_args()
    set_seed(args.seed)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    test_ids = load_test_ids(args.test_ids)
    if args.max_patients > 0:
        test_ids = test_ids[: args.max_patients]

    if not test_ids:
        raise ValueError(f"No IDs found in {args.test_ids}")

    manifest: dict[str, Any] = {
        "test_ids_file": args.test_ids,
        "dicom_root": args.dicom_root,
        "methods": args.methods,
        "device": args.device,
        "seed": args.seed,
        "run_start_utc": run_start_utc,
        "defaults_profile": "b200",
        "fit_hparams": {
            "implicitvol": {
                "steps": args.implicitvol_steps,
                "batch_patches": args.implicitvol_batch_patches,
                "lr": args.implicitvol_lr,
                "lr_step_interval": args.implicitvol_lr_step_interval,
                "lr_gamma": args.implicitvol_lr_gamma,
                "hidden_dim": args.implicitvol_hidden_dim,
                "num_layers": args.implicitvol_num_layers,
                "num_frequencies": args.implicitvol_num_frequencies,
                "w0_first": args.implicitvol_w0_first,
                "w0_hidden": args.implicitvol_w0_hidden,
                "patch_size": args.implicitvol_patch_size,
                "ssim_window_size": args.implicitvol_ssim_window_size,
                "loss_mode": args.implicitvol_loss_mode,
            },
            "ultra_nerf": {
                "steps": args.ultra_nerf_steps,
                "batch_patches": args.ultra_nerf_batch_patches,
                "lr": args.ultra_nerf_lr,
                "lr_decay_steps": args.ultra_nerf_lr_decay_steps,
                "lr_decay_rate": args.ultra_nerf_lr_decay_rate,
                "hidden_dim": args.ultra_nerf_hidden_dim,
                "num_layers": args.ultra_nerf_num_layers,
                "num_frequencies": args.ultra_nerf_num_frequencies,
                "encoder_use_pi": args.ultra_nerf_encoder_use_pi,
                "patch_size": args.ultra_nerf_patch_size,
                "ssim_window_size": args.ultra_nerf_ssim_window_size,
                "loss_mode": args.ultra_nerf_loss_mode,
                "ssim_weight": args.ultra_nerf_ssim_weight,
                "mse_weight": args.ultra_nerf_mse_weight,
            },
        },
        "patients": [],
    }
    method_fit_totals: dict[str, float] = {m: 0.0 for m in args.methods}
    method_fit_counts: dict[str, int] = {m: 0 for m in args.methods}

    print(
        f"Fitting {len(test_ids)} patient volumes with methods={args.methods} on device={args.device}",
        flush=True,
    )
    print(
        "[defaults=b200] implicitvol: "
        f"steps={args.implicitvol_steps}, batch_patches={args.implicitvol_batch_patches}, "
        f"lr={args.implicitvol_lr}, hidden={args.implicitvol_hidden_dim}, "
        f"layers={args.implicitvol_num_layers}, freq={args.implicitvol_num_frequencies}, "
        f"patch={args.implicitvol_patch_size}, "
        f"ssim_win={args.implicitvol_ssim_window_size}, loss={args.implicitvol_loss_mode}, "
        f"lr_step={args.implicitvol_lr_step_interval}, lr_gamma={args.implicitvol_lr_gamma}",
        flush=True,
    )
    print(
        "[defaults=b200] ultra_nerf: "
        f"steps={args.ultra_nerf_steps}, batch_patches={args.ultra_nerf_batch_patches}, "
        f"lr={args.ultra_nerf_lr}, hidden={args.ultra_nerf_hidden_dim}, "
        f"layers={args.ultra_nerf_num_layers}, freq={args.ultra_nerf_num_frequencies}, "
        f"enc_pi={args.ultra_nerf_encoder_use_pi}, "
        f"patch={args.ultra_nerf_patch_size}, ssim_win={args.ultra_nerf_ssim_window_size}, "
        f"loss={args.ultra_nerf_loss_mode}, ssim_w={args.ultra_nerf_ssim_weight}, "
        f"mse_w={args.ultra_nerf_mse_weight}, "
        f"lr_decay_steps={args.ultra_nerf_lr_decay_steps}, lr_decay_rate={args.ultra_nerf_lr_decay_rate}",
        flush=True,
    )

    for patient_id in test_ids:
        patient_start = time.perf_counter()
        patient_dir = resolve_patient_dir(args.dicom_root, patient_id)
        print(f"\n==== Patient {patient_id} ({patient_dir}) ====", flush=True)
        slices_u8, angles_rad = load_patient_volume_from_dicom(patient_dir)
        print(
            f"Loaded volume: slices={slices_u8.shape[0]}, H={slices_u8.shape[1]}, W={slices_u8.shape[2]}",
            flush=True,
        )

        patient_record: dict[str, Any] = {
            "patient_id": patient_id,
            "patient_dir": str(patient_dir),
            "methods": {},
        }

        for method in args.methods:
            out_path = output_root / method / f"{patient_id}.pt"
            if out_path.is_file() and not args.overwrite:
                print(f"[{method}] Skipping existing checkpoint: {out_path}", flush=True)
                patient_record["methods"][method] = {
                    "checkpoint": str(out_path),
                    "status": "skipped_existing",
                }
                continue

            if method == "implicitvol":
                fit_start = time.perf_counter()
                model, stats = fit_patient_implicitvol(
                    slices_u8=slices_u8,
                    angles_rad=angles_rad,
                    steps=args.implicitvol_steps,
                    batch_patches=args.implicitvol_batch_patches,
                    lr=args.implicitvol_lr,
                    device=args.device,
                    hidden_dim=args.implicitvol_hidden_dim,
                    num_layers=args.implicitvol_num_layers,
                    num_frequencies=args.implicitvol_num_frequencies,
                    w0_first=args.implicitvol_w0_first,
                    w0_hidden=args.implicitvol_w0_hidden,
                    patch_size=args.implicitvol_patch_size,
                    ssim_window_size=args.implicitvol_ssim_window_size,
                    loss_mode=args.implicitvol_loss_mode,
                    lr_step_interval=args.implicitvol_lr_step_interval,
                    lr_gamma=args.implicitvol_lr_gamma,
                    log_every=args.log_every,
                )
            elif method == "ultra_nerf":
                fit_start = time.perf_counter()
                model, stats = fit_patient_ultra_nerf(
                    slices_u8=slices_u8,
                    angles_rad=angles_rad,
                    steps=args.ultra_nerf_steps,
                    batch_patches=args.ultra_nerf_batch_patches,
                    lr=args.ultra_nerf_lr,
                    device=args.device,
                    hidden_dim=args.ultra_nerf_hidden_dim,
                    num_layers=args.ultra_nerf_num_layers,
                    num_frequencies=args.ultra_nerf_num_frequencies,
                    encoder_use_pi=args.ultra_nerf_encoder_use_pi,
                    patch_size=args.ultra_nerf_patch_size,
                    ssim_window_size=args.ultra_nerf_ssim_window_size,
                    loss_mode=args.ultra_nerf_loss_mode,
                    ssim_weight=args.ultra_nerf_ssim_weight,
                    mse_weight=args.ultra_nerf_mse_weight,
                    lr_decay_steps=args.ultra_nerf_lr_decay_steps,
                    lr_decay_rate=args.ultra_nerf_lr_decay_rate,
                    log_every=args.log_every,
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

            fit_seconds = float(time.perf_counter() - fit_start)
            method_fit_totals[method] += fit_seconds
            method_fit_counts[method] += 1

            save_checkpoint(
                out_path=out_path,
                model=model,
                method=method,
                patient_id=patient_id,
                patient_dir=patient_dir,
                fit_stats=stats,
            )
            print(
                f"[{method}] Saved checkpoint to {out_path} (last_loss={stats['last_loss']:.6f})",
                flush=True,
            )
            print(
                f"[{method}] Fit time: {fit_seconds:.2f}s ({fit_seconds / 60.0:.2f} min)",
                flush=True,
            )
            patient_record["methods"][method] = {
                "checkpoint": str(out_path),
                "status": "fitted",
                "last_loss": stats["last_loss"],
                "steps": stats["steps"],
                "batch_patches": stats.get("batch_patches"),
                "batch_points": stats["batch_points"],
                "fit_seconds": fit_seconds,
            }

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        patient_record["patient_fit_seconds"] = float(time.perf_counter() - patient_start)
        manifest["patients"].append(patient_record)

    run_seconds = float(time.perf_counter() - run_start)
    run_end_utc = datetime.now(timezone.utc).isoformat()
    manifest["run_end_utc"] = run_end_utc
    manifest["run_seconds"] = run_seconds
    manifest["timing_summary"] = {
        "method_fit_seconds_total": method_fit_totals,
        "method_fit_count": method_fit_counts,
        "method_fit_seconds_mean": {
            m: (method_fit_totals[m] / method_fit_counts[m] if method_fit_counts[m] > 0 else 0.0)
            for m in args.methods
        },
    }

    manifest_path = output_root / "fit_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest: {manifest_path}", flush=True)
    print(
        f"Total fit run time: {run_seconds:.2f}s ({run_seconds / 60.0:.2f} min)",
        flush=True,
    )


if __name__ == "__main__":
    main()
