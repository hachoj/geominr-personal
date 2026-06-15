#!/usr/bin/env python3
import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np

DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_SHARD_SIZE = 1024
DEFAULT_PATCH_SIZE = (64, 64)
DEFAULT_TRAIN_CROPS = 16


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert NPZ splits into WebDataset tar+idx shards with offline patch crops."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/dataset"),
        help="Root containing NPZ splits (train/val/test).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/dataset_wds"),
        help="Output root for tar+idx shards.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Splits to convert.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="Number of output examples per tar shard.",
    )
    parser.add_argument(
        "--patch-size",
        nargs=2,
        type=int,
        default=list(DEFAULT_PATCH_SIZE),
        metavar=("HEIGHT", "WIDTH"),
        help="Offline patch size to store per sample in shards.",
    )
    parser.add_argument(
        "--train-crops-per-sample",
        type=int,
        default=DEFAULT_TRAIN_CROPS,
        help="Number of random crops generated per source sample for train split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for deterministic random crop generation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max source files per split (for smoke tests).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing shard artifacts in output split dirs before writing.",
    )
    return parser.parse_args()


def _trim_zero_border(slices: np.ndarray):
    mask = (slices != 0).any(axis=0)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    height = slices.shape[1]
    width = slices.shape[2]

    if rows.size == 0 or cols.size == 0:
        return slices, {"top": 0, "bottom": 0, "left": 0, "right": 0}

    top = int(rows[0])
    bottom = int(rows[-1])
    left = int(cols[0])
    right = int(cols[-1])

    trimmed = slices[:, top : bottom + 1, left : right + 1]
    info = {
        "top": top,
        "bottom": height - 1 - bottom,
        "left": left,
        "right": width - 1 - right,
    }
    return trimmed, info


def _serialize_npz(slices: np.ndarray, angles: np.ndarray) -> bytes:
    payload = io.BytesIO()
    np.savez(payload, slices=slices, angles=angles.astype(np.float32))
    return payload.getvalue()


def _center_crop(slices: np.ndarray, patch_h: int, patch_w: int) -> np.ndarray:
    _, height, width = slices.shape
    start_h = 0 if height <= patch_h else (height - patch_h) // 2
    start_w = 0 if width <= patch_w else (width - patch_w) // 2
    return slices[:, start_h : start_h + patch_h, start_w : start_w + patch_w]


def _non_overlapping_random_crops(
    slices: np.ndarray,
    patch_h: int,
    patch_w: int,
    crop_count: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    _, height, width = slices.shape
    h_starts = list(range(0, height - patch_h + 1, patch_h))
    w_starts = list(range(0, width - patch_w + 1, patch_w))
    candidates = [(h, w) for h in h_starts for w in w_starts]

    if len(candidates) < crop_count:
        raise ValueError(
            f"Requested {crop_count} non-overlapping crops, but only {len(candidates)} "
            f"fit in shape {(height, width)} with patch {(patch_h, patch_w)}."
        )

    chosen = rng.choice(len(candidates), size=crop_count, replace=False)
    patches = []
    for idx in chosen.tolist():
        start_h, start_w = candidates[int(idx)]
        patch = slices[:, start_h : start_h + patch_h, start_w : start_w + patch_w]
        patches.append(patch)
    return patches


def _seed_for_sample(base_seed: int, sample_key: str) -> int:
    digest = hashlib.blake2b(sample_key.encode("utf-8"), digest_size=8).digest()
    hashed = int.from_bytes(digest, byteorder="little", signed=False)
    return (int(base_seed) + hashed) % (2**32)


def _prepare_split_output(split_output_dir: Path, overwrite: bool) -> None:
    split_output_dir.mkdir(parents=True, exist_ok=True)

    shard_artifacts = list(split_output_dir.glob("*.tar")) + list(
        split_output_dir.glob("*.idx")
    )
    manifest_path = split_output_dir / "manifest.json"

    if shard_artifacts and not overwrite:
        raise RuntimeError(
            f"Output directory {split_output_dir} already contains shards. Use --overwrite."
        )

    if overwrite:
        for path in shard_artifacts:
            path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()


def _build_split(
    split: str,
    input_root: Path,
    output_root: Path,
    shard_size: int,
    patch_size: tuple[int, int],
    train_crops_per_sample: int,
    seed: int,
    limit: int | None,
    overwrite: bool,
):
    patch_h, patch_w = int(patch_size[0]), int(patch_size[1])

    input_dir = input_root / split
    if not input_dir.exists():
        raise FileNotFoundError(f"Input split directory does not exist: {input_dir}")

    source_files = sorted(input_dir.glob("*.npz"))
    if not source_files:
        raise RuntimeError(f"No NPZ files found in {input_dir}")

    if limit is not None:
        source_files = source_files[:limit]

    stems = [path.stem for path in source_files]
    if len(stems) != len(set(stems)):
        raise RuntimeError(f"Duplicate source sample stems found in split {split}")

    output_dir = output_root / split
    _prepare_split_output(output_dir, overwrite=overwrite)

    manifest = {
        "split": split,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_count": len(source_files),
        "output_count": 0,
        "shard_size": int(shard_size),
        "crop_config": {
            "patch_size": [patch_h, patch_w],
            "train_crops_per_sample": int(train_crops_per_sample),
            "val_test_policy": "center_crop",
        },
        "shards": [],
        "trim_summary": {
            "samples": 0,
            "orig_height_min": None,
            "orig_height_max": None,
            "orig_width_min": None,
            "orig_width_max": None,
            "trim_height_min": None,
            "trim_height_max": None,
            "trim_width_min": None,
            "trim_width_max": None,
            "removed_top_total": 0,
            "removed_bottom_total": 0,
            "removed_left_total": 0,
            "removed_right_total": 0,
        },
    }

    seen_keys = set()

    current_tar = None
    current_tar_path = None
    current_idx_path = None
    current_count = 0
    current_shard_id = -1

    def _open_new_shard(shard_id: int):
        tar_path = output_dir / f"microus-{split}-{shard_id:05d}.tar"
        idx_path = output_dir / f"microus-{split}-{shard_id:05d}.idx"
        tar_obj = tarfile.open(tar_path, mode="w")
        return tar_obj, tar_path, idx_path

    def _close_current_shard():
        nonlocal current_tar, current_tar_path, current_idx_path, current_count
        if current_tar is None:
            return

        current_tar.close()
        subprocess.run(
            ["wds2idx", str(current_tar_path), str(current_idx_path)],
            check=True,
        )

        manifest["shards"].append(
            {
                "shard_id": int(current_shard_id),
                "tar": current_tar_path.name,
                "idx": current_idx_path.name,
                "samples": int(current_count),
            }
        )
        print(
            f"[{split}] wrote {current_tar_path.name} ({current_count} samples) and {current_idx_path.name}"
        )

        current_tar = None
        current_tar_path = None
        current_idx_path = None
        current_count = 0

    def _write_sample(sample_key: str, sample_slices: np.ndarray, sample_angles: np.ndarray):
        nonlocal current_tar, current_tar_path, current_idx_path, current_count, current_shard_id

        if sample_key in seen_keys:
            raise RuntimeError(f"Duplicate output sample key in split {split}: {sample_key}")
        seen_keys.add(sample_key)

        if current_tar is None or current_count >= shard_size:
            _close_current_shard()
            current_shard_id += 1
            current_tar, current_tar_path, current_idx_path = _open_new_shard(current_shard_id)
            current_count = 0

        payload = _serialize_npz(sample_slices, sample_angles)
        tar_info = tarfile.TarInfo(name=f"{sample_key}.npz")
        tar_info.size = len(payload)
        tar_info.mode = 0o644
        current_tar.addfile(tar_info, io.BytesIO(payload))

        current_count += 1
        manifest["output_count"] += 1

    for src_path in source_files:
        with np.load(src_path, allow_pickle=False) as f:
            slices = f["slices"]
            angles = f["angles"]

        if slices.ndim != 3 or slices.shape[0] != 3:
            raise ValueError(f"Invalid slices shape for {src_path}: {slices.shape}")
        if angles.shape != (3,):
            raise ValueError(f"Invalid angles shape for {src_path}: {angles.shape}")

        original_h = int(slices.shape[1])
        original_w = int(slices.shape[2])

        trimmed_slices, trim_info = _trim_zero_border(slices)
        trim_h = int(trimmed_slices.shape[1])
        trim_w = int(trimmed_slices.shape[2])

        if trim_h < patch_h or trim_w < patch_w:
            raise ValueError(
                f"Trimmed sample {src_path} is smaller than patch {(patch_h, patch_w)}: "
                f"{trimmed_slices.shape}"
            )

        manifest["trim_summary"]["samples"] += 1

        for key, value in (
            ("orig_height_min", original_h),
            ("orig_height_max", original_h),
            ("orig_width_min", original_w),
            ("orig_width_max", original_w),
            ("trim_height_min", trim_h),
            ("trim_height_max", trim_h),
            ("trim_width_min", trim_w),
            ("trim_width_max", trim_w),
        ):
            if manifest["trim_summary"][key] is None:
                manifest["trim_summary"][key] = int(value)
            elif key.endswith("_min"):
                manifest["trim_summary"][key] = min(manifest["trim_summary"][key], int(value))
            else:
                manifest["trim_summary"][key] = max(manifest["trim_summary"][key], int(value))

        manifest["trim_summary"]["removed_top_total"] += int(trim_info["top"])
        manifest["trim_summary"]["removed_bottom_total"] += int(trim_info["bottom"])
        manifest["trim_summary"]["removed_left_total"] += int(trim_info["left"])
        manifest["trim_summary"]["removed_right_total"] += int(trim_info["right"])

        stem = src_path.stem
        if split == "train":
            rng = np.random.default_rng(_seed_for_sample(seed, stem))
            crop_count = max(1, int(train_crops_per_sample))
            patches = _non_overlapping_random_crops(
                trimmed_slices,
                patch_h,
                patch_w,
                crop_count,
                rng,
            )
            for crop_idx, patch in enumerate(patches):
                sample_key = f"{stem}_crop{crop_idx:02d}" if crop_count > 1 else stem
                _write_sample(sample_key, patch, angles)
        else:
            patch = _center_crop(trimmed_slices, patch_h, patch_w)
            _write_sample(stem, patch, angles)

    _close_current_shard()

    if manifest["output_count"] == 0:
        raise RuntimeError(f"No output samples generated for split {split}")

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"[{split}] manifest: {manifest_path}")


def main():
    args = _parse_args()

    if args.shard_size <= 0:
        raise ValueError("--shard-size must be > 0")
    if args.train_crops_per_sample <= 0:
        raise ValueError("--train-crops-per-sample must be > 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be > 0")

    patch_size = (int(args.patch_size[0]), int(args.patch_size[1]))

    for split in args.splits:
        _build_split(
            split=split,
            input_root=args.input_root,
            output_root=args.output_root,
            shard_size=int(args.shard_size),
            patch_size=patch_size,
            train_crops_per_sample=int(args.train_crops_per_sample),
            seed=int(args.seed),
            limit=args.limit,
            overwrite=bool(args.overwrite),
        )


if __name__ == "__main__":
    main()
