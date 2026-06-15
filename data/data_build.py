import os
import argparse
import numpy as np
from tqdm import tqdm
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from geominr.models.utils.reconstruct import extract_slices

# Raw-DICOM locations (the working symlink under the repo, by default).
DEFAULT_VOL_DIR_POS = "data/Original_Data_in_DICOM_Format/positive"
DEFAULT_VOL_DIR_NEG = "data/Original_Data_in_DICOM_Format/negative"


def compute_window_quality(angles, max_adjacent_gap, max_overall_span):
    if len(angles) != 3:
        return 0.0

    adjacent_gaps = np.abs(np.diff(angles))
    max_gap = np.max(adjacent_gaps)
    min_gap = np.min(adjacent_gaps)

    overall_span = abs(angles[-1] - angles[0])

    if max_gap > max_adjacent_gap or overall_span > max_overall_span:
        return 0.0

    if max_gap < 1e-8:
        return 0.0

    gap_uniformity = 1.0 - (max_gap - min_gap) / max_gap  # [0, 1]
    span_utilization = overall_span / max_overall_span if max_overall_span > 0 else 0.0
    quality = 0.7 * gap_uniformity + 0.3 * span_utilization
    return float(quality)


def validate_window(angles, max_adjacent_gap, min_adjacent_gap, max_overall_span, min_overall_span):
    if len(angles) != 3:
        return False, None

    if len(np.unique(angles)) < 3:
        return False, None

    adjacent_gaps = np.abs(np.diff(angles))
    max_adjacent = float(np.max(adjacent_gaps))
    min_adjacent = float(np.min(adjacent_gaps))

    if max_adjacent > max_adjacent_gap:
        return False, None
    if min_adjacent < min_adjacent_gap:
        return False, None

    overall_span = float(abs(angles[-1] - angles[0]))

    if overall_span > max_overall_span:
        return False, None
    if overall_span < min_overall_span:
        return False, None

    quality = compute_window_quality(angles, max_adjacent_gap, max_overall_span)
    return True, quality


def process_patient_slices(patient_dict):
    return dict(sorted(patient_dict.items(), key=lambda kv: kv[1]["angle"]))


def process_patient(patient_dict, out_path, patient_name, label, cfg):
    angles = sorted([value["angle"] for value in patient_dict.values()])
    angles = np.array(angles, dtype=np.float32)
    angle_patient_dict = {np.float32(value["angle"]): value for value in patient_dict.values()}

    patient_dir = os.path.join(out_path, f"{label}_{patient_name}")
    os.makedirs(patient_dir, exist_ok=True)

    slice_number = 0
    total_attempts = 0
    valid_windows = []

    for i in range(len(angles) - 2):
        window_angles = angles[i : i + 3]  # [prev, curr, next]
        total_attempts += 1

        is_valid, quality = validate_window(
            window_angles,
            cfg.max_adjacent_gap,
            cfg.min_adjacent_gap,
            cfg.max_overall_span,
            cfg.min_overall_span,
        )
        if not is_valid:
            continue
        if cfg.use_quality_scoring and quality is not None and quality < cfg.min_quality_score:
            continue

        valid_windows.append((i, window_angles, quality))

    for idx, window_angles, quality in valid_windows:
        slice_number += 1
        file_name = f"patch_{slice_number:03d}.npz"
        path = os.path.join(patient_dir, file_name)

        prev_slice = angle_patient_dict[window_angles[0]]["slice_volume"]  # [H,W]
        curr_slice = angle_patient_dict[window_angles[1]]["slice_volume"]  # [H,W]
        next_slice = angle_patient_dict[window_angles[2]]["slice_volume"]  # [H,W]
        slices = np.stack([prev_slice, curr_slice, next_slice], axis=0)  # [3,H,W]

        np.savez(path, slices=slices, angles=window_angles.astype(np.float32))
        np.savez(
            path.replace(".npz", "_rev.npz"),
            slices=slices[::-1],
            angles=(-1 * window_angles[::-1]).astype(np.float32),
        )

    acceptance_rate = (slice_number / total_attempts * 100) if total_attempts > 0 else 0.0
    print(
        f"Patient {patient_name} ({label}): {slice_number}/{total_attempts} windows saved "
        f"(acceptance rate: {acceptance_rate:.1f}%)"
    )
    return slice_number


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build 3-slice NPZ triplets from DICOM with gap/span filtering."
    )
    ap.add_argument("--vol-dir-pos", default=DEFAULT_VOL_DIR_POS)
    ap.add_argument("--vol-dir-neg", default=DEFAULT_VOL_DIR_NEG)
    ap.add_argument("--out-dir", default=None,
                    help="output dir (per-patient subdirs); default encodes the params")
    ap.add_argument("--max-adjacent-gap", type=float, default=1.3)
    ap.add_argument("--min-adjacent-gap", type=float, default=0.1)
    ap.add_argument("--max-overall-span", type=float, default=1.8)
    ap.add_argument("--min-overall-span", type=float, default=0.4)
    ap.add_argument("--min-quality-score", type=float, default=0.2)
    ap.add_argument("--no-quality-scoring", dest="use_quality_scoring", action="store_false")
    ap.add_argument("--max-patients", type=int, default=None,
                    help="cap patients per label (for smoke tests)")
    return ap.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    if cfg.out_dir is None:
        cfg.out_dir = (
            "data/datasets/data_3_slice_"
            f"adj{cfg.max_adjacent_gap}_minadj{cfg.min_adjacent_gap}"
            f"_span{cfg.max_overall_span}_min{cfg.min_overall_span}_rev"
        )
    os.makedirs(cfg.out_dir, exist_ok=True)
    print(f"Building -> {cfg.out_dir}")
    print(f"  filter: adj[{cfg.min_adjacent_gap}, {cfg.max_adjacent_gap}]  "
          f"span[{cfg.min_overall_span}, {cfg.max_overall_span}]  "
          f"minq={cfg.min_quality_score if cfg.use_quality_scoring else 'off'}")

    total_slices = 0
    for vol_dir, label in [(cfg.vol_dir_pos, "positive"), (cfg.vol_dir_neg, "negative")]:
        patient_dirs = sorted(
            d for d in os.listdir(vol_dir) if os.path.isdir(os.path.join(vol_dir, d))
        )
        if cfg.max_patients is not None:
            patient_dirs = patient_dirs[: cfg.max_patients]
        for patient in tqdm(patient_dirs, desc=f"{label} patients"):
            patient_dict = extract_slices(os.path.join(vol_dir, patient))
            sorted_dict = process_patient_slices(patient_dict)
            total_slices += process_patient(sorted_dict, cfg.out_dir, patient, label, cfg)

    print(f"Total slices saved: {total_slices}")
