import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import torch

from geominr.models.utils.reconstruct import (
    extract_slices,
    save_volume,
    reconstruct_volume_nn,
    reconstruct_volume_linear_fast,
    reconstruct_volume_sr_fast,
)
from geominr.models.utils.utils import strip_orig_mod
from geominr.config import load_config, instantiate


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    print(f"Using device: {device}")
    print("--------------------------------")

    print("Building model...")
    model = instantiate(cfg.model).to(device)

    model_path = cfg.inference.model_path
    print(f"Loading weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    model.load_state_dict(strip_orig_mod(state_dict))
    model = torch.compile(model)
    model.eval()
    print("Model built and loaded successfully")
    print("--------------------------------")

    image_path = cfg.inference.image_path
    save_path = cfg.inference.save_path
    patient_name = os.path.basename(image_path.rstrip("/"))
    depth = cfg.inference.get("volume_depth")
    ap_voxel_count = cfg.inference.get("ap_voxel_count")
    probe_radius = cfg.inference.get("probe_radius", 12.5)

    print(f"Loading data from {image_path}...")
    extracted_slices = extract_slices(image_path)
    print("Data loaded successfully")
    print("--------------------------------")

    def _timed(label, fn):
        _sync(device)
        t0 = time.perf_counter()
        res = fn()
        _sync(device)
        print(f"{label} reconstruction time: {time.perf_counter() - t0:.2f}s")
        return res

    nn_res = _timed(
        "NN",
        lambda: reconstruct_volume_nn(
            extracted_slices,
            depth=depth,
            ap_voxel_count=ap_voxel_count,
            probe_radius=probe_radius,
            device=device,
        ),
    )
    save_volume(extracted_slices, nn_res, os.path.join(save_path, f"{patient_name}_nn"))
    print("--------------------------------")

    lin_res = _timed(
        "Linear",
        lambda: reconstruct_volume_linear_fast(
            extracted_slices,
            depth=depth,
            ap_voxel_count=ap_voxel_count,
            probe_radius=probe_radius,
            device=device,
        ),
    )
    save_volume(
        extracted_slices, lin_res, os.path.join(save_path, f"{patient_name}_linear")
    )
    print("--------------------------------")

    sr_res = _timed(
        "Model SR",
        lambda: reconstruct_volume_sr_fast(
            model,
            extracted_slices,
            depth=depth,
            ap_voxel_count=ap_voxel_count,
            patch_size=cfg.data.patch_size,
            stride=cfg.inference.stride,
            probe_radius=probe_radius,
        ),
    )
    save_volume(extracted_slices, sr_res, os.path.join(save_path, f"{patient_name}_sr"))
    print("Done!")


if __name__ == "__main__":
    main()
