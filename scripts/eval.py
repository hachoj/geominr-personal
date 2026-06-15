import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import DictConfig
from tqdm import tqdm
from pathlib import Path
import lpips

from data.data import build_val_dataloader
from geominr.models.utils.utils import trilinear_sample, create_xrtheta, strip_orig_mod
from geominr.models.utils.metrics import PSNR, SSIM_slicewise, EPI, SSI
from geominr.config import load_config, instantiate


def prepare_batch(batch, device):
    conditioning_idx = [0, 2]
    slices, angles, arcs = batch
    B, _, H, W = slices.shape

    angles = angles.to(device, non_blocking=True)

    conditioning_slices = slices[:, conditioning_idx, :, :].to(
        device, non_blocking=True
    )

    xyz = create_xrtheta(B, H, W, angles[:, 1], angles[:, conditioning_idx], device)
    xyz_left = create_xrtheta(
        B, H, W, angles[:, 0], angles[:, conditioning_idx], device
    )
    xyz_right = create_xrtheta(
        B, H, W, angles[:, 2], angles[:, conditioning_idx], device
    )

    target_slice = (
        slices[:, 1, :, :]
        .contiguous()
        .reshape(B, 1, H, W)
        .to(device, non_blocking=True)
    )
    target_slice_left = (
        slices[:, 0, :, :]
        .contiguous()
        .reshape(B, 1, H, W)
        .to(device, non_blocking=True)
    )
    target_slice_right = (
        slices[:, 2, :, :]
        .contiguous()
        .reshape(B, 1, H, W)
        .to(device, non_blocking=True)
    )
    arcs = arcs.to(device, non_blocking=True)

    # Relative angular position of the middle (query) slice between the two
    # conditioning slices — used by the trilinear baseline (trilinear_sample).
    rel_mid = (angles[:, 1] - angles[:, 0]) / (angles[:, 2] - angles[:, 0])

    return (
        conditioning_slices,
        xyz,
        xyz_left,
        xyz_right,
        target_slice,
        target_slice_left,
        target_slice_right,
        arcs,
        rel_mid,
    )


def compute_stats(tensor_list):
    if not tensor_list:
        return 0.0, 0.0
    combined = torch.cat(tensor_list)
    return combined.mean().item(), combined.std().item()


def log_and_save_results(results, model_label, output_dir):
    metric_groups = [("Total", ""), ("Query Only", "_query"), ("Linear", "_linear")]
    metrics_list = ["PSNR", "SSIM", "EPI", "SSI", "LPIPS"]

    lines = []
    lines.append(f"Eval completed")
    lines.append(f"Model: {model_label}")
    lines.append(f"-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#")

    for group_name, suffix in metric_groups:
        if group_name != "Total":
            lines.append(f" --------- {group_name} ----------")
            lines.append(f"-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#")

        for metric in metrics_list:
            key_mean = f"avg_{metric}{suffix}"
            key_std = f"std_{metric}{suffix}"

            if key_mean in results:
                lines.append(
                    f"Average {metric}{' ('+group_name+')' if group_name == 'Linear' else ''}: {results[key_mean]:.5f} ± {results[key_std]:.5f}"
                )

        lines.append(f"-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#")

    for line in lines:
        print(line)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # Machine-readable copy of the results for downstream aggregation.
    import json

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(
            {"model_label": model_label, **{k: float(v) for k, v in results.items()}},
            f,
            indent=2,
        )


def run_evaluation(model, cfg: DictConfig, device):
    eval_cfg = cfg.eval

    batch_size = eval_cfg.get("batch_size", 256)
    num_workers = eval_cfg.get("num_workers", 4)
    pin_memory = eval_cfg.get("pin_memory", True)
    data_dir = eval_cfg.data_dir
    seed = eval_cfg.get("seed", 42)

    patch_size_cfg = cfg.data.patch_size
    if isinstance(patch_size_cfg, int):
        patch_h = patch_w = patch_size_cfg
    else:
        patch_h, patch_w = patch_size_cfg

    print(f"Building dataloaders...")
    test_loader = build_val_dataloader(
        data_dir=data_dir,
        patch_size=(patch_h, patch_w),
        batch_size=batch_size,
        mode="test",
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print(f"Dataloaders built successfully")
    print(f"--------------------------------")
    print(f"Beginning eval...")

    metric_names = ["psnr", "ssim", "epi", "ssi", "lpips"]
    storage = {name: {"all": [], "query": [], "linear": []} for name in metric_names}

    # --- add lpips ---
    lpips_loss_fn = lpips.LPIPS(net="alex").to(device)

    metric_fns = {
        "psnr": lambda pred, tgt: PSNR(pred, tgt, reduction="none"),
        "ssim": lambda pred, tgt: SSIM_slicewise(pred, tgt, reduction="none"),
        "epi": lambda pred, tgt: EPI(pred, tgt, reduction="none"),
        "ssi": lambda pred, tgt: SSI(pred, tgt, reduction="none"),
        "lpips": lambda pred, tgt: lpips_loss_fn(pred, tgt),
    }
    model.eval()
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
            (
                conditioning_slices,
                xyz,
                xyz_left,
                xyz_right,
                target_slice,
                target_slice_left,
                target_slice_right,
                arcs,
                rel_mid,
            ) = prepare_batch(batch, device)

            B, _, H, W = conditioning_slices.shape
            num_samples += B

            left_slice = conditioning_slices[:, 0].unsqueeze(1)
            right_slice = conditioning_slices[:, 1].unsqueeze(1)

            out = model(left_slice, right_slice, arcs, xyz).view(B, 1, H, W)
            out_left = model(left_slice, right_slice, arcs, xyz_left).view(B, 1, H, W)
            out_right = model(left_slice, right_slice, arcs, xyz_right).view(B, 1, H, W)

            out_linear = trilinear_sample(conditioning_slices, rel_mid)

            for name, metric_fn in metric_fns.items():
                val_query = metric_fn(out, target_slice).detach().flatten()
                storage[name]["query"].append(val_query)

                val_left = metric_fn(out_left, target_slice_left).detach().flatten()
                val_right = metric_fn(out_right, target_slice_right).detach().flatten()

                storage[name]["all"].append(val_left)
                storage[name]["all"].append(val_right)
                storage[name]["all"].append(val_query)

                val_linear = metric_fn(out_linear, target_slice).detach().flatten()
                storage[name]["linear"].append(val_linear)

    results = {"num_samples": num_samples}

    for name in metric_names:
        upper_name = name.upper()

        mean, std = compute_stats(storage[name]["all"])
        results[f"avg_{upper_name}"] = mean  # pyrefly:ignore
        results[f"std_{upper_name}"] = std  # pyrefly:ignore

        mean_q, std_q = compute_stats(storage[name]["query"])
        results[f"avg_{upper_name}_query"] = mean_q  # pyrefly:ignore
        results[f"std_{upper_name}_query"] = std_q  # pyrefly:ignore

        mean_l, std_l = compute_stats(storage[name]["linear"])
        results[f"avg_{upper_name}_linear"] = mean_l  # pyrefly:ignore
        results[f"std_{upper_name}_linear"] = std_l  # pyrefly:ignore

    return results


def main():
    cfg = load_config()
    torch.set_float32_matmul_precision("high")
    device = "cuda"

    model = instantiate(cfg.model).to(device)

    model_path = cfg.eval.model_path
    if model_path and os.path.isfile(model_path):
        print(f"Loading model from {model_path}")
        ckpt = torch.load(model_path, map_location="cpu")

        state_dict = ckpt
        if "model" in ckpt:
            state_dict = ckpt["model"]

        model.load_state_dict(strip_orig_mod(state_dict), strict=True)
        print(f"Loaded model state successfully")
    else:
        print(f"Warning: No valid model path found at {model_path}")
        return

    results = run_evaluation(model, cfg, device)

    model_label = Path(cfg.eval.model_path).name
    output_dir = os.path.join(cfg.eval.output_dir, model_label)

    log_and_save_results(results, model_label, output_dir)


if __name__ == "__main__":
    main()
