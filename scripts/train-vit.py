import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW, Muon

torch.set_float32_matmul_precision("high")

import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

from data.data import build_train_dataloader, build_val_dataloader
from geominr.config import load_config, instantiate, save_config
from geominr.models.utils.metrics import PSNR, SSIM_slicewise
from geominr.models.utils.utils import trilinear_sample
from geominr.optim_utils import build_param_groups, build_lr_scheduler


def set_seed(base_seed: int, rank: int = 0) -> int:
    seed = int(base_seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def train(
    model,
    muon,
    adamw,
    muon_scheduler,
    adamw_scheduler,
    train_dataloader,
    val_dataloader,
    cfg,
    device,
    is_main,
):
    """ViT-S/2 patch-interpolation baseline: pure MSE, no GAN/LPIPS/self-consistency."""
    grad_clip = float("inf") if cfg.train.grad_clip < 0 else cfg.train.grad_clip

    if cfg.wandb.enabled and is_main:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            dir=cfg.wandb.dir,
            config={
                "lr_muon": cfg.train.lr_muon,
                "lr_adamw": cfg.train.lr_adamw,
                "batch_size": cfg.train.batch_size,
                "num_epochs": cfg.train.num_epochs,
            },
        )
        wandb.define_metric("train_step", hidden=True)
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("epoch_step", hidden=True)
        wandb.define_metric("epoch/*", step_metric="epoch_step")
        wandb.define_metric("image/*", step_metric="epoch_step")

    optimizer_step = 0
    save_dir = Path(cfg.train.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.train.num_epochs):
        if hasattr(train_dataloader, "sampler") and hasattr(
            train_dataloader.sampler, "set_epoch"
        ):
            train_dataloader.sampler.set_epoch(epoch)

        start_time = time.time()

        grad_norm = torch.zeros((), device=device)
        accum_window_loss = torch.zeros((), device=device)
        accum_window_steps = torch.zeros((), device=device)

        model.train()
        micro_step = 0
        num_batches = len(train_dataloader)
        for i, batch in enumerate(train_dataloader):
            if micro_step == 0:
                muon.zero_grad(set_to_none=True)
                adamw.zero_grad(set_to_none=True)
                accum_window_loss = torch.zeros((), device=device)
                accum_window_steps = torch.zeros((), device=device)
            micro_step += 1

            slices, angles, _ = batch
            slices = slices.to(device, non_blocking=True)
            angles = angles.to(device, non_blocking=True)

            middle_slice = slices[:, 1].unsqueeze(1)
            conditioning_slices = slices[:, [0, 2]]

            relative_angle = (angles[:, 1] - angles[:, 0]) / (
                angles[:, 2] - angles[:, 0]
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(conditioning_slices, relative_angle)

            pred = pred.to(dtype=torch.float32)

            loss = F.mse_loss(pred, middle_slice) / cfg.train.grad_accum

            accum_window_loss += loss.detach() * cfg.train.grad_accum
            accum_window_steps += 1

            should_step = (micro_step == cfg.train.grad_accum) or (i == num_batches - 1)
            if not should_step:
                with model.no_sync():
                    loss.backward()
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            muon.step()
            adamw.step()
            muon_scheduler.step()
            adamw_scheduler.step()
            optimizer_step += 1
            micro_step = 0

            if cfg.wandb.enabled and (
                optimizer_step == 1
                or optimizer_step % cfg.train.every_n_steps == 0
            ):
                train_loss = accum_window_loss / torch.clamp(
                    accum_window_steps, min=1.0
                )
                train_loss_reduced = train_loss.detach().clone()
                dist.all_reduce(train_loss_reduced, op=dist.ReduceOp.SUM)
                train_loss_reduced /= dist.get_world_size()

                if not is_main:
                    continue

                wandb.log(
                    {
                        "train_step": optimizer_step,
                        "train/loss": train_loss_reduced.item(),
                        "train/grad_norm": grad_norm.item(),
                        "train/lr": adamw.param_groups[0]["lr"],
                    }
                )

        val_psnr_accum = torch.zeros((), device=device)
        val_ssim_accum = torch.zeros((), device=device)
        val_linear_psnr_accum = torch.zeros((), device=device)
        val_linear_ssim_accum = torch.zeros((), device=device)
        val_n_accum = torch.zeros((), device=device)

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                slices, angles, _ = batch
                slices = slices.to(device, non_blocking=True)
                angles = angles.to(device, non_blocking=True)
                B, _, H, W = slices.shape

                conditioning_slices = slices[:, [0, 2]]
                query_slice = slices[:, 1].unsqueeze(1)

                relative_angle = (angles[:, 1] - angles[:, 0]) / (
                    angles[:, 2] - angles[:, 0]
                )

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model(conditioning_slices, relative_angle)

                pred = pred.to(dtype=torch.float32)

                linear_pred = trilinear_sample(conditioning_slices, relative_angle)

                psnr = PSNR(pred, query_slice, reduction="sum")
                ssim = SSIM_slicewise(pred, query_slice, reduction="sum")
                linear_psnr = PSNR(linear_pred, query_slice, reduction="sum")
                linear_ssim = SSIM_slicewise(linear_pred, query_slice, reduction="sum")

                val_psnr_accum += psnr
                val_ssim_accum += ssim
                val_linear_psnr_accum += linear_psnr
                val_linear_ssim_accum += linear_ssim
                val_n_accum += B

        dist.all_reduce(val_psnr_accum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_ssim_accum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_linear_psnr_accum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_linear_ssim_accum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_n_accum, op=dist.ReduceOp.SUM)

        if cfg.wandb.enabled and is_main:
            psnr_val = (val_psnr_accum / val_n_accum).item()
            ssim_val = (val_ssim_accum / val_n_accum).item()
            linear_psnr_val = (val_linear_psnr_accum / val_n_accum).item()
            linear_ssim_val = (val_linear_ssim_accum / val_n_accum).item()
            delta_psnr = psnr_val - linear_psnr_val
            delta_ssim = ssim_val - linear_ssim_val

            # Data is already in [0, 1] (data/data.py uses /255); show it directly.
            vis_query = query_slice[0].clamp(0.0, 1.0)
            vis_pred = pred[0].clamp(0.0, 1.0)
            vis_linear = linear_pred[0].clamp(0.0, 1.0)
            comparison_image = (
                torch.cat(
                    [
                        vis_query.permute(1, 2, 0),
                        vis_pred.permute(1, 2, 0),
                        vis_linear.permute(1, 2, 0),
                    ],
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            wandb.log(
                {
                    "epoch/PSNR": psnr_val,
                    "epoch/SSIM": ssim_val,
                    "epoch/deltaPSNR": delta_psnr,
                    "epoch/deltaSSIM": delta_ssim,
                    "epoch/time": (time.time() - start_time),
                    "epoch_step": epoch + 1,
                    "image/comparison": wandb.Image(
                        comparison_image,
                        caption="GT, MODEL, LINEAR",
                    ),
                }
            )

        if is_main and (epoch + 1) % cfg.train.save_epoch == 0:
            checkpoint = {
                "model": model.module.state_dict(),
                "muon": muon.state_dict(),
                "adamw": adamw.state_dict(),
                "muon_scheduler": muon_scheduler.state_dict(),
                "adamw_scheduler": adamw_scheduler.state_dict(),
            }
            torch.save(checkpoint, save_dir / f"model_{epoch+1}.pt")


def main():
    cfg = load_config()
    # setup DDP
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    is_main: bool = rank == 0
    if is_main:
        save_config(cfg, cfg.train.save_dir)

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl", init_method="env://"
    )  # nvidia collective communications library

    device = torch.device("cuda", local_rank)
    set_seed(cfg.train.seed, rank)

    train_dataloader = build_train_dataloader(
        cfg.data.train_dir,
        cfg.data.patch_size,
        cfg.data.num_workers,
        cfg.data.pin_memory,
        cfg.data.batch_size,
        prefetch_factor=cfg.data.prefetch_factor,
        persistent_workers=cfg.data.persistent_workers,
    )
    val_dataloader = build_val_dataloader(
        cfg.data.val_dir,
        cfg.data.patch_size,
        cfg.data.batch_size,
        num_workers=cfg.data.val_num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.val_prefetch_factor,
        persistent_workers=cfg.data.val_persistent_workers,
    )

    model = instantiate(cfg.model)

    total_steps = cfg.train.num_epochs * len(train_dataloader)
    warmup_steps = cfg.train.warmup_epochs * len(train_dataloader)

    # Same Muon (2D) + AdamW (1D) recipe as the HAT canonical, for a fair baseline.
    groups = build_param_groups(
        model,
        lr_adamw=cfg.train.lr_adamw,
        lr_muon=cfg.train.lr_muon,
        wd=cfg.train.weight_decay,
    )
    muon = Muon(groups["muon"], adjust_lr_fn=cfg.train.muon_mode)
    adamw = AdamW(groups["adamw"], betas=tuple(cfg.train.adamw_betas))
    muon_scheduler = build_lr_scheduler(
        cfg, muon, total_steps, cfg.train.lr_muon, cfg.train.min_lr, warmup_steps
    )
    adamw_scheduler = build_lr_scheduler(
        cfg, adamw, total_steps, cfg.train.lr_adamw, cfg.train.min_lr, warmup_steps
    )

    if is_main:
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    model = model.to(device=device)
    model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank])

    train(
        model=model,
        muon=muon,
        adamw=adamw,
        muon_scheduler=muon_scheduler,
        adamw_scheduler=adamw_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        cfg=cfg,
        device=device,
        is_main=is_main,
    )


if __name__ == "__main__":
    main()
