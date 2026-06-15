import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.optim import AdamW, Muon

torch.set_float32_matmul_precision("high")

from geominr.config import load_config, instantiate, save_config
import wandb
import time
import lpips

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from geominr.models.utils.utils import create_xrtheta, trilinear_sample
from geominr.models.utils.metrics import PSNR, SSIM_slicewise
from geominr.optim_utils import build_param_groups, build_lr_scheduler
from data.data import build_train_dataloader
from data.data import build_val_dataloader


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def train(
    model,
    muon,
    adamw,
    muon_scheduler,
    adamw_scheduler,
    discriminator,
    discriminator_muon,
    discriminator_adamw,
    discriminator_muon_scheduler,
    discriminator_adamw_scheduler,
    train_dataloader,
    val_dataloader,
    num_epochs: int,
    cfg,
    is_main,
    local_rank,
):
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    train_cfg = cfg.train
    grad_clip = float("inf") if train_cfg.grad_clip < 0 else train_cfg.grad_clip
    os.makedirs(train_cfg.save_dir, exist_ok=True)

    lpips_loss_fn = lpips.LPIPS(net="alex").to(device)
    lpips_loss_fn.eval()
    for param in lpips_loss_fn.parameters():
        param.requires_grad = False

    if cfg.wandb.enabled and is_main:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            dir=cfg.wandb.dir,
            config={
                "run_name": cfg.wandb.name,
                "lr_muon": train_cfg.lr_muon,
                "lr_adamw": train_cfg.lr_adamw,
                "batch_size": train_cfg.batch_size,
                "num_epochs": train_cfg.num_epochs,
                "lpips_weight": train_cfg.lpips_weight,
                "gan_weight": train_cfg.gan_weight,
                "lr_weight": train_cfg.lr_weight,
                "gan_start_epoch": train_cfg.gan_start_epoch,
            },
        )
        wandb.define_metric("train_step", hidden=True)
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("epoch_step", hidden=True)
        wandb.define_metric("epoch/*", step_metric="epoch_step")
        wandb.define_metric("image/*", step_metric="epoch_step")

    for epoch in range(num_epochs):
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        start_time = time.time()
        every_n_steps = train_cfg.every_n_steps

        fake_acc_accum = torch.tensor(0.0, device=device)
        real_acc_accum = torch.tensor(0.0, device=device)
        train_n_accum = torch.tensor(0.0, device=device)

        gan_active = epoch >= train_cfg.gan_start_epoch

        model.train()
        for i, batch in enumerate(train_dataloader):
            slices, angles, arcs = batch
            slices = slices.to(device)
            angles = angles.to(device)
            arcs = arcs.to(device)
            B, _, H, W = slices.shape
            actual_step = epoch * len(train_dataloader) + i + 1

            left_slice = slices[:, 0].unsqueeze(1)
            middle_slice = slices[:, 1].unsqueeze(1)
            right_slice = slices[:, 2].unsqueeze(1)
            conditioning_angles = angles[:, [0, 2]]

            xrtheta = create_xrtheta(B, H, W, angles[:, 1], conditioning_angles, device)
            xrtheta_left = create_xrtheta(
                B, H, W, angles[:, 0], conditioning_angles, device
            )
            xrtheta_right = create_xrtheta(
                B, H, W, angles[:, 2], conditioning_angles, device
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred, left_pred, right_pred = model(
                    left_slice,
                    right_slice,
                    arcs,
                    xrtheta,
                    xrtheta_left,
                    xrtheta_right,
                )
            pred = pred.reshape(B, 1, H, W).float()
            left_pred = left_pred.reshape(B, 1, H, W).float()
            right_pred = right_pred.reshape(B, 1, H, W).float()

            # Discriminator update (only after the GAN warmup period)
            if gan_active:
                discriminator_muon.zero_grad()
                discriminator_adamw.zero_grad()
                pred_copy = pred.float().detach()

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    fake_pred = discriminator(pred_copy)
                    real_pred = discriminator(middle_slice)

                fake_pred = fake_pred.float()
                real_pred = real_pred.float()

                fake_acc = (fake_pred <= 0).float().mean()
                real_acc = (real_pred > 0).float().mean()

                discriminator_loss = (
                    F.relu(1 - real_pred).mean() + F.relu(1 + fake_pred).mean()
                ) / 2
                discriminator_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
                discriminator_muon.step()
                discriminator_adamw.step()
                discriminator_muon_scheduler.step()
                discriminator_adamw_scheduler.step()
            else:
                fake_acc = torch.zeros((), device=device)
                real_acc = torch.zeros((), device=device)

            fake_acc_accum += fake_acc * B
            real_acc_accum += real_acc * B
            train_n_accum += B

            muon.zero_grad()
            adamw.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                lpips_loss = lpips_loss_fn(middle_slice, pred).mean()
                lpips_loss_left = lpips_loss_fn(left_slice, left_pred).mean()
                lpips_loss_right = lpips_loss_fn(right_slice, right_pred).mean()

                if gan_active:
                    discriminator_pred = discriminator(pred).float()
                    discriminator_pred_left = discriminator(left_pred).float()
                    discriminator_pred_right = discriminator(right_pred).float()

            pred = pred.float()
            left_pred = left_pred.float()
            right_pred = right_pred.float()

            recon_loss = F.l1_loss(pred, middle_slice)
            left_recon_loss = F.l1_loss(left_pred, left_slice)
            right_recon_loss = F.l1_loss(right_pred, right_slice)

            if gan_active:
                gan_loss = -1 * discriminator_pred.mean()
                left_gan_loss = -1 * discriminator_pred_left.mean()
                right_gan_loss = -1 * discriminator_pred_right.mean()
            else:
                gan_loss = torch.zeros((), device=device)
                left_gan_loss = torch.zeros((), device=device)
                right_gan_loss = torch.zeros((), device=device)

            loss = (
                recon_loss
                + train_cfg.lpips_weight * lpips_loss
                + train_cfg.gan_weight * gan_loss
            )
            loss_left = (
                left_recon_loss
                + train_cfg.lpips_weight * lpips_loss_left
                + train_cfg.gan_weight * left_gan_loss
            )
            loss_right = (
                right_recon_loss
                + train_cfg.lpips_weight * lpips_loss_right
                + train_cfg.gan_weight * right_gan_loss
            )

            loss_full = loss + train_cfg.lr_weight * (loss_left + loss_right)

            loss_full.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            muon.step()
            adamw.step()
            muon_scheduler.step()
            adamw_scheduler.step()

            if cfg.wandb.enabled and (i + 1) % every_n_steps == 0:
                red_loss = reduce_tensor(loss.detach(), world_size)
                red_recon = reduce_tensor(recon_loss.detach(), world_size)
                red_lpips = reduce_tensor(lpips_loss.detach(), world_size)
                red_gan = reduce_tensor(gan_loss.detach(), world_size)
                red_full = reduce_tensor(loss_full.detach(), world_size)

                red_fake_acc = reduce_tensor(fake_acc, world_size)
                red_real_acc = reduce_tensor(real_acc, world_size)

                if is_main:
                    train_log = {
                        "train/loss": red_loss.item(),
                        "train/recon_loss": red_recon.item(),
                        "train/lpips_loss": red_lpips.item(),
                        "train/gan_loss": red_gan.item(),
                        "train/total_lr_loss": red_full.item(),
                        "train/discriminator_fake_accuracy": red_fake_acc.item(),
                        "train/discriminator_real_accuracy": red_real_acc.item(),
                        "train/lr": adamw_scheduler.get_last_lr()[0],
                        "train_step": actual_step,
                    }
                    wandb.log(train_log)

        val_psnr_accum = torch.tensor(0.0, device=device)
        val_ssim_accum = torch.tensor(0.0, device=device)
        val_linear_psnr_accum = torch.tensor(0.0, device=device)
        val_linear_ssim_accum = torch.tensor(0.0, device=device)
        val_n_accum = torch.tensor(0.0, device=device)

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                slices, angles, arcs = batch
                slices = slices.to(device)
                angles = angles.to(device)
                arcs = arcs.to(device)
                B, _, H, W = slices.shape

                left_slice = slices[:, 0].unsqueeze(1)
                query_slice = slices[:, 1].unsqueeze(1)
                right_slice = slices[:, 2].unsqueeze(1)

                conditioning_angles = angles[:, [0, 2]]
                xrtheta = create_xrtheta(
                    B, H, W, angles[:, 1], conditioning_angles, device
                )

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model(left_slice, right_slice, arcs, xrtheta)

                pred = pred.reshape(B, 1, H, W).float()

                conditioning_slices = torch.stack(
                    [left_slice.squeeze(1), right_slice.squeeze(1)], dim=1
                )
                relative_angle = (angles[:, 1] - angles[:, 0]) / (angles[:, 2] - angles[:, 0])
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
        dist.all_reduce(fake_acc_accum, op=dist.ReduceOp.SUM)
        dist.all_reduce(real_acc_accum, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_n_accum, op=dist.ReduceOp.SUM)

        if cfg.wandb.enabled and is_main:
            psnr_val = (val_psnr_accum / val_n_accum).item()
            ssim_val = (val_ssim_accum / val_n_accum).item()
            linear_psnr_val = (val_linear_psnr_accum / val_n_accum).item()
            linear_ssim_val = (val_linear_ssim_accum / val_n_accum).item()
            delta_psnr = psnr_val - linear_psnr_val
            delta_ssim = ssim_val - linear_ssim_val

            comparison_image = (
                torch.cat(
                    [
                        query_slice[1].permute(1, 2, 0),  # pyrefly:ignore
                        pred[1].permute(1, 2, 0),  # pyrefly:ignore
                        linear_pred[1].permute(1, 2, 0),  # pyrefly:ignore
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
                    "epoch/discriminator_fake_accuracy": (
                        fake_acc_accum / train_n_accum
                    ).item(),
                    "epoch/discriminator_real_accuracy": (
                        real_acc_accum / train_n_accum
                    ).item(),
                    "epoch/time": (time.time() - start_time),
                    "epoch_step": epoch + 1,
                    "image/comparison": wandb.Image(
                        comparison_image,
                        caption="GT, MODEL, LINEAR",
                    ),
                }
            )

        if is_main and (epoch + 1) % train_cfg.save_epoch == 0:
            torch.save(
                {
                    "model": model.module.state_dict(),
                    "muon": muon.state_dict(),
                    "adamw": adamw.state_dict(),
                    "discriminator": discriminator.module.state_dict(),
                    "discriminator_muon": discriminator_muon.state_dict(),
                    "discriminator_adamw": discriminator_adamw.state_dict(),
                    "muon_scheduler": muon_scheduler.state_dict(),
                    "adamw_scheduler": adamw_scheduler.state_dict(),
                    "discriminator_muon_scheduler": discriminator_muon_scheduler.state_dict(),
                    "discriminator_adamw_scheduler": discriminator_adamw_scheduler.state_dict(),
                },
                f"{train_cfg.save_dir}/model_{epoch+1}.pt",
            )


def main():
    cfg = load_config()
    local_rank = setup_ddp()
    global_rank = dist.get_rank()
    is_main = global_rank == 0
    if is_main:
        save_config(cfg, cfg.train.save_dir)
    device = f"cuda:{local_rank}"

    use_probe_radius = bool(getattr(cfg.data, "use_probe_radius", True))
    train_dataloader = build_train_dataloader(
        cfg.data.train_dir,
        cfg.data.patch_size,
        cfg.data.num_workers,
        cfg.data.pin_memory,
        cfg.data.batch_size,
        use_probe_radius=use_probe_radius,
    )
    val_dataloader = build_val_dataloader(
        cfg.data.val_dir,
        cfg.data.patch_size,
        cfg.data.batch_size,
        use_probe_radius=use_probe_radius,
    )

    model = instantiate(cfg.model).to(device)

    # Muon for 2D weight matrices, AdamW for 1D params / biases / norms (the
    # acftyl8r recipe; pure AdamW on the 2D weights diverges at ~epoch 10).
    groups = build_param_groups(
        model,
        lr_adamw=cfg.train.lr_adamw,
        lr_muon=cfg.train.lr_muon,
        wd=cfg.train.weight_decay,
    )
    muon = Muon(groups["muon"], adjust_lr_fn=cfg.train.muon_mode)
    adamw = AdamW(groups["adamw"], betas=tuple(cfg.train.adamw_betas))
    print(
        f"Muon params (2D): {sum(p.numel() for g in groups['muon'] for p in g['params'])}, "
        f"AdamW params: {sum(p.numel() for g in groups['adamw'] for p in g['params'])}"
    )

    model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank])

    discriminator = instantiate(cfg.discriminator).to(device)
    disc_groups = build_param_groups(
        discriminator,
        lr_adamw=cfg.train.discriminator_lr_adamw,
        lr_muon=cfg.train.discriminator_lr_muon,
        wd=cfg.train.discriminator_weight_decay,
    )
    discriminator_muon = Muon(disc_groups["muon"], adjust_lr_fn=cfg.train.muon_mode)
    discriminator_adamw = AdamW(disc_groups["adamw"], betas=tuple(cfg.train.adamw_betas))

    discriminator = torch.compile(discriminator)
    discriminator = DDP(discriminator, device_ids=[local_rank])

    total_steps = cfg.train.num_epochs * len(train_dataloader)
    warmup_steps = cfg.train.warmup_epochs * len(train_dataloader)
    disc_warmup_steps = cfg.train.discriminator_warmup_epochs * len(train_dataloader)

    muon_scheduler = build_lr_scheduler(
        cfg, muon, total_steps, cfg.train.lr_muon, cfg.train.min_lr, warmup_steps
    )
    adamw_scheduler = build_lr_scheduler(
        cfg, adamw, total_steps, cfg.train.lr_adamw, cfg.train.min_lr, warmup_steps
    )
    discriminator_muon_scheduler = build_lr_scheduler(
        cfg,
        discriminator_muon,
        total_steps,
        cfg.train.discriminator_lr_muon,
        cfg.train.discriminator_min_lr,
        disc_warmup_steps,
    )
    discriminator_adamw_scheduler = build_lr_scheduler(
        cfg,
        discriminator_adamw,
        total_steps,
        cfg.train.discriminator_lr_adamw,
        cfg.train.discriminator_min_lr,
        disc_warmup_steps,
    )

    train(
        model,
        muon,
        adamw,
        muon_scheduler,
        adamw_scheduler,
        discriminator,
        discriminator_muon,
        discriminator_adamw,
        discriminator_muon_scheduler,
        discriminator_adamw_scheduler,
        train_dataloader,
        val_dataloader,
        cfg.train.num_epochs,
        cfg,
        is_main,
        local_rank,
    )


if __name__ == "__main__":
    main()