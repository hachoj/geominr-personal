import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

SPACING = 0.0335860058309
PROBE_RADIUS = 12.5


def _build_item_index(data_directory_path, desc="indexing dataset", max_workers=None):
    with os.scandir(data_directory_path) as scan_it:
        npz_paths = sorted(
            entry.path
            for entry in scan_it
            if entry.is_file() and entry.name.endswith(".npz")
        )

    if not npz_paths:
        return []

    def _load_angles(path):
        with np.load(path, allow_pickle=False) as f:
            return {"path": path, "angles": f["angles"]}

    if max_workers is None:
        env_override = os.getenv("MICROUS_INDEX_WORKERS")
        if env_override:
            try:
                max_workers = max(1, int(env_override))
            except ValueError:
                max_workers = None

    max_workers = max_workers or min(8, os.cpu_count() or 1)
    if max_workers <= 1:
        iterator = (_load_angles(path) for path in npz_paths)
        return list(tqdm(iterator, total=len(npz_paths), desc=desc))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        iterator = executor.map(_load_angles, npz_paths)
        return list(tqdm(iterator, total=len(npz_paths), desc=desc))


class MicroUSTrain(Dataset):
    def __init__(
        self,
        data_directory_path,
        patch_size,
        mode: str = "train",
        use_probe_radius: bool = True,
    ):
        self.items = _build_item_index(data_directory_path)
        self.patch_size = patch_size  # (ph, pw)
        self.mode = mode
        self.use_probe_radius = use_probe_radius

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        path = rec["path"]
        angles = rec["angles"]
        angles = angles * torch.pi / 180.0

        # lazy load the slices + angles
        with np.load(path, allow_pickle=False) as f:
            slices = torch.from_numpy(f["slices"]).float() / 255.0
            angles = torch.from_numpy(angles).float()

        ph, pw = self.patch_size
        _, H, W = slices.shape
        assert ph <= H and pw <= W, "patch size > image size"
        h, w = _rand_crop2d_content_aware(slices, ph, pw, mode=self.mode)
        slices = slices[:, h : h + ph, w : w + pw]

        delta_angle = abs(angles[0] - angles[2])
        radii = torch.arange(h, h + ph) * SPACING + (PROBE_RADIUS if self.use_probe_radius else 0.0)  # radius from rotation axis; probe-radius offset dropped when use_probe_radius=False (-r_probe ablation)
        arcs = delta_angle * radii  # [ph]

        return slices, angles, arcs


def _worker_init_fn(_):
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _rand_crop2d_content_aware(
    slices,
    ph,
    pw,
    mode: str = "train",
    seed: int | None = None,
    jitter_h: int = 128,
    jitter_w: int = 128,
    attempt_count: int = 0,
):
    # Validity checks should operate in [0, 1] intensity space.
    # Training/eval tensors remain in [-1, 1] for model consumption.
    slices_check = torch.clamp((slices + 1.0) * 0.5, 0.0, 1.0)

    H = int(slices.shape[-2])
    W = int(slices.shape[-1])
    if mode == "train":
        h = 0 if H == ph else np.random.randint(0, H - ph + 1)
        w = 0 if W == pw else np.random.randint(0, W - pw + 1)
        if slices_check[:, h, w : w + pw].sum() == 0:
            return _rand_crop2d_content_aware(slices, ph, pw, mode="train")
        if slices_check[:, h + ph - 1, w : w + pw].sum() == 0:
            return _rand_crop2d_content_aware(slices, ph, pw, mode="train")
        if slices_check[:, h : h + ph, w].sum() == 0:
            return _rand_crop2d_content_aware(slices, ph, pw, mode="train")
        if slices_check[:, h : h + ph, w + pw - 1].sum() == 0:
            return _rand_crop2d_content_aware(slices, ph, pw, mode="train")
    elif mode == "val":
        # during val, it's just do center crop
        h = 0 if H == ph else int(H / 2) - int(ph / 2)
        w = 0 if W == pw else int(W / 2) - int(pw / 2)
        return int(h), int(w)
    elif mode == "test":
        base_w = 0 if W == pw else int(W / 2) - int(pw / 2)
        base_h = 0 if H == ph else min(int(H * 0.60) - int(ph / 2), H - ph)

        if seed is not None:
            rng = np.random.RandomState(seed + attempt_count)
            jitter_height = rng.randint(-jitter_h, jitter_h + 1) if H > ph else 0
            jitter_width = rng.randint(-jitter_w, jitter_w + 1) if W > pw else 0
            attempt_count += 1

            h = max(0, min(base_h + jitter_height, H - ph))
            w = max(0, min(base_w + jitter_width, W - pw))
            if slices_check[:, h, w : w + pw].sum() == 0:
                return _rand_crop2d_content_aware(
                    slices,
                    ph,
                    pw,
                    mode="test",
                    seed=seed,
                    jitter_h=jitter_h,
                    jitter_w=jitter_w,
                    attempt_count=attempt_count,
                )
            if slices_check[:, h + ph - 1, w : w + pw].sum() == 0:
                return _rand_crop2d_content_aware(
                    slices,
                    ph,
                    pw,
                    mode="test",
                    seed=seed,
                    jitter_h=jitter_h,
                    jitter_w=jitter_w,
                    attempt_count=attempt_count,
                )
            if slices_check[:, h : h + ph, w].sum() == 0:
                return _rand_crop2d_content_aware(
                    slices,
                    ph,
                    pw,
                    mode="test",
                    seed=seed,
                    jitter_h=jitter_h,
                    jitter_w=jitter_w,
                    attempt_count=attempt_count,
                )
            if slices_check[:, h : h + ph, w + pw - 1].sum() == 0:
                return _rand_crop2d_content_aware(
                    slices,
                    ph,
                    pw,
                    mode="test",
                    seed=seed,
                    jitter_h=jitter_h,
                    jitter_w=jitter_w,
                    attempt_count=attempt_count,
                )
            if slices_check[:, h : h + ph, w : w + pw].sum() < 500:
                return _rand_crop2d_content_aware(
                    slices,
                    ph,
                    pw,
                    mode="test",
                    seed=seed,
                    jitter_h=jitter_h,
                    jitter_w=jitter_w,
                    attempt_count=attempt_count,
                )
        else:
            h = base_h
            w = base_w

    return int(h), int(w)


def build_train_dataloader(
    data_dir,
    patch_size,
    num_workers,
    pin_memory,
    batch_size,
    mode: str = "train",
    prefetch_factor=4,
    persistent_workers=True,
    use_probe_radius: bool = True,
):
    dataset = MicroUSTrain(data_dir, patch_size, mode, use_probe_radius=use_probe_radius)
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        **loader_kwargs,
    )


class MicroUSVal(Dataset):
    def __init__(
        self,
        data_directory_path,
        patch_size,
        mode: str = "val",
        seed=42,
        use_probe_radius: bool = True,
        jitter_h: int = 128,
        jitter_w: int = 128,
    ):
        self.items = _build_item_index(data_directory_path)
        self.patch_size = patch_size  # (ph, pw)
        self.seed = seed
        self.mode = mode
        self.use_probe_radius = use_probe_radius
        self.jitter_h = jitter_h
        self.jitter_w = jitter_w

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        path = rec["path"]
        angles = rec["angles"]
        angles = angles * torch.pi / 180.0

        # lazy load the slices + angles
        with np.load(path, allow_pickle=False) as f:
            slices = torch.from_numpy(f["slices"]).float() / 255.0
            angles = torch.from_numpy(angles).float()

        ph, pw = self.patch_size
        _, H, W = slices.shape
        assert ph <= H and pw <= W, "patch size > image size"
        h, w = _rand_crop2d_content_aware(
            slices, ph, pw, mode=self.mode, seed=self.seed + idx,
            jitter_h=self.jitter_h, jitter_w=self.jitter_w,
        )
        slices = slices[:, h : h + ph, w : w + pw]

        delta_angle = abs(angles[0] - angles[2])
        radii = torch.arange(h, h + ph) * SPACING + (PROBE_RADIUS if self.use_probe_radius else 0.0)  # radius from rotation axis; probe-radius offset dropped when use_probe_radius=False (-r_probe ablation)
        arcs = delta_angle * radii  # [ph]

        return slices, angles, arcs


def build_val_dataloader(
    data_dir,
    patch_size,
    batch_size,
    mode: str = "val",
    seed=42,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    use_probe_radius: bool = True,
    jitter_h: int = 128,
    jitter_w: int = 128,
):
    dataset = MicroUSVal(data_dir, patch_size, mode, seed, use_probe_radius=use_probe_radius,
                         jitter_h=jitter_h, jitter_w=jitter_w)
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
        **loader_kwargs,
    )


class MicroUSTestLazy(Dataset):
    def __init__(
        self,
        data_directory_path,
    ):
        self.items = _build_item_index(data_directory_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        # Return path and angles, NOT the full image
        return rec["path"], torch.from_numpy(rec["angles"]).float()


def build_test_dataloader_lazy(
    data_dir,
    batch_size,
    num_workers=0,
    pin_memory=False,
):
    dataset = MicroUSTestLazy(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
