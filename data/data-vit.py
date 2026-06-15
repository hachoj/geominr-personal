import io
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

SPACING = 0.0335860058309
PROBE_RADIUS = 12.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_sharding() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _get_device_id() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))
    return 0


def _default_seed() -> int:
    return int(torch.initial_seed() % 2_147_483_647)


def _coerce_patch_size(patch_size) -> tuple[int, int]:
    if isinstance(patch_size, int):
        return patch_size, patch_size
    try:
        if len(patch_size) == 2:
            return int(patch_size[0]), int(patch_size[1])
    except Exception:
        pass
    raise ValueError(f"Invalid patch_size: {patch_size!r}")


def _resolve_shards(split_dir: str | Path) -> tuple[list[str], list[str]]:
    split_path = Path(split_dir)
    if not split_path.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_path}")

    tar_paths = sorted(split_path.glob("*.tar"))
    idx_by_stem = {path.stem: path for path in split_path.glob("*.idx")}

    if not tar_paths:
        raise RuntimeError(f"No .tar shards found in {split_path}")

    missing_idx = [tar.name for tar in tar_paths if tar.stem not in idx_by_stem]
    extra_idx = sorted(stem for stem in idx_by_stem if stem not in {tar.stem for tar in tar_paths})
    if missing_idx or extra_idx:
        raise RuntimeError(
            "Shard/index mismatch in {}. Missing idx for: {}. Extra idx stems: {}".format(
                split_path,
                missing_idx,
                extra_idx,
            )
        )

    tar_str = [str(path) for path in tar_paths]
    idx_str = [str(idx_by_stem[path.stem]) for path in tar_paths]
    return tar_str, idx_str


def _parse_npz(npz_bytes):
    with np.load(io.BytesIO(npz_bytes.tobytes()), allow_pickle=False) as f:
        slices = f["slices"].astype(np.float32)
        angles = f["angles"].astype(np.float32)

    slices = (slices / 127.5) - 1.0
    angles = angles * np.pi / 180.0
    return slices, angles


# ---------------------------------------------------------------------------
# DALI pipelines
# ---------------------------------------------------------------------------


@pipeline_def
def _train_pipeline(
    tar_paths,
    idx_paths,
    patch_size,
    shard_id,
    num_shards,
    reader_seed,
    aug_seed,
):
    npz_raw = fn.readers.webdataset(
        paths=tar_paths,
        index_paths=idx_paths,
        ext=["npz"],
        random_shuffle=True,
        initial_fill=4096,
        seed=reader_seed,
        shard_id=shard_id,
        num_shards=num_shards,
        name="Reader",
        missing_component_behavior="error",
    )

    slices, angles = fn.python_function(npz_raw, function=_parse_npz, num_outputs=2)
    slices = fn.cast(slices, dtype=types.FLOAT)
    angles = fn.cast(angles, dtype=types.FLOAT)

    # Shards are pre-cropped offline; avoid per-sample crop python callbacks.
    del patch_size, aug_seed
    slices = slices.gpu()
    angles = angles.gpu()

    return slices, angles


@pipeline_def
def _val_pipeline(
    tar_paths,
    idx_paths,
    patch_size,
    shard_id,
    num_shards,
    reader_seed,
):
    npz_raw = fn.readers.webdataset(
        paths=tar_paths,
        index_paths=idx_paths,
        ext=["npz"],
        random_shuffle=False,
        seed=reader_seed,
        shard_id=shard_id,
        num_shards=num_shards,
        name="Reader",
        missing_component_behavior="error",
    )

    slices, angles = fn.python_function(npz_raw, function=_parse_npz, num_outputs=2)
    slices = fn.cast(slices, dtype=types.FLOAT)
    angles = fn.cast(angles, dtype=types.FLOAT)

    del patch_size
    slices = slices.gpu()
    angles = angles.gpu()

    return slices, angles


# ---------------------------------------------------------------------------
# DALI wrapper
# ---------------------------------------------------------------------------


class DALILoader:
    def __init__(self, dali_iterator):
        self._iterator = dali_iterator
        self.sampler = None

    def __iter__(self):
        for batch in self._iterator:
            payload = batch[0]
            slices = payload["slices"].float()
            angles = payload["angles"].float()
            yield slices, angles

    def __len__(self):
        return len(self._iterator)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_loader(
    pipeline_fn,
    tar_paths: Iterable[str],
    idx_paths: Iterable[str],
    batch_size: int,
    patch_size: tuple[int, int],
    is_train: bool,
    num_threads: int,
    prefetch_queue_depth: int,
    seed: int,
):
    shard_id, num_shards = _get_sharding()
    device_id = _get_device_id()
    reader_seed = int(seed) + shard_id
    aug_seed = int(seed) + 10_000 + shard_id

    pipe_kwargs = dict(
        tar_paths=list(tar_paths),
        idx_paths=list(idx_paths),
        patch_size=list(patch_size),
        shard_id=shard_id,
        num_shards=num_shards,
        batch_size=int(batch_size),
        num_threads=max(1, int(num_threads)),
        prefetch_queue_depth=max(2, int(prefetch_queue_depth)),
        device_id=device_id,
        seed=reader_seed,
        reader_seed=reader_seed,
    )
    if is_train:
        pipe_kwargs["aug_seed"] = aug_seed

    pipe = pipeline_fn(**pipe_kwargs)
    pipe.build()

    iterator = DALIGenericIterator(
        pipe,
        ["slices", "angles"],
        reader_name="Reader",
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True,
    )
    return DALILoader(iterator)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_train_dataloader(
    data_dir,
    patch_size,
    num_workers,
    pin_memory,
    batch_size,
    mode: str = "train",
    prefetch_factor=4,
    persistent_workers=True,
):
    del pin_memory, mode, persistent_workers

    patch_hw = _coerce_patch_size(patch_size)
    tar_paths, idx_paths = _resolve_shards(data_dir)
    return _build_loader(
        _train_pipeline,
        tar_paths,
        idx_paths,
        batch_size=batch_size,
        patch_size=patch_hw,
        is_train=True,
        num_threads=max(4, int(num_workers) * 2),
        prefetch_queue_depth=max(2, int(prefetch_factor)),
        seed=_default_seed(),
    )


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
):
    del mode, pin_memory, persistent_workers

    patch_hw = _coerce_patch_size(patch_size)
    tar_paths, idx_paths = _resolve_shards(data_dir)
    return _build_loader(
        _val_pipeline,
        tar_paths,
        idx_paths,
        batch_size=batch_size,
        patch_size=patch_hw,
        is_train=False,
        num_threads=max(2, int(num_workers) * 2),
        prefetch_queue_depth=max(2, int(prefetch_factor)),
        seed=int(seed),
    )


def build_test_dataloader_lazy(
    data_dir,
    batch_size,
    num_workers=0,
    pin_memory=False,
):
    from data.data_old import (
        build_test_dataloader_lazy as _build_test_dataloader_lazy_legacy,
    )

    return _build_test_dataloader_lazy_legacy(
        data_dir,
        batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
