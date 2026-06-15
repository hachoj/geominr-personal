"""Lightweight config loading and object instantiation.

Replaces Hydra: each run is described by a single self-contained YAML under
``configs/runs/``. Scripts call :func:`load_config` to parse ``--config`` (plus
optional ``--set`` dotlist overrides) and :func:`instantiate` to build objects
from a ``_target_`` key, mirroring ``hydra.utils.instantiate`` without the
dependency.
"""

import argparse
import importlib
from pathlib import Path

from omegaconf import OmegaConf


def instantiate(cfg):
    """Build an object from a config node carrying a dotted ``_target_`` path.

    ``{_target_: pkg.mod.Cls, a: 1, b: 2}`` -> ``pkg.mod.Cls(a=1, b=2)``.
    """
    params = OmegaConf.to_container(cfg, resolve=True)
    target = params.pop("_target_")
    module_name, _, class_name = target.rpartition(".")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**params)


def load_config(argv=None):
    """Parse ``--config <path>`` (+ optional ``--set k=v ...``) and return the
    run config as an OmegaConf ``DictConfig``.

    Example::

        python scripts/train.py --config configs/runs/canonical.yaml \\
            --set train.num_epochs=50 wandb.enabled=false
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a run config YAML")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="dotlist overrides, e.g. train.lr=5e-4 wandb.enabled=false",
    )
    args = parser.parse_args(argv)
    cfg = OmegaConf.load(args.config)
    if args.set:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.set))
    return cfg


def save_config(cfg, out_dir, name="config.yaml"):
    """Snapshot the resolved run config into ``out_dir`` for provenance
    (replaces Hydra's ``.hydra/config.yaml``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / name)
