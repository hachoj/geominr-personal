from __future__ import annotations

import importlib
import sys
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

REQUIRED_IMPORTS = [
    "geominr.config",
    "geominr.optim_utils",
    "geominr.models.model",
    "geominr.models.discriminator",
    "geominr.models.utils.metrics",
    "geominr.models.utils.reconstruct",
    "geominr.baselines.implicitvol.model",
    "geominr.baselines.ultranerf.model",
]


def _resolve_target(target: str) -> None:
    module_name, _, attr_name = target.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid _target_ value: {target}")
    module = importlib.import_module(module_name)
    getattr(module, attr_name)


def main() -> None:
    for module_name in REQUIRED_IMPORTS:
        importlib.import_module(module_name)

    config_paths = sorted(Path("configs/runs").glob("*.yaml"))
    if not config_paths:
        raise RuntimeError("No run configs found under configs/runs.")

    for path in config_paths:
        cfg = OmegaConf.load(path)
        if "model" not in cfg or "_target_" not in cfg.model:
            raise ValueError(f"Missing model._target_ in {path}")
        _resolve_target(str(cfg.model._target_))
        if "discriminator" in cfg and "_target_" in cfg.discriminator:
            _resolve_target(str(cfg.discriminator._target_))

    print(f"Smoke check passed: {len(config_paths)} run configs validated.")


if __name__ == "__main__":
    main()
