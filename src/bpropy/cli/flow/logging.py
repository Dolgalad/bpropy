from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


@dataclass
class RunPaths:
    run_name: str
    log_dir: str
    ckpt_dir: str


def make_run_paths(cfg: Dict[str, Any]) -> RunPaths:
    log_root = cfg["logging"]["log_dir"]
    model_name = cfg["model"]["name"]
    seed = int(cfg.get("seed", 0))
    ts = int(time.time())
    run_name = cfg.get("run", {}).get("name") or f"{model_name}_{ts}_{seed}"
    log_dir = os.path.join(log_root, run_name)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return RunPaths(run_name=run_name, log_dir=log_dir, ckpt_dir=ckpt_dir)


class TrainLogger:
    def __init__(self, paths: RunPaths, cfg: Dict[str, Any]) -> None:
        self.paths = paths
        self.cfg = cfg
        self.writer: Optional[SummaryWriter] = None
        if cfg["logging"].get("tensorboard", True):
            self.writer = SummaryWriter(paths.log_dir)

    def write_hparams(self, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {"resolved_config": self.cfg}
        if extra:
            payload.update(extra)
        with open(os.path.join(self.paths.log_dir, "hparams.json"), "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def add_scalars(self, tag: str, scalars: Dict[str, float], step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalars(tag, scalars, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

