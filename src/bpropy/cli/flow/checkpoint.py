from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from .schedulers import _get_scheduler_state_dict, _set_scheduler_state_dict

@dataclass
class EpochState:
    epoch: int = 1
    best_gap: float = float("inf")
    best_epoch: int = 1
    epochs_bad: int = 0


def save_checkpoint(
    path: str,
    state: EpochState,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ckpt: Dict[str, Any] = {
        "epoch": int(state.epoch),
        "best_gap": float(state.best_gap),
        "best_epoch": int(state.best_epoch),
        "epochs_bad": int(state.epochs_bad),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": _get_scheduler_state_dict(scheduler),
    }
    if extra:
        ckpt.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
) -> Tuple[EpochState, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        _set_scheduler_state_dict(scheduler, ckpt["scheduler_state_dict"])

    st = EpochState()
    st.epoch = int(ckpt.get("epoch", 0)) + 1
    st.best_gap = float(ckpt.get("best_gap", float("inf")))
    st.best_epoch = int(ckpt.get("best_epoch", 1))
    st.epochs_bad = int(ckpt.get("epochs_bad", 0))
    return st, ckpt

