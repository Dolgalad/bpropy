from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def gap_from_solution(
    y_true_sum: float,
    y_pred_sum: float,
    maximize: bool,
    eps: float = 1e-8,
) -> float:
    # matches your usage: divide by y_true_sum, with an eps in the "timed" eval path :contentReference[oaicite:5]{index=5}
    denom = y_true_sum + eps
    if maximize:
        return (y_true_sum - y_pred_sum) / denom
    return (y_pred_sum - y_true_sum) / denom


@dataclass
class EvalStats:
    loss_mean: float
    loss_std: float
    gap_mean: float
    gap_std: float
    time_pred_mean: float
    time_pred_std: float
    time_greedy_mean: float
    time_greedy_std: float
    time_total_mean: float
    time_total_std: float
    time_preprocess_mean: float
    time_preprocess_std: float
    time_end_to_end_mean: float
    time_end_to_end_std: float
    count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loss": {"mean": self.loss_mean, "std": self.loss_std},
            "gap": {"mean": self.gap_mean, "std": self.gap_std},
            "time_pred_s": {"mean": self.time_pred_mean, "std": self.time_pred_std},
            "time_greedy_s": {"mean": self.time_greedy_mean, "std": self.time_greedy_std},
            "time_total_s": {"mean": self.time_total_mean, "std": self.time_total_std},
            "time_preprocess_s": {"mean": self.time_preprocess_mean, "std": self.time_preprocess_std},
            "time_end_to_end_s": {"mean": self.time_end_to_end_mean, "std": self.time_end_to_end_std},

            "count": self.count,
        }


def _mean_std(x: List[float]) -> Tuple[float, float]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def write_test_eval_json(log_dir: str, stats: EvalStats, filename: str = "test_eval.json") -> str:
    path = os.path.join(log_dir, filename)
    with open(path, "w") as f:
        json.dump(stats.to_dict(), f, indent=2, sort_keys=True)
    return path

