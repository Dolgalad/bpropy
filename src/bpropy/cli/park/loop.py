from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from progress.bar import Bar

from bpropy.datasets.prefetcher import CUDAPrefetcher

from bpropy.cli.train.data import make_loaders
from bpropy.cli.train.logging import TrainLogger, make_run_paths
from bpropy.cli.train.metrics import EvalStats, _sync, gap_from_solution, write_test_eval_json
from bpropy.cli.train.schedulers import build_scheduler, _apply_lr_schedulers

from bpropy.mrf.lbp import loopy_belief_propagation
from bpropy.mrf.naive_mean_field import naive_mean_field
from bpropy.mrf.canonical_parameters import canonical_parameters

from bpropy.gd.greedy_decimation import greedy_decimation

# ---- Method registry ----
from bpropy.cli.train.method_registry import METHOD_REGISTRY


def objective_value(pb, sol: np.ndarray) -> float:
    """Objective value for a problem solution.

    Prefer the problem weights/objective when available. Fall back to the
    number of selected variables to keep the old unweighted behavior valid.
    """
    if isinstance(sol, torch.Tensor):
        s = sol.detach().cpu().numpy()
    elif hasattr(sol, "x"):
        s = np.asarray(sol.x, dtype=np.float64)
    else:
        s = np.asarray(sol, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)

    if hasattr(pb, "w"):
        return float(np.dot(np.asarray(pb.w, dtype=np.float64), s))
    if hasattr(pb, "objective"):
        return float(pb.objective(s))
    return float(s.sum())


def relative_objective_gap(
    obj_true: float,
    obj_pred: float,
    maximize: bool,
    eps: float = 1e-8,
) -> float:
    denom = max(abs(obj_true), eps)
    if maximize:
        return (obj_true - obj_pred) / denom
    return (obj_pred - obj_true) / denom


def _solution_array(sol: Any) -> np.ndarray:
    if isinstance(sol, torch.Tensor):
        return sol.detach().cpu().numpy()
    if hasattr(sol, "x"):
        return np.asarray(sol.x)
    return np.asarray(sol)


def _ensure_greedy_decimation_cfg(cfg: Dict[str, Any]) -> None:
    cfg.setdefault("eval", {})
    cfg["eval"].setdefault("greedy_decimation", {})
    cfg["eval"]["greedy_decimation"].setdefault("args", {})
    cfg["eval"]["greedy_decimation"].setdefault(
        "adapter", {"pick_mode": "min_entropy", "value_mode": "argmax"}
    )
    cfg["eval"]["greedy_decimation"]["adapter"].setdefault("pick_mode", "min_entropy")
    cfg["eval"]["greedy_decimation"]["adapter"].setdefault("value_mode", "argmax")

def _park_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    park = cfg.setdefault("park", {})

    # Global defaults
    park.setdefault("M", 10.0)
    park.setdefault("inference", "loopy_belief_propagation")
    park.setdefault("add_unaries", True)
    park.setdefault("unary_bias", 0.0)
    park.setdefault("tol", 0.0)
    park.setdefault("bp_tol", 1e-6)
    park.setdefault("neg_inf", -100.0)

    # Shared default inference params
    park.setdefault("max_iter", 10)
    park.setdefault("damping", 0.5)

    # Method-specific overrides
    park.setdefault("egg", {})
    park["egg"].setdefault("max_iter", park["max_iter"])
    park["egg"].setdefault("damping", park["damping"])

    park.setdefault("gd", {})
    park["gd"].setdefault("max_iter", park["max_iter"])
    park["gd"].setdefault("damping", park["damping"])
    park["gd"].setdefault("run_every", 1)

    return park


def _run_park_inference(
    *,
    pb,
    data,
    cfg: Dict[str, Any],
    device: torch.device,
    max_iter: int,
    damping: float,
):
    park = _park_cfg(cfg)

    dtype = torch.float32
    theta = canonical_parameters(
        pb,
        M=float(park["M"]),
        device=device,
        dtype=dtype,
        add_unaries=bool(park["add_unaries"]),
        unary_bias=float(park["unary_bias"]),
        tol=float(park["tol"]),
    )

    inference = str(park["inference"])

    if inference == "loopy_belief_propagation":
        if not hasattr(data, "lbp_edges"):
            raise AttributeError(
                "Park LBP inference requested, but data has no lbp_edges. "
                "Set data.precompute.lbp=true in the config."
            )

        n_msg = int(data.lbp_edges.n_f2v_msg.sum().item())
        v2f = torch.zeros(n_msg, device=device, dtype=theta.dtype)

        res = loopy_belief_propagation(
            theta,
            v2f,
            data.lbp_edges,
            max_iter=int(max_iter),
            tol=float(park["bp_tol"]),
            damping=float(damping),
        )

        mu = torch.nan_to_num(res.mu_v.reshape(-1, 2), nan=0.0)
        pred = (res, theta)
        return pred, mu, theta

    if inference == "naive_mean_field":
        if not hasattr(data, "nmf_edges"):
            raise AttributeError(
                "Park NMF inference requested, but data has no nmf_edges. "
                "Set data.precompute.nmf=true in the config."
            )

        sol = naive_mean_field(
            theta,
            data.nmf_edges,
            max_iter=int(max_iter),
            tol=float(park["bp_tol"]),
            damping=float(damping),
        )

        mu = torch.nan_to_num(sol.mu.reshape(-1, 2), nan=0.0)
        pred = (sol, theta)
        return pred, mu, theta

    raise ValueError(f"Unknown park.inference={inference!r}")


def ms(x):
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return 0.0, 0.0
    if a.size == 1:
        return float(a.mean()), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def summarize_method(d: Dict[str, list]) -> Dict[str, float]:
    out = {}
    for k, vals in d.items():
        mean, std = ms(vals)
        out[f"{k}_mean"] = mean
        out[f"{k}_std"] = std
    return out


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _build_test_payload(
    *,
    status: str,
    instance_stats: list,
    summary_lists: Dict[str, Dict[str, list]],
    n_ss: list,
) -> Dict[str, Any]:
    n_ss_mean, n_ss_std = ms(n_ss)
    return {
        "summary": {
            "count": len(instance_stats),
            "status": status,
            "methods": {
                "greedy_pred": summarize_method(summary_lists["greedy_pred"]),
                "entropy_greedy": summarize_method(summary_lists["entropy_greedy"]),
                "guided_decimation": summarize_method(summary_lists["guided_decimation"]),
            },
            "num_ss_mean": n_ss_mean,
            "num_ss_std": n_ss_std,
        },
        "instances": instance_stats,
    }


def _run_best_of_n_decimation(
    *,
    pb,
    backend,
    padapter,
    n_samples: int,
    maximize: bool,
) -> np.ndarray:
    best_val = None
    best_sol = None
    for _ in range(max(1, int(n_samples))):
        sol = greedy_decimation(pb, backend, padapter)
        sol_arr = _solution_array(sol)
        val = objective_value(pb, sol_arr)
        if best_val is None or (maximize and val > best_val) or ((not maximize) and val < best_val):
            best_val = val
            best_sol = sol_arr
    return np.asarray(best_sol)

def final_test_eval(
    cfg: Dict[str, Any],
    adapter,
    loader,
    device: torch.device,
    log_dir: str,
    output_json: Optional[str] = None,
) -> str:
    """Final test evaluation with per-instance JSON logging.

    Logged solve methods:
      1. greedy_pred: greedy projection guided by one forward pass scores.
      2. entropy_greedy: greedy_decimation with min_entropy selection and
         run_every = infinity, so inference is performed only once.
      3. guided_decimation: greedy_decimation with min_entropy selection and
         run_every = 1, so marginals/scores are refreshed after each fixing step.

    The JSON schema mirrors the flow test evaluator: it writes a running JSON
    after each instance and a completed JSON at the end.
    """

    n_ss = []
    maximize = bool(cfg.get("problem", {}).get("maximize", True))
    test_limit = int(cfg.get("eval", {}).get("test_limit", 20000))

    run_cfg = cfg.setdefault("run", {})
    base = run_cfg.get("run_name") or run_cfg.get("name") or cfg.get("run_name") or "run"
    seed = int(cfg.get("seed", 0))
    unique_name = f"{base}_seed{seed}"
    run_cfg["run_name"] = unique_name
    run_cfg["name"] = unique_name
    cfg["run_name"] = unique_name

    if output_json is not None:
        out_path = output_json
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    else:
        paths = make_run_paths(cfg)
        log_dir = paths.log_dir
        os.makedirs(log_dir, exist_ok=True)
        out_path = os.path.join(log_dir, "test_eval.json")

    print("IN TEST logdir = ", os.path.dirname(out_path) or log_dir)

    _ensure_greedy_decimation_cfg(cfg)
    gd_cfg = cfg["eval"]["greedy_decimation"]
    n_samples = int(gd_cfg.get("adapter", {}).get("n_samples", 1))

    instance_stats = []
    summary_lists = {
        "greedy_pred": {"obj": [], "gap": [], "forward_t": [], "decode_t": [], "total_t": []},
        "entropy_greedy": {"obj": [], "gap": [], "forward_t": [], "decode_t": [], "total_t": []},
        "guided_decimation": {"obj": [], "gap": [], "forward_t": [], "decode_t": [], "total_t": []},
    }

    with torch.no_grad():
        bar = Bar("[Test]          ", max=len(loader))

        for idx, (pb, data, y) in enumerate(CUDAPrefetcher(loader, device)):
            pb0 = pb[0] if isinstance(pb, (list, tuple)) else pb
            stem = None
            if hasattr(loader, "dataset") and hasattr(loader.dataset, "stems"):
                try:
                    stem = Path(loader.dataset.stems[idx]).name
                except Exception:
                    stem = None

            pre_t = float(getattr(data, "preprocess_time_s", 0.0))
            ref_val = objective_value(pb0, y)

            # ----- shared Park inference pass -----
            park = _park_cfg(cfg)
            egg_max_iter = int(park["egg"]["max_iter"])
            egg_damping = float(park["egg"]["damping"])

            _sync(device)
            t0 = time.perf_counter()
            pred, mu, theta = _run_park_inference(
                pb=pb0,
                data=data,
                cfg=cfg,
                device=device,
                max_iter=egg_max_iter,
                damping=egg_damping,
            )
            _sync(device)
            t1 = time.perf_counter()
            shared_forward_t = t1 - t0

            # Direct greedy uses exactly the Park marginal P(x_j = 1).
            scores_np = mu[:, 1].detach().cpu().numpy()

            # ----- greedy_pred -----
            _sync(device)
            t2 = time.perf_counter()
            greedy_pred_sol = pb0.greedy(scores=scores_np).x
            _sync(device)
            t3 = time.perf_counter()
            greedy_pred_decode_t = t3 - t2
            greedy_pred_total_t = pre_t + shared_forward_t + greedy_pred_decode_t

            # ----- entropy_greedy: min_entropy, run_every = infinity -----
            # ----- entropy_greedy: min_entropy, run_every = infinity -----
            park = _park_cfg(cfg)
            gd_cfg["args"]["run_every"] = 1000000
            gd_cfg["adapter"]["pick_mode"] = "min_entropy"
            gd_cfg["adapter"]["value_mode"] = "argmax"
            gd_cfg["args"]["max_iter"] = int(park["egg"]["max_iter"])
            gd_cfg["args"]["damping"] = float(park["egg"]["damping"])
            gd_cfg["args"]["neg_inf"] = float(park["neg_inf"])

            _sync(device)
            t4 = time.perf_counter()
            backend = adapter.decimation_backend_fn(pb0, data, pred, cfg, device=device)
            padapter = adapter.decimation_adapter_fn(pb0, cfg)
            entropy_greedy_sol = _run_best_of_n_decimation(
                pb=pb0,
                backend=backend,
                padapter=padapter,
                n_samples=n_samples,
                maximize=maximize,
            )
            _sync(device)
            t5 = time.perf_counter()
            entropy_greedy_decode_t = t5 - t4
            entropy_greedy_total_t = pre_t + shared_forward_t + entropy_greedy_decode_t

            # ----- guided_decimation: min_entropy, run_every = 1 -----
            # ----- guided_decimation: min_entropy, configurable run_every -----
            park = _park_cfg(cfg)
            gd_run_every = int(park["gd"].get("run_every", 1))

            gd_cfg["args"]["run_every"] = gd_run_every
            gd_cfg["adapter"]["pick_mode"] = "min_entropy"
            gd_cfg["adapter"]["value_mode"] = "argmax"
            gd_cfg["args"]["max_iter"] = int(park["gd"]["max_iter"])
            gd_cfg["args"]["damping"] = float(park["gd"]["damping"])
            gd_cfg["args"]["neg_inf"] = float(park["neg_inf"])

            _sync(device)
            t6 = time.perf_counter()
            backend = adapter.decimation_backend_fn(pb0, data, pred, cfg, device=device)
            padapter = adapter.decimation_adapter_fn(pb0, cfg)
            guided_decimation_sol = _run_best_of_n_decimation(
                pb=pb0,
                backend=backend,
                padapter=padapter,
                n_samples=n_samples,
                maximize=maximize,
            )
            _sync(device)
            t7 = time.perf_counter()
            guided_decimation_decode_t = t7 - t6
            guided_decimation_total_t = pre_t + shared_forward_t + guided_decimation_decode_t

            if hasattr(data, "lbp_edges"):
                n_ss += data.lbp_edges.n_ss.detach().cpu().tolist()
            if hasattr(data, "nmf_edges"):
                n_ss += data.nmf_edges.n_ss.detach().cpu().tolist()

            greedy_pred_val = objective_value(pb0, greedy_pred_sol)
            entropy_greedy_val = objective_value(pb0, entropy_greedy_sol)
            guided_decimation_val = objective_value(pb0, guided_decimation_sol)

            greedy_pred_gap = relative_objective_gap(ref_val, greedy_pred_val, maximize=maximize)
            entropy_greedy_gap = relative_objective_gap(ref_val, entropy_greedy_val, maximize=maximize)
            guided_decimation_gap = relative_objective_gap(ref_val, guided_decimation_val, maximize=maximize)

            instance_record = {
                "instance_index": idx,
                "instance_stem": stem,
                "reference_objective": float(ref_val),
                "preprocess_time_s": float(pre_t),
                "greedy_pred": {
                    "objective": float(greedy_pred_val),
                    "gap": float(greedy_pred_gap),
                    "forward_time_s": float(shared_forward_t),
                    "decode_time_s": float(greedy_pred_decode_t),
                    "total_time_s": float(greedy_pred_total_t),
                },
                "entropy_greedy": {
                    "objective": float(entropy_greedy_val),
                    "gap": float(entropy_greedy_gap),
                    "forward_time_s": float(shared_forward_t),
                    "decode_time_s": float(entropy_greedy_decode_t),
                    "total_time_s": float(entropy_greedy_total_t),
                    "run_every": 1000000,
                    "pick_mode": "min_entropy",
                    "value_mode": "argmax",
                },
                "guided_decimation": {
                    "objective": float(guided_decimation_val),
                    "gap": float(guided_decimation_gap),
                    "forward_time_s": float(shared_forward_t),
                    "decode_time_s": float(guided_decimation_decode_t),
                    "total_time_s": float(guided_decimation_total_t),
                    "run_every": gd_run_every,
                    "pick_mode": "min_entropy",
                    "value_mode": "argmax",
                },
            }
            instance_stats.append(instance_record)

            for name, val, gap, dec_t, total_t in [
                ("greedy_pred", greedy_pred_val, greedy_pred_gap, greedy_pred_decode_t, greedy_pred_total_t),
                ("entropy_greedy", entropy_greedy_val, entropy_greedy_gap, entropy_greedy_decode_t, entropy_greedy_total_t),
                ("guided_decimation", guided_decimation_val, guided_decimation_gap, guided_decimation_decode_t, guided_decimation_total_t),
            ]:
                summary_lists[name]["obj"].append(float(val))
                summary_lists[name]["gap"].append(float(gap))
                summary_lists[name]["forward_t"].append(float(shared_forward_t))
                summary_lists[name]["decode_t"].append(float(dec_t))
                summary_lists[name]["total_t"].append(float(total_t))

            partial_payload = _build_test_payload(
                status="running",
                instance_stats=instance_stats,
                summary_lists=summary_lists,
                n_ss=n_ss,
            )
            atomic_write_json(out_path, partial_payload)

            print()
            print(
                f"[test {idx:04d}] ref={ref_val:.4f} | "
                f"GRD gap={greedy_pred_gap:.4f} t={greedy_pred_total_t:.4f}s | "
                f"EGG gap={entropy_greedy_gap:.4f} t={entropy_greedy_total_t:.4f}s | "
                f"GD gap={guided_decimation_gap:.4f} t={guided_decimation_total_t:.4f}s"
            )

            bar.next()
            if len(instance_stats) >= test_limit:
                break

        bar.finish()

    final_payload = _build_test_payload(
        status="completed",
        instance_stats=instance_stats,
        summary_lists=summary_lists,
        n_ss=n_ss,
    )
    atomic_write_json(out_path, final_payload)
    return out_path



