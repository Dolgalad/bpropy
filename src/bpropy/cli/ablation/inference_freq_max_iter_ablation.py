from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch

from bpropy.cli.train.config import load_run_config, validate_config
from bpropy.cli.train.loop import build_model, compute_loss_from_spec
from bpropy.cli.train.data import make_loaders
from bpropy.cli.train.checkpoint import load_checkpoint
from bpropy.cli.train.method_registry import METHOD_REGISTRY
from bpropy.datasets.prefetcher import CUDAPrefetcher
from bpropy.losses import LOSS_REGISTRY
from bpropy.cli.train.metrics import _sync, gap_from_solution
from bpropy.gd.greedy_decimation import greedy_decimation
from bpropy.gd.nmf_backend import NMFBackend
from bpropy.gd.lbp_backend import LBPBackend


CompletedKey = Tuple[str, str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bpropy-inference-ablation")
    parser.add_argument("--config", required=True, help="Path to ablation JSON config.")
    parser.add_argument(
        "--bpropy-root",
        default=None,
        help=(
            "Root used to resolve relative paths in the ablation config, "
            "especially dataset.test_config. Example: --bpropy-root /users/schulz/bpropy-jz/bpropy"
        ),
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Override evaluation.max_instances from the ablation config.",
    )
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def resolve_existing_or_root(path: str | Path, root: str | Path | None, *, prefer_root: bool = False) -> Path:
    """Resolve a config/checkpoint path robustly.

    Absolute paths are returned unchanged. For relative paths, this supports both
    common launch styles:
      1. relative to the current working directory;
      2. relative to --bpropy-root.

    If prefer_root=True, root/path is tried before cwd/path. This is useful for
    config paths such as configs/train/... when jobs are launched from the repo
    parent directory.
    """
    path = Path(path).expanduser()
    if path.is_absolute():
        return path

    candidates = []
    if root is not None:
        root = Path(root).expanduser().resolve()
        if prefer_root:
            candidates.extend([root / path, Path.cwd() / path])
        else:
            candidates.extend([Path.cwd() / path, root / path])
    else:
        candidates.append(Path.cwd() / path)

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    # Return the preferred candidate even if it does not exist, so the eventual
    # FileNotFoundError shows the path that was actually attempted.
    return candidates[0].resolve()


def resolve_output_root(path: str | Path, root: str | Path | None) -> Path:
    """Resolve output roots. Keep runs/... relative to cwd by default.

    This matches how the existing training/test scripts are usually launched
    from the experiment/repo root, while still allowing absolute output paths.
    """
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def atomic_write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())


def json_safe(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x
    if isinstance(x, np.generic):
        return json_safe(x.item())
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return json_safe(x.detach().cpu().item())
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return str(x)


def ablation_value_to_str(v: Any) -> str:
    return "inf" if v == "inf" or v is None else str(v)


def ablation_value_to_run_every(v: Any, n_vars: int) -> int:
    if v == "inf" or v is None:
        return int(n_vars + 1)
    return int(v)


def output_dir_from_cfg(ab_cfg: Dict[str, Any], bpropy_root: str | Path | None = None) -> Path:
    root = resolve_output_root(
        ab_cfg.get("output", {}).get("root", "runs/inference_freq_max_iter_ablation"),
        bpropy_root,
    )
    dataset_id = ab_cfg["dataset"]["dataset_id"]
    model_id = ab_cfg["model"]["model_id"]
    seed = int(ab_cfg["model"].get("seed", 0))
    return root / dataset_id / model_id / f"seed_{seed}"


def read_completed_keys(results_path: Path) -> Set[CompletedKey]:
    done: Set[CompletedKey] = set()
    if not results_path.exists():
        return done
    with open(results_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Ignore an incomplete final line after interruption.
                continue
            if rec.get("record_type") != "instance_result":
                continue
            if rec.get("status") != "completed":
                continue
            done.add((
                str(rec.get("ablation_type")),
                str(rec.get("ablation_value")),
                int(rec.get("instance_idx")),
            ))
    return done


def summarize_results(results_path: Path) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    n_completed = 0
    n_failed = 0
    if results_path.exists():
        with open(results_path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("record_type") != "instance_result":
                    continue
                if rec.get("status") == "failed":
                    n_failed += 1
                    continue
                if rec.get("status") != "completed":
                    continue
                n_completed += 1
                key = (str(rec.get("ablation_type")), str(rec.get("ablation_value")))
                g = groups.setdefault(key, {"gap": [], "runtime": [], "num_bp_calls": []})
                if rec.get("dec_gap") is not None:
                    g["gap"].append(float(rec["dec_gap"]))
                if rec.get("total_runtime") is not None:
                    g["runtime"].append(float(rec["total_runtime"]))
                if rec.get("num_bp_calls") is not None:
                    g["num_bp_calls"].append(float(rec["num_bp_calls"]))

    def ms(vals: List[float]) -> Dict[str, Any]:
        a = np.asarray(vals, dtype=np.float64)
        if a.size == 0:
            return {"mean": None, "std": None, "count": 0}
        return {
            "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "count": int(a.size),
        }

    by_ablation: Dict[str, Dict[str, Any]] = {}
    for (atype, aval), vals in groups.items():
        by_ablation.setdefault(atype, {})[aval] = {
            "dec_gap": ms(vals["gap"]),
            "total_runtime": ms(vals["runtime"]),
            "num_bp_calls": ms(vals["num_bp_calls"]),
        }
    return {"num_completed": n_completed, "num_failed": n_failed, "by_ablation": by_ablation}


def maybe_get_num_variables(pb: Any, data: Any, y: torch.Tensor) -> int:
    if hasattr(pb, "nv"):
        return int(pb.nv())
    if hasattr(data, "num_variables"):
        return int(data.num_variables)
    return int(y.numel())


def marginal_delta_stats(prev_mu: Optional[torch.Tensor], cur_mu: Optional[torch.Tensor], y: Optional[torch.Tensor]) -> Dict[str, Any]:
    empty = {
        "avg_delta_all": None,
        "avg_delta_gt0": None,
        "avg_delta_gt1": None,
        "mean_abs_delta_all": None,
        "mean_abs_delta_gt0": None,
        "mean_abs_delta_gt1": None,
        "max_abs_delta": None,
    }
    if prev_mu is None or cur_mu is None:
        return empty

    prev = prev_mu.detach().float().cpu()
    cur = cur_mu.detach().float().cpu()
    if prev.dim() == 2:
        prev = prev[:, 1]
    if cur.dim() == 2:
        cur = cur[:, 1]
    if prev.numel() != cur.numel():
        return empty

    delta = prev - cur
    out = {
        "avg_delta_all": float(delta.mean().item()),
        "mean_abs_delta_all": float(delta.abs().mean().item()),
        "max_abs_delta": float(delta.abs().max().item()),
        "avg_delta_gt0": None,
        "avg_delta_gt1": None,
        "mean_abs_delta_gt0": None,
        "mean_abs_delta_gt1": None,
    }
    if y is not None:
        yy = y.detach().cpu().reshape(-1)
        if yy.numel() == delta.numel():
            for label in (0, 1):
                mask = yy == label
                if bool(mask.any()):
                    out[f"avg_delta_gt{label}"] = float(delta[mask].mean().item())
                    out[f"mean_abs_delta_gt{label}"] = float(delta[mask].abs().mean().item())
    return out


def should_backend_run(backend: Any) -> bool:
    step = int(getattr(backend, "_step", 0))
    run_every = int(getattr(backend, "run_every", 1))
    mu = getattr(backend, "_mu", None)
    return (mu is None) or (run_every <= 1) or (step % run_every == 0)


def backend_info(backend: Any, runtime_s: float) -> Dict[str, Any]:
    """Best effort. Backends should set _last_info for residual logging."""
    info = getattr(backend, "_last_info", None)
    if info is None:
        info = getattr(backend, "last_info", None)
    if info is None:
        info = {}
    info = dict(info) if isinstance(info, dict) else {"raw_info": str(info)}

    # Normalize a few likely field names.
    residual_trace = info.get("residual_trace") or info.get("residuals") or info.get("trace")
    final_residual = info.get("final_residual") or info.get("residual") or info.get("err")
    num_iters = info.get("num_iters") or info.get("n_iter") or info.get("iters")

    return {
        "runtime": float(runtime_s),
        "num_iters": json_safe(num_iters),
        "final_residual": json_safe(final_residual),
        "residual_trace": json_safe(residual_trace),
        "raw_info": json_safe(info) if info else None,
    }



def adapter_accepts_return_log(padapter: Any) -> bool:
    """Return True if padapter.step can accept return_log as a keyword."""
    import inspect

    try:
        sig = inspect.signature(padapter.step)
    except Exception:
        return False
    if "return_log" in sig.parameters:
        return True
    return any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())


class ReturnLogAdapterProxy:
    """Small compatibility shim for adapters whose step() has no return_log kwarg.

    greedy_decimation calls adapter.step(out, return_log=return_log). The SCP
    adapter supports that directly, while the uploaded MIS adapter currently has
    step(out). This proxy lets us still use the real greedy_decimation routine,
    while preserving the original adapter state and info returned by step().
    """

    def __init__(self, adapter: Any):
        self._adapter = adapter

    def reset(self, pb: Any) -> None:
        return self._adapter.reset(pb)

    def is_done(self) -> bool:
        return self._adapter.is_done()

    def step(self, backend_output: Any, return_log: bool = False):
        result = self._adapter.step(backend_output)
        if isinstance(result, tuple) and len(result) == 2:
            evidence, info = result
        else:
            evidence, info = result, None
        if return_log:
            return evidence, info
        return evidence, info

    def solution(self):
        return self._adapter.solution()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)


def ensure_return_log_adapter(padapter: Any) -> Any:
    if adapter_accepts_return_log(padapter):
        return padapter
    return ReturnLogAdapterProxy(padapter)


class TracedBackendProxy:
    """Proxy that logs inference calls while greedy_decimation drives the solve.

    This preserves the exact control flow of bpropy.gd.greedy_decimation:
        backend.reset(pb)
        adapter.reset(pb)
        out = backend.observe()
        evidence, info = adapter.step(out, return_log=True)
        backend.update(evidence)

    The proxy only wraps backend.observe/update to record when inference was
    actually rerun, marginal deltas, runtimes, and any residual info exposed by
    the concrete backend through _last_info/last_info.
    """

    def __init__(
        self,
        backend: Any,
        *,
        y: torch.Tensor,
        device: torch.device,
        log_marginal_deltas: bool,
        log_bp_residuals: bool,
    ):
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_y", y)
        object.__setattr__(self, "_device", device)
        object.__setattr__(self, "_log_marginal_deltas", log_marginal_deltas)
        object.__setattr__(self, "_log_bp_residuals", log_bp_residuals)
        object.__setattr__(self, "inference_trace", [])
        object.__setattr__(self, "total_inference_runtime", 0.0)
        object.__setattr__(self, "num_inference_calls", 0)
        object.__setattr__(self, "_previous_run_mu", None)

    def reset(self, pb: Any) -> None:
        self.inference_trace.clear()
        object.__setattr__(self, "total_inference_runtime", 0.0)
        object.__setattr__(self, "num_inference_calls", 0)
        object.__setattr__(self, "_previous_run_mu", None)
        return self._backend.reset(pb)

    def observe(self):
        will_run = should_backend_run(self._backend)
        gd_iter = int(getattr(self._backend, "_step", 0))
        t0 = time.perf_counter()
        out = self._backend.observe()
        device = self._device
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
        runtime_s = time.perf_counter() - t0

        if will_run:
            idx = int(self.num_inference_calls)
            object.__setattr__(self, "total_inference_runtime", float(self.total_inference_runtime + runtime_s))
            object.__setattr__(self, "num_inference_calls", idx + 1)

            cur_mu = out.detach().clone() if torch.is_tensor(out) else None
            rec = {
                "bp_call_idx": idx,
                "gd_iter": gd_iter,
                "backend_step": gd_iter,
                "runtime": float(runtime_s),
                "run_every": json_safe(getattr(self._backend, "run_every", None)),
                "max_iter": json_safe(getattr(self._backend, "max_iter", None)),
            }
            if self._log_marginal_deltas:
                rec.update(marginal_delta_stats(self._previous_run_mu, cur_mu, self._y))
            if self._log_bp_residuals:
                rec.update(backend_info(self._backend, runtime_s))
            self.inference_trace.append(json_safe(rec))
            object.__setattr__(self, "_previous_run_mu", cur_mu)

        return out

    def update(self, evidence: Dict[int, int]) -> None:
        return self._backend.update(evidence)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in {"inference_trace", "total_inference_runtime", "num_inference_calls"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._backend, name, value)


def normalize_greedy_log(log: Optional[Dict[str, Any]], inference_trace: List[Dict[str, Any]], *, log_gd_trace: bool) -> List[Dict[str, Any]]:
    if not log_gd_trace or not log:
        return []

    # Map gd_iter -> bp_call_idx for steps where observe actually reran inference.
    bp_by_iter = {int(r.get("gd_iter")): int(r.get("bp_call_idx")) for r in inference_trace if r.get("gd_iter") is not None}
    steps = log.get("step", [])
    evidences = log.get("evidence", [])
    infos = log.get("info", [])

    out = []
    for k, step in enumerate(steps):
        gd_iter = int(step)
        bp_call_idx = bp_by_iter.get(gd_iter)
        out.append(json_safe({
            "gd_iter": gd_iter,
            "bp_was_run": bp_call_idx is not None,
            "bp_call_idx": bp_call_idx,
            "evidence": evidences[k] if k < len(evidences) else None,
            "info": infos[k] if k < len(infos) else None,
        }))
    return out


def solution_to_array(sol: Any) -> np.ndarray:
    # SCPAdapter can return (x, active_sets) when fix_prop < 1.0.
    if isinstance(sol, tuple):
        sol = sol[0]
    return np.asarray(sol)


def run_decimation_with_trace(
    *,
    pb: Any,
    backend: Any,
    padapter: Any,
    y: torch.Tensor,
    maximize: bool,
    device: torch.device,
    log_gd_trace: bool,
    log_marginal_deltas: bool,
    log_bp_residuals: bool,
) -> Dict[str, Any]:
    traced_backend = TracedBackendProxy(
        backend,
        y=y,
        device=device,
        log_marginal_deltas=log_marginal_deltas,
        log_bp_residuals=log_bp_residuals,
    )
    traced_adapter = ensure_return_log_adapter(padapter)

    sol, gd_log = greedy_decimation(pb, traced_backend, traced_adapter, return_log=True)
    sol_arr = solution_to_array(sol)

    objective_value = float(np.dot(pb.w, sol_arr)) if hasattr(pb, "w") else float(sol_arr.sum())
    solution_sum = float(sol_arr.sum())
    dec_gap = float(gap_from_solution(
        y_true_sum=float(y.sum().item()),
        y_pred_sum=solution_sum,
        maximize=maximize,
        eps=1e-8,
    ))

    gd_trace = normalize_greedy_log(
        gd_log,
        traced_backend.inference_trace,
        log_gd_trace=log_gd_trace,
    )

    return {
        "solution": sol_arr,
        "objective_value": objective_value,
        "solution_sum": solution_sum,
        "dec_gap": dec_gap,
        "num_gd_steps": int(len(gd_log.get("step", [])) if gd_log else 0),
        "num_bp_calls": int(traced_backend.num_inference_calls),
        "inference_runtime": float(traced_backend.total_inference_runtime),
        "gd_trace": gd_trace,
        "inference_trace": traced_backend.inference_trace,
    }


def configure_backend(backend: Any, *, run_every: int, max_iter: int) -> Any:
    if hasattr(backend, "run_every"):
        backend.run_every = int(run_every)
    if hasattr(backend, "max_iter"):
        backend.max_iter = int(max_iter)
    return backend

def force_nmf_neg_inf(cfg, ab_cfg):
    model_type = ab_cfg.get("model", {}).get("model_type")

    if model_type != "nmf":
        return cfg

    gd_cfg = cfg.setdefault("eval", {}).setdefault("greedy_decimation", {})

    # Depending on your config structure, backend args may live here.
    backend_cfg = gd_cfg.setdefault("backend", {})
    backend_args = backend_cfg.setdefault("args", {})

    # NMF does not behave well with true -inf; use finite clamp.
    backend_args["neg_inf"] = -100.0
    backend_cfg["neg_inf"] = -100.0
    print(backend_args["neg_inf"] )

    return cfg

def prepare_train_cfg(ab_cfg: Dict[str, Any], bpropy_root: str | Path | None = None) -> Dict[str, Any]:
    train_cfg_path = resolve_existing_or_root(
        ab_cfg["dataset"]["test_config"],
        bpropy_root,
        prefer_root=True,
    )
    cfg = load_run_config(str(train_cfg_path))
    validate_config(cfg)

    cfg = force_nmf_neg_inf(cfg, ab_cfg)
    cfg["seed"] = int(ab_cfg["model"].get("seed", cfg.get("seed", 0)))
    return cfg


def build_adapter(train_cfg: Dict[str, Any]) -> Any:
    method_name = train_cfg.get("train", {}).get("method", "var_logits")
    eval_method = train_cfg.get("eval", {}).get("method") or method_name
    if eval_method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown eval method '{eval_method}'. Known: {list(METHOD_REGISTRY.keys())}")
    adapter = METHOD_REGISTRY[eval_method]
    adapter.loss_name = train_cfg.get("eval", {}).get("loss-name") or train_cfg["loss"]["name"]
    return adapter


def iter_ablation_settings(ab_cfg: Dict[str, Any]) -> Iterable[Tuple[str, Any]]:
    enabled = ab_cfg.get("ablation", {}).get("enabled_ablations", ["run_every", "bp_max_iter"])
    if "run_every" in enabled:
        for v in ab_cfg["ablation"].get("run_every_values", []):
            yield "run_every", v
    if "bp_max_iter" in enabled:
        for v in ab_cfg["ablation"].get("bp_max_iter_values", []):
            yield "bp_max_iter", v


def resolve_run_params(ab_cfg: Dict[str, Any], ablation_type: str, ablation_value: Any, n_vars: int) -> Tuple[int, int]:
    ab = ab_cfg["ablation"]
    if ablation_type == "run_every":
        run_every = ablation_value_to_run_every(ablation_value, n_vars)
        max_iter = int(ab.get("fixed_bp_max_iter_for_run_every_ablation", 16))
    elif ablation_type == "bp_max_iter":
        run_every = int(ab.get("fixed_run_every_for_max_iter_ablation", 1))
        max_iter = int(ablation_value)
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")
    return run_every, max_iter


def main() -> int:
    args = parse_args()
    bpropy_root = Path(args.bpropy_root).expanduser().resolve() if args.bpropy_root else None
    ab_cfg_path = resolve_existing_or_root(args.config, bpropy_root, prefer_root=False)
    ab_cfg = load_json(ab_cfg_path)

    if args.max_instances is not None:
        ab_cfg.setdefault("evaluation", {})["max_instances"] = int(args.max_instances)

    # Resolve path-like entries once and keep the resolved values in the copied
    # config that is written to the output directory. dataset.test_config is
    # intentionally resolved relative to --bpropy-root first. Checkpoints are
    # resolved from cwd first because existing experiments usually live under
    # ./runs when launched from the repo/experiment root.
    ab_cfg = copy.deepcopy(ab_cfg)
    ab_cfg.setdefault("dataset", {})["test_config"] = str(
        resolve_existing_or_root(ab_cfg["dataset"]["test_config"], bpropy_root, prefer_root=True)
    )
    ab_cfg.setdefault("model", {})["checkpoint"] = str(
        resolve_existing_or_root(ab_cfg["model"]["checkpoint"], bpropy_root, prefer_root=False)
    )

    out_dir = output_dir_from_cfg(ab_cfg, bpropy_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_ab_cfg = copy.deepcopy(ab_cfg)
    resolved_ab_cfg["_source_config"] = str(ab_cfg_path)
    resolved_ab_cfg["_bpropy_root"] = str(bpropy_root) if bpropy_root else None
    atomic_write_json(out_dir / "config.json", resolved_ab_cfg)

    if args.print_config:
        print(json.dumps(resolved_ab_cfg, indent=2, sort_keys=True))
        return 0

    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"
    writer = JsonlWriter(results_path)

    resume = bool(ab_cfg.get("output", {}).get("resume", True))
    overwrite = bool(ab_cfg.get("output", {}).get("overwrite", False))
    if overwrite and results_path.exists():
        results_path.unlink()
    completed = read_completed_keys(results_path) if resume else set()

    train_cfg = prepare_train_cfg(ab_cfg, bpropy_root)
    device_name = ab_cfg.get("evaluation", {}).get("device", "cuda")
    device = torch.device(device_name if (device_name == "cpu" or torch.cuda.is_available()) else "cpu")

    torch.manual_seed(int(ab_cfg["model"].get("seed", train_cfg.get("seed", 0))))
    np.random.seed(int(ab_cfg["model"].get("seed", train_cfg.get("seed", 0))))

    loaders = make_loaders(train_cfg, splits=["test"])
    adapter = build_adapter(train_cfg)
    model = build_model(train_cfg, device)
    load_checkpoint(ab_cfg["model"]["checkpoint"], model, optimizer=None, map_location=device)
    model.eval()

    maximize = bool(train_cfg.get("problem", {}).get("maximize", True))
    max_instances = ab_cfg.get("evaluation", {}).get("max_instances", None)
    log_gd_trace = bool(ab_cfg.get("evaluation", {}).get("log_gd_trace", True))
    log_marginal_deltas = bool(ab_cfg.get("evaluation", {}).get("log_marginal_deltas", True))
    log_bp_residuals = bool(ab_cfg.get("evaluation", {}).get("log_bp_residuals", True))
    log_solution = bool(ab_cfg.get("evaluation", {}).get("log_solution", False))

    experiment_meta = {
        "record_type": "experiment_header",
        "experiment_id": ab_cfg.get("experiment_id"),
        "dataset_id": ab_cfg["dataset"]["dataset_id"],
        "model_id": ab_cfg["model"]["model_id"],
        "model_type": ab_cfg["model"].get("model_type"),
        "seed": int(ab_cfg["model"].get("seed", 0)),
        "checkpoint": ab_cfg["model"]["checkpoint"],
        "test_config": ab_cfg["dataset"]["test_config"],
        "output_dir": str(out_dir),
    }
    if not completed:
        writer.append(experiment_meta)

    with torch.no_grad():
        for instance_idx, batch in enumerate(CUDAPrefetcher(loaders.test_loader, device)):
            if max_instances is not None and instance_idx >= int(max_instances):
                break

            # Existing final_test_eval receives (pb, data, y), where pb is often a one-element list.
            pbs, data, y = batch
            pb = pbs[0] if isinstance(pbs, (list, tuple)) else pbs
            n_vars = maybe_get_num_variables(pb, data, y)

            pred_t0 = time.perf_counter()
            pred = adapter.forward_fn(model, data)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            pred_runtime = time.perf_counter() - pred_t0

            spec = adapter.loss_spec_fn(adapter, data, pred, y, train_cfg)
            try:
                loss = float(compute_loss_from_spec(spec, loss_registry=LOSS_REGISTRY).detach().cpu().item())
            except Exception:
                loss = None

            for ablation_type, ablation_value in iter_ablation_settings(ab_cfg):
                aval_str = ablation_value_to_str(ablation_value)
                key = (ablation_type, aval_str, instance_idx)
                if key in completed:
                    print(f"[skip] {ablation_type}={aval_str} instance={instance_idx}")
                    continue

                run_every, max_iter = resolve_run_params(ab_cfg, ablation_type, ablation_value, n_vars)
                t0 = time.perf_counter()
                try:
                    backend = adapter.decimation_backend_fn(pb, data, pred, train_cfg, device=device)
                    if isinstance(backend, NMFBackend):
                        backend.neg_inf = -100.0
                        backend.damping = 0.5
                    if isinstance(backend, LBPBackend):
                        backend.neg_inf = -100.0
                        backend.damping = 0.5

                    padapter = adapter.decimation_adapter_fn(pb, train_cfg)
                    configure_backend(backend, run_every=run_every, max_iter=max_iter)

                    result = run_decimation_with_trace(
                        pb=pb,
                        backend=backend,
                        padapter=padapter,
                        y=y,
                        maximize=maximize,
                        device=device,
                        log_gd_trace=log_gd_trace,
                        log_marginal_deltas=log_marginal_deltas,
                        log_bp_residuals=log_bp_residuals,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    total_runtime = time.perf_counter() - t0

                    record = {
                        "record_type": "instance_result",
                        "status": "completed",
                        "experiment_id": ab_cfg.get("experiment_id"),
                        "dataset_id": ab_cfg["dataset"]["dataset_id"],
                        "model_id": ab_cfg["model"]["model_id"],
                        "model_type": ab_cfg["model"].get("model_type"),
                        "seed": int(ab_cfg["model"].get("seed", 0)),
                        "checkpoint": ab_cfg["model"]["checkpoint"],
                        "test_config": ab_cfg["dataset"]["test_config"],
                        "ablation_type": ablation_type,
                        "ablation_value": aval_str,
                        "run_every": "inf" if aval_str == "inf" else int(run_every),
                        "effective_run_every": int(run_every),
                        "bp_max_iter": int(max_iter),
                        "instance_idx": int(instance_idx),
                        "instance_id": str(getattr(data, "instance_id", instance_idx)),
                        "n_vars": int(n_vars),
                        "loss": loss,
                        "objective_value": result["objective_value"],
                        "solution_sum": result["solution_sum"],
                        "best_known_value": float(y.sum().item()),
                        "dec_gap": result["dec_gap"],
                        "total_runtime": float(total_runtime),
                        "prediction_runtime": float(pred_runtime),
                        "inference_runtime": result["inference_runtime"],
                        "num_gd_steps": result["num_gd_steps"],
                        "num_bp_calls": result["num_bp_calls"],
                        "gd_trace": result["gd_trace"],
                        "inference_trace": result["inference_trace"],
                    }
                    if log_solution:
                        record["solution"] = np.asarray(result["solution"]).astype(int).tolist()
                    writer.append(json_safe(record))
                    print(
                        f"[done] {ablation_type}={aval_str} instance={instance_idx} "
                        f"gap={record['dec_gap']:.6f} time={record['total_runtime']:.3f}s "
                        f"bp_calls={record['num_bp_calls']}"
                    )
                except Exception as e:
                    err = {
                        "record_type": "instance_result",
                        "status": "failed",
                        "experiment_id": ab_cfg.get("experiment_id"),
                        "dataset_id": ab_cfg["dataset"]["dataset_id"],
                        "model_id": ab_cfg["model"]["model_id"],
                        "seed": int(ab_cfg["model"].get("seed", 0)),
                        "ablation_type": ablation_type,
                        "ablation_value": aval_str,
                        "instance_idx": int(instance_idx),
                        "error": repr(e),
                        "traceback": traceback.format_exc(),
                    }
                    writer.append(json_safe(err))
                    print(f"[failed] {ablation_type}={aval_str} instance={instance_idx}: {e}")

                summary = summarize_results(results_path)
                summary.update(experiment_meta)
                atomic_write_json(summary_path, json_safe(summary))

    summary = summarize_results(results_path)
    summary.update(experiment_meta)
    atomic_write_json(summary_path, json_safe(summary))
    print(f"[saved] results: {results_path}")
    print(f"[saved] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
