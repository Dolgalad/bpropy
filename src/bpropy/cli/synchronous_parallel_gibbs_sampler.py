# save as: synchronous_parallel_gibbs_sampler.py
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Literal

import numpy as np
import torch
from progress.bar import Bar

from bpropy.cli.train.config import load_run_config, merge_cli_overrides, validate_config
from bpropy.cli.train.data import make_loaders
from bpropy.cli.train.logging import make_run_paths
from bpropy.cli.train.checkpoint import load_checkpoint
from bpropy.cli.train.loop import build_model
from bpropy.cli.train.method_registry import METHOD_REGISTRY
from bpropy.datasets.prefetcher import CUDAPrefetcher
from bpropy.mrf.factor_graph import FactorGraph

import matplotlib.pyplot as plt

# IMPORTANT: use the synchronous sampler you added to bpropy.mrf.gibbs
from bpropy.mrf.gibbs import synchronous_parallel_gibbs_sample


# ---------------------------
# CLI (same structure as gibbs_sampler.py) :contentReference[oaicite:3]{index=3}
# ---------------------------

def build_base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bpropy-sync-gibbs", add_help=False)
    p.add_argument("--config", required=True, help="Path to JSON config.")
    p.add_argument("--print-config", action="store_true", help="Print resolved config and exit.")
    return p


def build_full_parser(_defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bpropy-sync-gibbs")
    p.add_argument("--config", required=True, help="Path to JSON config.")
    p.add_argument("--print-config", action="store_true", help="Print resolved config and exit.")

    # common overrides (subset of train/test)
    p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    p.add_argument("--run-name", dest="run_name", default=argparse.SUPPRESS)
    p.add_argument("--log-dir", dest="log_dir", default=argparse.SUPPRESS)
    p.add_argument("--data-root", dest="data_root", default=argparse.SUPPRESS)
    p.add_argument("--test-split", dest="test_split", default=argparse.SUPPRESS)
    p.add_argument("--batch-size", type=int, dest="batch_size", default=argparse.SUPPRESS)
    p.add_argument("--num-workers", type=int, dest="num_workers", default=argparse.SUPPRESS)
    p.add_argument("--model", dest="model_name", default=argparse.SUPPRESS)
    p.add_argument("--loss", dest="loss_name", default=argparse.SUPPRESS)

    # gibbs-specific (same as gibbs_sampler.py) :contentReference[oaicite:4]{index=4}
    p.add_argument("--checkpoint", dest="checkpoint", default=argparse.SUPPRESS,
                   help="Path to checkpoint (.pt). If omitted, uses <run>/checkpoints/best.pt then last.pt.")
    p.add_argument("--num-samples", type=int, default=2000, help="Number of samples per instance.")
    p.add_argument("--burn-in", type=int, default=200, help="Burn-in sweeps.")
    p.add_argument("--thinning", type=int, default=1, help="Keep one sample every k sweeps.")
    p.add_argument("--out", dest="out_path", default=None,
                   help="Output .npz path. Default: <log_dir>/gibbs_samples_test.npz")
    p.add_argument("--gibbs-seed", type=int, default=None, help="Seed for Gibbs RNG (defaults to cfg.seed).")
    p.add_argument("--test-data-root", dest="test_data_root", default=argparse.SUPPRESS)

    # sync-parallel specific
    p.add_argument("--sync-workers", type=int, default=0,
                   help="Thread workers for synchronous Gibbs (0 => auto).")

    return p


def _apply_test_only_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if hasattr(args, "test_data_root"):
        cfg.setdefault("data", {})["root"] = getattr(args, "test_data_root")
    if hasattr(args, "test_split"):
        cfg.setdefault("data", {})["test_split"] = getattr(args, "test_split")
    return cfg


def _pick_checkpoint_path(cfg: Dict[str, Any], paths, args: argparse.Namespace) -> str:
    if hasattr(args, "checkpoint"):
        return getattr(args, "checkpoint")

    best = os.path.join(paths.ckpt_dir, "best.pt")
    last = os.path.join(paths.ckpt_dir, "last.pt")
    if os.path.exists(best):
        return best
    if os.path.exists(last):
        return last
    raise FileNotFoundError(
        f"No checkpoint found. Looked for {best!r} and {last!r}. Pass --checkpoint to specify one."
    )


def fg_builder(pb, cfg):
    fg_approx_cfg = cfg.get("data", {}).get("precompute", {}).get("fg_approx", None)
    if fg_approx_cfg is not None:
        from bpropy.mrf.factor_graph_approx import APPROX_BUILDERS
        from bpropy.graphs.vc_graph import VCGraph
        vcg = VCGraph(pb)
        fg_true = FactorGraph(pb)
        fg_approx, _, _ = APPROX_BUILDERS[fg_approx_cfg.get("method")](
            pb=pb, vcg=vcg, fg_true=fg_true, approx=fg_approx_cfg
        )
        return fg_approx
    return FactorGraph(pb)


def _extract_factor_graph(pb, data, cfg):
    return fg_builder(pb, cfg)


def _pred_to_theta(pred: torch.Tensor) -> torch.Tensor:
    # keep your current behavior: pred[1] :contentReference[oaicite:5]{index=5}
    return pred[1]


def plot_gibbs_diagnostics(
    problem,
    y_ref: np.ndarray,
    samples: np.ndarray,
    *,
    sample_times: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
    sense: Literal["max", "min"] = "max",
    gap_mode: Literal["relative", "absolute"] = "relative",
    eps: float = 1e-12,
    out_dir: str = ".",
    tag: str = "run",
    dpi: int = 150,
    show: bool = False,
) -> Tuple[str, str]:
    samples = np.asarray(samples)
    y_ref = np.asarray(y_ref)

    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D (S, n), got shape {samples.shape}")

    S = samples.shape[0]

    if sample_times is not None:
        t = np.asarray(sample_times, dtype=float)
        if t.shape != (S,):
            raise ValueError(f"sample_times must have shape ({S},), got {t.shape}")
    elif dt is not None:
        t = np.arange(S, dtype=float) * float(dt)
    else:
        t = np.arange(S, dtype=float)

    obj_ref = float(np.dot(problem.w, y_ref))

    obj = np.empty(S, dtype=float)
    feas = np.empty(S, dtype=bool)
    for k in range(S):
        xk = samples[k]
        feas[k] = bool(problem.check_solution(xk))
        obj[k] = float(np.dot(problem.w, xk))

    if sense == "max":
        best_so_far = np.maximum.accumulate(obj)
        raw_gap = obj_ref - best_so_far
    elif sense == "min":
        best_so_far = np.minimum.accumulate(obj)
        raw_gap = best_so_far - obj_ref
    else:
        raise ValueError("sense must be 'max' or 'min'")

    raw_gap = np.maximum(raw_gap, 0.0)

    if gap_mode == "relative":
        gap = raw_gap / (abs(obj_ref) + eps)
        gap_ylabel = "Optimality gap (relative)"
    elif gap_mode == "absolute":
        gap = raw_gap
        gap_ylabel = "Optimality gap (absolute)"
    else:
        raise ValueError("gap_mode must be 'relative' or 'absolute'")

    feas_rate = np.cumsum(feas.astype(int)) / (np.arange(S) + 1)

    os.makedirs(out_dir, exist_ok=True)

    gap_fig = plt.figure()
    plt.plot(t, gap)
    plt.xlabel("Time" if (sample_times is not None or dt is not None) else "Sample index")
    plt.ylabel(gap_ylabel)
    plt.title(f"Best-so-far gap vs time ({tag})")
    plt.grid(True, which="both", alpha=0.3)
    gap_path = os.path.join(out_dir, f"gap_vs_time_{tag}.png")
    plt.savefig(gap_path, dpi=dpi, bbox_inches="tight")
    plt.close(gap_fig)

    feas_fig = plt.figure()
    plt.plot(t, feas_rate, label="Cumulative feasible fraction")
    plt.scatter(t, feas.astype(int), s=8, alpha=0.35, label="Per-sample feasibility (0/1)")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Time" if (sample_times is not None or dt is not None) else "Sample index")
    plt.ylabel("Feasibility")
    plt.title(f"Feasibility over time ({tag})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    feas_path = os.path.join(out_dir, f"feasibility_{tag}.png")
    plt.savefig(feas_path, dpi=dpi, bbox_inches="tight")
    plt.close(feas_fig)

    if show:
        plt.show()

    return gap_path, feas_path


def run_gibbs(cfg: Dict[str, Any], checkpoint_path: str, args: argparse.Namespace) -> str:
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    gibbs_seed = seed if args.gibbs_seed is None else int(args.gibbs_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = make_run_paths(cfg)
    out_path = args.out_path or os.path.join(paths.log_dir, "gibbs_samples_test.npz")

    loaders = make_loaders(cfg, splits=["test"])

    method_name = cfg.get("train", {}).get("method", "var_logits")
    eval_method = cfg.get("eval", {}).get("method") or method_name
    if eval_method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown eval method '{eval_method}'. Known: {list(METHOD_REGISTRY.keys())}")
    adapter_eval = METHOD_REGISTRY[eval_method]

    model = build_model(cfg, device)
    load_checkpoint(checkpoint_path, model, optimizer=None, map_location=device)
    model.eval()

    num_samples = int(args.num_samples)
    burn_in = int(args.burn_in)
    thinning = int(args.thinning)
    sync_workers = int(args.sync_workers)

    payload: Dict[str, Any] = {}
    meta: Dict[str, Any] = {
        "num_samples": num_samples,
        "burn_in_sweeps": burn_in,
        "thinning": thinning,
        "checkpoint": checkpoint_path,
        "eval_method": eval_method,
        "gibbs_seed": gibbs_seed,
        "sync_workers": sync_workers,
    }

    bar = Bar("[SyncGibbs] test set", max=len(loaders.test_loader))

    with torch.no_grad():
        idx = 0
        for pb, data, y in CUDAPrefetcher(loaders.test_loader, device):
            if isinstance(pb, (list, tuple)) and len(pb) != 1:
                raise RuntimeError(
                    f"bpropy-sync-gibbs currently expects batch_size=1 (got batch of {len(pb)} problems)."
                )

            pb0 = pb[0] if isinstance(pb, (list, tuple)) else pb

            pred = adapter_eval.forward_fn(model, data)
            theta = _pred_to_theta(pred).cpu()

            if theta.dim() == 2 and theta.size(0) == 1:
                theta = theta[0]
            if theta.dim() != 1:
                raise RuntimeError(
                    f"Expected theta as 1D tensor (or [1, D]), got shape {tuple(theta.shape)}."
                )

            fg = _extract_factor_graph(pb0, data, cfg)

            samples = synchronous_parallel_gibbs_sample(
                fg,
                theta,
                max_samples=num_samples,
                burn_in_sweeps=burn_in,
                thinning=thinning,
                seed=gibbs_seed + idx,
                n_workers=sync_workers,
                pb=pb0,
            )

            gap_path, feas_path = plot_gibbs_diagnostics(
                problem=pb0,
                y_ref=y.cpu(),
                samples=samples,
                sample_times=None,
                sense="min",
                gap_mode="relative",
                out_dir="figs",
                tag=f"test_{idx:06d}",
                show=False,
            )
            print(gap_path, feas_path)

            payload[f"samples_{idx:06d}"] = samples
            try:
                payload[f"y_{idx:06d}"] = y.detach().cpu().numpy()
            except Exception:
                pass

            idx += 1
            bar.next()

        bar.finish()

    payload["__meta__"] = json.dumps(meta, indent=2, sort_keys=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    base = build_base_parser()
    base_args, _ = base.parse_known_args(argv)

    cfg = load_run_config(base_args.config)
    validate_config(cfg)

    parser = build_full_parser(cfg)
    args = parser.parse_args(argv)

    cfg = merge_cli_overrides(cfg, args)
    cfg = _apply_test_only_overrides(cfg, args)

    if getattr(args, "print_config", False):
        print(json.dumps(cfg, indent=2, sort_keys=True))
        return 0

    paths = make_run_paths(cfg)
    ckpt_path = _pick_checkpoint_path(cfg, paths, args)
    print("checkpoint path = ", ckpt_path)

    out_path = run_gibbs(cfg, ckpt_path, args)
    print("Saved Sync Gibbs samples to:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

