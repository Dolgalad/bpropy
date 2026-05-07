#!/usr/bin/env python3
"""
generate_inference_theta_distribution.py

Collect predicted theta values from a trained LBP/NMF model and split them into
constraint-factor assignments that are locally feasible vs infeasible.

The feasibility mask is built in the same overcomplete sufficient-statistic
ordering used by canonical_parameters / FactorGraph:

  - unary factors first, if add_unaries=True
  - constraint rows sorted by (scope size, lexicographic scope)
  - assignments enumerated little-endian

Example
-------
python generate_inference_theta_distribution.py \
    --config configs/train/scp/...json \
    --checkpoint runs/.../checkpoints/best.pt \
    --output-json runs/.../theta_distribution.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from progress.bar import Bar

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

from bpropy.datasets.prefetcher import CUDAPrefetcher

from bpropy.cli.train.config import (
    load_run_config,
    merge_cli_overrides,
    validate_config,
)
from bpropy.cli.train.loop import build_model
from bpropy.cli.train.data import make_loaders
from bpropy.cli.train.logging import make_run_paths
from bpropy.cli.train.checkpoint import load_checkpoint
from bpropy.cli.train.method_registry import METHOD_REGISTRY


# ---------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------


def build_base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate-inference-theta-distribution",
        add_help=False,
    )
    p.add_argument("--config", required=True, help="Path to JSON train config.")
    p.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved config and exit.",
    )
    return p


def build_full_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="generate-inference-theta-distribution")

    p.add_argument("--config", required=True, help="Path to JSON train config.")
    p.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved config and exit.",
    )

    p.add_argument(
        "--checkpoint",
        dest="checkpoint",
        default=argparse.SUPPRESS,
        help=(
            "Path to checkpoint. If omitted, uses <run>/checkpoints/best.pt "
            "then <run>/checkpoints/last.pt."
        ),
    )

    p.add_argument(
        "--output-json",
        dest="output_json",
        required=True,
        help="Path to output JSON file.",
    )

    p.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Optional maximum number of test instances to process.",
    )

    p.add_argument(
        "--include-unaries",
        action="store_true",
        help=(
            "Include unary theta values in the feasible group. "
            "By default, only constraint-factor assignments are used."
        ),
    )

    p.add_argument(
        "--save-per-instance-values",
        action="store_true",
        help=(
            "Store theta values per instance as well as globally. "
            "This can make the JSON much larger."
        ),
    )

    p.add_argument(
        "--tol",
        type=float,
        default=0.0,
        help="Tolerance for local constraint feasibility.",
    )

    # Same common overrides as bpropy-test.
    p.add_argument("--seed", type=int, default=argparse.SUPPRESS)

    p.add_argument("--run-name", dest="run_name", default=argparse.SUPPRESS)
    p.add_argument("--log-dir", dest="log_dir", default=argparse.SUPPRESS)

    p.add_argument("--data-root", dest="data_root", default=argparse.SUPPRESS)
    p.add_argument("--split", dest="split", default=argparse.SUPPRESS)
    p.add_argument("--val-split", dest="val_split", default=argparse.SUPPRESS)
    p.add_argument("--test-split", dest="test_split", default=argparse.SUPPRESS)
    p.add_argument("--batch-size", type=int, dest="batch_size", default=argparse.SUPPRESS)
    p.add_argument("--num-workers", type=int, dest="num_workers", default=argparse.SUPPRESS)
    p.add_argument("--feature-mode", dest="feature_mode", default=argparse.SUPPRESS)
    p.add_argument("--hetero", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--no-hetero", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--with-lbp", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--with-nmf", action="store_true", default=argparse.SUPPRESS)

    p.add_argument("--model", dest="model_name", default=argparse.SUPPRESS)
    p.add_argument("--loss", dest="loss_name", default=argparse.SUPPRESS)

    p.add_argument(
        "--test-data-root",
        dest="test_data_root",
        default=argparse.SUPPRESS,
        help="Override cfg.data.root for this test diagnostic only.",
    )

    return p


def _apply_test_only_overrides(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if hasattr(args, "test_data_root"):
        cfg.setdefault("data", {})["root"] = getattr(args, "test_data_root")

    if hasattr(args, "test_split"):
        cfg.setdefault("data", {})["test_split"] = getattr(args, "test_split")

    return cfg


def _pick_checkpoint_path(
    cfg: Dict[str, Any],
    paths,
    args: argparse.Namespace,
) -> str:
    if hasattr(args, "checkpoint"):
        return getattr(args, "checkpoint")

    best = os.path.join(paths.ckpt_dir, "best.pt")
    last = os.path.join(paths.ckpt_dir, "last.pt")

    if os.path.exists(best):
        return best
    if os.path.exists(last):
        return last

    raise FileNotFoundError(
        f"No checkpoint found. Looked for {best!r} and {last!r}. "
        "Pass --checkpoint explicitly."
    )


# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = str(out_path) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, out_path)


def ms(x: List[float]) -> Tuple[float, float]:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return 0.0, 0.0
    if a.size == 1:
        return float(a.mean()), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def values_summary(values: List[float]) -> Dict[str, Optional[float]]:
    a = np.asarray(values, dtype=np.float64)

    if a.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    if a.size == 1:
        std = 0.0
    else:
        std = float(a.std(ddof=1))

    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "std": std,
        "min": float(a.min()),
        "max": float(a.max()),
    }


def build_payload(
    *,
    status: str,
    config_path: str,
    checkpoint_path: str,
    include_unaries: bool,
    tol: float,
    theta_feasible: List[float],
    theta_infeasible: List[float],
    theta_unary: List[float],
    instances: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "status": status,
        "config": config_path,
        "checkpoint": checkpoint_path,
        "include_unaries": bool(include_unaries),
        "tol": float(tol),
        "summary": {
            "count": len(instances),
            "theta_feasible": values_summary(theta_feasible),
            "theta_infeasible": values_summary(theta_infeasible),
            "theta_unary": values_summary(theta_unary),
        },
        "theta_feasible": theta_feasible,
        "theta_infeasible": theta_infeasible,
        "theta_unary": theta_unary,
        "instances": instances,
    }


# ---------------------------------------------------------------------
# Theta extraction
# ---------------------------------------------------------------------


def extract_pred_theta(pred: Any) -> torch.Tensor:
    """
    Extract predicted theta from the adapter forward output.

    In your loop.py debugging code, pred can be a tuple where pred[1] is theta.
    This function prioritizes that convention.
    """

    if isinstance(pred, tuple):
        if len(pred) >= 2 and torch.is_tensor(pred[1]):
            return pred[1]
        for item in pred:
            if torch.is_tensor(item):
                return item

    if torch.is_tensor(pred):
        return pred

    if isinstance(pred, dict):
        for key in (
            "theta",
            "pred_theta",
            "theta_pred",
            "canonical_parameters",
            "pred_canonical_parameters",
        ):
            if key in pred and torch.is_tensor(pred[key]):
                return pred[key]

    for attr in (
        "theta",
        "pred_theta",
        "theta_pred",
        "canonical_parameters",
        "pred_canonical_parameters",
    ):
        if hasattr(pred, attr):
            value = getattr(pred, attr)
            if torch.is_tensor(value):
                return value

    raise RuntimeError(
        "Could not extract predicted theta from adapter output. "
        "Expected either pred[1], pred['theta'], pred.theta, or a tensor output."
    )


# ---------------------------------------------------------------------
# Feasibility masks
# ---------------------------------------------------------------------


def _extract_sorted_constraint_rows(
    pb: Any,
    *,
    include_empty_constraints: bool = False,
) -> Tuple[Any, np.ndarray, np.ndarray, List[Tuple[Tuple[int, ...], int]]]:
    A, lb, ub = pb.linear_constraints_matrix()

    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)

    rows: List[Tuple[Tuple[int, ...], int]] = []

    if sp is not None and sp.issparse(A):
        A_csr = A.tocsr()
        for i in range(A_csr.shape[0]):
            start, end = A_csr.indptr[i], A_csr.indptr[i + 1]
            cols = A_csr.indices[start:end]

            if cols.size == 0 and not include_empty_constraints:
                continue

            scope = tuple(sorted(int(j) for j in cols))
            rows.append((scope, i))
    else:
        A_arr = np.asarray(A, dtype=float)
        for i in range(A_arr.shape[0]):
            cols = np.flatnonzero(A_arr[i] != 0)

            if cols.size == 0 and not include_empty_constraints:
                continue

            scope = tuple(sorted(int(j) for j in cols))
            rows.append((scope, i))

    rows.sort(key=lambda t: (len(t[0]), list(t[0])))

    return A, lb, ub, rows


def _row_dot(
    A: Any,
    scope: Tuple[int, ...],
    row_idx: int,
    bits: int,
) -> float:
    """
    Compute A[row_idx, scope] @ x_scope.

    Little-endian convention:
      bit k corresponds to scope[k].
    """
    if sp is not None and sp.issparse(A):
        A_csr = A if sp.isspmatrix_csr(A) else A.tocsr()

        start, end = A_csr.indptr[row_idx], A_csr.indptr[row_idx + 1]
        cols = A_csr.indices[start:end]
        data = A_csr.data[start:end]
        coef_map = {int(j): float(a) for j, a in zip(cols, data)}

        val = 0.0
        for k, j in enumerate(scope):
            if (bits >> k) & 1:
                val += coef_map.get(j, 0.0)
        return val

    A_arr = np.asarray(A, dtype=float)

    val = 0.0
    for k, j in enumerate(scope):
        if (bits >> k) & 1:
            val += float(A_arr[row_idx, j])
    return val


def factor_assignment_masks(
    pb: Any,
    *,
    device: torch.device,
    add_unaries: bool = True,
    include_empty_constraints: bool = False,
    tol: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Build masks in the same overcomplete sufficient-statistic order as theta.

    Returns
    -------
    feasible:
        True for locally feasible assignments.
        Unary assignments are marked feasible.

    infeasible:
        Logical inverse of feasible.

    unary:
        True for unary sufficient-statistic entries.

    constraint:
        True for constraint sufficient-statistic entries.

    constraint_feasible:
        True only for feasible constraint assignments.

    constraint_infeasible:
        True only for infeasible constraint assignments.
    """
    A, lb, ub, rows = _extract_sorted_constraint_rows(
        pb,
        include_empty_constraints=include_empty_constraints,
    )

    if hasattr(pb, "nv"):
        n = int(pb.nv())
    elif hasattr(pb, "n"):
        n = int(pb.n)
    else:
        n = int(A.shape[1])

    ss_len = 0

    if add_unaries:
        ss_len += 2 * n

    for scope, _row_idx in rows:
        ss_len += 1 << len(scope)

    feasible = torch.ones(ss_len, device=device, dtype=torch.bool)
    unary = torch.zeros(ss_len, device=device, dtype=torch.bool)
    constraint = torch.zeros(ss_len, device=device, dtype=torch.bool)

    c = 0

    if add_unaries:
        unary[c : c + 2 * n] = True
        c += 2 * n

    for scope, row_idx in rows:
        arity = len(scope)
        num_assignments = 1 << arity

        constraint[c : c + num_assignments] = True

        if arity == 0:
            Ax = 0.0
            violated = (
                (np.isfinite(lb[row_idx]) and Ax < lb[row_idx] - tol)
                or (np.isfinite(ub[row_idx]) and Ax > ub[row_idx] + tol)
            )
            feasible[c] = not violated
            c += 1
            continue

        for bits in range(num_assignments):
            Ax = _row_dot(A, scope, row_idx, bits)

            violated = (
                (np.isfinite(lb[row_idx]) and Ax < lb[row_idx] - tol)
                or (np.isfinite(ub[row_idx]) and Ax > ub[row_idx] + tol)
            )

            feasible[c] = not violated
            c += 1

    infeasible = ~feasible

    return {
        "feasible": feasible,
        "infeasible": infeasible,
        "unary": unary,
        "constraint": constraint,
        "constraint_feasible": constraint & feasible,
        "constraint_infeasible": constraint & infeasible,
    }


# ---------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------


def run_theta_distribution(
    cfg: Dict[str, Any],
    *,
    config_path: str,
    checkpoint_path: str,
    output_json: str,
    max_instances: Optional[int] = None,
    include_unaries: bool = False,
    save_per_instance_values: bool = False,
    tol: float = 0.0,
) -> Dict[str, Any]:
    torch.manual_seed(int(cfg.get("seed", 0)))
    np.random.seed(int(cfg.get("seed", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = make_loaders(cfg, splits=["test"])

    method_name = cfg.get("train", {}).get("method", "var_logits")
    eval_method = cfg.get("eval", {}).get("method") or method_name

    if eval_method not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown eval method '{eval_method}'. "
            f"Known methods: {list(METHOD_REGISTRY.keys())}"
        )

    adapter_eval = METHOD_REGISTRY[eval_method]
    adapter_eval.loss_name = cfg.get("eval", {}).get("loss-name") or cfg["loss"]["name"]

    model = build_model(cfg, device)
    load_checkpoint(checkpoint_path, model, optimizer=None, map_location=device)
    model.eval()

    theta_feasible_all: List[float] = []
    theta_infeasible_all: List[float] = []
    theta_unary_all: List[float] = []
    instances: List[Dict[str, Any]] = []

    initial_payload = build_payload(
        status="running",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        include_unaries=include_unaries,
        tol=tol,
        theta_feasible=theta_feasible_all,
        theta_infeasible=theta_infeasible_all,
        theta_unary=theta_unary_all,
        instances=instances,
    )
    atomic_write_json(output_json, initial_payload)

    with torch.no_grad():
        bar = Bar("[ThetaDist]     ", max=len(loaders.test_loader))

        for idx, batch in enumerate(CUDAPrefetcher(loaders.test_loader, device)):
            if max_instances is not None and idx >= max_instances:
                break

            # Your test loader yields (pb, data, y), same as final_test_eval.
            pb, data, y = batch
            pb0 = pb[0] if isinstance(pb, (list, tuple)) else pb

            stem = None
            if hasattr(loaders.test_loader, "dataset") and hasattr(loaders.test_loader.dataset, "stems"):
                try:
                    stem = Path(loaders.test_loader.dataset.stems[idx]).name
                except Exception:
                    stem = None

            pred = adapter_eval.forward_fn(model, data)
            pred_theta = extract_pred_theta(pred).detach().flatten()

            masks = factor_assignment_masks(
                pb0,
                device=pred_theta.device,
                add_unaries=True,
                tol=tol,
            )

            if masks["feasible"].numel() != pred_theta.numel():
                raise RuntimeError(
                    f"Theta/mask length mismatch on instance {idx}: "
                    f"pred_theta has length {pred_theta.numel()}, "
                    f"mask has length {masks['feasible'].numel()}. "
                    "This usually means factor ordering, add_unaries, or model output "
                    "does not match the LP factor graph."
                )

            theta_constraint_feasible = pred_theta[masks["constraint_feasible"]]
            theta_constraint_infeasible = pred_theta[masks["constraint_infeasible"]]
            theta_unary = pred_theta[masks["unary"]]

            if include_unaries:
                theta_feasible_for_hist = torch.cat(
                    [theta_constraint_feasible, theta_unary],
                    dim=0,
                )
            else:
                theta_feasible_for_hist = theta_constraint_feasible

            feasible_values = theta_feasible_for_hist.detach().cpu().float().tolist()
            infeasible_values = theta_constraint_infeasible.detach().cpu().float().tolist()
            unary_values = theta_unary.detach().cpu().float().tolist()

            theta_feasible_all.extend(feasible_values)
            theta_infeasible_all.extend(infeasible_values)
            theta_unary_all.extend(unary_values)

            instance_record: Dict[str, Any] = {
                "instance_index": int(idx),
                "instance_stem": stem,
                "num_theta": int(pred_theta.numel()),
                "num_constraint_feasible": int(theta_constraint_feasible.numel()),
                "num_constraint_infeasible": int(theta_constraint_infeasible.numel()),
                "num_unary": int(theta_unary.numel()),
                "constraint_feasible": values_summary(
                    theta_constraint_feasible.detach().cpu().float().tolist()
                ),
                "constraint_infeasible": values_summary(
                    theta_constraint_infeasible.detach().cpu().float().tolist()
                ),
                "unary": values_summary(unary_values),
            }

            if include_unaries:
                instance_record["feasible_with_unaries"] = values_summary(feasible_values)

            if save_per_instance_values:
                instance_record["theta_feasible"] = feasible_values
                instance_record["theta_infeasible"] = infeasible_values
                instance_record["theta_unary"] = unary_values

            instances.append(instance_record)

            partial_payload = build_payload(
                status="running",
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                include_unaries=include_unaries,
                tol=tol,
                theta_feasible=theta_feasible_all,
                theta_infeasible=theta_infeasible_all,
                theta_unary=theta_unary_all,
                instances=instances,
            )
            atomic_write_json(output_json, partial_payload)

            print()
            print(
                f"[theta {idx:04d}] "
                f"feas={len(feasible_values)} "
                f"infeas={len(infeasible_values)} "
                f"unary={len(unary_values)}"
            )

            bar.next()

        bar.finish()

    final_payload = build_payload(
        status="completed",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        include_unaries=include_unaries,
        tol=tol,
        theta_feasible=theta_feasible_all,
        theta_infeasible=theta_infeasible_all,
        theta_unary=theta_unary_all,
        instances=instances,
    )
    atomic_write_json(output_json, final_payload)

    return final_payload


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    base_parser = build_base_parser()
    base_args, _remaining = base_parser.parse_known_args(argv)

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
    checkpoint_path = _pick_checkpoint_path(cfg, paths, args)

    print("checkpoint path = ", checkpoint_path)
    print("output json     = ", args.output_json)

    stats = run_theta_distribution(
        cfg,
        config_path=args.config,
        checkpoint_path=checkpoint_path,
        output_json=args.output_json,
        max_instances=args.max_instances,
        include_unaries=args.include_unaries,
        save_per_instance_values=args.save_per_instance_values,
        tol=args.tol,
    )

    print(json.dumps(stats["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
