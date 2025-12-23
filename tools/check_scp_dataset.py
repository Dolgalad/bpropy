#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from bpropy.co.scp import SCP


# -----------------------------
# Helpers: config parsing
# -----------------------------
def parse_int_range(s: str) -> Tuple[int, int]:
    """
    Parse "k" or "a:b" (inclusive) into (a, b).
    """
    s = str(s).strip()
    if ":" in s:
        a, b = s.split(":")
        lo, hi = int(a), int(b)
    else:
        lo = hi = int(s)
    if lo > hi:
        raise ValueError(f"Invalid int range: {s!r}")
    return lo, hi


def pick_first_key(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None:
            return k
    return None


# -----------------------------
# Helpers: file IO (serde-like)
# -----------------------------
def _open_text(path: Path, mode: str = "rt"):
    """
    Supports .gz json manifests.
    """
    if str(path).endswith(".gz"):
        return gzip.open(str(path), mode)
    return open(path, mode, encoding="utf-8")


# -----------------------------
# Instance discovery
# -----------------------------
def iter_instance_files(root: Path) -> Iterable[Path]:
    """
    Recursively yields candidate instance manifests.
    Includes .json and .json.gz, and skips obvious non-instance files.
    """
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in {".json", ".gz"}:
            continue
        name = p.name.lower()
        if name == "metadata.json" or name.endswith("metadata.json.gz"):
            continue
        if "config" in name:
            continue
        yield p


# -----------------------------
# Assignment requirements
# -----------------------------
def required_assignment_names_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    """
    Determine which assignment names must exist in the instance file.

    Policy (mirrors your MIS checker intent):
    - If solver/solve is missing or "none": require "greedy"
    - Otherwise: require the solver name (lowercased)
      (You can extend mapping here if your assignment names differ from solver names.)
    """
    solver_key = pick_first_key(cfg, ["solver", "solve"])
    solver = str(cfg.get(solver_key, "none") if solver_key else "none").lower().strip()

    if solver in {"", "none", "null"}:
        return ["greedy"]

    # If you have special solver naming conventions for assignments, map them here.
    # Example:
    # if solver == "scg3": return ["scg3"]
    return [solver]


def try_load_assignments(path: Path) -> Dict[str, np.ndarray]:
    """
    Try to load assignments the same way your MIS checker did (some serializers
    expect a prefix/path without .json, others accept the full path).
    """
    # 1) full path
    try:
        return SCP.load_assignments(str(path))
    except Exception:
        pass

    # 2) strip .json (and possibly .gz)
    s = str(path)
    if s.endswith(".json.gz"):
        s2 = s[: -len(".json.gz")]
    elif s.endswith(".json"):
        s2 = s[: -len(".json")]
    else:
        s2 = s
    try:
        return SCP.load_assignments(s2)
    except Exception:
        return {}


# -----------------------------
# Validation result
# -----------------------------
@dataclass
class InstanceCheckResult:
    path: Path
    ok: bool
    errors: List[str]
    warnings: List[str]


def compute_item_degrees(pb: SCP) -> np.ndarray:
    """
    item degree = number of sets containing each item.
    Returns array of length ne.
    """
    ne = pb.ne()
    deg = np.zeros(ne, dtype=int)
    for f in pb.F:
        for a in f:
            if 0 <= int(a) < ne:
                deg[int(a)] += 1
    return deg


def check_instance(
    *,
    path: Path,
    cfg_n_items: Tuple[int, int],
    cfg_n_sets: Tuple[int, int],
    cfg_item_deg: Optional[Tuple[int, int]],
    cfg: Dict[str, Any],
) -> InstanceCheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    # Load SCP instance
    try:
        pb = SCP.load(path)
    except Exception as e:
        return InstanceCheckResult(path=path, ok=False, errors=[f"Failed to load SCP instance: {e}"], warnings=[])

    ne = pb.ne()  # items
    nv = pb.nv()  # sets

    # Range checks for ne/nv

    lo_i, hi_i = cfg_n_items
    if not (lo_i <= ne <= hi_i):
        errors.append(f"ne (n_items)={ne} not in config range [{lo_i}, {hi_i}].")
        print(errors[-1])

    lo_s, hi_s = cfg_n_sets
    if not (lo_s <= nv <= hi_s):
        errors.append(f"nv (n_sets)={nv} not in config range [{lo_s}, {hi_s}].")
        print(errors[-1])

    # Empty set checks
    empty_sets = [j for j, f in enumerate(pb.F) if len(f) == 0]
    if empty_sets:
        # show only first few indices
        head = empty_sets[:10]
        tail = "..." if len(empty_sets) > 10 else ""
        errors.append(f"Found {len(empty_sets)} empty sets (indices: {head}{tail}).")
        print(errors[-1])

    # Basic sanity: all elements in sets within [0, ne-1]
    bad_entries = 0
    for j, f in enumerate(pb.F):
        for a in f:
            ai = int(a)
            if ai < 0 or ai >= ne:
                bad_entries += 1
                if bad_entries <= 5:
                    errors.append(f"Set F[{j}] contains out-of-range item {ai} (valid: 0..{ne-1}).")
                    print(errors[-1])

    if bad_entries > 5:
        errors.append(f"Found {bad_entries} out-of-range item occurrences in total.")
        print(errors[-1])

    # Item coverage degree checks (if specified)
    if cfg_item_deg is not None:
        dlo, dhi = cfg_item_deg
        deg = compute_item_degrees(pb)
        if deg.size != ne:
            errors.append(f"Internal error: computed item degree length {deg.size} != ne {ne}.")
            print(errors[-1])

        else:
            too_low = np.where(deg < dlo)[0]
            too_high = np.where(deg > dhi)[0]
            if too_low.size:
                show = too_low[:10].tolist()
                tail = "..." if too_low.size > 10 else ""
                errors.append(
                    f"{too_low.size} items have degree < {dlo} (first indices: {show}{tail}); "
                    f"min_degree={int(deg.min())}."
                )
                print(errors[-1])

            if too_high.size:
                show = too_high[:10].tolist()
                tail = "..." if too_high.size > 10 else ""
                errors.append(
                    f"{too_high.size} items have degree > {dhi} (first indices: {show}{tail}); "
                    f"max_degree={int(deg.max())}."
                )
                print(errors[-1])

    # Assignment presence checks
    req_assign = required_assignment_names_from_cfg(cfg)
    assignments = try_load_assignments(path)
    for name in req_assign:
        if name not in assignments:
            errors.append(
                f"Missing required assignment '{name}'. "
                f"(Config solver/solve implies required={req_assign}; found={sorted(assignments.keys())})"
            )
            print(errors[-1])

    return InstanceCheckResult(path=path, ok=(len(errors) == 0), errors=errors, warnings=warnings)


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate SCP dataset: n_items/n_sets ranges, empty sets, per-item coverage degree, and required assignments."
    )
    ap.add_argument("--config", required=True, help="Path to SCP dataset config JSON.")
    ap.add_argument("--data", required=True, help="Dataset folder to scan recursively.")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of instances checked.")
    ap.add_argument("--report", default="scp_validation_report.txt", help="Where to write a report.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    data_root = Path(args.data)

    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 2
    if not data_root.exists():
        print(f"Dataset folder not found: {data_root}", file=sys.stderr)
        return 2

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Required config keys
    k_items = pick_first_key(cfg, ["n_items", "ne", "items"])
    k_sets = pick_first_key(cfg, ["n_sets", "nv", "sets"])
    if k_items is None or k_sets is None:
        print(
            "Config must include item and set counts. Expected one of:\n"
            "  items:  n_items / ne / items\n"
            "  sets:   n_sets / nv / sets",
            file=sys.stderr,
        )
        return 2

    cfg_n_items = parse_int_range(str(cfg[k_items]))
    cfg_n_sets = parse_int_range(str(cfg[k_sets]))

    # Optional per-item degree (coverage count) key
    k_deg = pick_first_key(cfg, ["k", "deg", "item_deg", "item_degree"])
    cfg_item_deg: Optional[Tuple[int, int]] = parse_int_range(str(cfg[k_deg])) if k_deg is not None else None

    files = sorted(iter_instance_files(data_root))
    if args.limit is not None:
        files = files[: max(0, int(args.limit))]

    if not files:
        print(f"No instance files found under: {data_root}", file=sys.stderr)
        return 1

    n_ok = 0
    n_err = 0
    n_warn = 0
    failures: List[InstanceCheckResult] = []

    for i, path in enumerate(files, 1):
        res = check_instance(
            path=path,
            cfg_n_items=cfg_n_items,
            cfg_n_sets=cfg_n_sets,
            cfg_item_deg=cfg_item_deg,
            cfg=cfg,
        )
        n_warn += len(res.warnings)
        if res.ok:
            n_ok += 1
        else:
            n_err += 1
            failures.append(res)

        if i % 200 == 0:
            print(f"Checked {i}/{len(files)} ... ok={n_ok} err={n_err} warnings={n_warn}", flush=True)

    print(f"\nDone. Checked {len(files)} instances.")
    print(f"OK: {n_ok}")
    print(f"Errors: {n_err}")
    print(f"Warnings: {n_warn}")

    report_path = Path(args.report)
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"Config: {cfg_path}\nData: {data_root}\n")
        f.write(f"n_items range: {cfg_n_items}, n_sets range: {cfg_n_sets}\n")
        if cfg_item_deg is not None:
            f.write(f"item degree (coverage) range: {cfg_item_deg}\n")
        else:
            f.write("item degree (coverage) range: (not specified in config)\n")
        f.write(f"Required assignments (by config): {required_assignment_names_from_cfg(cfg)}\n\n")
        f.write(f"Checked {len(files)} instances. OK={n_ok} ERR={n_err} WARN={n_warn}\n\n")

        if failures:
            f.write("=== FAILURES ===\n")
            for res in failures:
                f.write(f"\n{res.path}\n")
                for e in res.errors:
                    f.write(f"  ERROR: {e}\n")
                for w in res.warnings:
                    f.write(f"  WARN:  {w}\n")
        else:
            f.write("No failures.\n")

    if failures:
        print(f"Failure report written to: {report_path}")
        return 1

    print(f"Report written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

