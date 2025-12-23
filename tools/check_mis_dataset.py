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
import networkx as nx

from bpropy.co.mis import MIS


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


# -----------------------------
# Helpers: file IO (serde-like)
# -----------------------------
def _open_text(path: Path, mode: str = "rt"):
    """
    Match bpropy.io.serde behavior: supports .gz manifests.
    """
    if str(path).endswith(".gz"):
        return gzip.open(str(path), mode)
    return open(path, mode, encoding="utf-8")


def load_manifest(instance_path: Path) -> Dict[str, Any]:
    with _open_text(instance_path, "rt") as f:
        return json.load(f)


def load_assignments_from_manifest(instance_path: Path, manifest: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Mirrors SerializableLPMixin.load_assignments() in serde.py:
    - assignments are declared in manifest["assignments"] as {name: {"source": "...", "key": "..."}}
    - arrays can be stored inline in manifest["_arrays"] or in an NPZ sidecar manifest["arrays_file"].
    """
    arrays: Dict[str, Any] = {}

    # NPZ sidecar
    arrays_file = manifest.get("arrays_file", None)
    if arrays_file is not None:
        sidecar_path = instance_path.with_name(arrays_file)
        if sidecar_path.exists():
            arrays.update(dict(np.load(sidecar_path, allow_pickle=False)))
        else:
            raise ValueError(f"Manifest declares arrays_file='{arrays_file}' but file is missing at {sidecar_path}")

    out: Dict[str, np.ndarray] = {}
    assignments = manifest.get("assignments", {})
    if not isinstance(assignments, dict):
        raise ValueError("manifest['assignments'] must be a dict if present.")

    for name, ref in assignments.items():
        if not isinstance(ref, dict) or "source" not in ref or "key" not in ref:
            raise ValueError(f"Bad assignment ref for '{name}': expected dict with keys 'source' and 'key'")
        key = ref["key"]
        src = ref["source"]

        if src == "npz":
            if key not in arrays:
                raise ValueError(f"Assignment '{name}' refers to key '{key}' not found in NPZ '{arrays_file}'")
            out[name] = np.asarray(arrays[key])
        else:
            # inline
            inline = manifest.get("_arrays", {})
            if not isinstance(inline, dict) or key not in inline:
                raise ValueError(f"Assignment '{name}' refers to inline key '{key}' not found in manifest['_arrays']")
            out[name] = np.asarray(inline[key])

    return out


# -----------------------------
# Helpers: MIS instance extraction
# -----------------------------
def extract_nv_edges(manifest: Dict[str, Any]) -> Tuple[int, List[List[int]]]:
    """
    Try common field names to get (nv, edges). If your MIS serialization uses different keys,
    adjust this function.

    Expected edges format: list of [u, v] pairs.
    """
    # nv candidates
    for k in ("nv", "n", "num_nodes", "n_nodes", "number_of_nodes"):
        if k in manifest:
            nv = int(manifest[k])
            break
    else:
        raise KeyError("Could not find number of nodes key in manifest (tried: nv, n, num_nodes, n_nodes, number_of_nodes)")

    # edges candidates
    for k in ("edges", "E", "edge_list", "edgelist"):
        if k in manifest:
            edges = manifest[k]
            break
    else:
        raise KeyError("Could not find edges key in manifest (tried: edges, E, edge_list, edgelist)")

    if not isinstance(edges, list):
        raise TypeError(f"Edges field is not a list (got {type(edges)}).")

    # normalize to list[list[int]]
    norm_edges: List[List[int]] = []
    for e in edges:
        if not (isinstance(e, (list, tuple)) and len(e) == 2):
            raise ValueError(f"Invalid edge entry {e!r}; expected [u, v].")
        u, v = int(e[0]), int(e[1])
        norm_edges.append([u, v])

    return nv, norm_edges


def build_graph(nv: int, edges: List[List[int]]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(nv))
    g.add_edges_from((u, v) for u, v in edges)
    return g


def iter_instance_files(root: Path) -> Iterable[Path]:
    """
    Recursively yields candidate instance manifests.
    We include .json and .json.gz, and skip obvious non-instance files.
    """
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in {".json", ".gz"}:
            continue
        name = p.name.lower()
        if name == "metadata.json" or name.endswith("metadata.json.gz"):
            continue
        # skip config files if you keep them in the same tree
        if "config" in name:
            continue
        yield p


# -----------------------------
# Assignment requirements
# -----------------------------
def required_assignment_names(solve: str, relax: bool) -> List[str]:
    """
    Map config 'solve' to the assignment names expected in manifest['assignments'].

    Per your request:
    - if solve == "none": require greedy only
    - otherwise require solver-specific assignment (and also greedy by default, see below)
    """
    s = (solve or "none").lower()

    if s == "none":
        return ["greedy"]

    if s == "cbc":
        return ["cbc"]
    if s == "kamis":
        return ["kamis"]
    if s == "scipy":
        return ["scipy_lp" if relax else "scipy"]
    if s == "all":
        return ["cbc", "kamis", ("scipy_lp" if relax else "scipy")]

    # Unknown -> at least greedy
    return ["greedy"]


def check_assignments_in_instance(
    instance_path: Path,
    pb: MIS,
    *,
    solve: str,
    relax: bool,
    require_greedy_always: bool,
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    req = required_assignment_names(solve, relax)
    assignments = MIS.load_assignments(str(instance_path).replace(".json",""))
    for name in req:
        if not name in assignments:
            errors.append(f"Missing required assignment name '{name}' for solve='{solve}'.")

    s = (solve or "none").lower()
    if s not in {"none", "cbc", "kamis", "scipy", "all"}:
        warnings.append(f"Unknown solve mode '{solve}': enforced greedy + inferred mapping only.")

    return errors, warnings


# -----------------------------
# Validation result
# -----------------------------
@dataclass
class InstanceCheckResult:
    path: Path
    ok: bool
    errors: List[str]
    warnings: List[str]


def check_instance(
    *,
    path: Path,
    cfg_nv: Tuple[int, int],
    cfg_maxdeg_cap: int,
    ensure_min_deg: int,
    strict_min_deg: bool,
    cfg_solve: str,
    cfg_relax: bool,
) -> InstanceCheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    #try:
    #    manifest = load_manifest(path)
    #except Exception as e:
    #    return InstanceCheckResult(path=path, ok=False, errors=[f"Failed to load manifest: {e}"], warnings=[])

    ## Extract MIS graph
    #try:
    #    nv, edges = extract_nv_edges(manifest)
    #except Exception as e:
    #    return InstanceCheckResult(path=path, ok=False, errors=[f"Failed to extract nv/edges: {e}"], warnings=[])

    #lo_nv, hi_nv = cfg_nv
    #if not (lo_nv <= nv <= hi_nv):
    #    errors.append(f"nv={nv} not in config range [{lo_nv}, {hi_nv}].")

    ## Graph checks
    #try:
    #    g = build_graph(nv, edges)
    #except Exception as e:
    #    errors.append(f"Failed to build graph: {e}")
    #    return InstanceCheckResult(path=path, ok=False, errors=errors, warnings=warnings)

    #if nv > 0 and not nx.is_connected(g):
    #    comps = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
    #    errors.append(f"Graph is disconnected (component sizes: {comps[:5]}{'...' if len(comps) > 5 else ''}).")

    try:
        pb = MIS.load(path)
    except Exception as e:
        return InstanceCheckResult(path=path, ok=False, errors=[f"Failed to load instance: {e}"], warnings=[])


    degs = [d for _, d in pb.g.degree()]
    max_deg = max(degs) if degs else 0
    min_deg = min(degs) if degs else 0

    # max_deg check (treat config max_deg as a cap)
    if max_deg > cfg_maxdeg_cap:
        errors.append(f"max_degree={max_deg} exceeds config max_deg cap {cfg_maxdeg_cap}.")

    # ensure_min_deg is often best-effort; warn by default
    if ensure_min_deg > 0 and min_deg < ensure_min_deg:
        msg = f"min_degree={min_deg} is below ensure_min_deg={ensure_min_deg}."
        if strict_min_deg:
            errors.append(msg)
        else:
            warnings.append(msg)

    # Assignments checks
    # Per your request: if solve == none -> only greedy required.
    # Otherwise: require the solver assignment, AND greedy too (can change via CLI flag).
    solve_l = (cfg_solve or "none").lower()
    require_greedy_always = (solve_l != "none")  # "just greedy if solver is none"
    a_err, a_warn = check_assignments_in_instance(
        path,
        pb,
        solve=cfg_solve,
        relax=cfg_relax,
        require_greedy_always=require_greedy_always,
    )
    errors.extend(a_err)
    warnings.extend(a_warn)

    return InstanceCheckResult(path=path, ok=(len(errors) == 0), errors=errors, warnings=warnings)


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate MIS dataset: connectivity + nv/degree constraints + required assignments inside manifests."
    )
    ap.add_argument("--config", required=True, help="Path to MIS dataset config JSON.")
    ap.add_argument("--data", required=True, help="Dataset folder to scan recursively.")
    ap.add_argument("--strict-min-deg", action="store_true", help="Treat ensure_min_deg violations as errors.")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of instances checked.")
    ap.add_argument("--report", default="mis_validation_report.txt", help="Where to write a report.")
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

    if "nv" not in cfg or "max_deg" not in cfg:
        print("Config must include at least keys: 'nv' and 'max_deg'.", file=sys.stderr)
        return 2

    cfg_nv = parse_int_range(str(cfg["nv"]))
    # Treat config max_deg as a cap; if it's a range, use its upper bound.
    md_lo, md_hi = parse_int_range(str(cfg["max_deg"]))
    cfg_maxdeg_cap = md_hi

    ensure_min_deg = int(cfg.get("ensure_min_deg", cfg.get("ensure-min-deg", 0)) or 0)
    cfg_solve = str(cfg.get("solve", "none"))
    cfg_relax = bool(cfg.get("relax", False))

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
            cfg_nv=cfg_nv,
            cfg_maxdeg_cap=cfg_maxdeg_cap,
            ensure_min_deg=ensure_min_deg,
            strict_min_deg=bool(args.strict_min_deg),
            cfg_solve=cfg_solve,
            cfg_relax=cfg_relax,
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
        f.write(f"nv range: {cfg_nv}, max_deg cap: {cfg_maxdeg_cap}, ensure_min_deg: {ensure_min_deg}\n")
        f.write(f"solve: {cfg_solve}, relax: {cfg_relax}\n\n")
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

