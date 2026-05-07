#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


PLOT_KEYS = [
    "entropy",
    "value",
    "mu1",
    "chosen_entropy",
    "chosen_mu1",
    "chosen_eff",
    "entropy_gap_to_mean",
    "mu1_gap_to_mean",
    "eff_gap_to_mean",
    "entropy_rank_asc",
    "mu1_rank_desc",
    "eff_rank_desc",
    "entropy_gap_best_second",
    "mu_gap_best_second",
    "eff_gap_best_second",
    "candidate_entropy_mean",
    "candidate_entropy_std",
    "candidate_mu_mean",
    "candidate_mu_std",
    "candidate_entropy_min",
    "candidate_entropy_max",
    "candidate_mu_min",
    "candidate_mu_max",
    "n_candidates",
    "newly_covered_count",
    "cumulative_newly_covered",
    "value_times_newly_covered",
    "cumulative_value_times_newly_covered",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract selected-variable diagnostics per iteration from GD/EGG log JSON files "
            "and save aggregated CSV files. "
            "Expected layout: ROOT/<dataset_id>/<model_id>/<seed>.json"
        )
    )
    p.add_argument("root", type=Path)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--min-support", type=int, default=1)

    # kept for CLI compatibility with the plotting script
    p.add_argument("--formats", nargs="+", default=["png"])
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--title-prefix", type=str, default="Selected-variable diagnostics")
    return p.parse_args()


def safe_entropy_from_mu1(mu1: float) -> float:
    p = np.clip(float(mu1), 1e-12, 1.0 - 1e-12)
    return float(-p * np.log(p) - (1.0 - p) * np.log(1.0 - p))


def pad_nan(seqs: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    if not seqs:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=int)

    max_len = max(len(s) for s in seqs)
    arr = np.full((len(seqs), max_len), np.nan, dtype=float)

    for i, seq in enumerate(seqs):
        arr[i, : len(seq)] = seq

    support = np.sum(~np.isnan(arr), axis=0)
    return arr, support


def aggregate_mean_std_support(
    seqs: List[List[float]],
    min_support: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr, support = pad_nan(seqs)
    if arr.size == 0:
        return np.empty((0,)), np.empty((0,)), np.empty((0,), dtype=int)

    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)

    keep = support >= min_support
    return mean[keep], std[keep], support[keep]


def sanitize_column_name(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "model"


def load_metrics_from_run(json_path: Path) -> Dict[str, Dict[str, List[List[float]]]]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    out: Dict[str, Dict[str, List[List[float]]]] = {
        "egg": {k: [] for k in PLOT_KEYS},
        "gd": {k: [] for k in PLOT_KEYS},
    }

    for inst in payload.get("instances", []):
        for variant in ("egg", "gd"):
            block = inst.get(variant)
            if not block:
                continue

            info_list = block.get("info", [])
            if not info_list:
                continue

            seqs = {k: [] for k in PLOT_KEYS}

            cumulative_cov = 0.0
            cumulative_value_gain = 0.0

            for step in info_list:
                if not step:
                    continue

                score = step.get("score")
                mu1 = step.get("mu1")
                value = step.get("value")

                if score is not None:
                    seqs["entropy"].append(float(score))
                elif mu1 is not None:
                    seqs["entropy"].append(safe_entropy_from_mu1(mu1))

                if value is not None:
                    seqs["value"].append(float(value))

                if mu1 is not None:
                    seqs["mu1"].append(float(mu1))

                for k in PLOT_KEYS:
                    if k in {
                        "entropy",
                        "value",
                        "mu1",
                        "newly_covered_count",
                        "cumulative_newly_covered",
                        "value_times_newly_covered",
                        "cumulative_value_times_newly_covered",
                    }:
                        continue
                    if k in step and step[k] is not None:
                        seqs[k].append(float(step[k]))

                newly_covered_count = None
                if "newly_covered" in step and step["newly_covered"] is not None:
                    newly_covered_count = len(step["newly_covered"])
                elif "newly_covered_count" in step and step["newly_covered_count"] is not None:
                    newly_covered_count = int(step["newly_covered_count"])

                if newly_covered_count is not None:
                    newly_covered_count = float(newly_covered_count)
                    seqs["newly_covered_count"].append(newly_covered_count)
                    cumulative_cov += newly_covered_count
                    seqs["cumulative_newly_covered"].append(cumulative_cov)

                    if value is not None:
                        vtg = float(value) * newly_covered_count
                        seqs["value_times_newly_covered"].append(vtg)
                        cumulative_value_gain += vtg
                        seqs["cumulative_value_times_newly_covered"].append(cumulative_value_gain)

            for k in PLOT_KEYS:
                if seqs[k]:
                    out[variant][k].append(seqs[k])

    return out


def collect_all_runs(
    root: Path,
    ignored_dirnames: set[str] | None = None,
) -> Dict[str, Dict[str, Dict[str, Dict[str, List[List[float]]]]]]:
    if ignored_dirnames is None:
        ignored_dirnames = set()

    data: Dict[str, Dict[str, Dict[str, Dict[str, List[List[float]]]]]] = {}

    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if dataset_dir.name in ignored_dirnames:
            continue

        dataset_id = dataset_dir.name
        data.setdefault(dataset_id, {})

        for model_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            json_files = sorted(model_dir.glob("*.json"))
            if not json_files:
                continue

            model_id = model_dir.name
            agg = {
                "egg": {k: [] for k in PLOT_KEYS},
                "gd": {k: [] for k in PLOT_KEYS},
            }

            for json_path in json_files:
                run_metrics = load_metrics_from_run(json_path)
                for variant in ("egg", "gd"):
                    for metric in PLOT_KEYS:
                        agg[variant][metric].extend(run_metrics[variant][metric])

            if any(
                agg[variant][metric]
                for variant in ("egg", "gd")
                for metric in PLOT_KEYS
            ):
                data[dataset_id][model_id] = agg

    return data


def build_variant_table(
    dataset_data: Dict[str, Dict[str, Dict[str, List[List[float]]]]],
    variant: str,
    min_support: int,
) -> Tuple[List[str], List[List[object]]]:
    model_stats: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    for model_id in sorted(dataset_data):
        stats_for_model: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        has_any = False

        for metric in PLOT_KEYS:
            mean, std, support = aggregate_mean_std_support(
                dataset_data[model_id][variant][metric], min_support
            )
            stats_for_model[metric] = (mean, std, support)
            if mean.size > 0:
                has_any = True

        if has_any:
            model_stats[model_id] = stats_for_model

    if not model_stats:
        return [], []

    max_len = 0
    for stats in model_stats.values():
        for metric in PLOT_KEYS:
            max_len = max(max_len, stats[metric][0].size)

    ordered_models = sorted(model_stats)
    safe_names = {m: sanitize_column_name(m) for m in ordered_models}

    header = ["iter"]
    for model_id in ordered_models:
        col = safe_names[model_id]
        for metric in PLOT_KEYS:
            header.extend(
                [
                    f"{col}__{metric}__mean",
                    f"{col}__{metric}__std",
                    f"{col}__{metric}__support",
                ]
            )

    rows: List[List[object]] = []
    for i in range(max_len):
        row: List[object] = [i]

        for model_id in ordered_models:
            stats = model_stats[model_id]

            for metric in PLOT_KEYS:
                mean, std, support = stats[metric]

                row.extend(
                    [
                        float(mean[i]) if i < mean.size else float("nan"),
                        float(std[i]) if i < std.size else float("nan"),
                        float(support[i]) if i < support.size else float("nan"),
                    ]
                )

        rows.append(row)

    return header, rows


def write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[ok] wrote {path}")


def write_dataset_csvs(
    dataset_id: str,
    dataset_data: Dict[str, Dict[str, Dict[str, List[List[float]]]]],
    output_dir: Path,
    min_support: int,
) -> None:
    for variant in ("egg", "gd"):
        header, rows = build_variant_table(dataset_data, variant, min_support)
        if not header:
            print(f"[skip] {dataset_id}/{variant}: no usable trajectories found")
            continue

        out_path = output_dir / f"{dataset_id}_{variant}_selected_variable_diagnostics.csv"
        write_csv(out_path, header, rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else root / "csv"

    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")

    ignored = {output_dir.name} if output_dir.parent == root else set()
    all_data = collect_all_runs(root, ignored_dirnames=ignored)

    if not all_data:
        raise RuntimeError(f"No dataset directories found under {root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_id in sorted(all_data):
        write_dataset_csvs(
            dataset_id=dataset_id,
            dataset_data=all_data[dataset_id],
            output_dir=output_dir,
            min_support=args.min_support,
        )


if __name__ == "__main__":
    main()
