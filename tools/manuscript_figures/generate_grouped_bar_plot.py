#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as stats
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

Number = Union[int, float]

METHOD_ALIASES = {
    "EGG": "entropy_greedy",
    "egg": "entropy_greedy",
    "entropy_greedy": "entropy_greedy",
    "GD": "guided_decimation",
    "gd": "guided_decimation",
    "guided_decimation": "guided_decimation",
    "greedy": "greedy_pred",
    "greedy_pred": "greedy_pred",
}

METRIC_ALIASES = {
    "gap": "gap",
    "relative_gap": "gap",
    "rel_gap": "gap",
    "obj": "obj",
    "objective": "obj",
    "runtime": "total_t",
    "time": "total_t",
    "total_time": "total_t",
    "total_t": "total_t",
    "decode_time": "decode_t",
    "decode_t": "decode_t",
    "forward_time": "forward_t",
    "forward_t": "forward_t",
}


def normalize_method(method: Optional[str]) -> Optional[str]:
    if method is None:
        return None
    return METHOD_ALIASES.get(str(method), str(method))


def normalize_metric(metric: str) -> str:
    return METRIC_ALIASES.get(str(metric), str(metric))


def load_json(path_or_dir: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path_or_dir).expanduser()
    if p.is_dir():
        candidates = [
            p / "eval_test.json",
            p / "test_eval.json",
            p / "test_results.json",
            p / "eval.json",
        ]
        existing = [q for q in candidates if q.exists()]
        if not existing:
            for name in ("eval_test.json", "test_eval.json"):
                matches = sorted(p.rglob(name))
                if matches:
                    existing = [matches[0]]
                    break
        if not existing:
            raise FileNotFoundError(f"No eval_test.json or test_eval.json found in {p}")
        p = existing[0]
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_under_root(path: Union[str, Path], config: Mapping[str, Any]) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    root = config.get("root_dir")
    if root:
        return Path(str(root)).expanduser() / p
    return p


def _glob_existing(pattern: str) -> List[Path]:
    return [Path(p) for p in sorted(glob.glob(str(Path(pattern).expanduser())))]


def seed_json_paths(spec: Mapping[str, Any], config: Mapping[str, Any]) -> List[Path]:
    if "json_files" in spec:
        return [resolve_under_root(p, config) for p in spec["json_files"]]
    if "files" in spec:
        return [resolve_under_root(p, config) for p in spec["files"]]
    if "runs" in spec:
        return [resolve_under_root(run, config) for run in spec["runs"]]
    if "glob" in spec:
        glob_pattern = resolve_under_root(str(spec["glob"]), config)
        paths = _glob_existing(str(glob_pattern))
        if paths:
            return paths

    prefix = spec.get("run_prefix") or spec.get("prefix")
    if not prefix:
        raise KeyError("Each bar spec needs one of: json_files, files, runs, glob, run_prefix, prefix")

    prefix = str(resolve_under_root(str(prefix), config))
    seeds = spec.get("seeds", config.get("seeds"))
    seed_pattern = spec.get(
        "seed_pattern",
        config.get("seed_pattern", "{prefix}_seed{seed}/test_eval.json"),
    )

    if seeds is not None:
        return [Path(seed_pattern.format(prefix=prefix, seed=seed)).expanduser() for seed in seeds]

    discovered: List[Path] = []
    for name in ("eval_test.json", "test_eval.json"):
        discovered.extend(_glob_existing(f"{prefix}_seed*/{name}"))
    return sorted(set(discovered))


def summary_mean_std_from_payload(
    payload: Mapping[str, Any],
    metric: str,
    method: Optional[str],
) -> Optional[Tuple[float, float]]:
    method = normalize_method(method)
    metric = normalize_metric(metric)

    if method is None:
        return None

    method_summary = payload.get("summary", {}).get("methods", {}).get(method)
    if not isinstance(method_summary, Mapping):
        return None

    mean = method_summary.get(f"{metric}_mean")
    std = method_summary.get(f"{metric}_std", 0.0)
    if mean is None:
        return None
    return float(mean), float(std)


def metric_scale(metric: str, spec: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    metric = normalize_metric(metric)
    if "scale" in spec:
        return float(spec["scale"])
    metric_scales = config.get("metric_scales", {})
    if metric in metric_scales:
        return float(metric_scales[metric])
    if metric == "gap":
        return 100.0
    return 1.0


def aggregate_seed_mean_std(
    payloads: Sequence[Mapping[str, Any]],
    method: Optional[str],
    metric: str,
    scale: float,
    std_mode: str = "pooled",
) -> Optional[Tuple[float, float]]:
    per_seed = [summary_mean_std_from_payload(payload, metric=metric, method=method) for payload in payloads]
    per_seed = [x for x in per_seed if x is not None]
    if not per_seed:
        return None

    means = [x[0] for x in per_seed]
    stds = [x[1] for x in per_seed]
    mean = stats.fmean(means)

    if len(per_seed) == 1:
        std = stds[0]
    elif std_mode == "seed_means":
        std = stats.pstdev(means)
    else:
        std = math.sqrt(stats.fmean([s * s for s in stds]))

    return mean * scale, std * scale


def load_payloads_for_bar(bar_spec: Mapping[str, Any], config: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    paths = seed_json_paths(bar_spec, config)
    existing = [p for p in paths if p.exists() or p.is_dir()]
    payloads: List[Mapping[str, Any]] = []
    for p in existing:
        try:
            payloads.append(load_json(p))
        except FileNotFoundError:
            pass
    return payloads


def bar_mean_std(bar_spec: Mapping[str, Any], config: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    payloads = load_payloads_for_bar(bar_spec, config)
    if not payloads:
        return None

    method = bar_spec.get("method", config.get("method"))
    metric = bar_spec.get("metric", config.get("metric", "gap"))
    scale = metric_scale(str(metric), bar_spec, config)
    std_mode = str(bar_spec.get("std_mode", config.get("std_mode", "pooled")))
    return aggregate_seed_mean_std(payloads, method=method, metric=str(metric), scale=scale, std_mode=std_mode)


def make_grouped_bar_plot(config: Dict[str, Any], output: Path) -> None:
    groups = config["groups"]
    if not groups:
        raise ValueError("Config must contain a non-empty 'groups' list.")

    group_labels = [str(g["label"]) for g in groups]
    series_defs = config.get("series")
    if not series_defs:
        raise ValueError("Config must contain a non-empty 'series' list.")

    series_keys = [str(s["key"]) for s in series_defs]
    series_titles = [str(s.get("title", s["key"])) for s in series_defs]

    values_by_series: List[List[float]] = []
    stds_by_series: List[List[float]] = []

    for sdef in series_defs:
        skey = str(sdef["key"])
        vals: List[float] = []
        errs: List[float] = []
        for group in groups:
            results = group.get("results", {})
            spec0 = results.get(skey)
            if spec0 is None:
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            if isinstance(spec0, str):
                spec: Dict[str, Any] = {"run_prefix": spec0}
            else:
                spec = dict(spec0)

            merged: Dict[str, Any] = dict(sdef)
            merged.update(spec)

            ms = bar_mean_std(merged, config)
            if ms is None:
                vals.append(np.nan)
                errs.append(np.nan)
            else:
                vals.append(ms[0])
                errs.append(ms[1])

        values_by_series.append(vals)
        stds_by_series.append(errs)

    n_groups = len(groups)
    n_series = len(series_defs)
    x = np.arange(n_groups)
    total_group_width = float(config.get("group_width", 0.8))
    bar_width = total_group_width / max(n_series, 1)
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * bar_width

    fig_width = float(config.get("figsize", [max(6.0, 1.6 * n_groups + 1.2 * n_series), 4.8])[0])
    fig_height = float(config.get("figsize", [max(6.0, 1.6 * n_groups + 1.2 * n_series), 4.8])[1])
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    show_error_bars = bool(config.get("show_error_bars", False))

    for i, title in enumerate(series_titles):
        y = np.array(values_by_series[i], dtype=float)
        yerr = np.array(stds_by_series[i], dtype=float)
        if show_error_bars:
            ax.bar(x + offsets[i], y, width=bar_width * 0.95, label=title, yerr=yerr, capsize=float(config.get("capsize", 4.0)))
        else:
            ax.bar(x + offsets[i], y, width=bar_width * 0.95, label=title)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=float(config.get("xtick_rotation", 0)))
    ax.set_xlabel(str(config.get("xlabel", "Dataset")))
    ax.set_ylabel(str(config.get("ylabel", "Value")))

    title = config.get("title")
    if title:
        ax.set_title(str(title))

    if config.get("ylim") is not None:
        ylim = config.get("ylim")
        ax.set_ylim(float(ylim[0]), float(ylim[1]))

    if bool(config.get("show_grid", True)):
        ax.grid(axis="y", alpha=0.3)

    legend_loc = str(config.get("legend_loc", "best"))
    ax.legend(loc=legend_loc, ncols=int(config.get("legend_ncols", 1)))

    if bool(config.get("annotate_bars", False)):
        decimals = int(config.get("decimals", 2))
        for i in range(n_series):
            y = values_by_series[i]
            for j, v in enumerate(y):
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                ax.text(x[j] + offsets[i], v, f"{v:.{decimals}f}", ha="center", va="bottom", fontsize=float(config.get("annotation_fontsize", 8)))

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(config.get("dpi", 300)), bbox_inches="tight")
    plt.close(fig)


def example_config() -> Dict[str, Any]:
    return {
        "figure_type": "grouped_bar",
        "title": "Average optimality gap by dataset",
        "xlabel": "Dataset",
        "ylabel": "Optimality gap (%)",
        "show_error_bars": False,
        "metric_scales": {"gap": 100.0, "relative_gap": 100.0},
        "seeds": [0, 1, 2],
        "seed_pattern": "{prefix}_seed{seed}/test_eval.json",
        "series": [
            {"key": "gnn", "title": "GNN", "method": "entropy_greedy", "metric": "gap"},
            {"key": "lbp", "title": "LBP", "method": "entropy_greedy", "metric": "gap"},
            {"key": "lbp_gd", "title": "LBP+GD", "method": "guided_decimation", "metric": "gap"},
            {"key": "nmf", "title": "NMF", "method": "entropy_greedy", "metric": "gap"},
        ],
        "groups": [
            {
                "label": "SC-S",
                "results": {
                    "gnn": {"run_prefix": "runs/scp_n50_m50_k5_var_logits_bce"},
                    "lbp": {"run_prefix": "runs/scp_n50_m50_k5_lbp_fce"},
                    "lbp_gd": {"run_prefix": "runs/scp_n50_m50_k5_lbp_fce"},
                    "nmf": {"run_prefix": "runs/scp_n50_m50_k5_nmf_fce"},
                },
            },
            {
                "label": "MIS-S",
                "results": {
                    "gnn": {"run_prefix": "runs/mis_n100_p0.15_var_logits_bce"},
                    "lbp": {"run_prefix": "runs/mis_n100_p0.15_lbp_fce"},
                    "lbp_gd": {"run_prefix": "runs/mis_n100_p0.15_lbp_fce"},
                    "nmf": {"run_prefix": "runs/mis_n100_p0.15_nmf_fce"},
                },
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grouped bar plots from bpropy JSON evaluation files.")
    parser.add_argument("--config", required=False, help="JSON config describing the grouped bar plot.")
    parser.add_argument("--output", required=False, help="Output image path (.png, .pdf, ...).")
    parser.add_argument("--root-dir", default=None, help="Root directory containing all run directories.")
    parser.add_argument("--write-example-config", help="Write an example grouped-bar config and exit.")
    args = parser.parse_args()

    if args.write_example_config:
        out = Path(args.write_example_config)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(example_config(), indent=2), encoding="utf-8")
        print(out)
        return

    if not args.config:
        raise SystemExit("Either --config or --write-example-config is required.")

    with Path(args.config).expanduser().open("r", encoding="utf-8") as f:
        config = json.load(f)

    if args.root_dir is not None:
        config["root_dir"] = args.root_dir

    if args.output:
        output = Path(args.output).expanduser()
    else:
        output = Path("grouped_bar_plot.png")

    make_grouped_bar_plot(config, output)
    print(output)


if __name__ == "__main__":
    main()
