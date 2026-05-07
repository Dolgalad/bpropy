#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot selected-variable diagnostics per iteration from GD/EGG log JSON files. "
            "Expected layout: ROOT/<dataset_id>/<model_id>/<seed>.json"
        )
    )
    p.add_argument("root", type=Path)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--formats", nargs="+", default=["png"])
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--title-prefix", type=str, default="Selected-variable diagnostics")
    p.add_argument("--min-support", type=int, default=1)
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


def aggregate_mean_std(
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


def plot_with_band(ax, mean: np.ndarray, std: np.ndarray, label: str) -> None:
    x = np.arange(mean.size)
    ax.plot(x, mean, label=label)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)


def plot_simple(ax, seqs: List[List[float]], label: str, min_support: int) -> bool:
    mean, std, _ = aggregate_mean_std(seqs, min_support)
    if mean.size == 0:
        return False
    plot_with_band(ax, mean, std, label)
    return True


def add_legends_to_nonempty_axes(axs) -> None:
    for ax in np.ravel(axs):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()


def load_metrics_from_run(json_path: Path) -> Dict[str, Dict[str, List[List[float]]]]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    keys = [
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

    out: Dict[str, Dict[str, List[List[float]]]] = {
        "egg": {k: [] for k in keys},
        "gd": {k: [] for k in keys},
    }

    for inst in payload.get("instances", []):
        for variant in ("egg", "gd"):
            block = inst.get(variant)
            if not block:
                continue

            info_list = block.get("info", [])
            if not info_list:
                continue

            seqs = {k: [] for k in keys}

            cumulative_cov = 0.0
            cumulative_value_gain = 0.0

            for step in info_list:
                if not step:
                    continue

                score = step.get("score", None)
                mu1 = step.get("mu1", None)
                value = step.get("value", None)

                if score is not None:
                    seqs["entropy"].append(float(score))
                elif mu1 is not None:
                    seqs["entropy"].append(safe_entropy_from_mu1(mu1))

                if value is not None:
                    seqs["value"].append(float(value))

                if mu1 is not None:
                    seqs["mu1"].append(float(mu1))

                for k in keys:
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
                    seqs["newly_covered_count"].append(float(newly_covered_count))
                    cumulative_cov += float(newly_covered_count)
                    seqs["cumulative_newly_covered"].append(cumulative_cov)

                    if value is not None:
                        vtg = float(value) * float(newly_covered_count)
                        seqs["value_times_newly_covered"].append(vtg)
                        cumulative_value_gain += vtg
                        seqs["cumulative_value_times_newly_covered"].append(cumulative_value_gain)

            for k in keys:
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
            data[dataset_id].setdefault(
                model_id,
                {
                    "egg": {},
                    "gd": {},
                },
            )

            agg = None
            for json_path in json_files:
                run_metrics = load_metrics_from_run(json_path)
                if agg is None:
                    agg = run_metrics
                else:
                    for variant in ("egg", "gd"):
                        for metric, seqs in run_metrics[variant].items():
                            agg[variant].setdefault(metric, [])
                            agg[variant][metric].extend(seqs)

            data[dataset_id][model_id] = agg

    return data


def _patch_tikzplotlib_legend(fig) -> None:
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None and hasattr(legend, "_ncols") and not hasattr(legend, "_ncol"):
            legend._ncol = legend._ncols


def save_figure(
    fig,
    base_path: Path,
    formats: List[str],
    dpi: int,
    tikz_axis_width: str = r"\figurewidth",
    tikz_axis_height: str = r"\figureheight",
) -> None:
    for fmt in formats:
        out_path = base_path.with_suffix(f".{fmt}")

        if fmt == "tex":
            try:
                import tikzplotlib
            except ImportError as e:
                raise RuntimeError(
                    "TikZ export requested but tikzplotlib is not installed. "
                    "Install it with: pip install tikzplotlib"
                ) from e

            _patch_tikzplotlib_legend(fig)

            tikzplotlib.save(
                out_path,
                figure=fig,
                axis_width=tikz_axis_width,
                axis_height=tikz_axis_height,
            )
        else:
            fig.savefig(out_path, dpi=dpi if fmt == "png" else None)

        print(f"[ok] wrote {out_path}")

def style_axes(axs) -> None:
    for ax in np.ravel(axs):
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Iteration")


def plot_dataset(
    dataset_id: str,
    dataset_data: Dict[str, Dict[str, Dict[str, List[List[float]]]]],
    output_dir: Path,
    formats: List[str],
    dpi: int,
    title_prefix: str,
    min_support: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Main 3x2 grid
    # =========================================================
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=False)

    ax_entropy_egg = axes[0, 0]
    ax_entropy_gd = axes[0, 1]
    ax_value_egg = axes[1, 0]
    ax_value_gd = axes[1, 1]
    ax_mu1_egg = axes[2, 0]
    ax_mu1_gd = axes[2, 1]

    plotted_any = False

    for model_id in sorted(dataset_data):
        d = dataset_data[model_id]
        print(
            dataset_id,
            model_id,
            "egg entropy:", len(d["egg"].get("entropy", [])),
            "gd entropy:", len(d["gd"].get("entropy", [])),
            "egg value:", len(d["egg"].get("value", [])),
            "gd value:", len(d["gd"].get("value", [])),
            "egg mu1:", len(d["egg"].get("mu1", [])),
            "gd mu1:", len(d["gd"].get("mu1", [])),
            "gd best2:", len(d["gd"].get("entropy_gap_best_second", [])),
            )

        if plot_simple(ax_entropy_egg, d["egg"]["entropy"], model_id, min_support):
            plotted_any = True
        if plot_simple(ax_entropy_gd, d["gd"]["entropy"], model_id, min_support):
            plotted_any = True

        plot_simple(ax_value_egg, d["egg"]["value"], model_id, min_support)
        plot_simple(ax_value_gd, d["gd"]["value"], model_id, min_support)

        plot_simple(ax_mu1_egg, d["egg"]["mu1"], model_id, min_support)
        plot_simple(ax_mu1_gd, d["gd"]["mu1"], model_id, min_support)

    if not plotted_any:
        print(f"[skip] {dataset_id}: no usable trajectories found")
        plt.close(fig)
        return

    ax_entropy_egg.set_title("EGG - entropy")
    ax_entropy_gd.set_title("GD - entropy")
    ax_value_egg.set_title("EGG - selected value")
    ax_value_gd.set_title("GD - selected value")
    ax_mu1_egg.set_title("EGG - selected $\\mu_1$")
    ax_mu1_gd.set_title("GD - selected $\\mu_1$")

    ax_entropy_egg.set_ylabel("Entropy")
    ax_entropy_gd.set_ylabel("Entropy")
    ax_value_egg.set_ylabel("Selected value")
    ax_value_gd.set_ylabel("Selected value")
    ax_mu1_egg.set_ylabel("$\\mu_1$")
    ax_mu1_gd.set_ylabel("$\\mu_1$")

    ax_value_egg.set_ylim(-0.02, 1.02)
    ax_value_gd.set_ylim(-0.02, 1.02)
    ax_mu1_egg.set_ylim(-0.02, 1.02)
    ax_mu1_gd.set_ylim(-0.02, 1.02)

    style_axes(axes)
    add_legends_to_nonempty_axes(axes)

    fig.suptitle(f"{title_prefix} - {dataset_id}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    save_figure(
        fig,
        output_dir / dataset_id,
        formats,
        dpi,
        tikz_axis_width=r"\figurewidth",
        tikz_axis_height=r"\figureheight",
    )
    plt.close(fig)

    # =========================================================
    # Choice vs pool
    # =========================================================
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=False)

    variants = [("egg", "EGG"), ("gd", "GD")]
    metrics = [
        ("chosen_entropy", "chosen entropy", "Chosen entropy"),
        ("chosen_eff", "chosen efficiency", "Chosen efficiency"),
        ("entropy_gap_to_mean", "entropy gap to candidate mean", "Gap"),
        ("eff_gap_to_mean", "efficiency gap to candidate mean", "Gap"),
    ]

    for row, (variant_key, variant_label) in enumerate(variants):
        for model_id in sorted(dataset_data):
            d = dataset_data[model_id]
            for col, (metric_key, title_suffix, ylabel) in enumerate(metrics):
                plot_simple(axes[row, col], d[variant_key][metric_key], model_id, min_support)

        for col, (_, title_suffix, ylabel) in enumerate(metrics):
            axes[row, col].set_title(f"{variant_label} - {title_suffix}")
            axes[row, col].set_ylabel(ylabel)

    style_axes(axes)
    add_legends_to_nonempty_axes(axes)
    fig.suptitle(f"{title_prefix} - {dataset_id} - choice vs pool", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_figure(
        fig,
        output_dir / f"{dataset_id}_choice_vs_pool",
        formats,
        dpi,
        tikz_axis_width=r"\figurewidth",
        tikz_axis_height=r"\figureheight",
    )
    plt.close(fig)

    # =========================================================
    # Ranks
    # =========================================================
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False)

    variants = [("egg", "EGG"), ("gd", "GD")]
    metrics = [
        ("entropy_rank_asc", "entropy rank (ascending)", "Rank"),
        ("mu1_rank_desc", r"$\mu_1$ rank (descending)", "Rank"),
        ("eff_rank_desc", "efficiency rank (descending)", "Rank"),
    ]

    for row, (variant_key, variant_label) in enumerate(variants):
        for model_id in sorted(dataset_data):
            d = dataset_data[model_id]
            for col, (metric_key, title_suffix, ylabel) in enumerate(metrics):
                plot_simple(axes[row, col], d[variant_key][metric_key], model_id, min_support)

        for col, (_, title_suffix, ylabel) in enumerate(metrics):
            axes[row, col].set_title(f"{variant_label} - {title_suffix}")
            axes[row, col].set_ylabel(ylabel)

    style_axes(axes)
    add_legends_to_nonempty_axes(axes)
    fig.suptitle(f"{title_prefix} - {dataset_id} - ranks", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_figure(
        fig,
        output_dir / f"{dataset_id}_ranks",
        formats,
        dpi,
        tikz_axis_width=r"\figurewidth",
        tikz_axis_height=r"\figureheight",
    )
    plt.close(fig)
    # =========================================================
    # Best vs second-best gaps
    # =========================================================
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False)

    variants = [("egg", "EGG"), ("gd", "GD")]
    metrics = [
        ("entropy_gap_best_second", "entropy best/second gap", "Gap"),
        ("mu_gap_best_second", r"$\mu_1$ best/second gap", "Gap"),
        ("eff_gap_best_second", "efficiency best/second gap", "Gap"),
    ]

    for row, (variant_key, variant_label) in enumerate(variants):
        for model_id in sorted(dataset_data):
            d = dataset_data[model_id]
            for col, (metric_key, title_suffix, ylabel) in enumerate(metrics):
                plot_simple(axes[row, col], d[variant_key][metric_key], model_id, min_support)

        for col, (_, title_suffix, ylabel) in enumerate(metrics):
            axes[row, col].set_title(f"{variant_label} - {title_suffix}")
            axes[row, col].set_ylabel(ylabel)

    style_axes(axes)
    add_legends_to_nonempty_axes(axes)
    fig.suptitle(f"{title_prefix} - {dataset_id} - best vs second", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_figure(
        fig,
        output_dir / f"{dataset_id}_best_vs_second",
        formats,
        dpi,
        tikz_axis_width=r"\figurewidth",
        tikz_axis_height=r"\figureheight",
    )
    plt.close(fig)
    
    # =========================================================
    # Candidate pool statistics
    # =========================================================
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=False)

    variants = [("egg", "EGG"), ("gd", "GD")]
    metrics = [
        ("candidate_entropy_mean", "candidate entropy mean", "Value"),
        ("candidate_entropy_std", "candidate entropy std", "Value"),
        ("candidate_mu_mean", r"candidate $\mu_1$ mean", "Value"),
        ("candidate_mu_std", r"candidate $\mu_1$ std", "Value"),
    ]

    for row, (variant_key, variant_label) in enumerate(variants):
        for model_id in sorted(dataset_data):
            d = dataset_data[model_id]
            for col, (metric_key, title_suffix, ylabel) in enumerate(metrics):
                plot_simple(axes[row, col], d[variant_key][metric_key], model_id, min_support)

        for col, (_, title_suffix, ylabel) in enumerate(metrics):
            axes[row, col].set_title(f"{variant_label} - {title_suffix}")
            axes[row, col].set_ylabel(ylabel)

    style_axes(axes)
    add_legends_to_nonempty_axes(axes)
    fig.suptitle(f"{title_prefix} - {dataset_id} - pool stats", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_figure(
        fig,
        output_dir / f"{dataset_id}_pool_stats",
        formats,
        dpi,
        tikz_axis_width=r"\figurewidth",
        tikz_axis_height=r"\figureheight",
    )
    plt.close(fig)

    # =========================================================
    # Coverage dynamics
    # =========================================================
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False)

    variants = [("egg", "EGG"), ("gd", "GD")]
    metrics = [
        ("newly_covered_count", "newly covered per step", "Count"),
        ("cumulative_newly_covered", "cumulative newly covered", "Count"),
    ]

    for row, (variant_key, variant_label) in enumerate(variants):
        for model_id in sorted(dataset_data):
            d = dataset_data[model_id]
            for col, (metric_key, title_suffix, ylabel) in enumerate(metrics):
                plot_simple(axes[row, col], d[variant_key][metric_key], model_id, min_support)

        for col, (_, title_suffix, ylabel) in enumerate(metrics):
            axes[row, col].set_title(f"{variant_label} - {title_suffix}")
            axes[row, col].set_ylabel(ylabel)

    style_axes(axes)
    add_legends_to_nonempty_axes(axes)
    fig.suptitle(f"{title_prefix} - {dataset_id} - coverage", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_figure(
        fig,
        output_dir / f"{dataset_id}_coverage",
        formats,
        dpi,
        tikz_axis_width=r"\figurewidth",
        tikz_axis_height=r"\figureheight",
    )
    plt.close(fig)

    # =========================================================
    # Candidate extrema and count
    # =========================================================
    fig, axes = plt.subplots(2, 6, figsize=(18, 8), sharex=False)

    variants = [("egg", "EGG"), ("gd", "GD")]
    metrics = [
        ("candidate_entropy_min", "candidate entropy min"),
        ("candidate_entropy_max", "candidate entropy max"),
        ("n_candidates", "number of candidates"),
        ("candidate_mu_min", r"candidate $\mu_1$ min"),
        ("candidate_mu_max", r"candidate $\mu_1$ max"),
        ("chosen_mu1", r"chosen $\mu_1$"),
    ]

    for row, (variant_key, variant_label) in enumerate(variants):
        for model_id in sorted(dataset_data):
            d = dataset_data[model_id]
            for col, (metric_key, title_suffix) in enumerate(metrics):
                plot_simple(axes[row, col], d[variant_key][metric_key], model_id, min_support)

        for col, (_, title_suffix) in enumerate(metrics):
            axes[row, col].set_title(f"{variant_label} - {title_suffix}")

    style_axes(axes)
    add_legends_to_nonempty_axes(axes)
    fig.suptitle(f"{title_prefix} - {dataset_id} - extrema", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_figure(
        fig,
        output_dir / f"{dataset_id}_extrema",
        formats,
        dpi,
        tikz_axis_width=r"\figurewidth",
        tikz_axis_height=r"\figureheight",
    )
    plt.close(fig)
    # =========================================================
    # Gain diagnostics
    # =========================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)

    variants = [("egg", "EGG"), ("gd", "GD")]
    metrics = [
        ("value_times_newly_covered", "value × newly covered"),
        ("cumulative_value_times_newly_covered", "cumulative value × newly covered"),
        ("mu1_gap_to_mean", r"$\mu_1$ gap to candidate mean"),
    ]

    for row, (variant_key, variant_label) in enumerate(variants):
        for model_id in sorted(dataset_data):
            d = dataset_data[model_id]
            for col, (metric_key, title_suffix) in enumerate(metrics):
                plot_simple(axes[row, col], d[variant_key][metric_key], model_id, min_support)

        for col, (_, title_suffix) in enumerate(metrics):
            axes[row, col].set_title(f"{variant_label} - {title_suffix}")

    style_axes(axes)
    add_legends_to_nonempty_axes(axes)
    fig.suptitle(f"{title_prefix} - {dataset_id} - gain diagnostics", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_figure(
        fig,
        output_dir / f"{dataset_id}_gains",
        formats,
        dpi,
        tikz_axis_width=r"\figurewidth",
        tikz_axis_height=r"\figureheight",
    )
    plt.close(fig)

def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else root / "plots"

    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")

    ignored = {output_dir.name} if output_dir.parent == root else set()

    all_data = collect_all_runs(root, ignored_dirnames=ignored)
    if not all_data:
        raise RuntimeError(f"No dataset directories found under {root}")

    for dataset_id in sorted(all_data):
        plot_dataset(
            dataset_id=dataset_id,
            dataset_data=all_data[dataset_id],
            output_dir=output_dir,
            formats=args.formats,
            dpi=args.dpi,
            title_prefix=args.title_prefix,
            min_support=args.min_support,
        )


if __name__ == "__main__":
    main()
