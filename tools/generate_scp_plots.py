#!/usr/bin/env python3

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
import numpy as np
import pandas as pd

from adjustText import adjust_text
import tikzplotlib

OPT_RE = re.compile(
    r"OPT\s*:\s*\[\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s*\]|"
    r"OPT\s*:\s*\[\s*([0-9eE+\-.]+)\s*,\s*([0-9eE+\-.]+)\s*\]"
)

ROW_RE_TEMPLATE = (
    r"{kind}\s+p=([0-9.]+)\s*:\s*\[\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s*\]|"
    r"{kind}\s+p=([0-9.]+)\s*:\s*\[\s*([0-9eE+\-.]+)\s*,\s*([0-9eE+\-.]+)\s*\]"
)


@dataclass
class SeriesSpec:
    label: str
    row_kind: str
    inputs: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate two 2x2 SCP comparison figures from labeled collections of .out files: "
            "(1) average over all supplied files, and "
            "(2) best file per label. "
            "Each plotted series specifies its own row type (GRD or GD)."
        )
    )
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        metavar="LABEL:ROW_KIND=PATH[,PATH...]",
        help=(
            "Repeat once per plotted series. "
            "Example: --series 'GNN:GRD=results/gnn/*.out' "
            "--series 'GNN+BP Greedy:GRD=results/bp/*.out' "
            "--series 'GNN+BP Decimation:GD=results/bp/*.out'"
        ),
    )
    parser.add_argument("--title", default="SCP comparison", help="Base figure title.")
    parser.add_argument(
        "--output-prefix",
        default="scp_comparison",
        help="Prefix for output images. Writes '<prefix>_avg.png' and '<prefix>_best.png'.",
    )
    parser.add_argument(
        "--exclude-seed0",
        action="store_true",
        help="Ignore files whose basename ends with '_0.out'.",
    )
    parser.add_argument(
        "--gap-scale",
        type=float,
        default=100.0,
        help="Scale factor applied to gaps before plotting. Use 100 if gaps are stored in [0,1].",
    )
    parser.add_argument(
        "--annotate-p",
        action="store_true",
        help="Annotate Pareto points with their p values.",
    )
    parser.add_argument(
        "--opt-runtime-mode",
        choices=["mean", "median", "none"],
        default="mean",
        help="How to summarize OPT runtime for the reference line.",
    )
    parser.add_argument(
            "--colors",
            type=str,
            default=None,
            help="color map")

    parser.add_argument(
            "--output-format",
            choices=["png", "tikz", "both"],
            default = "png",
            help="output format",
            )
    parser.add_argument(
            "--tikz-standalone",
            action="store_true",
            help="write standalone tikz",
            )
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI.")
    return parser.parse_args()


def parse_series_specs(raw_specs: Sequence[str]) -> List[SeriesSpec]:
    specs: List[SeriesSpec] = []

    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(
                f"Invalid --series value: {raw!r}. Expected LABEL:ROW_KIND=PATH[,PATH...]."
            )

        left, raw_paths = raw.split("=", 1)
        if ":" not in left:
            raise ValueError(
                f"Invalid --series value: {raw!r}. Expected LABEL:ROW_KIND=PATH[,PATH...]."
            )

        label, row_kind = left.rsplit(":", 1)
        label = label.strip()
        row_kind = row_kind.strip().upper()
        paths = [p.strip() for p in raw_paths.split(",") if p.strip()]

        if row_kind not in {"GRD", "GD"}:
            raise ValueError(
                f"Invalid row kind in {raw!r}. Expected GRD or GD."
            )
        if not label or not paths:
            raise ValueError(
                f"Invalid --series value: {raw!r}. Expected LABEL:ROW_KIND=PATH[,PATH...]."
            )

        specs.append(SeriesSpec(label=label, row_kind=row_kind, inputs=paths))

    return specs


def expand_paths(inputs: Iterable[str], exclude_seed0: bool) -> List[str]:
    print("in expand_paths")
    files: List[str] = []

    for item in inputs:
        matches: List[str] = []

        if any(ch in item for ch in "*?[]"):
            matches.extend(glob.glob(item, recursive=True))
        elif os.path.isdir(item):
            for root, _, names in os.walk(item):
                for name in names:
                    if name.endswith(".out"):
                        matches.append(os.path.join(root, name))
        elif os.path.isfile(item):
            matches.append(item)
        else:
            matches.extend(glob.glob(item, recursive=True))

        for path in matches:
            if not os.path.isfile(path):
                print("skip1 path = ", path)
                continue
            if not path.endswith(".out"):
                print("skip2 path = ", path)
                continue
            if exclude_seed0 and os.path.basename(path).endswith("_0.out"):
                print("skip3 path = ", path)

                continue
            files.append(os.path.abspath(path))

    return sorted(set(files))


def _extract_triplet(match: re.Match) -> Tuple[float, float, float]:
    groups = match.groups()
    if groups[0] is not None:
        return float(groups[0]), float(groups[1]), float(groups[2])
    return float(groups[3]), float(groups[4]), float(groups[5])


def parse_file(path: str, row_kind: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    row_re = re.compile(ROW_RE_TEMPLATE.format(kind=re.escape(row_kind)))

    rows = []
    for match in row_re.finditer(text):
        p, gap, runtime = _extract_triplet(match)
        rows.append((p, gap, runtime))

    opts = []
    for match in OPT_RE.finditer(text):
        g = match.groups()
        if g[0] is not None:
            opts.append((float(g[0]), float(g[1])))
        else:
            opts.append((float(g[2]), float(g[3])))

    return rows, opts


def build_dataframes(specs: Sequence[SeriesSpec], exclude_seed0: bool):
    row_records = []
    opt_records = []

    for spec in specs:
        files = expand_paths(spec.inputs, exclude_seed0)
        print("inputs : exclude seed0 : ", spec.inputs, exclude_seed0)
        if not files:
            raise FileNotFoundError(f"No .out files found for series {spec.label!r}.")

        for path in files:
            rows, opts = parse_file(path, spec.row_kind)

            for p, gap, runtime in rows:
                row_records.append(
                    {
                        "label": spec.label,
                        "row_kind": spec.row_kind,
                        "file": path,
                        "p": p,
                        "gap": gap,
                        "runtime": runtime,
                    }
                )

            for opt_gap, opt_runtime in opts:
                opt_records.append(
                    {
                        "label": spec.label,
                        "row_kind": spec.row_kind,
                        "file": path,
                        "opt_gap": opt_gap,
                        "opt_runtime": opt_runtime,
                    }
                )

    if not row_records:
        raise ValueError("No matching GRD/GD rows found in the provided files.")

    return pd.DataFrame(row_records), pd.DataFrame(opt_records)


def summarize_all_files(rows_df: pd.DataFrame) -> pd.DataFrame:
    return (
        rows_df.groupby(["label", "p"], as_index=False)
        .agg(
            gap_mean=("gap", "mean"),
            gap_std=("gap", "std"),
            runtime_mean=("runtime", "mean"),
            runtime_std=("runtime", "std"),
            count=("file", "count"),
        )
        .sort_values(["label", "p"])
    )


def select_best_files(rows_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pick one file per label.
    Best = smallest average gap for the largest p value present for that label.
    Ties are broken by smaller average runtime at that same p, then filename.
    """
    best_files = []

    for label in rows_df["label"].drop_duplicates():
        label_df = rows_df[rows_df["label"] == label].copy()
        max_p = label_df["p"].max()
        at_max_p = label_df[label_df["p"] == max_p]

        ranking = (
            at_max_p.groupby("file", as_index=False)
            .agg(gap_mean=("gap", "mean"), runtime_mean=("runtime", "mean"))
            .sort_values(["gap_mean", "runtime_mean", "file"])
        )

        best_files.append({"label": label, "file": ranking.iloc[0]["file"]})

    best_files_df = pd.DataFrame(best_files)
    return rows_df.merge(best_files_df, on=["label", "file"], how="inner")


def summarize_opt_runtime(opts_df: pd.DataFrame, mode: str):
    if mode == "none" or opts_df.empty:
        return None
    if mode == "mean":
        return float(opts_df["opt_runtime"].mean())
    if mode == "median":
        return float(opts_df["opt_runtime"].median())
    raise ValueError(f"Unsupported opt runtime mode: {mode}")


def filter_opts_for_best_files(opts_df: pd.DataFrame, best_rows_df: pd.DataFrame) -> pd.DataFrame:
    best_file_pairs = best_rows_df[["label", "file"]].drop_duplicates()
    return opts_df.merge(best_file_pairs, on=["label", "file"], how="inner")

def global_pareto_frontier(summary: pd.DataFrame, gap_scale: float):
    """
    Return the globally non-dominated points across all series
    for (runtime_mean, gap_mean * gap_scale), assuming both are minimized.

    The returned dataframe is sorted by runtime_mean.
    """
    pts = summary.copy()
    pts["gap_plot"] = pts["gap_mean"] * gap_scale
    pts = pts.sort_values(["runtime_mean", "gap_plot"]).reset_index(drop=True)

    keep = []
    best_gap = float("inf")

    for _, row in pts.iterrows():
        if row["gap_plot"] < best_gap:
            keep.append(True)
            best_gap = row["gap_plot"]
        else:
            keep.append(False)

    return pts.loc[keep].copy()

def parse_color_map(color_str: str):
    if color_str is None:
        return {}

    mapping = {}
    for item in color_str.split(","):
        if "=" not in item:
            raise ValueError(f"Invalid color spec: {item!r}")
        label, color = item.split("=", 1)
        mapping[label.strip()] = color.strip()
    return mapping

def tikzplotlib_fix_ncols(obj):
    """
    Work around matplotlib >= 3.6 renaming legend._ncol to legend._ncols,
    which breaks tikzplotlib.
    """
    if hasattr(obj, "_ncols") and not hasattr(obj, "_ncol"):
        obj._ncol = obj._ncols
        for child in obj.get_children():
            tikzplotlib_fix_ncols(child)
def tikzplotlib_fix_legend(fig):
    """
    Matplotlib >= 3.6 uses _ncols, while tikzplotlib expects _ncol.
    Patch all legends in the figure before tikz export.
    """
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None and hasattr(legend, "_ncols") and not hasattr(legend, "_ncol"):
            legend._ncol = legend._ncols
def save_figure(fig, output_base, dpi, output_format, tikz_standalone):
    if output_format in {"png", "both"}:
        fig.savefig(f"{output_base}.png", dpi=dpi, bbox_inches="tight")

    if output_format in {"tikz", "both"}:
        fig.canvas.draw()
        tikzplotlib_fix_ncols(fig)
        tikzplotlib_fix_legend(fig)
        tikzplotlib.save(
            f"{output_base}.tex",
            figure=fig,
            standalone=tikz_standalone,
        )

def make_plot(
    summary: pd.DataFrame,
    title: str,
    output_base: str,
    gap_scale: float,
    annotate_p: bool,
    opt_runtime,
    dpi: int,
    output_format: str,
    tikz_standalone: bool,
    color_map=None,
) -> None:
    labels = list(summary["label"].drop_duplicates())
    p_vals = sorted(summary["p"].drop_duplicates())

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2)
    ax_pareto = fig.add_subplot(gs[0,:])
    ax_runtime = fig.add_subplot(gs[1,0])
    ax_bar = fig.add_subplot(gs[1,1])

    fig.suptitle(title, fontsize=18)

    if color_map is None:
        color_map = {}

    # Pareto
    ax = ax_pareto
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_minor_locator(NullLocator())

    texts = []
    for label in labels:
        d = summary[summary["label"] == label].sort_values("p")
        color = color_map.get(label, None)
        ax.plot(d["runtime_mean"], d["gap_mean"] * gap_scale, marker="o", label=label, color=color)
        if annotate_p:
            for _, row in d.iterrows():
                texts.append(
                        ax.text(
                            row["runtime_mean"],
                            row["gap_mean"] * gap_scale,
                            f"{row['p']:.1f}",
                            fontsize=9,
                            )
                        )
                #ax.annotate(
                #    f"{row['p']:.1f}",
                #    (row["runtime_mean"], row["gap_mean"] * gap_scale),
                #    xytext=(4, 2),
                #    textcoords="offset points",
                #)
        if annotate_p and texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=1))
    # Global non-dominated frontier across all series
    frontier = global_pareto_frontier(summary, gap_scale)
    if output_format in ["both","tikz"]:
        pareto_frontier_linestyle="-"
    else:
        pareto_frontier_linestyle="--"
    ax.plot(
        frontier["runtime_mean"],
        frontier["gap_plot"],
        linestyle=pareto_frontier_linestyle,
        linewidth=2,
        color="black",
        label="Global non-dominated frontier",
    )
    if opt_runtime is not None:
        ax.axvline(opt_runtime, linestyle=pareto_frontier_linestyle, label="OPT runtime")
    ax.set_title("Pareto: Runtime vs Gap")
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Gap (%)")
    ax.grid(True, alpha=0.4)
    ax.set_xscale("log")
    #ax.legend()

    # Runtime vs p
    ax = ax_runtime
    for label in labels:
        d = summary[summary["label"] == label].sort_values("p")
        color = color_map.get(label, None)
        ax.plot(d["p"], d["runtime_mean"], marker="o", label=label, color=color)
    if opt_runtime is not None:
        ax.axhline(opt_runtime, linestyle=pareto_frontier_linestyle, label="OPT runtime")
    ax.set_title("Runtime vs p")
    ax.set_xlabel("p")
    ax.set_ylabel("Runtime (s)")
    ax.grid(True, alpha=0.4)
    ax.set_yscale("log")
    #ax.legend()

    # Gap comparison
    ax = ax_bar
    x = np.arange(len(p_vals))
    width = 0.8 / max(len(labels), 1)

    for i, label in enumerate(labels):
        d = summary[summary["label"] == label].set_index("p").reindex(p_vals)
        offset = (i - (len(labels) - 1) / 2.0) * width
        color = color_map.get(label, None)
        ax.bar(x + offset, d["gap_mean"].values * gap_scale, width=width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.1f}" for p in p_vals])
    ax.set_title("Gap Comparison")
    ax.set_xlabel("p")
    ax.set_ylabel("Gap (%)")
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_base, dpi, output_format, tikz_standalone)
    #fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    specs = parse_series_specs(args.series)
    rows_df, opts_df = build_dataframes(specs, args.exclude_seed0)

    color_map = parse_color_map(args.colors)

    # Plot 1: average over all supplied files
    summary_avg = summarize_all_files(rows_df)
    opt_runtime_avg = summarize_opt_runtime(opts_df, args.opt_runtime_mode)
    output_avg = f"{args.output_prefix}_avg"
    make_plot(
        summary=summary_avg,
        title=f"{args.title} — average over all files",
        output_base=output_avg,
        gap_scale=args.gap_scale,
        annotate_p=args.annotate_p,
        opt_runtime=opt_runtime_avg,
        dpi=args.dpi,
        color_map=color_map,
        output_format=args.output_format,
        tikz_standalone=args.tikz_standalone,
    )

    # Plot 2: best file per label
    best_rows_df = select_best_files(rows_df)
    summary_best = summarize_all_files(best_rows_df)
    best_opts_df = filter_opts_for_best_files(opts_df, best_rows_df)
    opt_runtime_best = summarize_opt_runtime(best_opts_df, args.opt_runtime_mode)
    output_best = f"{args.output_prefix}_best"
    make_plot(
        summary=summary_best,
        title=f"{args.title} — best file per label",
        output_base=output_best,
        gap_scale=args.gap_scale,
        annotate_p=args.annotate_p,
        opt_runtime=opt_runtime_best,
        dpi=args.dpi,
        color_map=color_map,
        output_format=args.output_format,
        tikz_standalone=args.tikz_standalone,
    )

    best_file_table = best_rows_df[["label", "file"]].drop_duplicates().sort_values(["label", "file"])

    if args.output_format in {"png", "both"}:
        print(f"Wrote plot to: {output_avg}.png")
        print(f"Wrote plot to: {output_best}.png")
    if args.output_format in {"tikz", "both"}:
        print(f"Wrote plot to: {output_avg}.tex")
        print(f"Wrote plot to: {output_best}.tex")
    print()
    print("=== Average over all files ===")
    print(summary_avg.to_string(index=False))

    if opt_runtime_avg is not None:
        print()
        print(f"OPT runtime ({args.opt_runtime_mode}, average plot): {opt_runtime_avg:.6f} s")

    print()
    print("=== Best file per label ===")
    print(best_file_table.to_string(index=False))
    print()
    print(summary_best.to_string(index=False))

    if opt_runtime_best is not None:
        print()
        print(f"OPT runtime ({args.opt_runtime_mode}, best plot): {opt_runtime_best:.6f} s")


if __name__ == "__main__":
    main()
