#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
import numpy as np
import pandas as pd

from adjustText import adjust_text
import tikzplotlib

FILE_RE = re.compile(
    r"^(?P<strategy>.+)_p(?P<p>[0-9.]+)_seed_?(?P<seed>-?[0-9]+)\.json$"
    )

@dataclass
class SeriesSpec:
    label: str
    row_kind: str
    inputs: List[str]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SCP comparison plots from the new JSON-based result layout. "
            "Each --series points to one or more <dataset>/<model> directories and a filename prefix "
            "(called ROW_KIND for backwards compatibility). Model files are expected to follow "
            "<ROW_KIND>_p<p>_seed<seed>.json."
        )
    )
    parser.add_argument(
        "--gurobi-dir",
        required=True,
        help=(
            "Directory containing one <stem>.json per successfully solved instance. "
            "Only instances with an optimal Gurobi solution are kept."
        ),
    )
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        metavar="LABEL:ROW_KIND=PATH[,PATH...]",
        help=(
            "Repeat once per plotted series. Example: "
            "--series 'GNN EGG:GD=test_fix_and_solve/scp_n50/gnn' "
            "--series 'MRF GD:GRD=test_fix_and_solve/scp_n50/gnn_lbp_full'"
        ),
    )
    parser.add_argument("--title", default="SCP comparison", help="Base figure title.")
    parser.add_argument(
        "--output-prefix",
        default="scp_comparison",
        help="Prefix for output images. Writes '<prefix>_avg.*' and '<prefix>_best.*'.",
    )
    parser.add_argument(
        "--exclude-seed0",
        action="store_true",
        help="Ignore model files whose filename ends with '_seed0.json'.",
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
        help="How to summarize Gurobi runtime for the reference line.",
    )
    parser.add_argument(
        "--colors",
        type=str,
        default=None,
        help="Comma-separated color map, e.g. 'A=#1f77b4,B=#ff7f0e'",
    )
    parser.add_argument(
        "--output-format",
        choices=["png", "tikz", "both"],
        default="png",
        help="Output format.",
    )
    parser.add_argument(
        "--tikz-standalone",
        action="store_true",
        help="Write standalone TikZ output.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI.")
    parser.add_argument(
        "--verbose-common-set",
        action="store_true",
        help="Print the common stem set and per-configuration coverage diagnostics.",
    )
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
        row_kind = row_kind.strip()
        paths = [p.strip() for p in raw_paths.split(",") if p.strip()]

        if not label or not row_kind or not paths:
            raise ValueError(
                f"Invalid --series value: {raw!r}. Expected LABEL:ROW_KIND=PATH[,PATH...]."
            )

        specs.append(SeriesSpec(label=label, row_kind=row_kind, inputs=paths))

    return specs



def parse_color_map(color_str: Optional[str]) -> Dict[str, str]:
    if color_str is None:
        return {}

    mapping: Dict[str, str] = {}
    for item in color_str.split(","):
        if "=" not in item:
            raise ValueError(f"Invalid color spec: {item!r}")
        label, color = item.split("=", 1)
        mapping[label.strip()] = color.strip()
    return mapping



def tikzplotlib_fix_ncols(obj):
    if hasattr(obj, "_ncols") and not hasattr(obj, "_ncol"):
        obj._ncol = obj._ncols
        for child in obj.get_children():
            tikzplotlib_fix_ncols(child)



def tikzplotlib_fix_legend(fig):
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



def first_present(obj, keys, default=None):
    for key in keys:
        if isinstance(obj, dict) and key in obj and obj[key] is not None:
            return obj[key]
    return default



def as_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return float(value)
    if isinstance(value, str):
        try:
            out = float(value)
        except ValueError:
            return None
        return out if math.isfinite(out) else None
    return None



def as_int(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None



def normalize_status(value) -> str:
    return str(value).strip().lower() if value is not None else ""



def is_optimal_gurobi_record(obj: dict) -> bool:
    if first_present(obj, ["feasible"], False) is True:
        return True

    status_raw = first_present(obj, ["status", "Status", "solve_status", "solver_status"])
    status = normalize_status(status_raw)
    if status in {"optimal", "opt", "grb.optimal"}:
        return True

    status_int = as_int(status_raw)
    if status_int == 2:
        return True

    if first_present(obj, ["is_optimal", "optimal"], False) is True:
        return True

    return False


def is_successful_model_result(entry: dict) -> bool:
    status = normalize_status(first_present(entry, ["status", "solve_status", "solver_status"]))
    if status in {"optimal", "feasible", "success", "solved", "ok"}:
        return True

    if first_present(entry, ["success", "solved"], False) is True:
        return True

    if as_float(first_present(entry, ["solution_value", "value", "objective_value", "obj_val", "objective"])) is not None:
        return True

    solution = first_present(entry, ["solution", "best_solution", "x"])
    if solution is not None:
        return True

    return False


def get_runtime(obj: dict) -> Optional[float]:
    return as_float(first_present(obj, [
        "runtime",
        "Runtime",
        "solve_time",
        "solver_runtime",
        "time",
        "wall_time",
        "runtime_wall_s",
        ]))


def get_objective_value(obj: dict) -> Optional[float]:
    return as_float(first_present(obj, [
        "objective_value",
        "ObjectiveValue",
        "obj_val",
        "ObjVal",
        "value",
        "objective",
        "optimal_value",
        "objective_value_from_x",
        ]))


def load_gurobi_optima(gurobi_dir: Path) -> Tuple[Dict[str, dict], pd.DataFrame]:
    optima: Dict[str, dict] = {}
    diagnostics = []

    for path in sorted(gurobi_dir.rglob("*.json")):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            diagnostics.append({"file": str(path), "reason": f"json_error: {exc}"})
            continue

        raw_stem = first_present(payload, ["stem"], path.stem)
        stem = Path(str(raw_stem)).name
        if not isinstance(payload, dict):
            diagnostics.append({"file": str(path), "reason": "not_a_dict"})
            continue

        if not is_optimal_gurobi_record(payload):
            diagnostics.append({"file": str(path), "reason": "not_optimal"})
            continue

        value = get_objective_value(payload)
        runtime = get_runtime(payload)
        if value is None:
            diagnostics.append({"file": str(path), "reason": "missing_objective_value"})
            continue

        optima[stem] = {
            "stem": stem,
            "objective_value": value,
            "runtime": runtime,
            "path": str(path),
        }

    if not optima:
        raise ValueError(f"No optimal Gurobi instance JSON files found under {gurobi_dir}")

    return optima, pd.DataFrame(diagnostics)



import glob

def expand_series_files(inputs: Iterable[str], row_kind: str, exclude_seed0: bool) -> List[Path]:
    files: List[Path] = []
    prefix = f"{row_kind}_p"

    for item in inputs:
        p = Path(item)
        candidates: List[Path] = []

        if p.is_file():
            candidates = [p]
        elif p.is_dir():
            candidates = sorted(x for x in p.rglob("*.json") if x.is_file())
        else:
            candidates = [Path(x) for x in glob.glob(item)]

        for path in candidates:
            name = path.name
            if not name.startswith(prefix):
                continue
            if exclude_seed0 and re.search(r"_seed_?0\.json$", name):
                continue
            if FILE_RE.match(name) is None:
                continue
            files.append(path.resolve())

    unique = sorted(set(files))
    if not unique:
        raise FileNotFoundError(
            f"No model JSON files found for row kind {row_kind!r} in {list(inputs)!r}"
        )
    return unique


def parse_model_file_metadata(path: Path, expected_row_kind: str) -> Tuple[float, int]:
    match = FILE_RE.match(path.name)
    if match is None:
        raise ValueError(f"Filename does not match expected pattern: {path}")

    row_kind = match.group("strategy")
    if row_kind != expected_row_kind:
        raise ValueError(
            f"Filename prefix mismatch for {path}. Expected {expected_row_kind!r}, got {row_kind!r}."
        )

    p_value = float(match.group("p"))
    seed = int(match.group("seed"))
    return p_value, seed



def load_model_results(path: Path) -> Dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    results = payload.get("results")
    if not isinstance(results, list):
        results = payload.get("instances")
    if not isinstance(results, list):
        raise ValueError(f"Expected a 'results' or 'instances' list in {path}")

    by_stem: Dict[str, dict] = {}
    for entry in results:
        if not isinstance(entry, dict):
            continue

        stem = first_present(entry, ["instance_stem", "stem", "instance", "name"])
        if stem is None:
            continue
        stem = str(stem)

        if not is_successful_model_result(entry):
            continue

        value = as_float(first_present(entry, [
            "solution_value",
            "value",
            "objective_value",
            "obj_val",
            "objective",
            ]))
        if value is None:
            continue

        runtime = get_runtime(entry)
        if runtime is None and isinstance(entry.get("timings_s"), dict):
            runtime = as_float(first_present(entry["timings_s"], [
                "total",
                "reduced_problem_solver",
                ]))

            by_stem[stem] = {
                "stem": stem,
                "objective_value": value,
                "runtime": runtime,
                "raw": entry,
                }

    return by_stem


def compute_gap(model_value: float, opt_value: float) -> float:
    denom = abs(opt_value)
    if denom <= 1e-12:
        return 0.0 if abs(model_value - opt_value) <= 1e-12 else float("nan")
    return (model_value - opt_value) / denom



def build_dataframes(
    specs: Sequence[SeriesSpec],
    gurobi_optima: Dict[str, dict],
    exclude_seed0: bool,
    verbose_common_set: bool,
):
    config_records = []
    per_instance_records = []
    coverage_sets: Dict[Tuple[str, str], Set[str]] = {}

    optimal_stems = set(gurobi_optima.keys())

    for spec in specs:
        files = expand_series_files(spec.inputs, spec.row_kind, exclude_seed0)

        for path in files:
            p_value, seed = parse_model_file_metadata(path, spec.row_kind)
            model_results = load_model_results(path)
            solved_stems = set(model_results.keys()) & optimal_stems

            coverage_key = (spec.label, str(path))
            coverage_sets[coverage_key] = solved_stems

            config_records.append(
                {
                    "label": spec.label,
                    "row_kind": spec.row_kind,
                    "file": str(path),
                    "seed": seed,
                    "p": p_value,
                    "solved_with_opt": len(solved_stems),
                }
            )

            for stem in solved_stems:
                opt = gurobi_optima[stem]
                model = model_results[stem]
                gap = compute_gap(model["objective_value"], opt["objective_value"])
                per_instance_records.append(
                    {
                        "label": spec.label,
                        "row_kind": spec.row_kind,
                        "file": str(path),
                        "seed": seed,
                        "p": p_value,
                        "stem": stem,
                        "opt_value": opt["objective_value"],
                        "model_value": model["objective_value"],
                        "gap": gap,
                        "runtime": model["runtime"],
                        "opt_runtime": opt["runtime"],
                    }
                )

    if not coverage_sets:
        raise ValueError("No model configuration files were found.")

    common_stems = set(optimal_stems)
    for solved_stems in coverage_sets.values():
        common_stems &= solved_stems

    if not common_stems:
        raise ValueError(
            "The common evaluation set is empty: no instance has both an optimal Gurobi solution "
            "and successful model results for every configuration."
        )

    per_instance_df = pd.DataFrame(per_instance_records)
    if per_instance_df.empty:
        raise ValueError("No per-instance records could be constructed from the JSON inputs.")

    per_instance_df = per_instance_df[per_instance_df["stem"].isin(common_stems)].copy()
    config_df = pd.DataFrame(config_records)
    config_df["common_count"] = len(common_stems)

    summary_df = (
        per_instance_df.groupby(["label", "row_kind", "file", "seed", "p"], as_index=False)
        .agg(
            gap_mean=("gap", "mean"),
            gap_std=("gap", "std"),
            runtime_mean=("runtime", "mean"),
            runtime_std=("runtime", "std"),
            opt_runtime_mean=("opt_runtime", "mean"),
            count=("stem", "count"),
        )
        .sort_values(["label", "seed", "p", "file"])
    )

    expected_count = len(common_stems)
    bad_counts = summary_df[summary_df["count"] != expected_count]
    if not bad_counts.empty:
        raise ValueError(
            "After intersecting the common stem set, at least one configuration still does not "
            "cover the full test set. This indicates duplicate/malformed result entries."
        )

    if verbose_common_set:
        print(f"Optimal Gurobi instances found: {len(optimal_stems)}")
        print(f"Common evaluation set size: {len(common_stems)}")
        print("Common stems:")
        for stem in sorted(common_stems):
            print(f"  {stem}")
        print()
        print("Coverage by configuration before intersection:")
        print(config_df.sort_values(["label", "seed", "p", "file"]).to_string(index=False))
        print()

    return summary_df, per_instance_df, sorted(common_stems)



def summarize_all_files(rows_df: pd.DataFrame) -> pd.DataFrame:
    return (
        rows_df.groupby(["label", "p"], as_index=False)
        .agg(
            gap_mean=("gap_mean", "mean"),
            gap_std=("gap_mean", "std"),
            runtime_mean=("runtime_mean", "mean"),
            runtime_std=("runtime_mean", "std"),
            count=("file", "count"),
        )
        .sort_values(["label", "p"])
    )



def select_best_seed_curves(rows_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pick one seed per label.
    Best = smallest average gap at the largest p value for that label.
    Ties are broken by smaller average runtime at that same p, then smaller seed.
    """
    best_seed_rows = []

    for label in rows_df["label"].drop_duplicates():
        label_df = rows_df[rows_df["label"] == label].copy()
        max_p = label_df["p"].max()
        at_max_p = label_df[label_df["p"] == max_p]

        ranking = (
            at_max_p.groupby("seed", as_index=False)
            .agg(gap_mean=("gap_mean", "mean"), runtime_mean=("runtime_mean", "mean"))
            .sort_values(["gap_mean", "runtime_mean", "seed"])
        )

        if ranking.empty:
            continue
        best_seed = int(ranking.iloc[0]["seed"])
        best_seed_rows.append({"label": label, "seed": best_seed})

    best_seeds_df = pd.DataFrame(best_seed_rows)
    return rows_df.merge(best_seeds_df, on=["label", "seed"], how="inner")



def summarize_opt_runtime(per_instance_df: pd.DataFrame, mode: str):
    if mode == "none" or per_instance_df.empty:
        return None

    opt_values = per_instance_df[["stem", "opt_runtime"]].drop_duplicates()["opt_runtime"].dropna()
    if opt_values.empty:
        return None
    if mode == "mean":
        return float(opt_values.mean())
    if mode == "median":
        return float(opt_values.median())
    raise ValueError(f"Unsupported opt runtime mode: {mode}")



def filter_instances_for_best_seed(per_instance_df: pd.DataFrame, best_rows_df: pd.DataFrame) -> pd.DataFrame:
    keep = best_rows_df[["label", "seed"]].drop_duplicates()
    return per_instance_df.merge(keep, on=["label", "seed"], how="inner")



def global_pareto_frontier(summary: pd.DataFrame, gap_scale: float):
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
    ax_pareto = fig.add_subplot(gs[0, :])
    ax_runtime = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])

    fig.suptitle(title, fontsize=18)

    if color_map is None:
        color_map = {}

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
    if annotate_p and texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=1))

    frontier = global_pareto_frontier(summary, gap_scale)
    pareto_frontier_linestyle = "-" if output_format in {"both", "tikz"} else "--"
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
    plt.close(fig)



def main() -> None:
    args = parse_args()
    specs = parse_series_specs(args.series)
    color_map = parse_color_map(args.colors)

    gurobi_optima, _ = load_gurobi_optima(Path(args.gurobi_dir))
    rows_df, per_instance_df, common_stems = build_dataframes(
        specs=specs,
        gurobi_optima=gurobi_optima,
        exclude_seed0=args.exclude_seed0,
        verbose_common_set=args.verbose_common_set,
    )

    summary_avg = summarize_all_files(rows_df)
    opt_runtime_avg = summarize_opt_runtime(per_instance_df, args.opt_runtime_mode)
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

    best_rows_df = select_best_seed_curves(rows_df)
    summary_best = summarize_all_files(best_rows_df)
    best_instances_df = filter_instances_for_best_seed(per_instance_df, best_rows_df)
    opt_runtime_best = summarize_opt_runtime(best_instances_df, args.opt_runtime_mode)
    output_best = f"{args.output_prefix}_best"
    make_plot(
        summary=summary_best,
        title=f"{args.title} — best seed per label",
        output_base=output_best,
        gap_scale=args.gap_scale,
        annotate_p=args.annotate_p,
        opt_runtime=opt_runtime_best,
        dpi=args.dpi,
        color_map=color_map,
        output_format=args.output_format,
        tikz_standalone=args.tikz_standalone,
    )

    best_seed_table = (
        best_rows_df[["label", "seed"]]
        .drop_duplicates()
        .sort_values(["label", "seed"])
    )

    if args.output_format in {"png", "both"}:
        print(f"Wrote plot to: {output_avg}.png")
        print(f"Wrote plot to: {output_best}.png")
    if args.output_format in {"tikz", "both"}:
        print(f"Wrote plot to: {output_avg}.tex")
        print(f"Wrote plot to: {output_best}.tex")

    print()
    print(f"Common evaluation set size: {len(common_stems)}")
    print()
    print("=== Average over all files ===")
    print(summary_avg.to_string(index=False))

    if opt_runtime_avg is not None:
        print()
        print(f"OPT runtime ({args.opt_runtime_mode}, average plot): {opt_runtime_avg:.6f} s")

    print()
    print("=== Best seed per label ===")
    print(best_seed_table.to_string(index=False))
    print()
    print(summary_best.to_string(index=False))

    if opt_runtime_best is not None:
        print()
        print(f"OPT runtime ({args.opt_runtime_mode}, best plot): {opt_runtime_best:.6f} s")


if __name__ == "__main__":
    main()
