#!/usr/bin/env python3
from __future__ import annotations

import math
import statistics as stats
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import argparse
import json
from pathlib import Path

from latex_figure_common import (
    DEFAULT_COLORS,
    aggregate_bar_value,
    default_ymax_from_values,
    emit_manual_legend_block,
    generate_color_defs,
    is_number,
    latex_escape_label,
    latex_list,
    series_color,
)

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
        raise KeyError("Each result source needs one of: json_files, files, runs, glob, run_prefix, prefix")

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


def latex_list(items: Sequence[str]) -> str:
    return ",".join(items)


def build_values(config: Mapping[str, Any]) -> Tuple[List[str], List[Mapping[str, Any]], List[List[Optional[float]]]]:
    groups = config["groups"]
    series = config["series"]
    group_labels = [str(g["label"]) for g in groups]

    values: List[List[Optional[float]]] = []
    for s in series:
        skey = str(s["key"])
        row_values: List[Optional[float]] = []
        for group in groups:
            results = group.get("results", {})
            source = results.get(skey)
            if source is None:
                row_values.append(None)
                continue
            if isinstance(source, str):
                source_spec: Dict[str, Any] = {"run_prefix": source}
            else:
                source_spec = dict(source)
            merged = dict(s)
            merged.update(source_spec)
            row_values.append(aggregate_bar_value(merged, config))
        values.append(row_values)

    return group_labels, list(series), values


def default_ymax(values: Sequence[Sequence[Optional[float]]]) -> float:
    vals = [float(v) for row in values for v in row if v is not None and is_number(v)]
    if not vals:
        return 1.0
    top = max(vals)
    if top <= 0:
        return 1.0
    return top * 1.15


def generate_grouped_bar_latex(config: Dict[str, Any]) -> str:
    group_labels, series, values = build_values(config)
    n_groups = len(group_labels)
    n_series = len(series)

    label = str(config.get("label", "fig:grouped_bar"))
    caption = config.get("caption")
    placement = str(config.get("placement", "t"))
    colors = config.get("colors", DEFAULT_COLORS)

    group_width = float(config.get("group_width", 0.8))
    bar_width = group_width / max(n_series, 1)
    x_min = float(config.get("xmin", -0.6))
    x_max = float(config.get("xmax", (n_groups - 1) + 0.6))
    y_min = float(config.get("ymin", 0.0))
    y_max = float(config.get("ymax", default_ymax_from_values(values)))

    xtick = [str(i) for i in range(n_groups)]
    latex_labels = []
    for g in config["groups"]:
        raw_label = str(g["label"])
        latex_labels.append(raw_label if g.get("latex", config.get("latex_group_labels", False)) else latex_escape_label(raw_label))

    width = str(config.get("width", r"0.82\textwidth"))
    height = str(config.get("height", r"0.42\textwidth"))
    ylabel = str(config.get("ylabel", r"Gap (\%)"))
    xlabel = str(config.get("xlabel", "Dataset"))
    title = config.get("title")

    lines: List[str] = []
    lines.append(rf"\begin{{figure}}[{placement}]")
    lines.append(r"\centering")
    lines.append("")
    lines.extend(generate_color_defs(config, series))
    lines.append("")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\begin{axis}[")
    lines.append(rf"    width={width},")
    lines.append(rf"    height={height},")
    lines.append(r"    scale only axis,")
    lines.append(r"    tick align=outside,")
    lines.append(r"    tick pos=left,")
    lines.append(r"    x grid style={darkgray176},")
    lines.append(r"    y grid style={darkgray176},")
    lines.append(r"    ymajorgrids,")
    lines.append(rf"    xmin={x_min:.6g}, xmax={x_max:.6g},")
    lines.append(rf"    ymin={y_min:.6g}, ymax={y_max:.6g},")
    lines.append(rf"    xtick={{{latex_list(xtick)}}},")
    lines.append(rf"    xticklabels={{{latex_list(latex_labels)}}},")
    lines.append(rf"    xlabel={{{xlabel}}},")
    lines.append(rf"    ylabel={{{ylabel}}},")
    if title:
        lines.append(rf"    title={{{title}}},")
    lines.append(r"    xtick style={color=black},")
    lines.append(r"    ytick style={color=black},")
    lines.append(rf"    legend columns={int(config.get('legend_columns', min(n_series, 4)))},")
    lines.append(r"    legend cell align=left,")
    lines.append(r"    legend style={")
    lines.append(r"      draw=lightgray204,")
    lines.append(r"      /tikz/every even column/.append style={column sep=0.4cm}")
    lines.append(r"    },")
    lines.append(r"]")
    lines.append("")

    for series_idx, s in enumerate(series):
        color = s.get("color", f"Series{series_idx}Color")
        offset_left = -group_width / 2.0 + series_idx * bar_width
        offset_right = offset_left + bar_width * float(config.get("bar_width_fraction", 0.95))
        lines.append(rf"% --- {s.get('title', s.get('key', series_idx))} ---")
        for group_idx in range(n_groups):
            value = values[series_idx][group_idx]
            if value is None or not is_number(value):
                lines.append(rf"% missing: group={group_labels[group_idx]}, series={s.get('key', series_idx)}")
                continue
            left = group_idx + offset_left
            right = group_idx + offset_right
            #lines.append(
            #    rf"\draw[draw=none,fill={color}] "
            #    rf"(axis cs:{left:.12g},0) rectangle (axis cs:{right:.12g},{float(value):.12g});"
            #)
            bar_draw = str(s.get("bar_draw", config.get("bar_draw", config.get("segment_draw", "black"))))
            bar_line_width = str(
                s.get(
                    "bar_line_width",
                    config.get("bar_line_width", config.get("segment_line_width", "0.3pt")),
                )
            )
            fill_opacity = float(s.get("fill_opacity", config.get("fill_opacity", 0.55)))
            
            lines.append(
                rf"\draw[draw={bar_draw},line width={bar_line_width},"
                rf"fill={color},fill opacity={fill_opacity}] "
                rf"(axis cs:{left:.12g},0) rectangle "
                rf"(axis cs:{right:.12g},{float(value):.12g});"
            )
        lines.append("")

    for idx, s in enumerate(series):
        color = series_color(s, idx, colors)
        title_text = str(s.get("title", s.get("key", idx)))
        if not bool(config.get("manual_legend", True)):
            lines.append(rf"\addlegendimage{{ybar,ybar legend,draw=none,fill={color}}}")
            lines.append(rf"\addlegendentry{{{title_text}}}")

    lines.append("")
    lines.append(r"\end{axis}")
    lines.extend(emit_manual_legend_block(series, config))

    lines.append(r"\end{tikzpicture}")
    if caption:
        lines.append("")
        lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def example_config() -> Dict[str, Any]:
    return {
        "figure_type": "grouped_bar_latex",
        "label": "fig:chapter_3_mrf_nair_gap_bars",
        "caption": "Average optimality gap (\\%) by dataset and method. Lower is better.",
        "xlabel": "Dataset",
        "ylabel": "Gap (\\%)",
        "width": "0.82\\textwidth",
        "height": "0.42\\textwidth",
        "legend_columns": 4,
        "seeds": [0, 1, 2],
        "seed_pattern": "{prefix}_seed{seed}/test_eval.json",
        "metric_scales": {"gap": 100.0, "relative_gap": 100.0},
        "series": [
            {"key": "gnn_egg", "title": "GNN EGG", "method": "entropy_greedy", "metric": "gap", "color": "GNNEGGColor"},
            {"key": "lbp_egg", "title": "LBP EGG", "method": "entropy_greedy", "metric": "gap", "color": "MRFEGGColor"},
            {"key": "lbp_gd", "title": "LBP GD", "method": "guided_decimation", "metric": "gap", "color": "MRFGDColor"},
            {"key": "nmf_egg", "title": "NMF EGG", "method": "entropy_greedy", "metric": "gap", "color": "PairMRFEGGColor"},
        ],
        "groups": [
            {
                "label": "SC-S",
                "results": {
                    "gnn_egg": {"run_prefix": "runs/scp_n50_m50_k5_var_logits_bce"},
                    "lbp_egg": {"run_prefix": "runs/scp_n50_m50_k5_lbp_fce"},
                    "lbp_gd": {"run_prefix": "runs/scp_n50_m50_k5_lbp_fce"},
                    "nmf_egg": {"run_prefix": "runs/scp_n50_m50_k5_nmf_fce"},
                },
            },
            {
                "label": "MIS-S",
                "results": {
                    "gnn_egg": {"run_prefix": "runs/mis_n100_p0.15_var_logits_bce"},
                    "lbp_egg": {"run_prefix": "runs/mis_n100_p0.15_lbp_fce"},
                    "lbp_gd": {"run_prefix": "runs/mis_n100_p0.15_lbp_fce"},
                    "nmf_egg": {"run_prefix": "runs/mis_n100_p0.15_nmf_fce"},
                },
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PGFPlots/TikZ grouped bar figure from bpropy JSON evaluation files.")
    parser.add_argument("--config", required=False, help="JSON config describing the figure.")
    parser.add_argument("--output", required=False, help="Output .tex file. Prints to stdout if omitted.")
    parser.add_argument("--root-dir", default=None, help="Root directory containing all run directories.")
    parser.add_argument("--write-example-config", help="Write an example config and exit.")
    args = parser.parse_args()

    if args.write_example_config:
        out = Path(args.write_example_config).expanduser()
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

    latex = generate_grouped_bar_latex(config)
    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(latex + "\n", encoding="utf-8")
        print(out)
    else:
        print(latex)


if __name__ == "__main__":
    main()
