from __future__ import annotations

import glob
import json
import math
import statistics as stats
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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

DEFAULT_COLORS = []


def normalize_method(method: Optional[str]) -> Optional[str]:
    if method is None:
        return None
    return METHOD_ALIASES.get(str(method), str(method))


def normalize_metric(metric: str) -> str:
    return METRIC_ALIASES.get(str(metric), str(metric))


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


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


def load_payloads_for_source(spec: Mapping[str, Any], config: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    paths = seed_json_paths(spec, config)
    payloads: List[Mapping[str, Any]] = []

    for p in paths:
        if not (p.exists() or p.is_dir()):
            continue
        try:
            payloads.append(load_json(p))
        except FileNotFoundError:
            continue

    return payloads


def aggregate_bar_value(
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Optional[float]:
    payloads = load_payloads_for_source(spec, config)
    if not payloads:
        return None

    method = spec.get("method", config.get("method"))
    metric = str(spec.get("metric", config.get("metric", "gap")))
    stat = str(spec.get("stat", config.get("stat", "mean"))).lower()

    per_seed = [summary_mean_std_from_payload(p, metric=metric, method=method) for p in payloads]
    per_seed = [x for x in per_seed if x is not None]
    if not per_seed:
        return None

    means = [x[0] for x in per_seed]

    if stat in {"mean", "avg", "average"}:
        value = stats.fmean(means)
    elif stat == "best":
        maximize = bool(spec.get("maximize", config.get("maximize", False)))
        value = max(means) if maximize else min(means)
    elif stat == "min":
        value = min(means)
    elif stat == "max":
        value = max(means)
    else:
        raise ValueError(f"Unsupported stat for bar plot: {stat!r}")

    return value * metric_scale(metric, spec, config)


def aggregate_mean_for_metric(
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    metric: str,
) -> Optional[float]:
    merged = dict(spec)
    merged["metric"] = metric
    merged["stat"] = "mean"
    return aggregate_bar_value(merged, config)


def latex_escape_label(text: str) -> str:
    repl = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
    }
    return "".join(repl.get(ch, ch) for ch in str(text))


def latex_list(items: Sequence[str]) -> str:
    return ",".join(items)


def series_color(series: Mapping[str, Any], idx: int, colors: Sequence[Mapping[str, Any]]) -> str:
    if "color" in series:
        return str(series["color"])
    if idx < len(colors):
        return str(colors[idx]["name"])
    return f"AutoBarColor{idx}"


def generate_color_defs(config: Mapping[str, Any], series: Sequence[Mapping[str, Any]]) -> List[str]:
    colors = config.get("colors", DEFAULT_COLORS)
    lines = [
        r"\definecolor{darkgray176}{RGB}{110,110,110}",
        r"\definecolor{lightgray204}{RGB}{200,200,200}",
    ]

    defined = {"darkgray176", "lightgray204"}
    for color in colors:
        name = str(color["name"])
        if name in defined:
            continue
        rgb = color.get("rgb", [0, 0, 0])
        lines.append(rf"\definecolor{{{name}}}{{RGB}}{{{int(rgb[0])},{int(rgb[1])},{int(rgb[2])}}}")
        defined.add(name)

    return lines


def default_ymax_from_values(values: Sequence[Sequence[Optional[float]]]) -> float:
    vals = [float(v) for row in values for v in row if v is not None and is_number(v)]
    if not vals:
        return 1.0
    top = max(vals)
    if top <= 0:
        return 1.0
    return top * 1.15

def emit_manual_legend_block(
    series: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    segments: Optional[list[Mapping[str, Any]]] = None,
) -> list[str]:
    if not bool(config.get("manual_legend", True)):
        return []

    anchor_expr = str(
        config.get(
            "legend_anchor",
            r"$(current bounding box.south) + (0,-0.38cm)$",
        )
    )

    node_opts = str(
        config.get(
            "legend_node_options",
            "anchor=north, draw=lightgray204, fill=white, fill opacity=0.9, text opacity=1, inner sep=3pt",
        )
    )

    method_entries_per_row = int(
        config.get("legend_entries_per_row", len(series))
    )
    pattern_entries_per_row = int(
        config.get("pattern_legend_entries_per_row", len(segments) if segments else 0)
    )

    method_marker_width = float(config.get("legend_marker_width", 0.42))
    method_marker_height = float(config.get("legend_marker_height", 0.16))

    pattern_marker_width = float(
        config.get("pattern_legend_marker_width", method_marker_width)
    )
    pattern_marker_height = float(
        config.get("pattern_legend_marker_height", method_marker_height)
    )

    legend_fill_opacity = float(
        config.get("legend_fill_opacity", config.get("fill_opacity", 0.55))
    )
    legend_draw = str(
        config.get("legend_draw", config.get("bar_draw", config.get("segment_draw", "black")))
    )
    legend_line_width = str(
        config.get(
            "legend_line_width",
            config.get("bar_line_width", config.get("segment_line_width", "0.3pt")),
        )
    )

    pattern_color = str(config.get("pattern_color", "black"))

    lines: list[str] = []
    lines.append(rf"\node[{node_opts}] at ({anchor_expr}) {{")
    lines.append(r"\begin{tikzpicture}[baseline]")
    lines.append(r"\matrix[")
    lines.append(r"    matrix of nodes,")
    lines.append(r"    nodes={anchor=west, font=\scriptsize},")
    lines.append(r"    column sep=8pt,")
    lines.append(r"    row sep=3pt")
    lines.append(r"] {")

    # ---- first row(s): method/color legend ----
    method_cells: list[str] = []
    for i, s in enumerate(series):
        color = str(s.get("color", f"Series{i}Color"))
        title = str(s.get("title", s.get("key", f"Series {i}")))

        entry = (
            rf"\draw[draw={legend_draw}, line width={legend_line_width}, "
            rf"fill={color}, fill opacity={legend_fill_opacity}] "
            rf"(0,0) rectangle ({method_marker_width:.3f},{method_marker_height:.3f}); & {title}"
        )
        method_cells.append(entry)

        is_end_of_row = len(method_cells) == method_entries_per_row
        is_last = i == len(series) - 1

        if is_end_of_row or is_last:
            line = " &\n".join(method_cells)
            if is_last and len(method_cells) < method_entries_per_row:
                missing = method_entries_per_row - len(method_cells)
                if missing > 0:
                    line += " &" + " &".join([""] * (2 * missing - 1))
            line += r" \\"
            lines.append(line)
            method_cells = []

    # ---- spacer row before pattern legend ----
    if segments:
        total_cols = max(method_entries_per_row, pattern_entries_per_row) * 2
        lines.append(" &".join([""] * (total_cols - 1)) + r" \\")

        # ---- second row(s): pattern legend ----
        pattern_cells: list[str] = []
        for i, seg in enumerate(segments):
            title = str(seg.get("title", seg.get("key", f"Segment {i}")))
            pattern = str(seg.get("pattern", "none"))
            seg_pattern_color = str(seg.get("pattern_color", pattern_color))

            if pattern == "none":
                entry = (
                    rf"\draw[draw={legend_draw}, line width={legend_line_width}, fill=white] "
                    rf"(0,0) rectangle ({pattern_marker_width:.3f},{pattern_marker_height:.3f}); & {title}"
                )
            else:
                entry = (
                    rf"\draw[draw={legend_draw}, line width={legend_line_width}, fill=white, "
                    rf"postaction={{pattern={pattern},pattern color={seg_pattern_color}}}] "
                    rf"(0,0) rectangle ({pattern_marker_width:.3f},{pattern_marker_height:.3f}); & {title}"
                )

            pattern_cells.append(entry)

            is_end_of_row = len(pattern_cells) == pattern_entries_per_row
            is_last = i == len(segments) - 1

            if is_end_of_row or is_last:
                line = " &\n".join(pattern_cells)
                if is_last and len(pattern_cells) < pattern_entries_per_row:
                    missing = pattern_entries_per_row - len(pattern_cells)
                    if missing > 0:
                        line += " &" + " &".join([""] * (2 * missing - 1))
                line += r" \\"
                lines.append(line)
                pattern_cells = []

    lines.append(r"};")
    lines.append(r"\end{tikzpicture}")
    lines.append(r"};")
    return lines
#def emit_manual_bar_legend(series: list[Mapping[str, Any]], config: Mapping[str, Any]) -> list[str]:
#    if not bool(config.get("manual_legend", True)):
#        return []
#
#    entries_per_row = int(config.get("legend_entries_per_row", 3))
#    anchor_expr = str(
#        config.get(
#            "legend_anchor",
#            r"$(current bounding box.south) + (0,-0.25cm)$",
#        )
#    )
#
#    node_opts = str(
#        config.get(
#            "legend_node_options",
#            "anchor=north, draw=lightgray204, fill=white, fill opacity=0.85, text opacity=1, inner sep=1pt",
#        )
#    )
#
#    marker_width = float(config.get("legend_marker_width", 0.35))
#    marker_height = float(config.get("legend_marker_height", 0.12))
#
#    lines: list[str] = []
#    lines.append(rf"\node[{node_opts}] at ({anchor_expr}) {{")
#    lines.append(r"\begin{tikzpicture}[baseline]")
#    lines.append(r"\matrix[")
#    lines.append(r"    matrix of nodes,")
#    lines.append(r"    nodes={anchor=west, font=\scriptsize},")
#    lines.append(r"    column sep=10pt,")
#    lines.append(r"    row sep=0pt")
#    lines.append(r"] {")
#
#    row_cells: list[str] = []
#    for i, s in enumerate(series):
#        color = str(s.get("color", f"Series{i}Color"))
#        title = str(s.get("title", s.get("key", f"Series {i}")))
#
#        #entry = (
#        #    rf"    \draw[draw=none, fill={color}] "
#        #    rf"(0,0) rectangle ({marker_width:.3f},{marker_height:.3f}); & {title}"
#        #)
#
#        legend_fill_opacity = float(config.get("legend_fill_opacity", config.get("fill_opacity", 0.55)))
#        legend_draw = str(config.get("legend_segment_draw", config.get("segment_draw", "black")))
#        legend_line_width = str(config.get("legend_segment_line_width", config.get("segment_line_width", "0.3pt")))
#        
#        entry = (
#            rf"    \draw[draw={legend_draw}, line width={legend_line_width}, "
#            rf"fill={color}, fill opacity={legend_fill_opacity}] "
#            rf"(0,0) rectangle ({marker_width:.3f},{marker_height:.3f}); & {title}"
#        )
#        row_cells.append(entry)
#
#        is_end_of_row = len(row_cells) == entries_per_row
#        is_last = i == len(series) - 1
#
#        if is_end_of_row or is_last:
#            line = " &\n".join(row_cells)
#            if is_last:
#                missing = entries_per_row - len(row_cells)
#                if missing > 0:
#                    line += " &" + " &".join([""] * (2 * missing - 1))
#            line += r" \\"
#            lines.append(line)
#            row_cells = []
#
#    lines.append(r"};")
#    lines.append(r"\end{tikzpicture}")
#    lines.append(r"};")
#    return lines
