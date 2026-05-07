#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from latex_figure_common import (
    DEFAULT_COLORS,
    aggregate_mean_for_metric,
    default_ymax_from_values,
    emit_manual_legend_block,
    generate_color_defs,
    is_number,
    latex_escape_label,
    latex_list,
)


def component_value(
    source_spec: Mapping[str, Any],
    segment: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Optional[float]:
    if "metric" in segment:
        return aggregate_mean_for_metric(source_spec, config, str(segment["metric"]))

    if "remainder_of" in segment:
        total = aggregate_mean_for_metric(source_spec, config, str(segment["remainder_of"]))
        if total is None:
            return None

        subtract = 0.0
        for m in segment.get("minus", []):
            v = aggregate_mean_for_metric(source_spec, config, str(m))
            if v is not None:
                subtract += v

        return max(0.0, total - subtract)

    raise KeyError("Each segment needs either 'metric' or 'remainder_of'.")


def build_stacked_values(
    config: Mapping[str, Any],
) -> tuple[list[str], list[Mapping[str, Any]], list[Mapping[str, Any]], list[list[list[Optional[float]]]]]:
    groups = config["groups"]
    series = list(config["series"])
    segments = list(config["segments"])
    group_labels = [str(g["label"]) for g in groups]

    values: list[list[list[Optional[float]]]] = []

    for s in series:
        skey = str(s["key"])
        series_values: list[list[Optional[float]]] = []

        for group in groups:
            results = group.get("results", {})
            source = results.get(skey)

            if source is None:
                series_values.append([None for _ in segments])
                continue

            source_spec = {"run_prefix": source} if isinstance(source, str) else dict(source)
            merged_source = dict(s)
            merged_source.update(source_spec)

            stack = [component_value(merged_source, segment, config) for segment in segments]
            series_values.append(stack)

        values.append(series_values)

    return group_labels, series, segments, values


def default_ymax_stacked(values: list[list[list[Optional[float]]]]) -> float:
    totals: list[float] = []
    for series_values in values:
        for stack in series_values:
            clean = [float(v) for v in stack if v is not None and is_number(v)]
            if clean:
                totals.append(sum(clean))
    if not totals:
        return 1.0
    top = max(totals)
    return 1.0 if top <= 0 else top * 1.15


def emit_pattern_legend(segments: list[Mapping[str, Any]], config: Mapping[str, Any]) -> list[str]:
    if not bool(config.get("manual_pattern_legend", True)):
        return []

    entries_per_row = int(config.get("pattern_legend_entries_per_row", len(segments)))
    anchor_expr = str(
        config.get(
            "pattern_legend_anchor",
            r"$(current bounding box.south) + (0,-0.70cm)$",
        )
    )
    node_opts = str(
        config.get(
            "pattern_legend_node_options",
            "anchor=north, draw=lightgray204, fill=white, fill opacity=0.85, text opacity=1, inner sep=1pt",
        )
    )

    marker_width = float(config.get("legend_marker_width", 0.35))
    marker_height = float(config.get("legend_marker_height", 0.12))

    lines: list[str] = []
    lines.append(rf"\node[{node_opts}] at ({anchor_expr}) {{")
    lines.append(r"\begin{tikzpicture}[baseline]")
    lines.append(r"\matrix[")
    lines.append(r"    matrix of nodes,")
    lines.append(r"    nodes={anchor=west, font=\scriptsize},")
    lines.append(r"    column sep=10pt,")
    lines.append(r"    row sep=0pt")
    lines.append(r"] {")

    row_cells: list[str] = []
    for i, seg in enumerate(segments):
        title = str(seg.get("title", seg.get("key", f"Segment {i}")))
        pattern = str(seg.get("pattern", "none"))
        pattern_option = "" if pattern == "none" else f", pattern={pattern}"

        entry = (
            rf"    \draw[draw=black, fill=white{pattern_option}] "
            rf"(0,0) rectangle ({marker_width:.3f},{marker_height:.3f}); & {title}"
        )
        row_cells.append(entry)

        is_end_of_row = len(row_cells) == entries_per_row
        is_last = i == len(segments) - 1
        if is_end_of_row or is_last:
            line = " &\n".join(row_cells)
            if is_last:
                missing = entries_per_row - len(row_cells)
                if missing > 0:
                    line += " &" + " &".join([""] * (2 * missing - 1))
            line += r" \\"
            lines.append(line)
            row_cells = []

    lines.append(r"};")
    lines.append(r"\end{tikzpicture}")
    lines.append(r"};")
    return lines


def generate_stacked_grouped_bar_latex(config: Dict[str, Any]) -> str:
    group_labels, series, segments, values = build_stacked_values(config)

    n_groups = len(group_labels)
    n_series = len(series)

    label = str(config.get("label", "fig:stacked_grouped_bar"))
    caption = config.get("caption")
    placement = str(config.get("placement", "t"))
    colors = config.get("colors", DEFAULT_COLORS)

    group_width = float(config.get("group_width", 0.8))
    bar_width = group_width / max(n_series, 1)
    x_min = float(config.get("xmin", -0.6))
    x_max = float(config.get("xmax", (n_groups - 1) + 0.6))
    y_min = float(config.get("ymin", 0.0))
    y_max = float(config.get("ymax", default_ymax_stacked(values)))

    xtick = [str(i) for i in range(n_groups)]
    latex_labels = []
    for g in config["groups"]:
        raw_label = str(g["label"])
        latex_labels.append(
            raw_label
            if g.get("latex", config.get("latex_group_labels", False))
            else latex_escape_label(raw_label)
        )

    width = str(config.get("width", r"0.82\textwidth"))
    height = str(config.get("height", r"0.42\textwidth"))
    ylabel = str(config.get("ylabel", r"Runtime (s)"))
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
    lines.append(r"]")
    lines.append("")

    for series_idx, s in enumerate(series):
        color = s.get("color", f"Series{series_idx}Color")
        offset_left = -group_width / 2.0 + series_idx * bar_width
        offset_right = offset_left + bar_width * float(config.get("bar_width_fraction", 0.95))

        lines.append(rf"% --- {s.get('title', s.get('key', series_idx))} ---")
        for group_idx in range(n_groups):
            stack = values[series_idx][group_idx]
            if not stack or all(v is None for v in stack):
                lines.append(rf"% missing: group={group_labels[group_idx]}, series={s.get('key', series_idx)}")
                continue

            left = group_idx + offset_left
            right = group_idx + offset_right
            bottom = 0.0

            for seg_idx, seg in enumerate(segments):
                value = stack[seg_idx]
                if value is None or not is_number(value) or float(value) <= 0:
                    continue

                top = bottom + float(value)
                #pattern = str(seg.get("pattern", "none"))

                #if pattern == "none":
                #    draw_opts = f"draw=none,fill={color}"
                #else:
                #    pattern_color = str(seg.get("pattern_color", config.get("pattern_color", "black")))
                #    draw_opts = (
                #        f"draw=none,fill={color},"
                #        f"postaction={{pattern={pattern},pattern color={pattern_color}}}"
                #    )
                #
                #lines.append(
                #    rf"\draw[{draw_opts}] "
                #    rf"(axis cs:{left:.12g},{bottom:.12g}) rectangle "
                #    rf"(axis cs:{right:.12g},{top:.12g});"
                #)
                pattern = str(seg.get("pattern", "none"))
                pattern_color = str(seg.get("pattern_color", config.get("pattern_color", "black")))
                
                segment_draw = str(seg.get("segment_draw", config.get("segment_draw", "black")))
                segment_line_width = str(seg.get("segment_line_width", config.get("segment_line_width", "0.3pt")))
                fill_opacity = float(seg.get("fill_opacity", config.get("fill_opacity", 0.55)))
                
                # First draw the colored segment with transparency and a visible border.
                lines.append(
                    rf"\draw[draw={segment_draw},line width={segment_line_width},"
                    rf"fill={color},fill opacity={fill_opacity}] "
                    rf"(axis cs:{left:.12g},{bottom:.12g}) rectangle "
                    rf"(axis cs:{right:.12g},{top:.12g});"
                )
                
                # Then overlay the pattern only.
                if pattern != "none":
                    lines.append(
                        rf"\draw[draw=none,pattern={pattern},pattern color={pattern_color}] "
                        rf"(axis cs:{left:.12g},{bottom:.12g}) rectangle "
                        rf"(axis cs:{right:.12g},{top:.12g});"
                    )
                bottom = top

        lines.append("")

    lines.append(r"\end{axis}")

    lines.extend(emit_manual_legend_block(series, config, segments))
    #lines.extend(emit_pattern_legend(segments, config))

    lines.append(r"\end{tikzpicture}")
    if caption:
        lines.append("")
        lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def example_config() -> Dict[str, Any]:
    return {
        "label": "fig:chapter_3_mrf_nair_runtime_stacked_bars",
        "caption": "Average runtime by dataset and method. Bar height is total runtime; patterns show runtime components.",
        "xlabel": "Dataset",
        "ylabel": "Runtime (s)",
        "width": "0.82\\textwidth",
        "height": "0.42\\textwidth",
        "seeds": [0, 1, 2],
        "seed_pattern": "{prefix}_seed{seed}/test_eval.json",
        "manual_legend": True,
        "manual_pattern_legend": True,
        "legend_entries_per_row": 5,
        "series": [
            {"key": "gnn_egg", "title": "GNN EGG", "method": "entropy_greedy", "color": "GNNEGGColor"},
            {"key": "lbp_egg", "title": "LBP EGG", "method": "entropy_greedy", "color": "MRFEGGColor"},
            {"key": "lbp_gd", "title": "LBP GD", "method": "guided_decimation", "color": "MRFGDColor"},
            {"key": "nmf_egg", "title": "NMF EGG", "method": "entropy_greedy", "color": "PairMRFEGGColor"},
            {"key": "nmf_gd", "title": "NMF GD", "method": "guided_decimation", "color": "PairMRFGDColor"},
        ],
        "segments": [
            {"key": "forward", "title": "Forward", "metric": "forward_t", "pattern": "none"},
            {"key": "decode", "title": "Decode", "metric": "decode_t", "pattern": "north east lines"},
            {
                "key": "other",
                "title": "Other",
                "remainder_of": "total_t",
                "minus": ["forward_t", "decode_t"],
                "pattern": "dots",
            },
        ],
        "groups": [
            {
                "label": "SC-S",
                "results": {
                    "gnn_egg": {"run_prefix": "runs/scp_n50_m50_k5_var_logits_bce"},
                    "lbp_egg": {"run_prefix": "runs/scp_n50_m50_k5_lbp_fce"},
                    "lbp_gd": {"run_prefix": "runs/scp_n50_m50_k5_lbp_fce"},
                    "nmf_egg": {"run_prefix": "runs/scp_n50_m50_k5_nmf_fce"},
                    "nmf_gd": {"run_prefix": "runs/scp_n50_m50_k5_nmf_fce"},
                },
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a PGFPlots/TikZ stacked grouped bar figure from bpropy JSON evaluation files."
    )
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

    latex = generate_stacked_grouped_bar_latex(config)

    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(latex + "\n", encoding="utf-8")
        print(out)
    else:
        print(latex)


if __name__ == "__main__":
    main()
