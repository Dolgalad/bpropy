#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DEFAULT_METHOD_STYLES: Dict[str, Dict[str, str]] = {
    "lbp_gd": {"title": "LBP+GD", "mark": "*"},
    "nmf_gd": {"title": "NMF+GD", "mark": "square*"},
}

DEFAULT_DATASET_STYLES: Dict[str, Dict[str, str]] = {
    "mis_n100_p0.15": {"title": r"MIS $n=100$, $p=0.15$", "color": "MISColor"},
    "scp_n50_m50_k5": {"title": r"SCP $n=50$, $m=50$, $k=5$", "color": "SCPColor"},
}

# Backward-compatible aliases for old configs.
DEFAULT_MODEL_STYLES: Dict[str, Dict[str, str]] = {
    "lbp_gd": {"title": "LBP+GD", "color": "LBPGDColor", "mark": "*"},
    "nmf_gd": {"title": "NMF+GD", "color": "NMFGDColor", "mark": "square*"},
}

import math

def log_tick_exponents(vmin: float, vmax: float) -> list[int]:
    if vmin <= 0 or vmax <= 0:
        return []
    lo = int(math.floor(math.log10(vmin)))
    hi = int(math.ceil(math.log10(vmax)))
    return list(range(lo, hi + 1))

def latex_escape_label(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def is_number(x: Any) -> bool:
    try:
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False


def mean_std(vals: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    clean = [float(v) for v in vals if is_number(v)]
    n = len(clean)
    if n == 0:
        return None, None
    mean = sum(clean) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in clean) / (n - 1)
    return mean, math.sqrt(var)


def normalize_setting(v: Any) -> str:
    if v is None:
        return "inf"
    s = str(v)
    if s.lower() in {"inf", "infinity", "none"}:
        return "inf"
    if is_number(s):
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(f)
    return s


def setting_sort_key(v: Any) -> float:
    s = normalize_setting(v)
    if s == "inf":
        return float("inf")
    return float(s)


def setting_label(v: Any) -> str:
    s = normalize_setting(v)
    return r"$\infty$" if s == "inf" else s


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def configured_methods(config: Mapping[str, Any]) -> List[str]:
    if "methods" in config:
        return [str(m) for m in config.get("methods", [])]
    return [str(m) for m in config.get("models", ["lbp_gd", "nmf_gd"])]


def configured_datasets(config: Mapping[str, Any], root_dir: Path) -> List[Dict[str, str]]:
    raw = config.get("datasets")
    if raw:
        out: List[Dict[str, str]] = []
        for item in raw:
            if isinstance(item, str):
                ds_id = item
                out.append({"id": ds_id, "path": ds_id})
            else:
                ds_id = str(item.get("id") or item.get("name") or item.get("path"))
                out.append({
                    "id": ds_id,
                    "path": str(item.get("path", ds_id)),
                    "title": str(item.get("title", latex_escape_label(ds_id))),
                    "color": str(item.get("color", DEFAULT_DATASET_STYLES.get(ds_id, {}).get("color", "black"))),
                })
        return out

    ds_id = root_dir.name if not root_dir.is_file() else "dataset"
    return [{
        "id": ds_id,
        "path": ".",
        "title": DEFAULT_DATASET_STYLES.get(ds_id, {}).get("title", latex_escape_label(ds_id)),
        "color": DEFAULT_DATASET_STYLES.get(ds_id, {}).get("color", "black"),
    }]


def discover_results(root_dir: Path, datasets: Sequence[Mapping[str, str]], methods: Sequence[str]) -> List[Tuple[Path, str, str]]:
    if root_dir.is_file():
        return [(root_dir, "dataset", "method")]

    paths: List[Tuple[Path, str, str]] = []
    for dataset in datasets:
        dataset_id = str(dataset["id"])
        dataset_path = root_dir / str(dataset.get("path", dataset_id))
        search_root = dataset_path if dataset_path.exists() else root_dir

        if methods:
            for method_id in methods:
                for pat in (
                    f"{method_id}/seed_*/results.jsonl",
                    f"{method_id}/results.jsonl",
                    f"**/{method_id}/seed_*/results.jsonl",
                    f"**/{method_id}/results.jsonl",
                ):
                    for path in search_root.glob(pat):
                        paths.append((path.resolve(), dataset_id, method_id))
        else:
            for path in search_root.rglob("results.jsonl"):
                method_id = path.parent.parent.name if path.parent.name.startswith("seed_") else path.parent.name
                paths.append((path.resolve(), dataset_id, method_id))

    seen: set[Tuple[Path, str, str]] = set()
    out: List[Tuple[Path, str, str]] = []
    for item in sorted(paths, key=lambda x: str(x[0])):
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def load_instance_records(root_dir: Path, datasets: Sequence[Mapping[str, str]], methods: Sequence[str], *, prefer_record_ids: bool = False) -> List[Dict[str, Any]]:
    load_instance_records.prefer_record_ids = prefer_record_ids
    records: List[Dict[str, Any]] = []
    for path, dataset_id, method_id in discover_results(root_dir, datasets, methods):
        for rec in read_jsonl(path):
            if rec.get("record_type") != "instance_result":
                continue
            if rec.get("status") != "completed":
                continue
            row = dict(rec)
            if bool(getattr(load_instance_records, "prefer_record_ids", False)):
                row["dataset_id"] = str(rec.get("dataset_id") or rec.get("dataset") or dataset_id)
                row["method_id"] = str(rec.get("method_id") or rec.get("model_id") or method_id)
            else:
                row["dataset_id"] = str(dataset_id)
                row["method_id"] = str(method_id)
            records.append(row)
    return records


def aggregate_pareto(records: Sequence[Mapping[str, Any]], ablation_type: str, gap_scale: float = 1.0) -> Dict[str, Dict[str, Dict[str, Any]]]:
    groups: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {}
    counts: Dict[Tuple[str, str, str], int] = {}

    for rec in records:
        if str(rec.get("ablation_type")) != ablation_type:
            continue
        dataset_id = str(rec.get("dataset_id", "dataset"))
        method_id = str(rec.get("method_id") or rec.get("model_id", "method"))
        setting = normalize_setting(rec.get("ablation_value"))
        key = (dataset_id, method_id, setting)
        g = groups.setdefault(key, {"gap": [], "runtime": [], "bp_calls": []})
        counts[key] = counts.get(key, 0) + 1

        gap = rec.get("dec_gap")
        runtime = rec.get("total_runtime")
        bp_calls = rec.get("num_bp_calls")
        if is_number(gap):
            g["gap"].append(float(gap) * float(gap_scale))
        if is_number(runtime):
            g["runtime"].append(float(runtime))
        if is_number(bp_calls):
            g["bp_calls"].append(float(bp_calls))

    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (dataset_id, method_id, setting), vals in groups.items():
        gap_mean, gap_std = mean_std(vals["gap"])
        time_mean, time_std = mean_std(vals["runtime"])
        calls_mean, calls_std = mean_std(vals["bp_calls"])
        if gap_mean is None or time_mean is None:
            continue
        series_id = f"{dataset_id}::{method_id}"
        out.setdefault(series_id, {})[setting] = {
            "dataset_id": dataset_id,
            "method_id": method_id,
            "setting": setting,
            "gap_mean": gap_mean,
            "gap_std": gap_std,
            "runtime_mean": time_mean,
            "runtime_std": time_std,
            "bp_calls_mean": calls_mean,
            "bp_calls_std": calls_std,
            "count": counts[(dataset_id, method_id, setting)],
        }
    return out


def dataset_style(dataset_id: str, config: Mapping[str, Any]) -> Dict[str, str]:
    styles = dict(DEFAULT_DATASET_STYLES)
    styles.update({str(k): dict(v) for k, v in config.get("dataset_styles", {}).items()})
    for item in config.get("datasets", []):
        if isinstance(item, Mapping):
            ds_id = str(item.get("id") or item.get("name") or item.get("path"))
            styles.setdefault(ds_id, {})
            styles[ds_id].update({str(k): str(v) for k, v in item.items() if k in {"title", "color"}})
    if dataset_id in styles:
        return styles[dataset_id]
    return {"title": latex_escape_label(dataset_id), "color": "black"}


def method_style(method_id: str, config: Mapping[str, Any]) -> Dict[str, str]:
    styles = dict(DEFAULT_METHOD_STYLES)
    styles.update({str(k): {kk: str(vv) for kk, vv in dict(v).items() if kk != "color"} for k, v in config.get("model_styles", {}).items()})
    styles.update({str(k): dict(v) for k, v in config.get("method_styles", {}).items()})
    if method_id in styles:
        return styles[method_id]
    return {"title": latex_escape_label(method_id), "mark": "*"}


def series_style(series_id: str, config: Mapping[str, Any]) -> Dict[str, str]:
    dataset_id, method_id = series_id.split("::", 1)
    ds = dataset_style(dataset_id, config)
    mt = method_style(method_id, config)
    return {
        "title": f"{ds.get('title', latex_escape_label(dataset_id))} / {mt.get('title', latex_escape_label(method_id))}",
        "dataset_title": ds.get("title", latex_escape_label(dataset_id)),
        "method_title": mt.get("title", latex_escape_label(method_id)),
        "color": ds.get("color", "black"),
        "mark": mt.get("mark", "*"),
    }


def coordinates(rows: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(
        f"({float(r['runtime_mean']):.12g},{float(r['gap_mean']):.12g})"
        for r in rows
    )


def configured_series_ids(datasets: Sequence[Mapping[str, str]], methods: Sequence[str]) -> List[str]:
    return [f"{d['id']}::{m}" for d in datasets for m in methods]


def series_ids_for_dataset(dataset_id: str, methods: Sequence[str]) -> List[str]:
    return [f"{dataset_id}::{m}" for m in methods]


def active_series_ids(series_ids: Sequence[str], *aggregated: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for series_id in series_ids:
        if any(series_id in agg and agg.get(series_id) for agg in aggregated):
            out.append(series_id)
    return out


def active_dataset_ids(datasets: Sequence[Mapping[str, str]], methods: Sequence[str], *aggregated: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for dataset in datasets:
        ds_id = str(dataset["id"])
        ds_series = series_ids_for_dataset(ds_id, methods)
        if any(series_id in agg and agg.get(series_id) for agg in ds_series for agg in aggregated):
            out.append(ds_id)
    return out


def legend_mode(config: Mapping[str, Any], *, default_mode: str) -> str:
    mode = str(config.get("legend_mode", "auto"))
    if mode == "auto":
        return default_mode
    return mode


def emit_manual_legend(series_ids: Sequence[str], config: Mapping[str, Any], *, default_mode: str = "series") -> List[str]:
    mode = legend_mode(config, default_mode=default_mode)
    if not series_ids:
        return []

    entries: List[str] = []
    if mode == "methods":
        seen_methods: List[str] = []
        for series_id in series_ids:
            _ds_id, mt_id = series_id.split("::", 1)
            if mt_id not in seen_methods:
                seen_methods.append(mt_id)
        for mt_id in seen_methods:
            mt = method_style(mt_id, config)
            mark = mt.get("mark", "*")
            title = mt.get("title", latex_escape_label(mt_id))
            entries.append(
                rf"\tikz[baseline=-0.55ex]{{\draw[very thick, color=black] (0,0) -- (0.42,0); "
                rf"\draw[color=black] plot[only marks, mark={mark}, mark size=2.0pt] coordinates {{(0.21,0)}};}} & {title}"
            )
    elif mode == "datasets":
        seen_datasets: List[str] = []
        for series_id in series_ids:
            ds_id, _mt_id = series_id.split("::", 1)
            if ds_id not in seen_datasets:
                seen_datasets.append(ds_id)
        for ds_id in seen_datasets:
            ds = dataset_style(ds_id, config)
            color = ds.get("color", "black")
            title = ds.get("title", latex_escape_label(ds_id))
            entries.append(rf"\tikz[baseline=-0.55ex]{{\draw[very thick, color={color}] (0,0) -- (0.42,0);}} & {title}")
    elif mode == "separate":
        dataset_ids: List[str] = []
        method_ids: List[str] = []
        for series_id in series_ids:
            ds_id, mt_id = series_id.split("::", 1)
            if ds_id not in dataset_ids:
                dataset_ids.append(ds_id)
            if mt_id not in method_ids:
                method_ids.append(mt_id)
        for ds_id in dataset_ids:
            ds = dataset_style(ds_id, config)
            color = ds.get("color", "black")
            title = ds.get("title", latex_escape_label(ds_id))
            entries.append(rf"\tikz[baseline=-0.55ex]{{\draw[very thick, color={color}] (0,0) -- (0.42,0);}} & {title}")
        for mt_id in method_ids:
            mt = method_style(mt_id, config)
            mark = mt.get("mark", "*")
            title = mt.get("title", latex_escape_label(mt_id))
            entries.append(rf"\tikz[baseline=-0.55ex]{{\draw[color=black] plot[only marks, mark={mark}, mark size=2.0pt] coordinates {{(0,0)}};}} & {title}")
    else:
        for series_id in series_ids:
            st = series_style(series_id, config)
            color = st.get("color", "MISSColor")
            mark = st.get("mark", "*")
            title = st.get("title", latex_escape_label(series_id))
            entries.append(
                rf"\tikz[baseline=-0.55ex]{{\draw[very thick, color={color}] (0,0) -- (0.42,0); "
                rf"\draw[color={color}] plot[only marks, mark={mark}, mark size=2.0pt] coordinates {{(0.21,0)}};}} & {title}"
            )

    ncols_default = min(len(entries), 4) or 1
    ncols = int(config.get("legend_columns", ncols_default))
    xshift = str(config.get("legend_xshift", "0cm"))
    yshift = str(config.get("legend_yshift", "-0.25cm"))

    rows: List[List[str]] = []
    current: List[str] = []
    for entry in entries:
        current.append(entry)
        if len(current) == ncols:
            rows.append(current)
            current = []
    if current:
        rows.append(current)

    matrix_rows: List[str] = []
    for row in rows:
        missing = ncols - len(row)
        cells = list(row) + ([" & "] * missing)
        matrix_rows.append(" & ".join(cells) + r" \\")

    return [
        rf"\node[anchor=north, draw=lightgray204, fill=white, fill opacity=0.9, text opacity=1, inner sep=3pt] at ($(current bounding box.south) + ({xshift}, {yshift})$) {{",
        r"\begin{tikzpicture}[baseline]",
        r"\matrix[",
        r"    matrix of nodes,",
        r"    nodes={anchor=west, font=\scriptsize},",
        r"    column sep=8pt,",
        r"    row sep=3pt",
        r"] {",
        *matrix_rows,
        r"};",
        r"\end{tikzpicture}",
        r"};",
    ]


def axis_options(
    *,
    config: Mapping[str, Any],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    show_ylabel: bool,
    show_xlabel: bool,
    show_ytick_labels: bool = True,
    show_xtick_labels: bool = True,
    ylabel_text: Optional[str] = None,
    title: Optional[str] = None,
) -> List[str]:
    width = str(config.get("panel_width", r"0.45\textwidth"))
    height = str(config.get("panel_height", r"0.34\textwidth"))
    xlabel = str(config.get("xlabel", r"Runtime (s)"))
    ylabel = str(ylabel_text if ylabel_text is not None else config.get("ylabel", r"Optimality gap"))

    lines = [
        rf"    width={width},",
        rf"    height={height},",
        r"    scale only axis,",
        r"    tick align=outside,",
        r"    tick pos=left,",
        r"    xmajorgrids,",
        r"    ymajorgrids,",
        r"    xmode=log,",
        r"    ymode=log,",
        r"    x grid style={darkgray176},",
        r"    y grid style={darkgray176},",
        rf"    xmin={xmin:.12g}, xmax={xmax:.12g},",
        rf"    ymin={ymin:.12g}, ymax={ymax:.12g},",
        r"    xtick style={color=black},",
        r"    ytick style={color=black},",
    ]
    if show_xlabel:
        lines.insert(11, rf"    xlabel={{{xlabel}}},")
    if show_ylabel:
        lines.insert(11, rf"    ylabel={{{ylabel}}},")
        lines.insert(12, r"    ylabel style={align=center},")
    if not show_xtick_labels:
        lines.append(r"    xticklabels=\empty,")
    if not show_ytick_labels:
        lines.append(r"    yticklabels=\empty,")
        lines.append(r"    ytick style={draw=none},")
    if title:
        insert_at = 13 if show_ylabel else 12
        if show_xlabel:
            insert_at += 1
        lines.insert(insert_at, rf"    title={{{title}}},")
    return lines


def emit_panel_body(
    *,
    aggregated: Dict[str, Dict[str, Dict[str, Any]]],
    series_ids: Sequence[str],
    config: Mapping[str, Any],
) -> List[str]:
    lines: List[str] = []
    for series_id in series_ids:
        by_setting = aggregated.get(series_id, {})
        rows = sorted(by_setting.values(), key=lambda r: setting_sort_key(r["setting"]))
        if not rows:
            continue

        style = series_style(series_id, config)
        color = style.get("color", "black")
        mark = style.get("mark", "*")
        connect_all_points = bool(config.get("connect_all_points", True))

        lines.append(rf"% {series_id}: all settings")
        if connect_all_points:
            lines.append(
                rf"\addplot+[mark={mark}, mark size=2.0pt, very thick, color={color}] "
                rf"coordinates {{{coordinates(rows)}}};"
            )
        else:
            lines.append(
                rf"\addplot+[only marks, mark={mark}, mark size=2.0pt, color={color}] "
                rf"coordinates {{{coordinates(rows)}}};"
            )
        lines.append("")

        if bool(config.get("annotate_settings", False)):
            for row in rows:
                label = setting_label(row["setting"])
                lines.append(
                    rf"\node[font=\tiny, anchor=south west, text={color}] at "
                    rf"(axis cs:{float(row['runtime_mean']):.12g},{float(row['gap_mean']):.12g}) {{{label}}};"
                )
            lines.append("")
    return lines


def y_limits_global(all_aggs: Sequence[Dict[str, Dict[str, Dict[str, Any]]]], config: Mapping[str, Any]) -> Tuple[float, float]:
    gaps: List[float] = []
    for agg in all_aggs:
        for by_setting in agg.values():
            for row in by_setting.values():
                gaps.append(float(row["gap_mean"]))

    if not gaps:
        return 0.0, 1.0
    ymin = float(config.get("ymin", 0.0))
    ymax = float(config.get("ymax", max(gaps) * 1.15 if max(gaps) > 0 else 1.0))
    return ymin, ymax


def y_limits_for_series_ids(
    series_ids: Sequence[str],
    aggregated_list: Sequence[Dict[str, Dict[str, Dict[str, Any]]]],
    config: Mapping[str, Any],
) -> Tuple[float, float]:
    gaps: List[float] = []
    for aggregated in aggregated_list:
        for series_id in series_ids:
            by_setting = aggregated.get(series_id, {})
            for row in by_setting.values():
                gaps.append(float(row["gap_mean"]))

    ymin = float(config.get("ymin", 0.0))
    if not gaps:
        return ymin, float(config.get("empty_ymax", 1.0))
    ymax = float(config.get("ymax", max(gaps) * 1.15 if max(gaps) > 0 else 1.0))
    return ymin, ymax


def x_limits_for_series_ids(
    series_ids: Sequence[str],
    aggregated: Dict[str, Dict[str, Dict[str, Any]]],
    config: Mapping[str, Any],
) -> Tuple[float, float]:
    runtimes: List[float] = []
    for series_id in series_ids:
        by_setting = aggregated.get(series_id, {})
        for row in by_setting.values():
            runtimes.append(float(row["runtime_mean"]))

    xmin = float(config.get("xmin", 0.0))
    if not runtimes:
        return xmin, float(config.get("empty_xmax", 1.0))
    xmax = float(config.get("xmax", max(runtimes) * 1.15 if max(runtimes) > 0 else 1.0))
    return xmin, xmax


def emit_combined_layout(
    *,
    config: Mapping[str, Any],
    series_ids: Sequence[str],
    run_every_agg: Dict[str, Dict[str, Dict[str, Any]]],
    max_iter_agg: Dict[str, Dict[str, Dict[str, Any]]],
    ymin: float,
    ymax: float,
) -> List[str]:
    lines: List[str] = []
    lines.append(r"\begin{groupplot}[")
    lines.append(
        rf"    group style={{group size=2 by 1, horizontal sep={str(config.get('horizontal_sep', '1.0cm'))}}},"
    )
    lines.append(r"]")
    lines.append("")

    lines.append(r"\nextgroupplot[")
    run_xmin, run_xmax = x_limits_for_series_ids(series_ids, run_every_agg, config)
    lines.extend(axis_options(
        config=config,
        xmin=run_xmin,
        xmax=run_xmax,
        ymin=ymin,
        ymax=ymax,
        show_ylabel=True,
        show_xlabel=True,
        show_ytick_labels=True,
        show_xtick_labels=True,
        title=str(config.get("run_every_title", r"Varying reinference interval $k$")) if bool(config.get("show_panel_titles", True)) else None,
    ))
    lines.append(r"]")
    lines.extend(emit_panel_body(aggregated=run_every_agg, series_ids=series_ids, config=config))
    lines.append("")

    lines.append(r"\nextgroupplot[")
    bp_xmin, bp_xmax = x_limits_for_series_ids(series_ids, max_iter_agg, config)
    lines.extend(axis_options(
        config=config,
        xmin=bp_xmin,
        xmax=bp_xmax,
        ymin=ymin,
        ymax=ymax,
        show_ylabel=False,
        show_xlabel=True,
        show_ytick_labels=bool(config.get("show_ytick_labels_on_non_left", False)),
        show_xtick_labels=True,
        title=str(config.get("bp_max_iter_title", r"Varying message-passing iterations")) if bool(config.get("show_panel_titles", True)) else None,
    ))
    lines.append(r"]")
    lines.extend(emit_panel_body(aggregated=max_iter_agg, series_ids=series_ids, config=config))
    lines.append("")

    lines.append(r"\end{groupplot}")
    lines.extend(emit_manual_legend(series_ids, config, default_mode="series"))
    return lines


def emit_dataset_grid_layout(
    *,
    config: Mapping[str, Any],
    datasets: Sequence[Mapping[str, str]],
    methods: Sequence[str],
    run_every_agg: Dict[str, Dict[str, Dict[str, Any]]],
    max_iter_agg: Dict[str, Dict[str, Dict[str, Any]]],
    ymin: float,
    ymax: float,
) -> List[str]:
    dataset_ids = [str(d["id"]) for d in datasets]
    if not bool(config.get("show_empty_datasets", True)):
        active_ids = set(active_dataset_ids(datasets, methods, run_every_agg, max_iter_agg))
        dataset_ids = [ds_id for ds_id in dataset_ids if ds_id in active_ids]

    nrows = max(1, len(dataset_ids))
    hsep = str(config.get("horizontal_sep", "1.0cm"))
    vsep = str(config.get("vertical_sep", "0.85cm"))
    show_xlabel_only_bottom = bool(config.get("show_xlabel_only_bottom", False))
    show_ylabel_every_row = bool(config.get("show_ylabel_every_row", True))

    lines: List[str] = []
    lines.append(r"\begin{groupplot}[")
    lines.append(rf"    group style={{group size=2 by {nrows}, horizontal sep={hsep}, vertical sep={vsep}}},")
    lines.append(r"]")
    lines.append("")

    for row_idx, dataset_id in enumerate(dataset_ids):
        ds = dataset_style(dataset_id, config)
        ds_title = ds.get("title", latex_escape_label(dataset_id))
        ds_series_ids = series_ids_for_dataset(dataset_id, methods)
        bottom_row = row_idx == (nrows - 1)
        show_xlabel = (not show_xlabel_only_bottom) or bottom_row

        row_ymin, row_ymax = y_limits_for_series_ids(ds_series_ids, [run_every_agg, max_iter_agg], config)
        run_xmin, run_xmax = x_limits_for_series_ids(ds_series_ids, run_every_agg, config)
        base_ylabel = str(config.get("ylabel", r"Optimality gap"))
        row_ylabel = ds_title + r"\\" + base_ylabel
        lines.append(r"\nextgroupplot[")
        lines.extend(axis_options(
            config=config,
            xmin=run_xmin,
            xmax=run_xmax,
            ymin=row_ymin,
            ymax=row_ymax,
            show_ylabel=show_ylabel_every_row,
            show_xlabel=show_xlabel,
            show_ytick_labels=True,
            show_xtick_labels=True,
            ylabel_text=row_ylabel if show_ylabel_every_row else None,
            title=None,
        ))
        lines.append(r"]")
        lines.extend(emit_panel_body(aggregated=run_every_agg, series_ids=ds_series_ids, config=config))
        lines.append("")

        bp_xmin, bp_xmax = x_limits_for_series_ids(ds_series_ids, max_iter_agg, config)
        lines.append(r"\nextgroupplot[")
        lines.extend(axis_options(
            config=config,
            xmin=bp_xmin,
            xmax=bp_xmax,
            ymin=row_ymin,
            ymax=row_ymax,
            show_ylabel=False,
            show_xlabel=show_xlabel,
            show_ytick_labels=bool(config.get("show_ytick_labels_on_non_left", False)),
            show_xtick_labels=True,
            title=None,
        ))
        lines.append(r"]")
        lines.extend(emit_panel_body(aggregated=max_iter_agg, series_ids=ds_series_ids, config=config))
        lines.append("")

    lines.append(r"\end{groupplot}")
    all_series = configured_series_ids(datasets, methods)
    lines.extend(emit_manual_legend(all_series, config, default_mode="methods"))
    return lines


def generate_latex(config: Dict[str, Any]) -> str:
    root_dir = Path(config.get("root_dir", ".")).expanduser()
    methods = configured_methods(config)
    datasets = configured_datasets(config, root_dir)
    all_series = configured_series_ids(datasets, methods)
    records = load_instance_records(
        root_dir,
        datasets,
        methods,
        prefer_record_ids=bool(config.get("prefer_record_ids", False)),
    )

    gap_scale = float(config.get("gap_scale", 100.0))
    run_every_agg = aggregate_pareto(records, "run_every", gap_scale=gap_scale)
    max_iter_agg = aggregate_pareto(records, "bp_max_iter", gap_scale=gap_scale)
    plotted_series = active_series_ids(all_series, run_every_agg, max_iter_agg)
    ymin, ymax = y_limits_global([run_every_agg, max_iter_agg], config)

    label = str(config.get("label", "fig:inference_freq_max_iter_ablation_pareto"))
    caption = config.get("caption")
    placement = str(config.get("placement", "t"))
    layout = str(config.get("layout", "dataset_grid"))

    lines: List[str] = []
    lines.append(rf"\begin{{figure}}[{placement}]")
    lines.append(r"\centering")
    lines.append("")
    lines.append(r"% Requires: \usepackage{pgfplots}")
    lines.append(r"% Requires: \usepgfplotslibrary{groupplots}")
    lines.append(r"% Requires: \usetikzlibrary{calc, matrix}")
    lines.append(r"% Requires: \pgfplotsset{compat=1.18}")
    lines.append(r"\definecolor{darkgray176}{RGB}{176,176,176}")
    lines.append(r"\definecolor{lightgray204}{RGB}{204,204,204}")
    lines.append(r"% Dataset colors are assumed to be defined outside this file.")
    lines.append("")
    lines.append(r"\begin{tikzpicture}")

    if layout == "combined":
        lines.extend(emit_combined_layout(
            config=config,
            series_ids=plotted_series,
            run_every_agg=run_every_agg,
            max_iter_agg=max_iter_agg,
            ymin=ymin,
            ymax=ymax,
        ))
    else:
        lines.extend(emit_dataset_grid_layout(
            config=config,
            datasets=datasets,
            methods=methods,
            run_every_agg=run_every_agg,
            max_iter_agg=max_iter_agg,
            ymin=ymin,
            ymax=ymax,
        ))

    lines.append(r"\end{tikzpicture}")
    if caption:
        lines.append("")
        lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def example_config() -> Dict[str, Any]:
    return {
        "root_dir": "runs/inference_freq_max_iter_ablation",
        "layout": "dataset_grid",
        "label": "fig:inference_freq_max_iter_ablation_pareto",
        "caption": "Pareto frontiers for guided decimation ablations. Each row corresponds to one dataset. Left: varying the reinference interval with fixed message-passing depth. Right: varying the number of message-passing iterations with fixed reinference interval.",
        "datasets": [
            {
                "id": "mis_n100_p0.15",
                "path": "mis_n100_p0.15",
                "title": "MIS-S",
                "color": "MISSColor",
            },
            {
                "id": "scp_n50_m50_k5",
                "path": "scp_n50_m50_k5",
                "title": "SC-S",
                "color": "SCSColor",
            },
        ],
        "methods": ["lbp_gd", "nmf_gd"],
        "method_styles": {
            "lbp_gd": {"title": "LBP+GD", "mark": "*"},
            "nmf_gd": {"title": "NMF+GD", "mark": "square*"},
        },
        "panel_width": "0.42\\textwidth",
        "panel_height": "0.16\\textwidth",
        "horizontal_sep": "0.9cm",
        "vertical_sep": "0.95cm",
        "xlabel": "Runtime (s)",
        "ylabel": "Optimality gap (\\%)",
        "show_xlabel_only_bottom": True,
        "show_ylabel_every_row": True,
        "show_empty_datasets": True,
        "show_panel_titles": False,
        "show_ytick_labels_on_non_left": False,
        "annotate_settings": False,
        "connect_all_points": True,
        "legend_mode": "methods",
        "legend_columns": 2,
        "legend_xshift": "0cm",
        "legend_yshift": "-0.35cm",
        "gap_scale": 100.0,
        "prefer_record_ids": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a PGFPlots/TikZ Pareto figure for inference frequency / max-iteration ablations."
    )
    parser.add_argument("--config", required=False, help="JSON config describing the figure.")
    parser.add_argument("--output", required=False, help="Output .tex file. Prints to stdout if omitted.")
    parser.add_argument("--root-dir", default=None, help="Directory containing ablation results.jsonl files.")
    parser.add_argument("--write-example-config", help="Write an example config and exit.")
    args = parser.parse_args()

    if args.write_example_config:
        out = Path(args.write_example_config).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(example_config(), indent=2) + "\n", encoding="utf-8")
        print(out)
        return

    if args.config:
        with Path(args.config).expanduser().open("r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = example_config()

    if args.root_dir is not None:
        config["root_dir"] = args.root_dir

    latex = generate_latex(config)

    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(latex + "\n", encoding="utf-8")
        print(out)
    else:
        print(latex)


if __name__ == "__main__":
    main()
