#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_DATASET_STYLES: Dict[str, Dict[str, Any]] = {
    "scp_n50_m50_k5": {
        "title": r"\tiny SCP $\left(\begin{array}{l}n=50\\ m=50\\ k=5\end{array}\right)$",
        "maximize": False,
    },
    "scp_n120-180_m150-250_k5": {
        "title": r"\tiny SCP $\left(\begin{array}{l}n\in\mathcal{U}[120,180]\\ m\in\mathcal{U}[150,250]\\ k=5\end{array}\right)$",
        "maximize": False,
    },
    "scp_n150-300_m150-300_k15": {
        "title": r"\tiny SCP $\left(\begin{array}{l}n\in[150,300]\\ m\in[150,300]\\ k=15\end{array}\right)$",
        "maximize": False,
    },
    "mis_n100_p0.15": {
        "title": r"\tiny MIS $\left(\begin{array}{l}n=100\\ p=0.15\end{array}\right)$",
        "maximize": True,
    },
    "mis_n200-300_p0.15": {
        "title": r"\tiny MIS $\left(\begin{array}{l}n\in[200,300]\\ p=0.15\end{array}\right)$",
        "maximize": True,
    },
}

DEFAULT_SERIES_STYLES: Dict[str, Dict[str, str]] = {
    "var_logits_bce::EGG": {"title": "GNN+EGG", "color": "GNNEGGColor", "mark": "+"},
    "var_logits_bce::GD": {"title": "GNN+GD", "color": "GNNGDColor", "mark": "x"},
    "lbp_fce::EGG": {"title": "LBP+EGG", "color": "MRFEGGColor", "mark": "+"},
    "lbp_fce::GD": {"title": "LBP+GD", "color": "MRFGDColor", "mark": "x"},
    "nmf_fce::EGG": {"title": "NMF+EGG", "color": "NMFEGGColor", "mark": "+"},
    "nmf_fce::GD": {"title": "NMF+GD", "color": "NMFGDColor", "mark": "x"},
}


def is_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def mean_std(vals: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    clean = [float(v) for v in vals if is_number(v)]
    if not clean:
        return None, None
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1)
    return mean, math.sqrt(var)


def latex_escape_label(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def configured_datasets(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = config.get("datasets")
    if raw:
        out: List[Dict[str, Any]] = []
        for item in raw:
            if isinstance(item, str):
                base = dict(DEFAULT_DATASET_STYLES.get(item, {}))
                base["id"] = item
                base.setdefault("title", latex_escape_label(item))
                out.append(base)
            else:
                ds_id = str(item.get("id") or item.get("name"))
                base = dict(DEFAULT_DATASET_STYLES.get(ds_id, {}))
                base.update(dict(item))
                base["id"] = ds_id
                base.setdefault("title", latex_escape_label(ds_id))
                out.append(base)
        return out
    return []


def configured_methods(config: Mapping[str, Any]) -> List[str]:
    return [str(x) for x in config.get("methods", ["var_logits_bce", "lbp_fce", "nmf_fce"])]


def configured_decodings(config: Mapping[str, Any]) -> List[str]:
    return [str(x) for x in config.get("decodings", ["EGG", "GD"])]


def ordered_2x2_datasets(datasets: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """Order panels as:
    top row: SCP datasets
    bottom row: MIS datasets
    """
    scp = [d for d in datasets if str(d["id"]).startswith("scp")]
    mis = [d for d in datasets if str(d["id"]).startswith("mis")]
    other = [d for d in datasets if not str(d["id"]).startswith(("scp", "mis"))]

    ordered = list(scp) + list(mis) + list(other)

    if len(ordered) > 4:
        print(f"[warn] 2x2 layout requested but {len(ordered)} datasets are configured. Keeping first 4.")
        ordered = ordered[:4]

    return ordered


def dataset_title(dataset_id: str, config: Mapping[str, Any]) -> str:
    for d in configured_datasets(config):
        if str(d["id"]) == dataset_id:
            return str(d.get("title", latex_escape_label(dataset_id)))
    return str(DEFAULT_DATASET_STYLES.get(dataset_id, {}).get("title", latex_escape_label(dataset_id)))


def dataset_maximize(dataset_id: str, config: Mapping[str, Any]) -> bool:
    for d in configured_datasets(config):
        if str(d["id"]) == dataset_id and "maximize" in d:
            return bool(d["maximize"])
    if dataset_id.startswith("mis"):
        return True
    if dataset_id.startswith("scp"):
        return False
    return bool(config.get("maximize", False))


def infer_dataset_id(result_stem: str, datasets: Sequence[Mapping[str, Any]]) -> str:
    candidates = sorted([str(d["id"]) for d in datasets], key=len, reverse=True)
    for ds in candidates:
        if result_stem.startswith(ds + "_") or result_stem == ds:
            return ds

    m = re.match(r"^(scp_.+?_k\d+|mis_.+?_p[0-9.]+)", result_stem)
    if m:
        return m.group(1)

    return "dataset"


def infer_method_id(result_stem: str, dataset_id: str, methods: Sequence[str]) -> str:
    mid = result_stem
    if mid.startswith(dataset_id + "_"):
        mid = mid[len(dataset_id) + 1 :]

    mid = re.sub(r"_seed\d+_.*$", "", mid)

    for method in sorted(methods, key=len, reverse=True):
        if mid == method or mid.endswith("_" + method):
            return method

    return mid


def parse_result_filename(
    path: Path,
    datasets: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
) -> Tuple[str, str, Optional[str], Optional[float]]:
    stem = path.stem

    dataset_id = infer_dataset_id(stem, datasets)
    method_id = infer_method_id(stem, dataset_id, methods)

    decoding = None
    m_dec = re.search(r"_(GD|EGG)(?:_|$)", stem)
    if m_dec:
        decoding = m_dec.group(1)

    fix_prop = None
    m_p = re.search(r"_p([0-9]+(?:\.[0-9]+)?)$", stem)
    if m_p:
        fix_prop = float(m_p.group(1))

    return dataset_id, method_id, decoding, fix_prop


def discover_result_files(results_dir: Path) -> List[Path]:
    if results_dir.is_file():
        return [results_dir]
    return sorted(p for p in results_dir.rglob("*.json") if p.is_file())


def load_gurobi_optimal_value(
    *,
    gurobi_solutions_dir: Path,
    dataset_id: str,
    instance_stem: str,
    split: str,
    optimal_value_key: str,
) -> float:
    stem = Path(str(instance_stem)).stem
    path = gurobi_solutions_dir / f"gurobi_{dataset_id}_{split}" / f"{stem}.json"

    if not path.exists():
        raise FileNotFoundError(f"missing gurobi solution file: {path}")

    payload = read_json(path)
    if payload is None:
        raise ValueError(f"could not read gurobi solution JSON: {path}")

    if not bool(payload.get("feasible", True)):
        raise ValueError(f"gurobi solution marked infeasible: {path}")

    if optimal_value_key not in payload or not is_number(payload[optimal_value_key]):
        raise KeyError(
            f"missing numeric key {optimal_value_key!r} in {path}. "
            f"Available keys: {sorted(payload.keys())}"
        )

    return float(payload[optimal_value_key])


def compute_gap(
    *,
    value: float,
    opt: float,
    maximize: bool,
    as_percent: bool,
    clamp_negative: bool,
) -> float:
    denom = max(abs(float(opt)), 1e-12)

    if maximize:
        gap = (float(opt) - float(value)) / denom
    else:
        gap = (float(value) - float(opt)) / denom

    if clamp_negative:
        gap = max(0.0, gap)

    return 100.0 * gap if as_percent else gap


def load_records(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    results_dir = Path(str(config.get("results_dir", "."))).expanduser()
    gurobi_solutions_dir = Path(str(config.get("gurobi_solutions_dir", "gurobi_solutions"))).expanduser()

    split = str(config.get("gurobi_split", "test"))
    optimal_value_key = str(config.get("optimal_value_key", "objective_value_from_x"))
    result_value_key = str(config.get("result_value_key", "solution_value"))
    runtime_key = str(config.get("runtime_key", "total"))

    gap_as_percent = bool(config.get("gap_as_percent", True))
    clamp_negative_gap = bool(config.get("clamp_negative_gap", True))
    strict_missing_opt = bool(config.get("strict_missing_opt", True))

    datasets = configured_datasets(config)
    methods = configured_methods(config)

    records: List[Dict[str, Any]] = []
    missing_opt: List[str] = []

    for path in discover_result_files(results_dir):
        payload = read_json(path)
        if payload is None:
            continue

        dataset_id, method_id, decoding_from_name, p_from_name = parse_result_filename(
            path, datasets, methods
        )

        summary_obj = payload.get("summary", {})
        summary = summary_obj if isinstance(summary_obj, dict) else {}

        decoding = str(summary.get("decoding") or decoding_from_name or "")
        fix_prop = summary.get("fix_prop", p_from_name)

        instances = payload.get("instances", [])
        if not isinstance(instances, list):
            continue

        for inst in instances:
            if not isinstance(inst, Mapping):
                continue

            instance_stem = Path(str(inst.get("instance_stem", ""))).stem
            if not instance_stem:
                continue

            try:
                opt_value = load_gurobi_optimal_value(
                    gurobi_solutions_dir=gurobi_solutions_dir,
                    dataset_id=dataset_id,
                    instance_stem=instance_stem,
                    split=split,
                    optimal_value_key=optimal_value_key,
                )
            except Exception as exc:
                missing_opt.append(f"{dataset_id}/{instance_stem}: {exc}")
                if strict_missing_opt:
                    raise
                continue

            solution_value = inst.get(result_value_key)
            timings = inst.get("timings_s", {})
            runtime = timings.get(runtime_key) if isinstance(timings, Mapping) else None

            if runtime is None:
                runtime = inst.get("runtime") or inst.get("total_runtime")

            if not (is_number(solution_value) and is_number(runtime) and is_number(fix_prop)):
                continue

            maximize = dataset_maximize(dataset_id, config)
            gap = compute_gap(
                value=float(solution_value),
                opt=float(opt_value),
                maximize=maximize,
                as_percent=gap_as_percent,
                clamp_negative=clamp_negative_gap,
            )

            records.append(
                {
                    "dataset_id": dataset_id,
                    "method_id": method_id,
                    "decoding": decoding,
                    "series_id": f"{method_id}::{decoding}",
                    "fix_prop": float(fix_prop),
                    "instance_stem": instance_stem,
                    "solution_value": float(solution_value),
                    "optimal_value": float(opt_value),
                    "gap": float(gap),
                    "runtime": float(runtime),
                    "source_file": str(path),
                }
            )

    if missing_opt and bool(config.get("warn_missing_opt", True)):
        shown = "\n".join(missing_opt[:10])
        suffix = "" if len(missing_opt) <= 10 else f"\n... ({len(missing_opt)} total)"
        print(f"[warn] missing gurobi solution files:\n{shown}{suffix}")

    return records


def aggregate(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Dict[float, Dict[str, Any]]]]:
    groups: Dict[Tuple[str, str, float], Dict[str, List[float]]] = {}
    counts: Dict[Tuple[str, str, float], int] = {}

    for rec in records:
        key = (str(rec["dataset_id"]), str(rec["series_id"]), float(rec["fix_prop"]))
        group = groups.setdefault(key, {"gap": [], "runtime": []})
        group["gap"].append(float(rec["gap"]))
        group["runtime"].append(float(rec["runtime"]))
        counts[key] = counts.get(key, 0) + 1

    out: Dict[str, Dict[str, Dict[float, Dict[str, Any]]]] = {}
    for (dataset_id, series_id, p), vals in groups.items():
        gap_mean, gap_std = mean_std(vals["gap"])
        runtime_mean, runtime_std = mean_std(vals["runtime"])
        if gap_mean is None or runtime_mean is None:
            continue

        out.setdefault(dataset_id, {}).setdefault(series_id, {})[p] = {
            "dataset_id": dataset_id,
            "series_id": series_id,
            "fix_prop": p,
            "gap_mean": gap_mean,
            "gap_std": gap_std,
            "runtime_mean": runtime_mean,
            "runtime_std": runtime_std,
            "count": counts[(dataset_id, series_id, p)],
        }

    return out


def configured_series_ids(config: Mapping[str, Any]) -> List[str]:
    series = [f"{m}::{d}" for m in configured_methods(config) for d in configured_decodings(config)]

    include_series = config.get("include_series")
    exclude_series = set(str(x) for x in config.get("exclude_series", []))

    if include_series:
        series = [str(x) for x in include_series]

    series = [s for s in series if s not in exclude_series]
    return series

def series_style(series_id: str, config: Mapping[str, Any]) -> Dict[str, str]:
    styles = dict(DEFAULT_SERIES_STYLES)
    styles.update(
        {
            str(k): {str(kk): str(vv) for kk, vv in dict(v).items()}
            for k, v in config.get("series_styles", {}).items()
        }
    )
    return styles.get(series_id, {"title": latex_escape_label(series_id), "color": "black", "mark": "*"})


def fmt(x: float) -> str:
    return f"{float(x):.12g}"


def coordinates(rows: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(f"({fmt(r['runtime_mean'])},{fmt(r['gap_mean'])})" for r in rows)


def limits(
    vals: Sequence[float],
    *,
    log: bool,
    pad: float,
    default: Tuple[float, float],
) -> Tuple[float, float]:
    xs = [float(v) for v in vals if is_number(v)]
    if log:
        xs = [v for v in xs if v > 0]

    if not xs:
        return default

    lo, hi = min(xs), max(xs)

    if lo == hi:
        if log:
            return lo / 1.5, hi * 1.5
        delta = max(abs(lo) * 0.1, 1.0)
        return lo - delta, hi + delta

    if log:
        return lo / pad, hi * pad

    delta = (hi - lo) * (pad - 1.0)
    return lo - delta, hi + delta


def pareto_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    best_gap = float("inf")
    out: List[Mapping[str, Any]] = []

    for row in sorted(rows, key=lambda r: (float(r["runtime_mean"]), float(r["gap_mean"]))):
        gap = float(row["gap_mean"])
        if gap < best_gap:
            out.append(row)
            best_gap = gap

    return out


def emit_legend(series_ids: Sequence[str], config: Mapping[str, Any]) -> List[str]:
    if not series_ids:
        return []

    ncols = int(config.get("legend_columns", min(max(len(series_ids), 1), 4)))
    xshift = str(config.get("legend_xshift", "0cm"))
    yshift = str(config.get("legend_yshift", "-0.35cm"))

    entries: List[str] = []
    for sid in series_ids:
        st = series_style(sid, config)
        color = st.get("color", "black")
        mark = st.get("mark", "*")
        title = st.get("title", sid)
        entries.append(
            rf"\tikz[baseline=-0.55ex]{{\draw[very thick, color={color}] (0,0) -- (0.42,0); "
            rf"\draw[color={color}] plot[only marks, mark={mark}, mark size=2.0pt] coordinates {{(0.21,0)}};}} & {title}"
        )

    rows: List[str] = []
    for i in range(0, len(entries), ncols):
        chunk = entries[i : i + ncols]
        chunk = chunk + ([" & "] * (ncols - len(chunk)))
        rows.append(" & ".join(chunk) + r" \\")

    return [
        rf"\node[anchor=north, draw=lightgray204, fill=white, fill opacity=0.9, text opacity=1, inner sep=3pt] at ($(current bounding box.south) + ({xshift}, {yshift})$) {{",
        r"\begin{tikzpicture}[baseline]",
        r"\matrix[matrix of nodes, nodes={anchor=west, font=\scriptsize}, column sep=8pt, row sep=3pt] {",
        *rows,
        r"};",
        r"\end{tikzpicture}",
        r"};",
    ]


def generate_latex(config: Dict[str, Any]) -> str:
    records = load_records(config)
    agg = aggregate(records)

    datasets = configured_datasets(config)
    if not datasets:
        datasets = [{"id": ds, "title": dataset_title(ds, config)} for ds in sorted(agg)]

    layout = str(config.get("layout", "2x2"))
    if layout == "2x2":
        datasets = ordered_2x2_datasets(datasets)

    series_ids = configured_series_ids(config)
    if not bool(config.get("show_empty_series", False)):
        active: List[str] = []
        for sid in series_ids:
            if any(sid in agg.get(str(d["id"]), {}) for d in datasets):
                active.append(sid)
        series_ids = active

    placement = str(config.get("placement", "t"))
    label = str(config.get("label", "fig:inference_test_nair_pareto"))
    caption = config.get("caption")

    if layout == "2x2":
        group_size = "2 by 2"
        default_width = "0.42\\textwidth"
        default_height = "0.30\\textwidth"
        default_hsep = "0.75cm"
        default_vsep = "0.75cm"
    else:
        group_size = f"{len(datasets)} by 1"
        default_width = "0.29\\textwidth"
        default_height = "0.30\\textwidth"
        default_hsep = "0.2cm"
        default_vsep = "0.0cm"

    width = str(config.get("panel_width", default_width))
    height = str(config.get("panel_height", default_height))
    hsep = str(config.get("horizontal_sep", default_hsep))
    vsep = str(config.get("vertical_sep", default_vsep))
    xlabel = str(config.get("xlabel", "Runtime (s)"))
    ylabel = str(config.get("ylabel", "Gap (\\%)"))
    xmode = str(config.get("xmode", "log"))
    ymode = str(config.get("ymode", "normal"))

    lines: List[str] = []
    lines.append(rf"\begin{{figure}}[{placement}]")
    lines.append(r"\centering")
    lines.append(r"% Requires: \usepackage{pgfplots}")
    lines.append(r"% Requires: \usepgfplotslibrary{groupplots}")
    lines.append(r"% Requires: \usetikzlibrary{calc, matrix}")
    lines.append(r"\definecolor{darkgray176}{RGB}{110,110,110}")
    lines.append(r"\definecolor{lightgray204}{RGB}{200,200,200}")

    for name, rgb in config.get("define_colors", {}).items():
        lines.append(rf"\definecolor{{{name}}}{{RGB}}{{{rgb}}}")

    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\begin{groupplot}[")
    lines.append(
        rf"    group style={{group size={group_size}, horizontal sep={hsep}, vertical sep={vsep}, y descriptions at=edge left, ylabels at=edge left}},"
    )
    lines.append(rf"    width={width},")
    lines.append(rf"    height={height},")
    lines.append(r"    scale only axis,")
    lines.append(r"    tick align=outside,")
    lines.append(r"    tick pos=left,")
    lines.append(r"    xmajorgrids,")
    lines.append(r"    ymajorgrids,")
    lines.append(r"    x grid style={darkgray176},")
    lines.append(r"    y grid style={darkgray176},")
    lines.append(r"]")
    lines.append("")

    for idx, d in enumerate(datasets):
        ds_id = str(d["id"])
        by_series = agg.get(ds_id, {})
        all_rows = [row for sid in series_ids for row in by_series.get(sid, {}).values()]

        xmin, xmax = limits(
            [r["runtime_mean"] for r in all_rows],
            log=(xmode == "log"),
            pad=float(config.get("xpad", 1.2)),
            default=(0.1, 1.0),
        )
        ymin, ymax = limits(
            [r["gap_mean"] for r in all_rows],
            log=(ymode == "log"),
            pad=float(config.get("ypad", 1.15)),
            default=(0.0, 1.0),
        )

        if "xmin" in config:
            xmin = float(config["xmin"])
        if "xmax" in config:
            xmax = float(config["xmax"])
        if "ymin" in config:
            ymin = float(config["ymin"])
        if "ymax" in config:
            ymax = float(config["ymax"])

        if layout == "2x2":
            col = idx % 2
            row = idx // 2
            is_left_col = col == 0
            is_bottom_row = row == 1
        else:
            is_left_col = idx == 0
            is_bottom_row = True

        lines.append(r"\nextgroupplot[")
        lines.append(rf"    title={{{dataset_title(ds_id, config)}}},")

        if is_bottom_row:
            lines.append(rf"    xlabel={{{xlabel}}},")
        else:
            #lines.append(r"    xticklabels=\empty,")
            lines.append(r"    xtick style={draw=none},")

        if is_left_col:
            lines.append(rf"    ylabel={{{ylabel}}},")
        else:
            #lines.append(r"    yticklabels=\empty,")
            lines.append(r"    ytick style={draw=none},")

        if xmode == "log":
            lines.append(r"    xmode=log,")
            lines.append(r"    log basis x={10},")
        if ymode == "log":
            lines.append(r"    ymode=log,")
            lines.append(r"    log basis y={10},")

        lines.append(rf"    xmin={fmt(xmin)}, xmax={fmt(xmax)},")
        lines.append(rf"    ymin={fmt(ymin)}, ymax={fmt(ymax)},")
        lines.append(r"]")

        for sid in series_ids:
            rows = sorted(by_series.get(sid, {}).values(), key=lambda r: float(r["fix_prop"]))
            if not rows:
                continue

            st = series_style(sid, config)
            plot_mode = "" if bool(config.get("connect_all_points", True)) else "only marks, "

            lines.append(rf"% {ds_id} / {sid}")
            lines.append(
                rf"\addplot+[semithick, {plot_mode}color={st.get('color', 'black')}, "
                rf"mark={st.get('mark', '*')}, mark size=3, mark options={{solid}}]"
            )
            lines.append(rf"coordinates {{{coordinates(rows)}}};")

            if bool(config.get("annotate_fix_prop", False)):
                for row_data in rows:
                    lines.append(
                        rf"\node[font=\tiny, anchor=south west, text={st.get('color', 'black')}] "
                        rf"at (axis cs:{fmt(row_data['runtime_mean'])},{fmt(row_data['gap_mean'])}) "
                        rf"{{{fmt(row_data['fix_prop'])}}};"
                    )

        if bool(config.get("draw_pareto_frontier", True)) and all_rows:
            frontier = pareto_rows(all_rows)
            lines.append(r"\addplot [thick, black, dashed]")
            lines.append(rf"coordinates {{{coordinates(frontier)}}};")

        lines.append("")

    lines.append(r"\end{groupplot}")
    lines.extend(emit_legend(series_ids, config))
    lines.append(r"\end{tikzpicture}")

    if caption:
        lines.append(rf"\caption{{{caption}}}")

    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{figure}")

    return "\n".join(lines)


def example_config() -> Dict[str, Any]:
    return {
        "results_dir": "test_inference_fix_and_solve",
        "gurobi_solutions_dir": "gurobi_solutions",
        "gurobi_split": "test",
        "optimal_value_key": "objective_value_from_x",
        "layout": "2x2",
        "label": "fig:inference_test_nair_pareto",
        "caption": "Fix-and-solve Pareto frontiers. Each point is averaged over the tested instances for one fixed-variable proportion.",
        "datasets": [
            {
                "id": "scp_n50_m50_k5",
                "title": r"\tiny SCP $\left(\begin{array}{l}n=50\\ m=50\\ k=5\end{array}\right)$",
                "maximize": False,
            },
            {
                "id": "scp_n120-180_m150-250_k5",
                "title": r"\tiny SCP $\left(\begin{array}{l}n\in\mathcal{U}[120,180]\\ m\in\mathcal{U}[150,250]\\ k=5\end{array}\right)$",
                "maximize": False,
            },
            {
                "id": "mis_n100_p0.15",
                "title": r"\tiny MIS $\left(\begin{array}{l}n=100\\ p=0.15\end{array}\right)$",
                "maximize": True,
            },
            {
                "id": "mis_n200-300_p0.15",
                "title": r"\tiny MIS $\left(\begin{array}{l}n\in[200,300]\\ p=0.15\end{array}\right)$",
                "maximize": True,
            },
        ],
        "methods": ["var_logits_bce", "lbp_fce", "nmf_fce"],
        "decodings": ["EGG", "GD"],
        "series_styles": DEFAULT_SERIES_STYLES,
        "define_colors": {
            "GNNEGGColor": "0,102,204",
            "GNNGDColor": "120,200,150",
            "MRFEGGColor": "200,60,40",
            "MRFGDColor": "140,80,160",
            "NMFEGGColor": "230,120,20",
            "NMFGDColor": "0,150,180",
        },
        "result_value_key": "solution_value",
        "runtime_key": "total",
        "gap_as_percent": True,
        "clamp_negative_gap": True,
        "strict_missing_opt": True,
        "xmode": "log",
        "ymode": "normal",
        "xlabel": "Runtime (s)",
        "ylabel": "Gap (\\%)",
        "panel_width": "0.42\\textwidth",
        "panel_height": "0.30\\textwidth",
        "horizontal_sep": "0.75cm",
        "vertical_sep": "0.75cm",
        "legend_columns": 3,
        "legend_yshift": "-0.35cm",
        "connect_all_points": True,
        "draw_pareto_frontier": True,
        "annotate_fix_prop": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a PGFPlots/TikZ Pareto figure for Nair fix-and-solve inference tests."
    )
    parser.add_argument("--config", help="Optional JSON config describing the figure.")
    parser.add_argument("--output", help="Output .tex file. Prints to stdout if omitted.")
    parser.add_argument("--results-dir", help="Directory containing bpropy-test-nair JSON outputs.")
    parser.add_argument(
        "--gurobi-solutions-dir",
        help="Directory containing folders named gurobi_<dataset>_<split>.",
    )
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

    if args.results_dir:
        config["results_dir"] = args.results_dir
    if args.gurobi_solutions_dir:
        config["gurobi_solutions_dir"] = args.gurobi_solutions_dir

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
