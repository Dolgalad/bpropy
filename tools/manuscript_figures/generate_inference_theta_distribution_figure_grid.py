#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_DATASET_STYLES: Dict[str, Dict[str, str]] = {
    "mis_n100_p0.15": {
        "title": r"MIS-S",
    },
    "mis_n200-300_p0.15": {
        "title": r"MIS-L",
    },
    "scp_n50_m50_k5": {
        "title": r"SC-S",
    },
    "scp_n120-180_m150-250_k5": {
        "title": r"SC-L",
    },
}

DEFAULT_METHOD_STYLES: Dict[str, Dict[str, str]] = {
    "lbp": {
        "title": "LBP",
        "color": "LBPGDColor",
    },
    "nmf": {
        "title": "NMF",
        "color": "NMFGDColor",
    },
    "lbp_gd": {
        "title": "LBP",
        "color": "LBPGDColor",
    },
    "nmf_gd": {
        "title": "NMF",
        "color": "NMFGDColor",
    },
}


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


def clean_values(values: Any, *, max_values: Optional[int] = None) -> List[float]:
    if not isinstance(values, list):
        return []

    out = [float(v) for v in values if is_number(v)]

    if max_values is not None and max_values > 0 and len(out) > max_values:
        # Deterministic downsampling to keep emitted TikZ compact when needed.
        idx = np.linspace(0, len(out) - 1, int(max_values)).astype(int)
        out = [out[int(i)] for i in idx]

    return out


def configured_methods(config: Mapping[str, Any]) -> List[Dict[str, str]]:
    raw = config.get("methods", config.get("models", ["lbp", "nmf"]))

    out: List[Dict[str, str]] = []
    for item in raw:
        if isinstance(item, str):
            out.append({"id": item, "path": item})
        else:
            method_id = str(item.get("id") or item.get("name") or item.get("path"))
            out.append(
                {
                    "id": method_id,
                    "path": str(item.get("path", method_id)),
                    "title": str(
                        item.get(
                            "title",
                            DEFAULT_METHOD_STYLES.get(method_id, {}).get(
                                "title",
                                latex_escape_label(method_id),
                            ),
                        )
                    ),
                    "color": str(
                        item.get(
                            "color",
                            DEFAULT_METHOD_STYLES.get(method_id, {}).get(
                                "color",
                                "black",
                            ),
                        )
                    ),
                }
            )

    return out


def configured_datasets(config: Mapping[str, Any], root_dir: Path) -> List[Dict[str, str]]:
    raw = config.get("datasets", [])

    if not raw:
        ds_id = root_dir.name
        return [
            {
                "id": ds_id,
                "path": ".",
                "title": DEFAULT_DATASET_STYLES.get(ds_id, {}).get(
                    "title",
                    latex_escape_label(ds_id),
                ),
            }
        ]

    out: List[Dict[str, str]] = []
    for item in raw:
        if isinstance(item, str):
            ds_id = item
            out.append(
                {
                    "id": ds_id,
                    "path": ds_id,
                    "title": DEFAULT_DATASET_STYLES.get(ds_id, {}).get(
                        "title",
                        latex_escape_label(ds_id),
                    ),
                }
            )
        else:
            ds_id = str(item.get("id") or item.get("name") or item.get("path"))
            out.append(
                {
                    "id": ds_id,
                    "path": str(item.get("path", ds_id)),
                    "title": str(
                        item.get(
                            "title",
                            DEFAULT_DATASET_STYLES.get(ds_id, {}).get(
                                "title",
                                latex_escape_label(ds_id),
                            ),
                        )
                    ),
                }
            )

    return out


def dataset_style(dataset_id: str, config: Mapping[str, Any]) -> Dict[str, str]:
    styles = dict(DEFAULT_DATASET_STYLES)
    styles.update(
        {str(k): {str(kk): str(vv) for kk, vv in dict(v).items()} for k, v in config.get("dataset_styles", {}).items()}
    )

    for item in config.get("datasets", []):
        if isinstance(item, Mapping):
            ds_id = str(item.get("id") or item.get("name") or item.get("path"))
            styles.setdefault(ds_id, {})
            for key in ("title",):
                if key in item:
                    styles[ds_id][key] = str(item[key])

    if dataset_id in styles:
        return styles[dataset_id]

    return {"title": latex_escape_label(dataset_id)}


def method_style(method_id: str, config: Mapping[str, Any]) -> Dict[str, str]:
    styles = dict(DEFAULT_METHOD_STYLES)

    # Backward compatibility with your previous configs.
    styles.update(
        {str(k): {str(kk): str(vv) for kk, vv in dict(v).items()} for k, v in config.get("model_styles", {}).items()}
    )
    styles.update(
        {str(k): {str(kk): str(vv) for kk, vv in dict(v).items()} for k, v in config.get("method_styles", {}).items()}
    )

    for item in config.get("methods", config.get("models", [])):
        if isinstance(item, Mapping):
            mt_id = str(item.get("id") or item.get("name") or item.get("path"))
            styles.setdefault(mt_id, {})
            for key in ("title", "color"):
                if key in item:
                    styles[mt_id][key] = str(item[key])

    if method_id in styles:
        return styles[method_id]

    return {"title": latex_escape_label(method_id), "color": "black"}


def discover_theta_files(
    root_dir: Path,
    dataset: Mapping[str, str],
    method: Mapping[str, str],
    config: Mapping[str, Any],
) -> List[Path]:
    dataset_id = str(dataset["id"])
    dataset_path = str(dataset.get("path", dataset_id))

    method_id = str(method["id"])
    method_path = str(method.get("path", method_id))

    # New flat-file convention:
    #   "{dataset_id}_{method_id}_fce.json"
    #
    # Example:
    #   scp_n50_m50_k5_lbp_fce.json
    if "file_template" in config:
        filename = str(config["file_template"]).format(
            dataset_id=dataset_id,
            dataset_path=dataset_path,
            method_id=method_id,
            method_path=method_path,
        )

        path = root_dir / filename
        return [path.resolve()] if path.is_file() else []

    # Old nested convention.
    dataset_dir = root_dir / dataset_path
    filename = str(config.get("theta_filename", "theta_distribution.json"))
    patterns = config.get("theta_file_patterns")

    if patterns is None:
        patterns = [
            f"{method_path}/{filename}",
            f"{method_path}/seed_*/{filename}",
            f"{method_path}/seed*/{filename}",
            f"{method_path}/**/{filename}",
            f"**/{method_path}/{filename}",
            f"**/{method_path}/**/{filename}",
        ]

    search_root = dataset_dir if dataset_dir.exists() else root_dir

    paths: List[Path] = []
    for pat in patterns:
        for p in search_root.glob(str(pat)):
            if p.is_file():
                paths.append(p.resolve())

    seen = set()
    unique: List[Path] = []
    for p in sorted(paths, key=lambda x: str(x)):
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)

    return unique

def load_theta_values_from_file(
    path: Path,
    *,
    max_values_per_file: Optional[int] = None,
) -> Tuple[List[float], List[float]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return [], []

    feasible = clean_values(payload.get("theta_feasible", []), max_values=max_values_per_file)
    infeasible = clean_values(payload.get("theta_infeasible", []), max_values=max_values_per_file)

    return feasible, infeasible


def load_theta_values(
    root_dir: Path,
    datasets: Sequence[Mapping[str, str]],
    methods: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    max_values_per_file = config.get("max_values_per_file")
    if max_values_per_file is not None:
        max_values_per_file = int(max_values_per_file)

    out: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for dataset in datasets:
        dataset_id = str(dataset["id"])
        out.setdefault(dataset_id, {})

        for method in methods:
            method_id = str(method["id"])
            files = discover_theta_files(root_dir, dataset, method, config)

            feasible_all: List[float] = []
            infeasible_all: List[float] = []

            for path in files:
                feasible, infeasible = load_theta_values_from_file(
                    path,
                    max_values_per_file=max_values_per_file,
                )
                feasible_all.extend(feasible)
                infeasible_all.extend(infeasible)

            out[dataset_id][method_id] = {
                "files": [str(p) for p in files],
                "theta_feasible": feasible_all,
                "theta_infeasible": infeasible_all,
            }

    return out


def dataset_value_range(
    by_method: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Tuple[float, float]:
    values: List[float] = []
    for payload in by_method.values():
        values.extend(payload.get("theta_feasible", []))
        values.extend(payload.get("theta_infeasible", []))

    if not values:
        return (
            float(config.get("empty_xmin", -1.0)),
            float(config.get("empty_xmax", 1.0)),
        )

    lo = float(np.min(values))
    hi = float(np.max(values))

    if lo == hi:
        lo -= 1.0
        hi += 1.0

    pad_frac = float(config.get("xpad_frac", 0.03))
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad

def histogram_coordinates(
    values: Sequence[float],
    *,
    bins: np.ndarray,
    density: bool,
) -> List[Tuple[float, float]]:
    if not values:
        return []

    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=bins, density=density)

    # Step histogram coordinates. We use const plot with left bin edges,
    # and append the last edge with zero height to close the curve.
    coords: List[Tuple[float, float]] = []
    for i, c in enumerate(counts):
        coords.append((float(edges[i]), float(c)))
    coords.append((float(edges[-1]), 0.0))
    return coords


def coordinates_to_tex(coords: Sequence[Tuple[float, float]]) -> str:
    return " ".join(f"({x:.12g},{y:.12g})" for x, y in coords)


def compute_all_histograms(
    data: Mapping[str, Mapping[str, Mapping[str, Any]]],
    datasets: Sequence[Mapping[str, str]],
    methods: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> Tuple[
    Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]],
    Dict[str, Dict[str, float]],
]:
    density = bool(config.get("density", True))

    hists: Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]] = {}
    limits: Dict[str, Dict[str, float]] = {}

    for dataset in datasets:
        dataset_id = str(dataset["id"])
        by_method = data.get(dataset_id, {})

        xmin, xmax = dataset_value_range(by_method, config)

        if "bin_width" in config:
            bin_width = float(config["bin_width"])
            bins = np.arange(xmin, xmax + bin_width, bin_width)
            if len(bins) < 2:
                bins = np.linspace(xmin, xmax, 2)
        else:
            nbins = int(config.get("bins", 50))
            bins = np.linspace(xmin, xmax, nbins + 1)

        hists.setdefault(dataset_id, {})
        dataset_ymax = 0.0

        for method in methods:
            method_id = str(method["id"])
            payload = by_method.get(method_id, {})

            feas = payload.get("theta_feasible", [])
            infeas = payload.get("theta_infeasible", [])

            feas_coords = histogram_coordinates(feas, bins=bins, density=density)
            infeas_coords = histogram_coordinates(infeas, bins=bins, density=density)

            for _x, y in feas_coords + infeas_coords:
                dataset_ymax = max(dataset_ymax, float(y))

            hists[dataset_id][method_id] = {
                "feasible": feas_coords,
                "infeasible": infeas_coords,
            }

        if dataset_ymax <= 0.0:
            dataset_ymax = float(config.get("empty_ymax", 1.0))
        else:
            dataset_ymax *= float(config.get("ypad_factor", 1.15))

        limits[dataset_id] = {
            "xmin": xmin,
            "xmax": xmax,
            "ymax": float(dataset_ymax),
        }

    return hists, limits

def axis_options(
    *,
    config: Mapping[str, Any],
    title: str,
    xmin: float,
    xmax: float,
    ymax: float,
    show_ylabel: bool,
    show_xlabel: bool,
    show_ytick_labels: bool,
    show_xtick_labels: bool,
) -> List[str]:
    width = str(config.get("panel_width", r"0.44\textwidth"))
    height = str(config.get("panel_height", r"0.28\textwidth"))
    xlabel = str(config.get("xlabel", r"Predicted $\theta$"))
    ylabel = str(config.get("ylabel", r"Density" if bool(config.get("density", True)) else r"Count"))

    lines = [
        rf"    width={width},",
        rf"    height={height},",
        r"    scale only axis,",
        r"    tick align=outside,",
        r"    tick pos=left,",
        r"    xmajorgrids,",
        r"    ymajorgrids,",
        r"    x grid style={darkgray176},",
        r"    y grid style={darkgray176},",
        rf"    xmin={xmin:.12g}, xmax={xmax:.12g},",
        rf"    ymin=0, ymax={ymax:.12g},",
        r"    xtick style={color=black},",
        r"    ytick style={color=black},",
        rf"    title={{{title}}},",
    ]

    if show_xlabel:
        lines.append(rf"    xlabel={{{xlabel}}},")
    if show_ylabel:
        lines.append(rf"    ylabel={{{ylabel}}},")
    if not show_xtick_labels:
        lines.append(r"    xticklabels=\empty,")
    if not show_ytick_labels:
        lines.append(r"    yticklabels=\empty,")
        lines.append(r"    ytick style={draw=none},")

    return lines


def emit_histogram_panel(
    *,
    dataset_id: str,
    methods: Sequence[Mapping[str, str]],
    hists: Mapping[str, Mapping[str, Mapping[str, List[Tuple[float, float]]]]],
    data: Mapping[str, Mapping[str, Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> List[str]:
    lines: List[str] = []

    feasible_line_width = str(config.get("feasible_line_width", "very thick"))
    infeasible_line_width = str(config.get("infeasible_line_width", "very thick"))
    feasible_style = str(config.get("feasible_style", "solid"))
    infeasible_style = str(config.get("infeasible_style", "densely dashed"))

    for method in methods:
        method_id = str(method["id"])
        st = method_style(method_id, config)
        color = st.get("color", "black")

        payload = data.get(dataset_id, {}).get(method_id, {})
        n_feas = len(payload.get("theta_feasible", []))
        n_infeas = len(payload.get("theta_infeasible", []))
        files = payload.get("files", [])

        lines.append(rf"% {dataset_id} / {method_id}: {len(files)} file(s), feasible={n_feas}, infeasible={n_infeas}")

        feas_coords = hists.get(dataset_id, {}).get(method_id, {}).get("feasible", [])
        infeas_coords = hists.get(dataset_id, {}).get(method_id, {}).get("infeasible", [])

        fill_opacity = str(config.get("fill_opacity", "0.5"))
        shade_under_curves = bool(config.get("shade_under_curves", True))
        
        if feas_coords:
            if shade_under_curves:
                lines.append(
                    rf"\addplot+[const plot, mark=none, {feasible_line_width}, {feasible_style}, "
                    rf"color={color}, fill={color}, fill opacity={fill_opacity}] "
                    rf"coordinates {{{coordinates_to_tex(feas_coords)}}} \closedcycle;"
                )
            else:
                lines.append(
                    rf"\addplot+[const plot, mark=none, {feasible_line_width}, {feasible_style}, color={color}] "
                    rf"coordinates {{{coordinates_to_tex(feas_coords)}}};"
                )
        
        if infeas_coords:
            if shade_under_curves:
                lines.append(
                    rf"\addplot+[const plot, mark=none, {infeasible_line_width}, {infeasible_style}, "
                    rf"color={color}, fill={color}, fill opacity={fill_opacity}] "
                    rf"coordinates {{{coordinates_to_tex(infeas_coords)}}} \closedcycle;"
                )
            else:
                lines.append(
                    rf"\addplot+[const plot, mark=none, {infeasible_line_width}, {infeasible_style}, color={color}] "
                    rf"coordinates {{{coordinates_to_tex(infeas_coords)}}};"
                )
        lines.append("")

    return lines


def emit_manual_legend(
    methods: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> List[str]:
    ncols = int(config.get("legend_columns", 4))
    xshift = str(config.get("legend_xshift", "0cm"))
    yshift = str(config.get("legend_yshift", "-0.35cm"))

    entries: List[str] = []

    for method in methods:
        method_id = str(method["id"])
        st = method_style(method_id, config)
        color = st.get("color", "black")
        title = st.get("title", latex_escape_label(method_id))
        entries.append(
            rf"\tikz[baseline=-0.55ex]{{\draw[very thick, color={color}] (0,0) -- (0.48,0);}} & {title}"
        )

    entries.append(
        r"\tikz[baseline=-0.55ex]{\draw[very thick, solid, color=black] (0,0) -- (0.48,0);} & feasible"
    )
    entries.append(
        r"\tikz[baseline=-0.55ex]{\draw[very thick, densely dashed, color=black] (0,0) -- (0.48,0);} & infeasible"
    )

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

def emit_dataset_grid_layout(
    *,
    datasets: Sequence[Mapping[str, str]],
    methods: Sequence[Mapping[str, str]],
    data: Mapping[str, Mapping[str, Mapping[str, Any]]],
    hists: Mapping[str, Mapping[str, Mapping[str, List[Tuple[float, float]]]]],
    limits: Mapping[str, Mapping[str, float]],
    config: Mapping[str, Any],
) -> List[str]:
    ncols = int(config.get("group_cols", 2))
    nrows = int(math.ceil(len(datasets) / ncols))
    group_size = str(config.get("group_size", f"{ncols} by {nrows}"))
    hsep = str(config.get("horizontal_sep", "0.85cm"))
    vsep = str(config.get("vertical_sep", "0.85cm"))

    show_xlabel_only_bottom = bool(config.get("show_xlabel_only_bottom", False))
    show_ylabel_only_left = bool(config.get("show_ylabel_only_left", False))

    lines: List[str] = []
    lines.append(r"\begin{groupplot}[")
    lines.append(rf"    group style={{group size={group_size}, horizontal sep={hsep}, vertical sep={vsep}}},")
    lines.append(r"]")
    lines.append("")

    for idx, dataset in enumerate(datasets):
        dataset_id = str(dataset["id"])
        ds = dataset_style(dataset_id, config)
        title = ds.get("title", latex_escape_label(dataset_id))

        panel_limits = limits.get(dataset_id, {})
        xmin = float(panel_limits.get("xmin", config.get("empty_xmin", -1.0)))
        xmax = float(panel_limits.get("xmax", config.get("empty_xmax", 1.0)))
        ymax = float(panel_limits.get("ymax", config.get("empty_ymax", 1.0)))


        row = idx // ncols
        col = idx % ncols
        bottom = row == (nrows - 1)
        left = col == 0

        show_xlabel = (not show_xlabel_only_bottom) or bottom
        show_ylabel = (not show_ylabel_only_left) or left
        show_xtick_labels = show_xlabel or bool(config.get("show_xtick_labels_on_non_bottom", False))
        show_ytick_labels = show_ylabel or bool(config.get("show_ytick_labels_on_non_left", False))

        lines.append(r"\nextgroupplot[")
        lines.extend(
            axis_options(
                config=config,
                title=title,
                xmin=xmin,
                xmax=xmax,
                ymax=ymax,
                show_ylabel=show_ylabel,
                show_xlabel=show_xlabel,
                show_ytick_labels=show_ytick_labels,
                show_xtick_labels=show_xtick_labels,
            )
        )
        lines.append(r"]")
        lines.extend(
            emit_histogram_panel(
                dataset_id=dataset_id,
                methods=methods,
                hists=hists,
                data=data,
                config=config,
            )
        )
        lines.append("")

    lines.append(r"\end{groupplot}")
    lines.extend(emit_manual_legend(methods, config))
    return lines


def generate_latex(config: Dict[str, Any]) -> str:
    root_dir = Path(config.get("root_dir", ".")).expanduser()
    datasets = configured_datasets(config, root_dir)
    methods = configured_methods(config)

    if len(datasets) == 0:
        raise ValueError("At least one dataset must be configured.")
    
    if len(datasets) > 4:
        raise ValueError(
            f"This figure supports at most 4 datasets, got {len(datasets)}."
        )

    data = load_theta_values(root_dir, datasets, methods, config)
    hists, limits = compute_all_histograms(data, datasets, methods, config)

    label = str(config.get("label", "fig:inference_theta_distribution"))
    caption = config.get("caption")
    placement = str(config.get("placement", "t"))

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
    lines.append(r"% Method colors are assumed to be defined outside this file.")
    lines.append("")
    lines.append(r"\begin{tikzpicture}")

    lines.extend(
        emit_dataset_grid_layout(
            datasets=datasets,
            methods=methods,
            data=data,
            hists=hists,
            limits=limits,
            config=config,
        )
    )
    lines.append(r"\end{tikzpicture}")

    if caption:
        lines.append("")
        lines.append(rf"\caption{{{caption}}}")

    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{figure}")

    return "\n".join(lines)


def example_config() -> Dict[str, Any]:
    return {
        "root_dir": "../runs/inference_theta_distribution",
        "label": "fig:inference_theta_distribution",
        "caption": (
            "Distribution of predicted local interaction parameters. "
            "For each dataset and model class, predicted non-unary factor potentials "
            "are grouped according to whether the corresponding local assignment is "
            "feasible or infeasible."
        ),
        "datasets": [
            {
                "id": "mis_n100_p0.15",
                "path": "mis_n100_p0.15",
                "title": "MIS-S",
            },
            {
                "id": "mis_n200-300_p0.15",
                "path": "mis_n200-300_p0.15",
                "title": "MIS-L",
            },
            {
                "id": "scp_n50_m50_k5",
                "path": "scp_n50_m50_k5",
                "title": "SC-S",
            },
            {
                "id": "scp_n120-180_m150-250_k5",
                "path": "scp_n120-180_m150-250_k5",
                "title": "SC-L",
            },
        ],
        "methods": [
            {
                "id": "lbp",
                "path": "lbp",
                "title": "LBP",
                "color": "LBPGDColor",
            },
            {
                "id": "nmf",
                "path": "nmf",
                "title": "NMF",
                "color": "NMFGDColor",
            },
        ],
        "theta_filename": "theta_distribution.json",
        "bins": 60,
        "density": True,
        "panel_width": "0.43\\textwidth",
        "panel_height": "0.25\\textwidth",
        "horizontal_sep": "0.85cm",
        "vertical_sep": "0.85cm",
        "xlabel": r"Predicted $\theta$",
        "ylabel": "Density",
        "show_xlabel_only_bottom": True,
        "show_ylabel_only_left": True,
        "show_xtick_labels_on_non_bottom": False,
        "show_ytick_labels_on_non_left": False,
        "legend_columns": 4,
        "legend_xshift": "0cm",
        "legend_yshift": "-0.35cm",
        "feasible_style": "solid",
        "infeasible_style": "densely dashed",
        "max_values_per_file": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a PGFPlots/TikZ 2x2 grid of predicted theta histograms "
            "grouped by local feasibility."
        )
    )
    parser.add_argument("--config", required=False, help="JSON config describing the figure.")
    parser.add_argument("--output", required=False, help="Output .tex file. Prints to stdout if omitted.")
    parser.add_argument("--root-dir", default=None, help="Override root_dir from the config.")
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
