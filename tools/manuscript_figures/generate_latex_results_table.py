#!/usr/bin/env python3
from __future__ import annotations
import math
import argparse
import glob
import json
import math
import re
import statistics as stats
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import statistics as stats
from typing import Mapping, Optional

Number = Union[int, float]

DEFAULT_COLUMNS = [
    {"key": "gnn", "title": "GNN"},
    {"key": "lbp", "title": "LBP"},
    {"key": "lbp_gd", "title": r"LBP$+$GD"},
    {"key": "nmf", "title": "NMF"},
    {"key": "nmf_gd", "title": r"NMF$+$GD"},
]

MRF_NAIR_COLUMNS = [
    {"title": r"Dataset", "source": "dataset_id", "align": "l"},
    {"title": r"Model", "source": "model_architecture", "align": "l"},
    {"title": r"EGG gap", "method": "egg", "metric": "relative_gap", "stat": "mean", "format": "number"},
    {"title": r"Best EGG gap", "method": "egg", "metric": "relative_gap", "stat": "best", "format": "number"},
    {"title": r"EGG time", "method": "egg", "metric": "runtime", "stat": "mean", "format": "number"},
    {"title": r"GD gap", "method": "gd", "metric": "relative_gap", "stat": "mean", "format": "number"},
    {"title": r"Best GD gap", "method": "gd", "metric": "relative_gap", "stat": "best", "format": "number"},
    {"title": r"GD time", "method": "gd", "metric": "runtime", "stat": "mean", "format": "number"},
]

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

LOWER_IS_BETTER = {"gap", "relative_gap", "runtime", "time", "loss", "error"}
HIGHER_IS_BETTER = {"score", "objective", "value", "accuracy"}


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

def latex_escape_text(s: str) -> str:
    repl = {
        "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "$": r"\$",
    }
    return "".join(repl.get(ch, ch) for ch in str(s))

def summary_mean_std_from_payload(
    payload: Mapping[str, Any],
    metric: str,
    method: Optional[str],
) -> Optional[Tuple[float, float]]:
    method = normalize_method(method)
    metric = normalize_metric(metric)

    if method is None:
        return None

    methods = (
        payload.get("summary", {})
        .get("methods", {})
    )

    method_summary = methods.get(method)
    if not isinstance(method_summary, Mapping):
        return None

    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"

    mean = method_summary.get(mean_key)
    std = method_summary.get(std_key)

    if mean is None:
        return None

    if std is None:
        std = 0.0

    return float(mean), float(std)


def metric_names(metric: str) -> List[str]:
    names = [metric]
    names.extend(METRIC_ALIASES.get(metric, []))
    out: List[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def first_number_for_names(d: Mapping[str, Any], names: Iterable[str]) -> Optional[float]:
    lower_to_key = {str(k).lower(): k for k in d.keys()}
    for name in names:
        key = lower_to_key.get(name.lower())
        if key is not None and is_number(d[key]):
            return float(d[key])
    return None


def summary_mean_std(payload: Mapping[str, Any], metric: str) -> Optional[Tuple[float, float]]:
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        return None

    names = metric_names(metric)
    mean_keys: List[str] = []
    std_keys: List[str] = []
    for n in names:
        mean_keys.extend([f"{n}_mean", f"mean_{n}", f"avg_{n}", f"{n}_avg"])
        std_keys.extend([f"{n}_std", f"std_{n}", f"{n}_stdev", f"stdev_{n}"])

    mean = first_number_for_names(summary, mean_keys)
    std = first_number_for_names(summary, std_keys)
    if mean is not None:
        return mean, (0.0 if std is None else std)
    return None


def final_log_entry(log: Any) -> Any:
    if isinstance(log, Mapping):
        info = log.get("info")
        if isinstance(info, list) and info:
            return info[-1]
        return log
    return log


def _method_object(instance: Mapping[str, Any], method: Optional[str]) -> Any:
    if not method:
        return None
    for key in (method, method.lower(), method.upper()):
        if key in instance:
            return instance[key]
    return None


def extract_instance_value(instance: Mapping[str, Any], metric: str, method: Optional[str]) -> Optional[float]:
    names = metric_names(metric)

    method_obj = _method_object(instance, method)
    if method_obj is not None:
        # Runtime is often stored on the method log itself; final score/gap is usually in info[-1].
        if isinstance(method_obj, Mapping):
            val = first_number_for_names(method_obj, names)
            if val is not None:
                return val
        final = final_log_entry(method_obj)
        if isinstance(final, Mapping):
            val = first_number_for_names(final, names)
            if val is not None:
                return val

    val = first_number_for_names(instance, names)
    if val is not None:
        return val

    for key in ("metrics", "final", "result", "stats", "summary"):
        obj = instance.get(key)
        if isinstance(obj, Mapping):
            val = first_number_for_names(obj, names)
            if val is not None:
                return val
    return None


def collect_values(payload: Mapping[str, Any], metric: str, method: Optional[str]) -> List[float]:
    instances = payload.get("instances", [])
    if not isinstance(instances, list):
        return []
    values: List[float] = []
    for inst in instances:
        if not isinstance(inst, Mapping):
            continue
        val = extract_instance_value(inst, metric=metric, method=method)
        if val is not None:
            values.append(float(val))
    return values


def mean_std_from_payload(payload: Mapping[str, Any], metric: str, method: Optional[str]) -> Optional[Tuple[float, float]]:
    if method is None:
        summarized = summary_mean_std(payload, metric)
        if summarized is not None:
            return summarized

    values = collect_values(payload, metric=metric, method=method)
    if not values:
        return None
    mean = stats.fmean(values)
    std = stats.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def mean_from_payload(payload: Mapping[str, Any], metric: str, method: Optional[str]) -> Optional[float]:
    ms = mean_std_from_payload(payload, metric=metric, method=method)
    return None if ms is None else ms[0]


def parse_cell_spec(spec: Any) -> Tuple[Optional[str], Optional[str]]:
    if spec is None or spec == "":
        return None, None
    if isinstance(spec, str):
        return spec, None
    if isinstance(spec, Mapping):
        return spec.get("path") or spec.get("dir") or spec.get("json"), spec.get("method")
    raise TypeError(f"Invalid cell spec: {spec!r}")


def format_mean_std(mean_std: Optional[Tuple[float, float]], decimals: int) -> str:
    if mean_std is None:
        return ""
    mean, std = mean_std
    return f"{mean:.{decimals}f} ({std:.{decimals}f})"


def format_number(value: Optional[float], decimals: int) -> str:
    if value is None:
        return ""
    return f"{value:.{decimals}f}"


def direction_for_metric(metric: str, config: Mapping[str, Any], row: Optional[Mapping[str, Any]] = None, col: Optional[Mapping[str, Any]] = None) -> str:
    for obj in (col, row, config):
        if isinstance(obj, Mapping) and "maximize" in obj:
            return "max" if bool(obj["maximize"]) else "min"
    m = metric.lower()
    if m in HIGHER_IS_BETTER:
        return "max"
    return "min"


def best_value(values: Sequence[float], metric: str, config: Mapping[str, Any], row: Optional[Mapping[str, Any]] = None, col: Optional[Mapping[str, Any]] = None) -> Optional[float]:
    clean = [float(v) for v in values if is_number(v)]
    if not clean:
        return None
    return max(clean) if direction_for_metric(metric, config, row, col) == "max" else min(clean)


def _glob_existing(pattern: str) -> List[Path]:
    return [Path(p) for p in sorted(glob.glob(str(Path(pattern).expanduser())))]


def seed_json_paths(row: Mapping[str, Any], config: Mapping[str, Any]) -> List[Path]:
    if "json_files" in row:
        return [resolve_under_root(p, config) for p in row["json_files"]]
    if "files" in row:
        return [resolve_under_root(p, config) for p in row["files"]]
    
    if "runs" in row:
        return [resolve_under_root(run, config) for run in row["runs"]]
    if "glob" in row:
        glob_pattern = resolve_under_root(str(row["glob"]), config)
        paths = _glob_existing(str(glob_pattern))
        if paths:
            return paths

    prefix = row.get("run_prefix") or row.get("prefix")
    if not prefix:
        raise KeyError(
            "Seed-aggregate rows need one of: json_files, files, runs, glob, run_prefix, prefix"
        )

    prefix = str(resolve_under_root(str(prefix), config))
    seeds = row.get("seeds", config.get("seeds"))
    seed_pattern = row.get("seed_pattern", config.get("seed_pattern", "{prefix}_seed{seed}/eval_test.json"))

    if seeds is not None:
        paths = []
        for seed in seeds:
            paths.append(Path(seed_pattern.format(prefix=prefix, seed=seed)).expanduser())
        return paths

    # No explicit seeds: discover all matching seed folders.
    discovered: List[Path] = []
    for name in ("eval_test.json", "test_eval.json"):
        discovered.extend(_glob_existing(f"{prefix}_seed*/{name}"))
    return sorted(set(discovered))


def common_seed_prefix(paths: Sequence[Path]) -> str:
    if not paths:
        return ""
    names = []
    for p in paths:
        parent = p.parent.name if p.suffix == ".json" else p.name
        names.append(re.sub(r"_seed[^_/]+$", "", parent))
    if not names:
        return ""
    return names[0]


def derive_fields_from_prefix(prefix: str, config: Mapping[str, Any], row: Mapping[str, Any]) -> Dict[str, str]:
    base = Path(prefix).name
    fields: Dict[str, str] = {}

    regex = row.get("prefix_regex") or config.get("prefix_regex")
    if regex:
        m = re.search(str(regex), base)
        if m:
            fields.update({k: v for k, v in m.groupdict().items() if v is not None})

    if "dataset_id" not in fields or "model_architecture" not in fields:
        model_tokens = row.get("model_tokens", config.get("model_tokens", ["gnn", "lbp", "nmf"]))
        parts = base.split("_")
        split_idx = None
        for i, token in enumerate(parts):
            if token.lower() in {str(t).lower() for t in model_tokens}:
                split_idx = i
                break
        if split_idx is not None:
            fields.setdefault("dataset_id", "_".join(parts[:split_idx]))
            fields.setdefault("model_architecture", "_".join(parts[split_idx:]))
        else:
            fields.setdefault("dataset_id", base)
            fields.setdefault("model_architecture", "")

    # Explicit row fields always win.
    for key in ("dataset_id", "model_architecture"):
        if key in row:
            fields[key] = str(row[key])

    return fields


def build_seed_context(row: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    paths = seed_json_paths(row, config)
    existing = [p for p in paths if p.exists() or p.is_dir()]
    print(existing)

    # Keep the row in the table even when no seed files have been produced yet.
    # This is useful while experiments are still running: metadata columns are
    # filled from the row/prefix, and numeric aggregate columns render empty.
    prefix = str(row.get("run_prefix") or row.get("prefix") or "")
    if not prefix and existing:
        prefix = common_seed_prefix(existing)

    fields = derive_fields_from_prefix(prefix, config, row) if prefix else {
        "dataset_id": str(row.get("dataset_id", "")),
        "model_architecture": str(row.get("model_architecture", "")),
    }

    payloads = [load_json(p) for p in existing]
    return {
        "paths": existing,
        "expected_paths": paths,
        "missing_paths": [p for p in paths if p not in existing],
        "payloads": payloads,
        "prefix": prefix,
        "missing": len(existing) == 0,
        **fields,
    }

def seed_stat(
    ctx: Mapping[str, Any],
    col: Mapping[str, Any],
    config: Mapping[str, Any],
    row: Mapping[str, Any],
) -> Optional[float]:
    payloads = ctx["payloads"]
    method = col.get("method")
    metric = col.get("metric", config.get("metric", "gap"))
    metric = normalize_metric(str(metric))

    per_seed = [
        summary_mean_std_from_payload(payload, metric=metric, method=method)
        for payload in payloads
    ]
    per_seed = [x for x in per_seed if x is not None]

    if not per_seed:
        return None

    # Use the per-seed *_mean values for scalar stats.
    values = [x[0] for x in per_seed]

    stat = str(col.get("stat", "mean")).lower()

    if stat in {"mean", "avg", "average"}:
        value = stats.fmean(values)
    elif stat == "best":
        maximize = bool(col.get("maximize", row.get("maximize", config.get("maximize", False))))
        value = max(values) if maximize else min(values)
    elif stat == "min":
        value = min(values)
    elif stat == "max":
        value = max(values)
    elif stat in {"std", "stdev"}:
        value = stats.pstdev(values) if len(values) > 1 else 0.0
    else:
        raise ValueError(f"Unknown seed aggregate stat: {stat!r}")

    #scale = metric_scale(metric, col, config)
    return value 

def metric_scale(metric: str, col: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    metric = normalize_metric(metric)

    if "scale" in col:
        return float(col["scale"])

    metric_scales = config.get("metric_scales", {})
    if metric in metric_scales:
        return float(metric_scales[metric])

    # Default: gaps are shown as percentages.
    if metric == "gap":
        return 100.0

    return 1.0

def seed_mean_std(
    ctx: Mapping[str, Any],
    col: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Optional[Tuple[float, float]]:
    payloads = ctx["payloads"]
    method = col.get("method")
    metric = col.get("metric", config.get("metric", "gap"))
    metric = normalize_metric(metric)

    per_seed = [
        summary_mean_std_from_payload(payload, metric=metric, method=method)
        for payload in payloads
    ]
    per_seed = [x for x in per_seed if x is not None]

    if not per_seed:
        return None

    means = [x[0] for x in per_seed]
    stds = [x[1] for x in per_seed]

    # Mean of the per-seed means.
    mean = stats.fmean(means)

    # Std is computed from the *_std fields, not from the seed means.
    #
    # This is a pooled within-seed std. Since each seed usually has the same
    # number of evaluated instances, this is the right default for reporting
    # instance-level variability aggregated over seeds.
    std = math.sqrt(stats.fmean([s * s for s in stds]))

    scale = metric_scale(metric, col, config)
    return mean * scale, std * scale
def cell_is_suppressed(
    ctx: Mapping[str, Any],
    row: Mapping[str, Any],
    col: Mapping[str, Any],
) -> bool:
    suppress_if = col.get("suppress_if")
    if not isinstance(suppress_if, Mapping):
        return False

    for key, expected in suppress_if.items():
        actual = ctx.get(str(key), row.get(str(key)))

        if isinstance(expected, list):
            if actual in expected:
                return True
        else:
            if actual == expected:
                return True

    return False
def render_seed_aggregate_cell(
    ctx: Mapping[str, Any],
    col: Mapping[str, Any],
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    decimals: int,
) -> str:
    if cell_is_suppressed(ctx, row, col):
        return str(col.get("suppressed_value", config.get("suppressed_value", r"\multicolumn{1}{c}{--}")))
    source = col.get("source")
    if source:
        val = ctx.get(str(source), row.get(str(source), ""))
        return str(val) if col.get("latex", False) else latex_escape_text(str(val))

    cell_decimals = int(col.get("decimals", decimals))

    # Default behavior: plain scalar number.
    # If cell_format is "mean_std", render as: value (std)
    cell_format = str(
        col.get(
            "cell_format",
            config.get("numeric_cell_format", "number"),
        )
    )

    if cell_format in {"mean_std", "value_std"}:
        return format_mean_std(
            seed_mean_std(ctx, col, config),
            cell_decimals,
        )

    value = seed_stat(ctx, col, config, row)
    
    metric = col.get("metric", config.get("metric", "gap"))
    scale = metric_scale(str(metric), col, config)
    
    if value is not None:
        value = value * scale
    
    return format_number(value, cell_decimals)

def build_seed_aggregate_table(config: Dict[str, Any], decimals: int) -> str:
    columns = config.get("columns", MRF_NAIR_COLUMNS)
    rows = config["rows"]
    label = config.get("label", "tab:mrf_nair_inference_results")
    caption = config.get("caption")
    placement = config.get("placement", "htbp")
    table_format = config.get("table_format", f"2.{decimals}")

    mean_std_table_format = config.get(
        "mean_std_table_format",
        f"2.{decimals}(2.{decimals})",
    )

    col_specs: List[str] = []
    for col in columns:
        align = col.get("align")
        if align:
            col_specs.append(str(align))
        elif col.get("format", "number") == "text" or col.get("source"):
            col_specs.append("l")
        else:
            cell_format = str(
                col.get(
                    "cell_format",
                    config.get("numeric_cell_format", "number"),
                )
            )

            default_col_format = (
                mean_std_table_format
                if cell_format in {"mean_std", "value_std"}
                else table_format
            )

            col_specs.append(
                f"S[table-format={col.get('table_format', default_col_format)}]"
            )

    colspec = "\n                    ".join(col_specs)
    header = " & ".join("{" + str(c["title"]) + "}" for c in columns) + r" \\" 

    body: List[str] = []
    
    group_key = config.get("group_key", "dataset_id")
    multirow_group_col = config.get("multirow_group_col", "dataset_id")
    group_rule = config.get("group_rule", r"\midrule")
    
    # First pass: build contexts once and compute group sizes.
    row_items = []
    for row in rows:
        ctx = build_seed_context(row, config)
        group_value = ctx.get(group_key, row.get(group_key, ""))
        row_items.append((row, ctx, group_value))
    
    group_sizes: Dict[Any, int] = {}
    for _row, _ctx, group_value in row_items:
        group_sizes[group_value] = group_sizes.get(group_value, 0) + 1
    
    seen_in_group: Dict[Any, int] = {}
    
    for row_idx, (row, ctx, group_value) in enumerate(row_items):
        group_pos = seen_in_group.get(group_value, 0)
        seen_in_group[group_value] = group_pos + 1
    
        cells = []
        for col in columns:
            source = col.get("source")
    
            if source == multirow_group_col:
                if group_pos == 0:
                    raw_val = ctx.get(str(source), row.get(str(source), ""))
                    text = str(raw_val) if col.get("latex", False) else latex_escape_text(str(raw_val))
                    cells.append(rf"\multirow{{{group_sizes[group_value]}}}{{*}}{{{text}}}")
                else:
                    cells.append("")
            else:
                cells.append(render_seed_aggregate_cell(ctx, col, config, row, decimals))
    
        body.append("        " + " & ".join(cells) + r" \\")
    
        # Add a horizontal rule after each dataset group, but not after the final group.
        is_last_row = row_idx == len(row_items) - 1
        is_last_in_group = seen_in_group[group_value] == group_sizes[group_value]
    
        if is_last_in_group and not is_last_row:
            body.append(f"        {group_rule}")
    out: List[str] = []
    out.append(rf"\begin{{table}}[{placement}]")
    out.append(r"\centering")
    if caption:
        out.append(rf"\caption{{{caption}}}")
    if config.get("resizebox", True):
        out.append(r"\resizebox{\textwidth}{!}{")
    out.append(rf"\begin{{tabular}}{{{colspec}}}")
    out.append(r"        \toprule")
    out.append("        " + header)
    out.append(r"        \midrule")
    out.extend(body)
    out.append(r"        \bottomrule")
    out.append(r"    \end{tabular}" + ("}" if config.get("resizebox", True) else ""))
    out.append(rf"\label{{{label}}}")
    out.append(r"\end{table}")
    return "\n".join(out)


def build_classic_table(config: Dict[str, Any], metric: str, decimals: int, bold_best: bool) -> str:
    columns = config.get("columns", DEFAULT_COLUMNS)
    rows = config["rows"]
    label = config.get("label", "tab:test_gap_results")
    caption = config.get("caption")
    placement = config.get("placement", "htbp")
    table_format = config.get("table_format", f"2.{decimals}({decimals}.{decimals})")

    colspec = "l\n" + "\n".join(
        f"                    S[table-format={table_format}]" for _ in columns
    )

    header = "Dataset & " + " & ".join("{" + c["title"] + "}" for c in columns) + r" \\" 
    lines: List[str] = []

    for row in rows:
        row_label = row["label"]
        results = row.get("results", {})
        cell_values: List[Optional[Tuple[float, float]]] = []

        for col in columns:
            path, method = parse_cell_spec(results.get(col["key"]))
            if path is None:
                cell_values.append(None)
                continue
            payload = load_json(path)
            cell_values.append(mean_std_from_payload(payload, metric=metric, method=method))

        best_idx: Optional[int] = None
        numeric_means = [(i, v[0]) for i, v in enumerate(cell_values) if v is not None]
        if bold_best and numeric_means:
            maximize = bool(row.get("maximize", config.get("maximize", metric in HIGHER_IS_BETTER)))
            best_idx = max(numeric_means, key=lambda x: x[1])[0] if maximize else min(numeric_means, key=lambda x: x[1])[0]

        cells: List[str] = []
        for i, val in enumerate(cell_values):
            text = format_mean_std(val, decimals)
            if best_idx == i and text:
                text = r"{\bfseries " + text + "}"
            cells.append(text)
        lines.append(f"        {row_label} & " + " & ".join(cells) + " \\\\")

    out: List[str] = []
    out.append(rf"\begin{{table}}[{placement}]")
    out.append(r"\centering")
    if caption:
        out.append(rf"\caption{{{caption}}}")
    out.append(r"\resizebox{\textwidth}{!}{")
    out.append(r"% S columns align numbers on the decimal and on the uncertainty")
    out.append(rf"\begin{{tabular}}{{{colspec}}}")
    out.append(r"        \toprule")
    out.append("        " + header)
    out.append(r"        \midrule")
    out.extend(lines)
    out.append(r"        \bottomrule")
    out.append(r"    \end{tabular}}")
    out.append(rf"\label{{{label}}}")
    out.append(r"\end{table}")
    return "\n".join(out)


def build_table(config: Dict[str, Any], metric: str, decimals: int, bold_best: bool) -> str:
    table_type = config.get("table_type", config.get("kind", "classic"))
    if table_type in {"seed_aggregate", "seed-aggregate", "seeds"}:
        return build_seed_aggregate_table(config, decimals=decimals)
    return build_classic_table(config, metric=metric, decimals=decimals, bold_best=bold_best)


def example_config() -> Dict[str, Any]:
    return {
        "label": "tab:lbp_nmf_bpgd_test_gap_results",
        "caption": "Test gaps on SC and MIS datasets.",
        "columns": DEFAULT_COLUMNS,
        "rows": [
            {
                "label": r"$SC\,(n=50, m=50, k=5)$",
                "results": {
                    "gnn": "runs/scp_small_gnn/test_eval.json",
                    "lbp": "runs/scp_small_lbp/test_eval.json",
                    "lbp_gd": {"path": "runs/scp_small_lbp_gd/test_eval.json", "method": "gd"},
                    "nmf": "runs/scp_small_nmf/test_eval.json",
                    "nmf_gd": {"path": "runs/scp_small_nmf_gd/test_eval.json", "method": "gd"},
                },
            }
        ],
    }


def mrf_nair_example_config() -> Dict[str, Any]:
    return {
        "table_type": "seed_aggregate",
        "label": "tab:mrf_nair_inference_results",
        "caption": "MRF inference results on NAIR instances.",
        "seeds": [0, 1, 2, 3, 4],
        "seed_pattern": "{prefix}_seed{seed}/eval_test.json",
        "model_tokens": ["gnn", "lbp", "nmf"],
        "columns": MRF_NAIR_COLUMNS,
        "rows": [
            {
                "run_prefix": "runs/scp_n50_m50_k5_lbp_vce_pairwise_approx_max_transformer",
                "dataset_id": "scp_n50_m50_k5",
                "model_architecture": "lbp_vce_pairwise_approx_max_transformer"
            },
            {
                "run_prefix": "runs/mis_n100_p0.15_lbp_cc_pairwise",
                "dataset_id": "mis_n100_p0.15",
                "model_architecture": "lbp_cc_pairwise"
            }
        ]
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generate LaTeX tables from bpropy JSON evaluation files.")
    p.add_argument("--config", required=False, help="JSON config describing rows, columns, and result paths.")
    p.add_argument("--output", required=False, help="Output .tex file. Prints to stdout if omitted.")
    p.add_argument("--metric", default="gap", help="Classic-table metric to extract, e.g. gap, score, runtime.")
    p.add_argument("--decimals", type=int, default=3)
    p.add_argument("--no-bold-best", action="store_true")
    p.add_argument("--write-example-config", help="Write a classic mean/std example config JSON and exit.")
    p.add_argument("--write-mrf-nair-config", help="Write an example config for tab:mrf_nair_inference_results and exit.")
    p.add_argument(
        "--root-dir",
        default=None,
        help="Root directory containing all run directories. Relative run paths in the config are resolved under this root.",
    )
    args = p.parse_args()

    if args.write_example_config:
        out = Path(args.write_example_config)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(example_config(), indent=2), encoding="utf-8")
        print(out)
        return

    if args.write_mrf_nair_config:
        out = Path(args.write_mrf_nair_config)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(mrf_nair_example_config(), indent=2), encoding="utf-8")
        print(out)
        return

    if not args.config:
        raise SystemExit("Either --config, --write-example-config, or --write-mrf-nair-config is required.")

    with Path(args.config).expanduser().open("r", encoding="utf-8") as f:
        config = json.load(f)
    
    if args.root_dir is not None:
        config["root_dir"] = args.root_dir
    
    table = build_table(config, metric=args.metric, decimals=args.decimals, bold_best=not args.no_bold_best)
    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n", encoding="utf-8")
        print(out)
    else:
        print(table)


if __name__ == "__main__":
    main()
