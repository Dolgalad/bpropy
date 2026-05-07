#!/usr/bin/env python3
from pathlib import Path
import re
import shutil
import os
import json
import math
from statistics import mean, stdev


RUNS_DIR = Path(os.environ["WORK"]) / Path("runs")
SEEDS = [0, 1, 2]

# ============================================================
# ONLY THESE EXPERIMENT FAMILIES WILL BE CHECKED
# model = "" means no extra model suffix after the loss
# ============================================================
EXPECTED_RUNS = [
    # ---------- n50 ----------
    ("scp_n50_m50_k5", "var_logits", "bce", ""),
    ("scp_n50_m50_k5", "lbp", "vce", ""),
    ("scp_n50_m50_k5", "lbp", "vce", "3subset_approx_max"),
    ("scp_n50_m50_k5", "lbp", "vce", "3subset_approx_max_attention_v1"),
    ("scp_n50_m50_k5", "lbp", "vce", "3subset_approx_max_attention_v2"),
    ("scp_n50_m50_k5", "lbp", "vce", "3subset_approx_max_attention_varonly"),
    ("scp_n50_m50_k5", "lbp", "vce", "3subset_approx_max_transformer"),
    ("scp_n50_m50_k5", "lbp", "vce", "4subset_approx_max"),
    ("scp_n50_m50_k5", "lbp", "vce", "4subset_approx_max_attention_v1"),
    ("scp_n50_m50_k5", "lbp", "vce", "4subset_approx_max_attention_v2"),
    ("scp_n50_m50_k5", "lbp", "vce", "4subset_approx_max_attention_varonly"),
    ("scp_n50_m50_k5", "lbp", "vce", "4subset_approx_max_transformer"),
    ("scp_n50_m50_k5", "lbp", "vce", "pairwise_approx_max"),
    ("scp_n50_m50_k5", "lbp", "vce", "pairwise_approx_max_attention_v1"),
    ("scp_n50_m50_k5", "lbp", "vce", "pairwise_approx_max_attention_v2"),
    ("scp_n50_m50_k5", "lbp", "vce", "pairwise_approx_max_attention_varonly"),
    ("scp_n50_m50_k5", "lbp", "vce", "pairwise_approx_max_transformer"),
    ("scp_n50_m50_k5", "lbp", "vce", "tree_approx_max"),
    ("scp_n50_m50_k5", "lbp", "vce", "tree_approx_max_attention_v1"),
    ("scp_n50_m50_k5", "lbp", "vce", "tree_approx_max_attention_v2"),
    ("scp_n50_m50_k5", "lbp", "vce", "tree_approx_max_attention_varonly"),
    ("scp_n50_m50_k5", "lbp", "vce", "tree_approx_max_transformer"),

    # ---------- n120-180 ----------
    # ("scp_n120-180_m150-250_k5", "lbp", "vce", ""),
    # ("scp_n120-180_m150-250_k5", "var_logits", "bce", ""),
    # ...

    # ---------- n150-300 ----------
    # ("scp_n150-300_m150-300_k15", "var_logits", "bce", ""),
    # ...
]


# ============================================================
# DISPLAY ORDER / NAMES FOR LATEX TABLES
# Only models listed here are emitted, and in this exact order.
# group_break_after=True inserts \cmidrule(lr){2-8} after the row.
# ============================================================
ROW_SPECS = {
    "scp_n50_m50_k5": [
        {"pred_method": "var_logits", "loss": "bce", "model": "", "label": "Var. Logits", "group_break_after": False},
        {"pred_method": "lbp", "loss": "vce", "model": "", "label": "MRF LBP", "group_break_after": True},

        {"pred_method": "lbp", "loss": "vce", "model": "tree_approx_max", "label": "P.0 MST", "group_break_after": False},
        {"pred_method": "lbp", "loss": "vce", "model": "pairwise_approx_max", "label": "P.0 ($\\alpha' \\leq 2$)", "group_break_after": False},
        {"pred_method": "lbp", "loss": "vce", "model": "3subset_approx_max", "label": "P.0 ($\\alpha' \\leq 3$)", "group_break_after": False},
        {"pred_method": "lbp", "loss": "vce", "model": "4subset_approx_max", "label": "P.0 ($\\alpha' \\leq 4$)", "group_break_after": True},

        {"pred_method": "lbp", "loss": "vce", "model": "tree_approx_max_attention_v1", "label": "P.1 MST", "group_break_after": False},
        {"pred_method": "lbp", "loss": "vce", "model": "pairwise_approx_max_attention_v1", "label": "P.1 ($\\alpha' \\leq 2$)", "group_break_after": False},
        {"pred_method": "lbp", "loss": "vce", "model": "3subset_approx_max_attention_v1", "label": "P.1 ($\\alpha' \\leq 3$)", "group_break_after": False},
        {"pred_method": "lbp", "loss": "vce", "model": "4subset_approx_max_attention_v1", "label": "P.1 ($\\alpha' \\leq 4$)", "group_break_after": True},

        {"pred_method": "lbp", "loss": "vce", "model": "tree_approx_max_attention_varonly", "label": "P.1 VO MST", "group_break_after": False},
        {"pred_method": "lbp", "loss": "vce", "model": "pairwise_approx_max_attention_varonly", "label": "P.1 VO ($\\alpha' \\leq 2$)", "group_break_after": False},
        {"pred_method": "lbp", "loss": "vce", "model": "3subset_approx_max_attention_varonly", "label": "P.1 VO ($\\alpha' \\leq 3$)", "group_break_after": False},
        {"pred_method": "lbp", "loss": "vce", "model": "4subset_approx_max_attention_varonly", "label": "P.1 VO ($\\alpha' \\leq 4$)", "group_break_after": True},

        {"pred_method": "lbp", "loss": "vce", "model": "tree_approx_max_attention_v2", "label": "P.2 MST", "group_break_after": False},
        {"pred_method": "lbp", "loss": "vce", "model": "pairwise_approx_max_attention_v2", "label": "P.2 ($\\alpha' \\leq 2$)", "group_break_after": False},
        {"pred_method": "lbp", "loss": "vce", "model": "3subset_approx_max_attention_v2", "label": "P.2 ($\\alpha' \\leq 3$)", "group_break_after": False},
        {"pred_method": "lbp", "loss": "vce", "model": "4subset_approx_max_attention_v2", "label": "P.2 ($\\alpha' \\leq 4$)", "group_break_after": True},

        {"pred_method": "lbp", "loss": "vce", "model": "tree_approx_max_transformer", "label": "P.3 MST", "group_break_after": False},
        {"pred_method": "lbp", "loss": "vce", "model": "pairwise_approx_max_transformer", "label": "P.3 ($\\alpha' \\leq 2$)", "group_break_after": False},
        {"pred_method": "lbp", "loss": "vce", "model": "3subset_approx_max_transformer", "label": "P.3 ($\\alpha' \\leq 3$)", "group_break_after": False},
        {"pred_method": "lbp", "loss": "vce", "model": "4subset_approx_max_transformer", "label": "P.3 ($\\alpha' \\leq 4$)", "group_break_after": False},
    ],
}

DATASET_LATEX_LABEL = {
    "scp_n50_m50_k5": r"SCP $\left(\begin{array}{l}n=50\\ m=50\\ k=5\end{array}\right)$",
    "scp_n120-180_m150-250_k5": r"SCP $\left(\begin{array}{l}n=120\text{-}180\\ m=150\text{-}250\\ k=5\end{array}\right)$",
    "scp_n150-300_m150-300_k15": r"SCP $\left(\begin{array}{l}n=150\text{-}300\\ m=150\text{-}300\\ k=15\end{array}\right)$",
}

TABLE_CAPTIONS = {
    "scp_n50_m50_k5": (
        r"Results on the SCP $(n=50,m=50,k=5)$ dataset. "
        r"Gap values are reported as mean $\pm$ standard deviation over seeds. "
        r"Best columns report the best (minimum) gap obtained among the seeds."
    ),
}

TABLE_LABELS = {
    "scp_n50_m50_k5": "tab:scp50_results",
}


# ============================================================
# JSON KEY PATHS
# Adjust these candidate paths to match your actual test_eval.json files.
# Each entry is a list of alternative nested key paths.
# ============================================================
METRIC_PATHS = {
    "dec_gap_avg": [
        ("decimation", "gap_avg_pct"),
        ("decimation", "avg_gap_pct"),
        ("decimation_gap_avg_pct",),
        ("decimation_gap_avg",),
    ],
    "dec_gap_best": [
        ("decimation", "best_gap_pct"),
        ("decimation", "gap_best_pct"),
        ("decimation_gap_best_pct",),
        ("decimation_gap_best",),
    ],
    "dec_time": [
        ("decimation", "time_s"),
        ("decimation", "total_time_s"),
        ("total_gd_t_s",),
        ("decimation_time_s",),
    ],
    "grd_gap_avg": [
        ("greedy", "gap_avg_pct"),
        ("greedy", "avg_gap_pct"),
        ("greedy_gap_avg_pct",),
        ("greedy_gap_avg",),
    ],
    "grd_gap_best": [
        ("greedy", "best_gap_pct"),
        ("greedy", "gap_best_pct"),
        ("greedy_gap_best_pct",),
        ("greedy_gap_best",),
    ],
    "grd_time": [
        ("greedy", "time_s"),
        ("greedy", "total_time_s"),
        ("total_grd_t_s",),
        ("greedy_time_s",),
    ],
}


SEED_RE = re.compile(r"^(?P<base>.+)_seed(?P<seed>\d+)$")


def build_run_name(dataset: str, pred_method: str, loss: str, model: str, seed: int) -> str:
    parts = [dataset, pred_method, loss]
    if model:
        parts.append(model)
    parts.append(f"seed{seed}")
    return "_".join(parts)


def inspect_run_dir(run_dir: Path) -> dict:
    best_pt = run_dir / "checkpoints" / "best.pt"
    test_eval = run_dir / "test_eval.json"
    return {
        "has_best_pt": best_pt.is_file(),
        "has_test_eval": test_eval.is_file(),
    }


def build_directory_index(runs_dir: Path):
    exact_dirs = {}
    duplicates = {}

    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue

        exact_dirs[p.name] = p

        m = SEED_RE.match(p.name)
        if not m:
            continue

        base = m.group("base")
        seed = m.group("seed")

        m2 = SEED_RE.match(base)
        if m2 and m2.group("seed") == seed:
            canonical = f"{m2.group('base')}_seed{seed}"
            duplicates.setdefault(canonical, []).append(p)

    return exact_dirs, duplicates


def repair_duplicate_test_eval(original_dir: Path, duplicate_dir: Path, dry_run: bool = True) -> str:
    original_test = original_dir / "test_eval.json"
    duplicate_test = duplicate_dir / "test_eval.json"

    if not duplicate_test.is_file():
        return "no_action: duplicate has no test_eval.json"

    if original_test.exists():
        return "no_action: original already has test_eval.json"

    if not dry_run:
        shutil.move(str(duplicate_test), str(original_test))
        shutil.rmtree(duplicate_dir)

    return "repaired"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def nested_get(d, path):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def get_metric(eval_data, metric_name):
    for path in METRIC_PATHS[metric_name]:
        value = nested_get(eval_data, path)
        if value is not None:
            return value
    return None


def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean_std(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def min_value(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None
    return min(values)


def fmt_mean_std(x, s, digits=3):
    if x is None:
        return "--"
    return f"${x:.{digits}f} (\\pm {s:.{digits}f})$"


def fmt_scalar(x, digits=3):
    if x is None:
        return "--"
    return f"${x:.{digits}f}$"


def collect_seed_metrics(run_dir: Path):
    eval_path = run_dir / "test_eval.json"
    if not eval_path.is_file():
        return None

    data = load_json(eval_path)
    return {
        "dec_gap_avg": safe_float(get_metric(data, "dec_gap_avg")),
        "dec_gap_best": safe_float(get_metric(data, "dec_gap_best")),
        "dec_time": safe_float(get_metric(data, "dec_time")),
        "grd_gap_avg": safe_float(get_metric(data, "grd_gap_avg")),
        "grd_gap_best": safe_float(get_metric(data, "grd_gap_best")),
        "grd_time": safe_float(get_metric(data, "grd_time")),
    }


def aggregate_family_results(dataset, pred_method, loss, model, exact_dirs):
    seed_rows = []
    for seed in SEEDS:
        run_name = build_run_name(dataset, pred_method, loss, model, seed)
        run_dir = exact_dirs.get(run_name)
        if run_dir is None:
            continue
        info = inspect_run_dir(run_dir)
        if not (info["has_best_pt"] and info["has_test_eval"]):
            continue
        m = collect_seed_metrics(run_dir)
        if m is not None:
            seed_rows.append(m)

    if not seed_rows:
        return None

    return {
        "dec_gap_avg_mean_std": mean_std([x["dec_gap_avg"] for x in seed_rows]),
        "dec_gap_best_min": min_value([x["dec_gap_best"] for x in seed_rows]),
        "dec_time_mean_std": mean_std([x["dec_time"] for x in seed_rows]),
        "grd_gap_avg_mean_std": mean_std([x["grd_gap_avg"] for x in seed_rows]),
        "grd_gap_best_min": min_value([x["grd_gap_best"] for x in seed_rows]),
        "grd_time_mean_std": mean_std([x["grd_time"] for x in seed_rows]),
        "n_seeds": len(seed_rows),
    }


def latex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def render_dataset_table(dataset, exact_dirs):
    row_specs = ROW_SPECS[dataset]
    dataset_label = DATASET_LATEX_LABEL.get(dataset, latex_escape(dataset))
    caption = TABLE_CAPTIONS.get(dataset, f"Results on {latex_escape(dataset)}.")
    label = TABLE_LABELS.get(dataset, f"tab:{dataset}")

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{.99\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Model & \multicolumn{3}{c}{Decimation} & \multicolumn{3}{c}{Greedy} \\")
    lines.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}")
    lines.append(r" &  & Gap Avg. (\%) $\downarrow$ & Best (\%) $\downarrow$ & t (s) & Gap Avg. (\%) $\downarrow$ & Best (\%) $\downarrow$ & t (s) \\")
    lines.append(r"\midrule")

    nrows = len(row_specs)
    first_row = True

    for spec in row_specs:
        agg = aggregate_family_results(
            dataset=dataset,
            pred_method=spec["pred_method"],
            loss=spec["loss"],
            model=spec["model"],
            exact_dirs=exact_dirs,
        )

        if agg is None:
            dec_gap_avg = "--"
            dec_gap_best = "--"
            dec_time = "--"
            grd_gap_avg = "--"
            grd_gap_best = "--"
            grd_time = "--"
        else:
            dec_gap_avg = fmt_mean_std(*agg["dec_gap_avg_mean_std"])
            dec_gap_best = fmt_scalar(agg["dec_gap_best_min"])
            dec_time = fmt_mean_std(*agg["dec_time_mean_std"])
            grd_gap_avg = fmt_mean_std(*agg["grd_gap_avg_mean_std"])
            grd_gap_best = fmt_scalar(agg["grd_gap_best_min"])
            grd_time = fmt_mean_std(*agg["grd_time_mean_std"])

        if first_row:
            left = rf"\multirow{{{nrows}}}{{*}}{{{dataset_label}}}"
            first_row = False
        else:
            left = ""

        lines.append(
            f"{left}\n"
            f"& {spec['label']}\n"
            f"& {dec_gap_avg}\n"
            f"& {dec_gap_best}\n"
            f"& {dec_time}\n"
            f"& {grd_gap_avg}\n"
            f"& {grd_gap_best}\n"
            f"& {grd_time} \\\\"
        )

        if spec.get("group_break_after", False):
            lines.append(r"\cmidrule(lr){2-8}")

    if lines[-1] == r"\cmidrule(lr){2-8}":
        lines.pop()

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def render_full_latex_document(datasets, exact_dirs):
    parts = []
    parts.append(r"\documentclass{article}")
    parts.append(r"\usepackage{booktabs}")
    parts.append(r"\usepackage{multirow}")
    parts.append(r"\usepackage{float}")
    parts.append(r"\usepackage{array}")
    parts.append(r"\usepackage[margin=1in]{geometry}")
    parts.append(r"\begin{document}")
    parts.append("")

    for dataset in datasets:
        if dataset not in ROW_SPECS:
            continue
        parts.append(render_dataset_table(dataset, exact_dirs))
        parts.append("")

    parts.append(r"\end{document}")
    return "\n".join(parts)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--apply", action="store_true",
                        help="Actually move duplicate test_eval.json files and delete duplicate folders")
    parser.add_argument("--latex-out", type=Path, default=Path("results_tables.tex"))
    args = parser.parse_args()

    runs_dir = args.runs_dir
    dry_run = not args.apply

    if not runs_dir.is_dir():
        raise SystemExit(f"Missing runs directory: {runs_dir}")

    exact_dirs, duplicates = build_directory_index(runs_dir)

    missing_dirs = []
    failed_training = []
    missing_test = []
    complete = []
    suspicious_duplicates = []
    repaired_duplicates = []
    unrepaired_duplicates = []

    expected_names = []
    for dataset, pred_method, loss, model in EXPECTED_RUNS:
        for seed in SEEDS:
            expected_names.append(build_run_name(dataset, pred_method, loss, model, seed))

    for run_name in expected_names:
        original_dir = exact_dirs.get(run_name)
        duplicate_dirs = duplicates.get(run_name, [])

        if original_dir is None or not duplicate_dirs:
            continue

        for dup_dir in duplicate_dirs:
            status = repair_duplicate_test_eval(original_dir, dup_dir, dry_run=dry_run)
            if status == "repaired":
                repaired_duplicates.append((run_name, dup_dir.name))
            else:
                unrepaired_duplicates.append((run_name, dup_dir.name, status))

    exact_dirs, duplicates = build_directory_index(runs_dir)

    for run_name in expected_names:
        if run_name in duplicates:
            suspicious_duplicates.append((run_name, [p.name for p in duplicates[run_name]]))

        run_dir = exact_dirs.get(run_name)
        if run_dir is None:
            missing_dirs.append(run_name)
            continue

        info = inspect_run_dir(run_dir)
        if not info["has_best_pt"]:
            failed_training.append(run_name)
        elif not info["has_test_eval"]:
            missing_test.append(run_name)
        else:
            complete.append(run_name)

    datasets_in_use = []
    seen = set()
    for dataset, _, _, _ in EXPECTED_RUNS:
        if dataset not in seen:
            seen.add(dataset)
            datasets_in_use.append(dataset)

    latex_doc = render_full_latex_document(datasets_in_use, exact_dirs)
    args.latex_out.write_text(latex_doc, encoding="utf-8")

    mode = "APPLY MODE" if not dry_run else "DRY RUN"
    print("=" * 100)
    print(f"Runs directory: {runs_dir}")
    print(f"Mode: {mode}")
    print(f"Expected runs: {len(expected_names)}")
    print(f"LaTeX written to: {args.latex_out}")
    print("=" * 100)

    print("\nREPAIRED DUPLICATE TEST RESULTS:")
    if repaired_duplicates:
        for canonical, dup in repaired_duplicates:
            if dry_run:
                print(f"  - would repair {canonical} using {dup}")
            else:
                print(f"  - repaired {canonical} using {dup}")
    else:
        print("  none")

    print("\nUNREPAIRED DUPLICATES:")
    if unrepaired_duplicates:
        for canonical, dup, reason in unrepaired_duplicates:
            print(f"  - {canonical} | duplicate={dup} | reason={reason}")
    else:
        print("  none")

    print("\nMISSING RUN DIRECTORIES:")
    if missing_dirs:
        for x in missing_dirs:
            print(f"  - {x}")
    else:
        print("  none")

    print("\nFAILED TRAINING (missing checkpoints/best.pt):")
    if failed_training:
        for x in failed_training:
            print(f"  - {x}")
    else:
        print("  none")

    print("\nMISSING TEST RESULTS (best.pt exists but test_eval.json missing):")
    if missing_test:
        for x in missing_test:
            print(f"  - {x}")
    else:
        print("  none")

    print("\nREMAINING SUSPICIOUS DUPLICATE DIRECTORIES:")
    if suspicious_duplicates:
        for canonical, dupes in suspicious_duplicates:
            print(f"  - {canonical}")
            for d in dupes:
                print(f"      duplicate: {d}")
    else:
        print("  none")

    print("\nCOMPLETE RUNS:")
    if complete:
        for x in complete:
            print(f"  - {x}")
    else:
        print("  none")

    print("\nSUMMARY:")
    print(f"  repaired duplicates : {len(repaired_duplicates)}")
    print(f"  unrepaired dupes    : {len(unrepaired_duplicates)}")
    print(f"  missing directories : {len(missing_dirs)}")
    print(f"  failed training     : {len(failed_training)}")
    print(f"  missing test eval   : {len(missing_test)}")
    print(f"  remaining dupes     : {len(suspicious_duplicates)}")
    print(f"  complete            : {len(complete)}")


if __name__ == "__main__":
    main()
