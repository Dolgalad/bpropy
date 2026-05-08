from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

from .config import load_run_config, merge_cli_overrides, validate_config
from .loop import run_training


def build_base_parser() -> argparse.ArgumentParser:
    """
    Stage-1 parser: ONLY --config (and maybe --print-config).
    We parse this first to load JSON config before building full parser.
    """
    p = argparse.ArgumentParser(prog="bpropy-train", add_help=False)
    p.add_argument("--config", required=True, help="Path to JSON config.")
    p.add_argument("--print-config", action="store_true", help="Print resolved config and exit.")
    return p


def build_full_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    """
    Stage-2 parser: all training flags.
    IMPORTANT: use argparse.SUPPRESS so that only explicitly provided args appear in parsed namespace.
    That lets us apply overrides cleanly.
    """
    p = argparse.ArgumentParser(prog="bpropy-train")

    # still include these so `-h` shows them and they can be re-parsed
    p.add_argument("--config", required=True, help="Path to JSON config.")
    p.add_argument("--print-config", action="store_true", help="Print resolved config and exit.")

    # --- common overrides (add more as you like) ---
    p.add_argument("--seed", type=int, default=argparse.SUPPRESS)

    # run/logging
    p.add_argument("--run-name", dest="run_name", default=argparse.SUPPRESS)
    p.add_argument("--log-dir", dest="log_dir", default=argparse.SUPPRESS)

    # data
    p.add_argument("--data-root", dest="data_root", default=argparse.SUPPRESS)
    p.add_argument("--cache-dir", dest="cache_dir", default=argparse.SUPPRESS)
    p.add_argument("--split", dest="split", default=argparse.SUPPRESS)
    p.add_argument("--val-split", dest="val_split", default=argparse.SUPPRESS)
    p.add_argument("--test-split", dest="test_split", default=argparse.SUPPRESS)
    p.add_argument("--batch-size", type=int, dest="batch_size", default=argparse.SUPPRESS)
    p.add_argument("--num-workers", type=int, dest="num_workers", default=argparse.SUPPRESS)
    p.add_argument("--feature-mode", dest="feature_mode", default=argparse.SUPPRESS)
    p.add_argument("--hetero", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--no-hetero", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--with-lbp", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--with-nmf", action="store_true", default=argparse.SUPPRESS)

    # model/loss
    p.add_argument("--model", dest="model_name", default=argparse.SUPPRESS)
    p.add_argument("--loss", dest="loss_name", default=argparse.SUPPRESS)

    # train
    p.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    p.add_argument("--steps-per-epoch", type=int, dest="steps_per_epoch", default=argparse.SUPPRESS)
    p.add_argument("--steps-per-val", type=int, dest="steps_per_val", default=argparse.SUPPRESS)
    p.add_argument("--lr", type=float, dest="lr", default=argparse.SUPPRESS)
    p.add_argument("--weight-decay", type=float, dest="weight_decay", default=argparse.SUPPRESS)
    p.add_argument("--resume", dest="resume", default=argparse.SUPPRESS)
    p.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Path to checkpoint to initialize model weights from (model only, no optimizer/scheduler)"
    )

    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # stage 1 parse to get config path
    base = build_base_parser()
    base_args, remaining = base.parse_known_args(argv)

    cfg = load_run_config(base_args.config)
    validate_config(cfg)  # ensure defaults exist / fill missing

    # stage 2 parse with full parser
    parser = build_full_parser(cfg)
    args = parser.parse_args(argv)

    # apply overrides only for flags that were explicitly provided
    cfg = merge_cli_overrides(cfg, args)

    if getattr(args, "print_config", False):
        from .config import dump_config_json
        print(dump_config_json(cfg))
        return 0

    run_training(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

