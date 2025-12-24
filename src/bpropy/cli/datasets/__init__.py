# src/bpropy/cli/datasets/__init__.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bpropy.cli.datasets.generate_mis_dataset import add_mis_subparser
from bpropy.cli.datasets.generate_scp_dataset import add_scp_subparser


def _dict_to_argv(d: Dict[str, Any]) -> List[str]:
    """
    Convert a dict of CLI options into a flat argv list.
    - bool True => '--flag'
    - list/tuple => '--flag v1 v2 ...'
    - None/False => omitted
    """
    argv: List[str] = []
    for k, v in d.items():
        if k in {"command", "problem", "output_dir"}:
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        elif v is None:
            continue
        elif isinstance(v, (list, tuple)):
            argv.append(flag)
            argv.extend(str(x) for x in v)
        else:
            argv.extend([flag, str(v)])
    return argv


def _load_config(path: str) -> Dict[str, Any]:
    p = Path(path).expanduser()
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a JSON object (dict).")
    return cfg


def _infer_slurm_sharding(cfg: Dict[str, Any]) -> None:
    """
    If shard_id/num_shards are missing, fill them from Slurm env vars.
    """
    if "shard_id" not in cfg and "SLURM_ARRAY_TASK_ID" in os.environ:
        cfg["shard_id"] = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if "num_shards" not in cfg and "SLURM_ARRAY_TASK_COUNT" in os.environ:
        cfg["num_shards"] = int(os.environ["SLURM_ARRAY_TASK_COUNT"])


def _expand_config_to_argv(cfg: Dict[str, Any], out_override: Optional[str]) -> List[str]:
    """
    Supported config schema (minimal):
      {
        "command": "generate",          # optional; default "generate"
        "problem": "mis" | "scp",       # required
        "output_dir": "path",           # required (unless overridden)
        ... generator args as keys ...
      }
    """
    command = cfg.get("command", "generate")
    problem = cfg.get("problem", None)
    output_dir = cfg.get("output_dir", None)

    if problem is None:
        raise ValueError("Config must include 'problem' (e.g. 'mis' or 'scp').")

    if out_override:
        output_dir = out_override
    if not output_dir:
        raise ValueError("Config must include 'output_dir' (or pass --out-override).")

    # Fill sharding from Slurm, if present and not explicitly set
    _infer_slurm_sharding(cfg)

    # Build argv matching the existing CLI structure:
    # bpropy-datasets generate <problem> <output_dir> [flags...]
    argv = [command, str(problem), str(output_dir)]
    argv += _dict_to_argv(cfg)
    return argv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bpropy-train",
        description="Dataset generation utilities for bpropy.",
    )

    # New: config-driven entry
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config file. If provided, subcommands/args are read from the config.",
    )
    p.add_argument(
        "--out-override",
        type=str,
        default=None,
        help="Override 'output_dir' from the config file (useful for scratch paths on clusters).",
    )

    sub = p.add_subparsers(dest="command", required=False)

    gen = sub.add_parser("generate", help="Generate datasets (MIS/SCP/...)")
    gen_sub = gen.add_subparsers(dest="problem", required=True)
    add_mis_subparser(gen_sub)
    add_scp_subparser(gen_sub)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)

    # If config is provided, expand to a synthetic argv and parse again fully.
    if args.config is not None:
        cfg = _load_config(args.config)
        synthetic_argv = _expand_config_to_argv(cfg, args.out_override)
        # allow extra CLI overrides after --config (unknown)
        synthetic_argv += unknown
        args = parser.parse_args(synthetic_argv)

    else:
        # Normal CLI usage (no config)
        if args.command is None:
            parser.error("Either provide --config, or use a subcommand like: bpropy-datasets generate mis ...")

        # If caller passed extras, make it a hard error (keeps behavior strict)
        if unknown:
            parser.error(f"Unrecognized arguments: {' '.join(unknown)}")

    return int(args._handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

