from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict
import warnings
import argparse
import os

Config = Dict[str, Any]

def load_run_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = json.load(f)
    # keep path for later (logging/debug)
    cfg.setdefault("_meta", {})
    cfg["_meta"]["config_path"] = os.path.abspath(path)
    return cfg

def dump_config_json(cfg: Config) -> str:
    return json.dumps(cfg, indent=2, sort_keys=True)


def merge_cli_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """
    Take explicit CLI args (SUPPRESS means absent) and overwrite cfg.

    This is where you decide the mapping from CLI flags -> config keys.
    Keep it small; you can always add flags later.
    """
    d = vars(args)

    # seed
    if "seed" in d:
        cfg["seed"] = int(d["seed"])

    # run/logging
    if "run_name" in d:
        cfg.setdefault("run", {})
        cfg["run"]["name"] = d["run_name"]
    if "log_dir" in d:
        cfg.setdefault("logging", {})
        cfg["logging"]["log_dir"] = d["log_dir"]

    # data
    if "data_root" in d:
        cfg.setdefault("data", {})
        cfg["data"]["root"] = d["data_root"]
    if "split" in d:
        cfg["data"]["split"] = d["split"]
    if "val_split" in d:
        cfg["data"]["val_split"] = d["val_split"]
    if "test_split" in d:
        cfg["data"]["test_split"] = d["test_split"]
    if "batch_size" in d:
        cfg.setdefault("data", {}).setdefault("loader", {})
        cfg["data"]["loader"]["batch_size"] = int(d["batch_size"])
    if "num_workers" in d:
        cfg.setdefault("data", {}).setdefault("loader", {})
        cfg["data"]["loader"]["num_workers"] = int(d["num_workers"])
    if "feature_mode" in d:
        cfg["data"]["feature_mode"] = d["feature_mode"]

    # booleans with explicit negation support
    if "hetero" in d:
        cfg["data"]["hetero"] = True
    if "no_hetero" in d:
        cfg["data"]["hetero"] = False
    if "with_lbp" in d:
        cfg.setdefault("data", {}).setdefault("precompute", {})
        cfg["data"]["precompute"]["lbp"] = True
    if "with_nmf" in d:
        cfg.setdefault("data", {}).setdefault("precompute", {})
        cfg["data"]["precompute"]["nmf"] = True

    # model/loss selection
    if "model_name" in d:
        cfg.setdefault("model", {})
        cfg["model"]["name"] = d["model_name"]
    if "loss_name" in d:
        cfg.setdefault("loss", {})
        cfg["loss"]["name"] = d["loss_name"]

    # train
    if "epochs" in d:
        cfg.setdefault("train", {})
        cfg["train"]["epochs"] = int(d["epochs"])
    if "steps_per_epoch" in d:
        cfg["train"]["steps_per_epoch"] = int(d["steps_per_epoch"])
    if "steps_per_val" in d:
        cfg["train"]["steps_per_val"] = int(d["steps_per_val"])
    if "resume" in d:
        cfg.setdefault("run", {})
        cfg["run"]["resume"] = d["resume"]
    if "init_from" in d:
        cfg["model"]["init_from"] = d["init_from"]

    # optimizer args (flattened flags)
    if "lr" in d:
        cfg.setdefault("train", {}).setdefault("optimizer", {}).setdefault("args", {})
        cfg["train"]["optimizer"].setdefault("name", "adam")
        cfg["train"]["optimizer"]["args"]["lr"] = float(d["lr"])
    if "weight_decay" in d:
        cfg.setdefault("train", {}).setdefault("optimizer", {}).setdefault("args", {})
        cfg["train"]["optimizer"].setdefault("name", "adam")
        cfg["train"]["optimizer"]["args"]["weight_decay"] = float(d["weight_decay"])

    return cfg

def validate_config(cfg: Config) -> None:
    """
    Keep this lightweight at first; just check required fields exist.
    """
    # required-ish
    for k in ["run", "data", "model", "train", "loss"]:
        if k not in cfg:
            raise ValueError(f"Config missing top-level key '{k}'.")

    if "root" not in cfg["data"]:
        raise ValueError("Config missing data.root")
    if "split" not in cfg["data"]:
        raise ValueError("Config missing data.split")

    if "name" not in cfg["model"]:
        raise ValueError("Config missing model.name")
    if "epochs" not in cfg["train"]:
        raise ValueError("Config missing train.epochs")

    # defaults (safe)
    cfg.setdefault("seed", 0)
    cfg["data"].setdefault("val_split", "val")
    cfg["data"].setdefault("test_split", "test")
    cfg["data"].setdefault("processed_dir", "processed")
    cfg["data"].setdefault("feature_mode", "EB")
    cfg["data"].setdefault("hetero", False)
    cfg["data"].setdefault("precompute", {})
    cfg["data"]["precompute"].setdefault("lbp", False)
    cfg["data"]["precompute"].setdefault("nmf", False)
    cfg["data"].setdefault("loader", {})
    cfg["data"]["loader"].setdefault("batch_size", 16)
    cfg["data"]["loader"].setdefault("num_workers", 2)
    cfg["data"]["loader"].setdefault("pin_memory", True)
    cfg["data"]["loader"].setdefault("prefetch_factor", 2)
    cfg["data"]["loader"].setdefault("persistent_workers", True)

    cfg.setdefault("problem", {})
    cfg["problem"].setdefault("maximize", True)
    cfg["problem"].setdefault("ground_truth", {"solver": "greedy", "args": {}})

    cfg["model"].setdefault("args", {})
    cfg["model"].setdefault("init_from", None)

    cfg["loss"].setdefault("args", {})
    cfg["loss"].setdefault("regularizers", [])
    cfg["train"].setdefault("steps_per_epoch", 10_000_000)
    cfg["train"].setdefault("steps_per_val", 10_000_000)
    cfg["train"].setdefault("amp", False)
    cfg["train"].setdefault("grad_clip_norm", None)
    cfg["train"].setdefault("optimizer", {"name": "adam", "args": {"lr": 1e-3}})
    cfg["train"].setdefault("early_stop", {"patience": 20, "metric": "val/gap", "mode": "min"})
    cfg["train"].setdefault("checkpoint", {"save_best": True})

    cfg.setdefault("logging", {})
    cfg["logging"].setdefault("log_dir", "runs")
    cfg["logging"].setdefault("tensorboard", True)
    cfg["logging"].setdefault("console_every_n_steps", 100)

    cfg.setdefault("eval", {})
    cfg["eval"].setdefault("enabled", True)
    cfg["eval"].setdefault("split", "test")  # final evaluation like your script writes test_eval.json

    # Apply requirement overwrite
    from .config_rules import apply_config_policies
    apply_config_policies(cfg)



