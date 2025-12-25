from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from torch.utils.data import DataLoader

from bpropy.datasets.mrf_dataset_2 import MRFDataset, MRFDatasetConfig
from bpropy.adapters.vc_to_data import problem_to_data
from bpropy.datasets.collator import mrf_collate


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def make_dataset(cfg: Dict[str, Any], split: str, yield_mode: str) -> MRFDataset:
    """
    yield_mode: "data" -> yields (data, y)
               "both" -> yields (pb, data, y)
    """
    print("in make dataset split ", split)
    #data_cfg = cfg["data"]
    gt = cfg.get("problem", {}).get("ground_truth", {"solver": "greedy", "args": {}})
    solver = gt.get("solver", "greedy")

    data_cfg_base = cfg["data"]
    split_over = data_cfg_base.get("splits", {}).get(split, {})
    data_cfg = _deep_merge(data_cfg_base, split_over)
    print(data_cfg.get("precompute", {}))

    preprocess_kwargs = {
                "hetero":bool(data_cfg.get("hetero", False)),
                "feature_mode":data_cfg["feature_mode"],
                "with_lbp":bool(data_cfg.get("precompute", {}).get("lbp", False)),
                "with_nmf":bool(data_cfg.get("precompute", {}).get("nmf", False)),
                "with_ss":bool(data_cfg.get("precompute", {}).get("ss", False)),
            }

    ds_cfg = MRFDatasetConfig(
        feature_mode=data_cfg["feature_mode"],
        yield_mode=yield_mode,
        refresh=data_cfg.get("refresh", False),
        return_assignment=True,
        assignment_solver=solver,
        preprocess_kwargs = preprocess_kwargs,
    )

    # prepare the preprocess function, if with_ss, with_lbp or with_nmf add the assignment as second argument
    if preprocess_kwargs["with_lbp"] or preprocess_kwargs["with_nmf"] or preprocess_kwargs["with_ss"]:
        preprocess_fn=lambda pb, y: problem_to_data(
            pb, y=y, **preprocess_kwargs
        )
    else:
        preprocess_fn=lambda pb: problem_to_data(
            pb, **preprocess_kwargs
        )



    return MRFDataset(
        data_cfg["root"],
        split,
        cfg=ds_cfg,
        preprocess=preprocess_fn,
        processed_dir=data_cfg.get("processed_dir", "processed"),
    )


def make_loaders(cfg: Dict[str, Any]) -> DataBundle:
    dl = cfg["data"]["loader"]

    # train yields (data, y)
    ds_train = make_dataset(cfg, cfg["data"]["split"], yield_mode="data")
    train_loader = DataLoader(
        ds_train,
        batch_size=int(dl["batch_size"]),
        shuffle=True,
        num_workers=int(dl["num_workers"]),
        pin_memory=bool(dl["pin_memory"]),
        persistent_workers=bool(dl["persistent_workers"]),
        prefetch_factor=int(dl["prefetch_factor"]),
        collate_fn=mrf_collate,
    )

    # val/test yield (pb, data, y) for greedy-gap evaluation
    ds_val = make_dataset(cfg, cfg["data"]["val_split"], yield_mode="both")
    val_loader = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=int(dl["num_workers"]),
        pin_memory=bool(dl["pin_memory"]),
        persistent_workers=bool(dl["persistent_workers"]),
        prefetch_factor=int(dl["prefetch_factor"]),
        collate_fn=mrf_collate,
    )

    ds_test = make_dataset(cfg, cfg["data"]["test_split"], yield_mode="both")
    test_loader = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        num_workers=int(dl["num_workers"]),
        pin_memory=bool(dl["pin_memory"]),
        persistent_workers=bool(dl["persistent_workers"]),
        prefetch_factor=int(dl["prefetch_factor"]),
        collate_fn=mrf_collate,
    )

    return DataBundle(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)

