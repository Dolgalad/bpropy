from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch

def _build_one_scheduler(name: str, optimizer, args=(), kwargs=None):
    kwargs = {} if kwargs is None else dict(kwargs)
    key = name.lower()

    # ---- common aliases / forgiving names ----
    alias = {
        "cosineannealingwarmrestart": "cosineannealingwarmrestarts",
        "cosineannealingwarmrestarts": "cosineannealingwarmrestarts",
        "cosineannealinglr": "cosineannealinglr",
        "cosineannealing": "cosineannealinglr",
        "reducelronplateau": "reducelronplateau",
        "multiplicativelr": "multiplicativelr",
        "constantlr": "constantlr",
    }
    key = alias.get(key, key)

    if key == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, *args, **kwargs)

    if key == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, *args, **kwargs)

    if key == "cosineannealingwarmrestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, *args, **kwargs)

    if key == "multiplicativelr":
        if "lr_lambda" in kwargs:
            kwargs["lr_lambda"] = eval(kwargs["lr_lambda"])
        return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, *args, **kwargs)

    if key == "constantlr":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, *args, **kwargs)

    raise ValueError(f"Unknown scheduler '{name}'")


def build_scheduler(cfg: Dict[str, Any], optimizer):
    sch_cfg = cfg["train"]["scheduler"]

    if sch_cfg is None:
        return None

    # -------- nested form (dict) --------
    if isinstance(sch_cfg, dict):
        name = sch_cfg["name"].lower()
        args = tuple(sch_cfg.get("args", ()))
        kwargs = dict(sch_cfg.get("kwargs", {}))

        # SequentialLR / ChainedScheduler: have nested "schedulers"
        if name.lower() in ("sequentiallr", "chainedscheduler"):
            child_cfgs = sch_cfg.get("schedulers", [])
            if not isinstance(child_cfgs, list) or len(child_cfgs) == 0:
                raise ValueError(f"{sch_cfg['name']} requires non-empty 'schedulers' list")

            children = []
            for c in child_cfgs:
                c_name = c["name"]
                c_args = tuple(c.get("args", ()))
                c_kwargs = dict(c.get("kwargs", {}))
                children.append(_build_one_scheduler(c_name, optimizer, c_args, c_kwargs))

            if name.lower() == "chainedscheduler":
                # PyTorch: ChainedScheduler takes only the schedulers list
                return torch.optim.lr_scheduler.ChainedScheduler(children)

            # name == SequentialLR:
            # milestones can come from kwargs["milestones"] OR from args (your example).
            milestones = kwargs.pop("milestones", None)
            if milestones is None:
                # interpret args as milestones, supporting both:
                # - args = [20]          (already correct)
                # - args = [20, 20]      (treat as stage lengths -> convert to cumulative milestones)
                if len(args) == 0:
                    raise ValueError("SequentialLR requires milestones (kwargs.milestones or args)")
                # If user gave stage lengths, convert to cumulative cut points:
                # e.g. [20, 20] -> [20] (for 2 schedulers) or [20, 40] (for 3 schedulers), etc.
                # We detect this by allowing len(args) == len(children) and converting.
                if len(args) == len(children):
                    # stage lengths -> cumulative milestones excluding final stage
                    cum = 0
                    ms = []
                    for i, length in enumerate(args[:-1]):
                        cum += int(length)
                        ms.append(cum)
                    milestones = ms
                else:
                    # assume args already is milestones list
                    milestones = [int(x) for x in args]

            # sanity: SequentialLR expects len(milestones) == len(children)-1
            if len(milestones) != len(children) - 1:
                raise ValueError(
                    f"SequentialLR milestones must have length len(schedulers)-1 = {len(children)-1}, "
                    f"got {len(milestones)}"
                )

            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=children,
                milestones=milestones,
                **kwargs,
            )

        # -------- existing single-scheduler dict form --------
        return _build_one_scheduler(sch_cfg["name"], optimizer, args, kwargs)

    # -------- existing list-of-schedulers form --------
    if isinstance(sch_cfg, list):
        schedulers = []
        for sch_k in sch_cfg:
            name = sch_k["name"]
            args = tuple(sch_k.get("args", ()))
            kwargs = dict(sch_k.get("kwargs", {}))
            schedulers.append(_build_one_scheduler(name, optimizer, args, kwargs))
        return schedulers

    raise ValueError("train.scheduler must be a dict or list")

def _apply_lr_schedulers(cfg: Dict[str, Any], schedulers, epoch, metrics):
    if schedulers is None:
        return 
    val = metrics[cfg["train"].get("scheduler", {}).get("metric", "val/gap")]

    if isinstance(schedulers, list):
        for sch in schedulers:
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch.step(val)
            #elif isinstance(sch, torch.optim.lr_scheduler.MultiplicativeLR):
            #    sch.step(epoch)
            else:
                sch.step()
    else:
        if isinstance(schedulers, torch.optim.lr_scheduler.ReduceLROnPlateau):
            schedulers.step(val)
        #elif isinstance(sch, torch.optim.lr_scheduler.MultiplicativeLR):
        #    schedulers.step(epoch)

        else:
            schedulers.step()

def _get_scheduler_state_dict(schedulers):
    if schedulers is None:
        return 
    if isinstance(schedulers, list):
        return [_get_scheduler_state_dict(sch) for sch in schedulers]
    return schedulers.state_dict()

def _set_scheduler_state_dict(schedulers, state_dict):
    if schedulers is None:
        return 
    if isinstance(schedulers, list):
        for sch,sd in zip(schedulers, state_dict):
            _set_scheduler_state_dict(sch, sd)
    else:
        schedulers.load_state_dict(state_dict)

