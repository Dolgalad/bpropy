from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from progress.bar import Bar

from bpropy.datasets.prefetcher import CUDAPrefetcher

from .checkpoint import EpochState, load_checkpoint, save_checkpoint
from .data import make_loaders
from .logging import TrainLogger, make_run_paths
from .metrics import EvalStats, _sync, gap_from_solution, write_test_eval_json
from .schedulers import build_scheduler, _apply_lr_schedulers

from bpropy.gd.greedy_decimation import greedy_decimation

# ---- Model registry ----
from bpropy.models import MODEL_REGISTRY

# ---- Loss registry ----
from bpropy.losses import LOSS_REGISTRY

# ---- Method registry ----
from bpropy.cli.train.method_registry import METHOD_REGISTRY


def build_model(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    name = cfg["model"]["name"]
    args = cfg["model"].get("args", {})
    Model = MODEL_REGISTRY[name]
    model = Model(**args).to(device)

    init_from = cfg["model"].get("init_from")
    if init_from:
        # load model weights only
        ckpt = torch.load(init_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    return model


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_cfg = cfg["train"]["optimizer"]
    name = opt_cfg["name"].lower()
    args = dict(opt_cfg.get("args", {}))

    if name == "adam":
        return torch.optim.Adam(model.parameters(), **args)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), **args)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **args)

    raise ValueError(f"Unknown optimizer '{name}'")

#def build_scheduler(cfg: Dict[str, Any], optimizer):
#    sch_cfg = cfg["train"]["scheduler"]
#    if isinstance(sch_cfg, dict):
#        name = sch_cfg["name"].lower()
#        args = tuple(sch_cfg.get("args", ()))
#        kwargs = dict(sch_cfg.get("kwargs", {}))
#
#        if name.lower() == "reducelronplateau":
#            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, *args, **kwargs)
#        if name.lower() == "cosineannealinglr":
#            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, *args, **kwargs)
#        if name.lower() == "cosineannealingwarmrestarts":
#            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, *args, **kwargs)
#        if name.lower() == "multiplicativelr":
#            kwargs["lr_lambda"] = eval(kwargs["lr_lambda"])
#            return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, *args, **kwargs)
#
#    if isinstance(sch_cfg, list):
#        schedulers = []
#        for sch_k in sch_cfg:
#            name = sch_k["name"].lower()
#            args = tuple(sch_k.get("args", ()))
#            kwargs = dict(sch_k.get("kwargs", {}))
#            if name.lower() == "reducelronplateau":
#                schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, *args, **kwargs))
#            if name.lower() == "cosineannealinglr":
#                schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, *args, **kwargs))
#            if name.lower() == "cosineannealingwarmrestarts":
#                schedulers.append(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, *args, **kwargs))
#            if name.lower() == "multiplicativelr":
#                if "lr_lambda" in kwargs:
#                    kwargs["lr_lambda"] = eval(kwargs["lr_lambda"])
#                schedulers.append(torch.optim.lr_scheduler.MultiplicativeLR(optimizer, *args, **kwargs))
#
#        return schedulers
#    raise ValueError(f"Unknown scheduler '{name}'")

#def _apply_lr_schedulers(cfg: Dict[str, Any], schedulers, val):
#    if isinstance(schedulers, list):
#        for sch in schedulers:
#            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
#                sch.step(val)
#            else:
#                sch.step()
#    else:
#        if isinstance(schedulers, torch.optim.lr_scheduler.ReduceLROnPlateau):
#            schedulers.step(val)
#        else:
#            schedulers.step()

def compute_loss_from_spec(
    spec,
    *,
    loss_registry: Dict[str, Callable],
) -> torch.Tensor:
    if spec.backward_fn is not None:
        raise RuntimeError("backward_fn should not be used in eval/no_grad mode.")
    if spec.loss_name is None or spec.pred is None or spec.target is None:
        raise ValueError("Incomplete LossSpec for standard loss path.")
    base = loss_registry[spec.loss_name](spec.pred, spec.target, **spec.loss_kwargs)
    if spec.extra_terms:
        base = base + sum(spec.extra_terms.values())
    return base



def train_one_epoch(
    cfg: Dict[str, Any],
    adapter: MethodAdapter,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    losses = []
    steps = 0
    max_steps = int(cfg["train"]["steps_per_epoch"])

    bar = Bar(f"[Train] epoch {epoch}", max=len(loader))
    for data, y in CUDAPrefetcher(loader, device):
        optimizer.zero_grad(set_to_none=True)
        pred = adapter.forward_fn(model, data)
        spec = adapter.loss_spec_fn(adapter, data, pred, y, cfg)
        
        if spec.backward_fn is not None:
            logged_loss = spec.backward_fn()
            loss_scalar = float(logged_loss.detach().cpu().item()) if torch.is_tensor(logged_loss) else float(logged_loss)
        else:
            loss = compute_loss_from_spec(spec, loss_registry=LOSS_REGISTRY)
            loss.backward()
            loss_scalar = float(loss.detach().cpu().item())
        
        clip = cfg["train"].get("grad_clip_norm")
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))

        optimizer.step()

        losses.append(loss_scalar)

        bar.next()

        steps += 1
        if steps >= max_steps:
            break
    bar.finish()

    return float(np.mean(losses)) if losses else 0.0


def validate_one_epoch(
    cfg: Dict[str, Any],
    adapter: MethodAdapter,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    model.eval()
    losses = []
    eval_loss_names = [eval_loss_cfg["name"] for eval_loss_cfg in cfg.get("eval", {}).get("losses", [])]
    eval_losses = {n: [] for n in eval_loss_names}
    gaps = []
    steps = 0
    max_steps = int(cfg["train"]["val_steps"])
    maximize = bool(cfg.get("problem", {}).get("maximize", True))

    with torch.no_grad():
        bar = Bar(f"[Val]   epoch {epoch}", max=len(loader))
        for pb, data, y in CUDAPrefetcher(loader, device):
            pred = adapter.forward_fn(model, data)
            spec = adapter.loss_spec_fn(adapter, data, pred, y, cfg)
            
            # In eval, backward_fn should be None; if not, you can log NaN or skip loss
            loss = compute_loss_from_spec(spec, loss_registry=LOSS_REGISTRY)
            losses.append(float(loss.detach().cpu().item()))

            for eval_loss_cfg in cfg.get("eval", {}).get("losses", []):
                name = eval_loss_cfg["name"]
                args = eval_loss_cfg.get("args", {})
                eval_loss = LOSS_REGISTRY[name](
                        pred,
                        y,
                        data=data,
                        **args,
                        )
                eval_losses[name].append(eval_loss.detach().float().cpu().item())

            scores = adapter.greedy_scores_fn(pb[0], data, pred)
            grd_sol = pb[0].greedy(scores=scores).x

            _0_idx, _1_idx = (y == 0).cpu().numpy(), (y == 1).cpu().numpy()

            _0_scores_mean, _0_scores_std, _1_scores_mean, _1_scores_std = (
                    np.mean( scores[_0_idx] ),
                    np.std( scores[_0_idx] ),
                    np.mean( scores[_1_idx] ),
                    np.std( scores[_1_idx] ),

                    )

            #print("0 mean, std : ", _0_scores_mean, _0_scores_std)
            #print("1 mean, std : ", _1_scores_mean, _1_scores_std)

            gap = gap_from_solution(
                y_true_sum=float(y.sum().item()),
                y_pred_sum=float(grd_sol.sum()),
                maximize=maximize,
                eps=1e-8,  # match your epoch validation divide-by-y.sum() style :contentReference[oaicite:10]{index=10}
            )
            gaps.append(float(gap))

            bar.next()
            steps += 1
            if steps >= max_steps:
                break
        bar.finish()

    return (float(np.mean(losses)) if losses else 0.0, 
            float(np.mean(gaps)) if gaps else 0.0, 
            {n:float(np.mean(eval_losses[n])) for n in eval_losses.keys()})


def final_test_eval(
    cfg: Dict[str, Any],
    adapter: MethodAdapter,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    log_dir: str,
) -> str:
    model.eval()
    losses, gaps, dec_gaps = [], [], []
    eval_loss_names = [eval_loss_cfg["name"] for eval_loss_cfg in cfg.get("eval", {}).get("losses", [])]
    eval_losses = {n: [] for n in eval_loss_names}
    t_pred, t_grd, t_tot, t_pre, t_e2e, t_gd = [], [], [], [], [], []
    maximize = bool(cfg.get("problem", {}).get("maximize", True))

    with torch.no_grad():
        bar = Bar("[Test]          ", max=len(loader))
        for pb, data, y in CUDAPrefetcher(loader, device):
            _sync(device)
            t0 = time.perf_counter()

            t1 = time.perf_counter()
            pred = adapter.forward_fn(model, data)
            _sync(device)
            t2 = time.perf_counter()

            scores = adapter.greedy_scores_fn(pb[0], data, pred)
            t3 = time.perf_counter()
            grd_sol = pb[0].greedy(scores=scores).x
            t4 = time.perf_counter()

            # Greedy decimation
            t5 = time.perf_counter()
            backend = adapter.decimation_backend_fn(pb[0], data, pred, cfg, device=device)
            padapter = adapter.decimation_adapter_fn(pb[0], cfg)
            dec_sol = greedy_decimation(pb[0], backend, padapter)
            t6 = time.perf_counter()
            
            spec = adapter.loss_spec_fn(adapter, data, pred, y, cfg)
           
            # In eval, backward_fn should be None; if not, you can log NaN or skip loss
            loss = compute_loss_from_spec(spec, loss_registry=LOSS_REGISTRY)
            losses.append(float(loss.detach().cpu().item()))

            for eval_loss_cfg in cfg.get("eval", {}).get("losses", []):
                name = eval_loss_cfg["name"]
                args = eval_loss_cfg.get("args", {})
                eval_loss = LOSS_REGISTRY[name](
                        pred,
                        y,
                        data=data,
                        **args,
                        )
                eval_losses[name].append(eval_loss.detach().float().cpu().item())


            gap = float(
                gap_from_solution(
                    y_true_sum=float(y.sum().item()),
                    y_pred_sum=float(grd_sol.sum()),
                    maximize=maximize,
                    eps=1e-8,  # match your timed eval path :contentReference[oaicite:11]{index=11}
                )
            )

            dec_gap = float(
                gap_from_solution(
                    y_true_sum=float(y.sum().item()),
                    y_pred_sum=float(dec_sol.sum()),
                    maximize=maximize,
                    eps=1e-8,
                )
            )

            gaps.append(gap)
            dec_gaps.append(dec_gap)
            t_pred.append(t2 - t1)
            t_grd.append(t4 - t3)
            t_tot.append(t4 - t0)
            t_pre.append(float(getattr(data, "preprocess_time_s", 0.0)))
            t_gd.append(t6 - t5)
            t_e2e.append(t_pre[-1] + (t4 - t0))

            bar.next()
        bar.finish()

    def ms(x):
        a = np.asarray(x, dtype=np.float64)
        if a.size == 0:
            return 0.0, 0.0
        if a.size == 1:
            return float(a.mean()), 0.0
        return float(a.mean()), float(a.std(ddof=1))

    eval_loss_stats = {}
    for n, vals in eval_losses.items():
        mean, std = ms(vals)
        eval_loss_stats[n] = {"mean": ms(vals)[0], "std": ms(vals)[1]}

    stats = EvalStats(
        loss_mean=ms(losses)[0], loss_std=ms(losses)[1],
        gap_mean=ms(gaps)[0], gap_std=ms(gaps)[1],
        dec_gap_mean=ms(dec_gaps)[0], dec_gap_std=ms(dec_gaps)[1],
        time_pred_mean=ms(t_pred)[0], time_pred_std=ms(t_pred)[1],
        time_greedy_mean=ms(t_grd)[0], time_greedy_std=ms(t_grd)[1],
        time_total_mean=ms(t_tot)[0], time_total_std=ms(t_tot)[1],
        time_preprocess_mean=ms(t_pre)[0], time_preprocess_std=ms(t_pre)[1],
        time_end_to_end_mean=ms(t_e2e)[0], time_end_to_end_std=ms(t_e2e)[1],
        time_gd_mean=ms(t_gd)[0], time_gd_std=ms(t_gd)[1],

        count=len(losses),
        eval_losses=eval_loss_stats,
    )
    out_path = write_test_eval_json(log_dir, stats, filename="test_eval.json")
    return out_path


def run_training(cfg: Dict[str, Any]) -> None:
    # seeds
    torch.manual_seed(int(cfg.get("seed", 0)))
    np.random.seed(int(cfg.get("seed", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run dirs + logger
    paths = make_run_paths(cfg)
    logger = TrainLogger(paths, cfg)
    logger.write_hparams(extra={"device": str(device)})

    # data
    loaders = make_loaders(cfg)

    # method adapter
    method_name = cfg.get("train", {}).get("method", "var_logits")
    if method_name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown train.method '{method_name}'. Known: {list(METHOD_REGISTRY.keys())}")
    adapter_train = METHOD_REGISTRY[method_name]
    eval_method = cfg.get("eval", {}).get("method") or method_name
    adapter_eval = METHOD_REGISTRY[eval_method] 
    adapter_eval.loss_name = cfg.get("eval", {}).get("loss-name") or cfg["loss"]["name"]

    # model/opt/loss
    model = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    loss_name = cfg["loss"]["name"]
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss.name '{loss_name}'. Known: {list(LOSS_REGISTRY.keys())}")

    # resume
    st = EpochState()
    resume_path = cfg.get("run", {}).get("resume")
    if resume_path:
        st, _ = load_checkpoint(resume_path, model, optimizer, scheduler, map_location=device)

    # training loop
    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"].get("early_stop", {}).get("patience", 20))

    for epoch in range(st.epoch, epochs + 1):
        tr = train_one_epoch(cfg, adapter_train, model, optimizer, loaders.train_loader, device, epoch)
        va, gap, eloss = validate_one_epoch(cfg, adapter_train, model, loaders.val_loader, device, epoch)

        logger.add_scalars("loss", {"train": tr, "val": va}, epoch)
        logger.add_scalars("lr", {f"pg{k}":pg["lr"] for k,pg in enumerate(optimizer.param_groups)}, epoch)
        logger.add_scalars("gap", {"val": gap}, epoch)
        epoch_out = f"Epoch {epoch:03d} | train {tr:.4f} | val {va:.4f} | gap {gap:.4f}"
        if len(eloss):
            eloss_out = " | ".join(f"{n} {eloss[n]:.4f}" for n in eloss.keys())
            epoch_out += " | " + eloss_out
            logger.add_scalars("extra_loss", eloss, epoch)
        print(epoch_out)

        # create dictionary of metrics
        metrics_dict = {
                "train/loss": tr,
                "val/loss": va,
                "val/gap": gap,
                **eloss
                }
        # schedule step
        _apply_lr_schedulers(cfg, scheduler, epoch, metrics_dict)

        # early stopping on gap (same “best_gap/epochs_bad” logic as your script) :contentReference[oaicite:12]{index=12}
        st.epoch = epoch
        improved = metrics_dict[cfg["train"].get("early_stop", {}).get("metric", "val/gap")] < st.best_gap
        if improved:
            st.best_gap = gap
            st.best_epoch = epoch
            st.epochs_bad = 0
        else:
            st.epochs_bad += 1

        # checkpoints
        last_path = os.path.join(paths.ckpt_dir, "last.pt")
        save_checkpoint(last_path, st, model, optimizer, scheduler)

        if improved and cfg["train"]["checkpoint"].get("save_best", True):
            best_path = os.path.join(paths.ckpt_dir, "best.pt")
            print("Saving new best checkpoint")
            save_checkpoint(best_path, st, model, optimizer, scheduler)

        if st.epochs_bad >= patience:
            print(f"[early-stop] No gap improvement for {st.epochs_bad} epochs (patience={patience}).")
            break

    # final eval: load best if present, then evaluate test split and write test_eval.json :contentReference[oaicite:13]{index=13}
    best_path = os.path.join(paths.ckpt_dir, "last.pt")
    if os.path.exists(best_path):
        print(f"[eval] Loading best checkpoint from {best_path}")
        load_checkpoint(best_path, model, optimizer=None, map_location=device)

    if cfg.get("eval", {}).get("enabled", True):
        out_path = final_test_eval(cfg, adapter_eval, model, loaders.test_loader, device, paths.log_dir)
        print(f"[test-eval] saved stats to {out_path}")

    logger.close()

