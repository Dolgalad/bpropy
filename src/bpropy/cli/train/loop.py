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
        print("Loading from ", init_from)
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

def _warmup_lr(cfg, optimizer, global_step: int):
    w = cfg.get("train", {}).get("warmup", None)
    if not w or not w.get("enabled", False):
        return

    warmup_steps = int(w.get("steps", 0))
    if warmup_steps <= 0 or global_step >= warmup_steps:
        return

    start_factor = float(w.get("start_factor", 0.05))
    # linear ramp: start_factor -> 1.0
    t = (global_step + 1) / warmup_steps
    factor = start_factor + t * (1.0 - start_factor)

    # apply factor to each param group base lr
    # best practice: store base_lrs once at start of training
    for pg, base_lr in zip(optimizer.param_groups, optimizer._base_lrs):
        pg["lr"] = base_lr * factor

def train_one_epoch(
    cfg: Dict[str, Any],
    adapter: MethodAdapter,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    device: torch.device,
    epoch: int,
    global_step: int,
) -> float:
    model.train()
    losses = []
    steps = 0
    max_steps = int(cfg["train"]["steps_per_epoch"])

    bar = Bar(f"[Train] epoch {epoch}", max=len(loader))

    score_deltas = []

    for data, y in CUDAPrefetcher(loader, device):
        optimizer.zero_grad(set_to_none=True)
        pred = adapter.forward_fn(model, data)
        spec = adapter.loss_spec_fn(adapter, data, pred, y, cfg)
        if epoch == 1 and steps == 0:
            from bpropy.debug_audit import audit_batch
            #audit_batch(data, pred = pred, cfg=cfg, strict=True)
       
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

        _warmup_lr(cfg, optimizer, global_step + steps)

        optimizer.step()


        losses.append(loss_scalar)

        bar.next()

        steps += 1
        if steps >= max_steps:
            break
    bar.finish()

    return float(np.mean(losses)) if losses else 0.0, steps


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


    sum_true, sum_pred = 0.0, 0.0


    score_deltas = []

    with torch.no_grad():
        bar = Bar(f"[Val]   epoch {epoch}", max=len(loader))
        for pbs, data, y in CUDAPrefetcher(loader, device):
            nv_offset = 0
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

            #scores = adapter.greedy_scores_fn(pb[0], data, pred)
            #grd_sol = pb[0].greedy(scores=scores).x
            scores = adapter.greedy_scores_fn(pbs, data, pred)

            _0_idx, _1_idx = (y == 0).cpu().numpy(), (y == 1).cpu().numpy()

            _0_scores_mean, _0_scores_std, _1_scores_mean, _1_scores_std = (
                    np.mean( scores[_0_idx] ),
                    np.std( scores[_0_idx] ),
                    np.mean( scores[_1_idx] ),
                    np.std( scores[_1_idx] ),

                    )


            score_deltas.append(np.mean(scores[_1_idx]) - np.mean(scores[_0_idx]))

            for pb in pbs:
                grd_sol = pb.greedy(scores=scores[nv_offset:nv_offset+pb.nv()]).x

                gap = gap_from_solution(
                    y_true_sum=float(y[nv_offset:nv_offset+pb.nv()].sum().item()),
                    y_pred_sum=float(grd_sol.sum()),
                    maximize=maximize,
                    eps=1e-8,  # match your epoch validation divide-by-y.sum() style :contentReference[oaicite:10]{index=10}
                )

                gaps.append(float(gap))
                nv_offset += pb.nv()
            #sum_true += y.sum().item()
            #sum_pred += grd_sol.sum()

            bar.next()
            steps += 1
            if steps >= max_steps:
                break
        bar.finish()
    #gap = gap_from_solution(
    #            y_true_sum=sum_true,
    #            y_pred_sum=sum_pred,
    #            maximize=maximize,
    #            eps=1e-8,  # match your epoch validation divide-by-y.sum() style :contentReference[oaicite:10]{index=10}
    #        )
    #return (float(np.mean(losses)) if losses else 0.0, 
    #        float(gap) if gap else 0.0, 
    #        {n:float(np.mean(eval_losses[n])) for n in eval_losses.keys()})
    eld = {n:float(np.mean(eval_losses[n])) for n in eval_losses.keys()}
    eld["score_delta"] = np.mean(score_deltas)

    return (float(np.mean(losses)) if losses else 0.0, 
            float(np.mean(gaps)) if gaps else 0.0, 
            eld)


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

    c = 0

    with torch.no_grad():
        bar = Bar("[Test]          ", max=len(loader))
        for pb, data, y in CUDAPrefetcher(loader, device):
            print("data.fg_map shape : ", data.fg_map.shape, data.fg_map[:,1].max())
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
            best_val = None
            dec_sol = None
            n_samples = cfg.get("eval", {}).get("greedy_decimation", {}).get("adapter", {}).get("n_samples", 1)
            for _ in range(n_samples):
                _dec_sol = greedy_decimation(pb[0], backend, padapter)
                dec_val = float(np.dot(pb[0].w, _dec_sol))
                print("sample val ", dec_val)
                if best_val is None:
                    best_val = dec_val
                    dec_sol = _dec_sol
                else:
                    if maximize and (best_val < dec_val):
                        best_val = dec_val
                        dec_sol = _dec_sol
                    if (not maximize) and (best_val > dec_val):
                        best_val = dec_val
                        dec_sol = _dec_sol
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

            print(f"gap : {gap}, dec_gap : {dec_gap}")
            print("\t", float(y.sum().item()), float(dec_sol.sum()))

            gaps.append(gap)
            dec_gaps.append(dec_gap)
            t_pred.append(t2 - t1)
            t_grd.append(t4 - t3)
            t_tot.append(t4 - t0)
            t_pre.append(float(getattr(data, "preprocess_time_s", 0.0)))
            t_gd.append(t6 - t5)
            t_e2e.append(t_pre[-1] + (t4 - t0))

            bar.next()

            #c += 1 
            #if c >= 10:
            #    break

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
    optimizer._base_lrs = [pg["lr"] for pg in optimizer.param_groups]

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
    global_step = 0
    warmup_steps = int(cfg.get("train", {}).get("warmup",{}).get("steps", 0))

    for epoch in range(st.epoch, epochs + 1):
        tr, tr_steps = train_one_epoch(cfg, adapter_train, model, optimizer, loaders.train_loader, device, epoch, global_step)
        global_step += tr_steps
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
        if global_step < warmup_steps:
            pass
        else:
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
    best_path = os.path.join(paths.ckpt_dir, "best.pt")
    if os.path.exists(best_path):
        print(f"[eval] Loading best checkpoint from {best_path}")
        load_checkpoint(best_path, model, optimizer=None, map_location=device)

    if cfg.get("eval", {}).get("enabled", True):
        out_path = final_test_eval(cfg, adapter_eval, model, loaders.test_loader, device, paths.log_dir)
        print(f"[test-eval] saved stats to {out_path}")

    logger.close()

