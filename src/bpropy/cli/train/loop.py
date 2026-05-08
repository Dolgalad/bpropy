from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
from pathlib import Path

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

def ensure_seeded_run_name(cfg: Dict[str, Any]) -> str:
    run_cfg = cfg.setdefault("run", {})
    seed = int(cfg.get("seed", 0))
    suffix = f"_seed{seed}"

    base = run_cfg.get("run_name") or run_cfg.get("name") or cfg.get("run_name") or "run"

    if not str(base).endswith(suffix):
        base = f"{base}{suffix}"

    run_cfg["run_name"] = base
    run_cfg["name"] = base
    cfg["run_name"] = base

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
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    losses = []
    score_deltas = []
    steps = 0
    max_steps = int(cfg["train"]["steps_per_epoch"])

    grad_accum_steps = int(cfg.get("train", {}).get("grad_accum_steps", 1))
    grad_accum_steps = max(1, grad_accum_steps)

    update_steps = 0
    optimizer.zero_grad(set_to_none=True)

    bar = Bar(f"[Train] epoch {epoch}", max=len(loader))

    score_deltas = []
    theta_deltas = []

    for data, y in CUDAPrefetcher(loader, device):
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred = adapter.forward_fn(model, data)
            spec = adapter.loss_spec_fn(adapter, data, pred, y, cfg)

        
        if spec.backward_fn is not None:
            raise RuntimeError(
                "Gradient accumulation is not implemented for LossSpec.backward_fn paths yet. "
                "Use the standard loss path or adapt backward_fn to support scaled loss."
            )
        
        loss = compute_loss_from_spec(spec, loss_registry=LOSS_REGISTRY)
        loss_scalar = float(loss.detach().cpu().item())
        scaler.scale(loss / grad_accum_steps).backward()
        
        is_update_step = ((steps + 1) % grad_accum_steps == 0)
       
        if is_update_step:
            clip = cfg["train"].get("grad_clip_norm")
            if clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
        
            _warmup_lr(cfg, optimizer, global_step + update_steps)
        
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
            update_steps += 1

        losses.append(loss_scalar)

        # debugging
        NV = (data.bipartite == 0).sum().item()
        if isinstance(pred, tuple):
            scores = pred[0].mu_v.reshape(NV,2)[:,1].detach().cpu().numpy()

            _0_idx, _1_idx = (y == 0).cpu().numpy(), (y == 1).cpu().numpy()

            _0_scores_mean, _0_scores_std, _1_scores_mean, _1_scores_std = (
                    np.mean( scores[_0_idx] ),
                    np.std( scores[_0_idx] ),
                    np.mean( scores[_1_idx] ),
                    np.std( scores[_1_idx] ),

                    )
            _0_theta_mean = pred[1][data.y_ss == 0].mean().item()
            _1_theta_mean = pred[1][data.y_ss == 1].mean().item()

            mu_v = pred[0].mu_v.reshape(NV,2)

            score_deltas.append(_1_scores_mean - _0_scores_mean)
            theta_deltas.append(_1_theta_mean - _0_theta_mean)

            if torch.any(mu_v < 0):
                print("negative probabilities")
                break

        bar.next()
        bar.suffix = f" loss: {np.mean(losses):.3f} sgap: {np.mean(score_deltas):.3f} tgap: {np.mean(theta_deltas):.3f}"

        steps += 1
        if steps >= max_steps:
            break

    if steps>0 and steps % grad_accum_steps != 0:
        clip = cfg["train"].get("grad_clip_norm")
        if clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
    
        _warmup_lr(cfg, optimizer, global_step + update_steps)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    
        update_steps += 1
    bar.finish()
    print("mean loss : ", np.mean(losses))

    return float(np.mean(losses)) if losses else 0.0, update_steps


def validate_one_epoch(
    cfg: Dict[str, Any],
    adapter: MethodAdapter,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    epoch: int,
    use_amp: bool = False,
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

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
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


def objective_value(pb, sol: np.ndarray) -> float:
    """Objective value for a problem solution.

    Prefer the problem weights/objective when available. Fall back to the
    number of selected variables to keep the old unweighted behavior valid.
    """
    if isinstance(sol, torch.Tensor):
        s = sol.detach().cpu().numpy()
    elif hasattr(sol, "x"):
        s = np.asarray(sol.x, dtype=np.float64)
    else:
        s = np.asarray(sol, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)

    if hasattr(pb, "w"):
        return float(np.dot(np.asarray(pb.w, dtype=np.float64), s))
    if hasattr(pb, "objective"):
        return float(pb.objective(s))
    return float(s.sum())


def relative_objective_gap(
    obj_true: float,
    obj_pred: float,
    maximize: bool,
    eps: float = 1e-8,
) -> float:
    denom = max(abs(obj_true), eps)
    if maximize:
        return (obj_true - obj_pred) / denom
    return (obj_pred - obj_true) / denom


def _solution_array(sol: Any) -> np.ndarray:
    if isinstance(sol, torch.Tensor):
        return sol.detach().cpu().numpy()
    if hasattr(sol, "x"):
        return np.asarray(sol.x)
    return np.asarray(sol)


def _ensure_greedy_decimation_cfg(cfg: Dict[str, Any]) -> None:
    cfg.setdefault("eval", {})
    cfg["eval"].setdefault("greedy_decimation", {})
    cfg["eval"]["greedy_decimation"].setdefault("args", {})
    cfg["eval"]["greedy_decimation"].setdefault(
        "adapter", {"pick_mode": "min_entropy", "value_mode": "argmax"}
    )
    cfg["eval"]["greedy_decimation"]["adapter"].setdefault("pick_mode", "min_entropy")
    cfg["eval"]["greedy_decimation"]["adapter"].setdefault("value_mode", "argmax")


def ms(x):
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return 0.0, 0.0
    if a.size == 1:
        return float(a.mean()), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def summarize_method(d: Dict[str, list]) -> Dict[str, float]:
    out = {}
    for k, vals in d.items():
        mean, std = ms(vals)
        out[f"{k}_mean"] = mean
        out[f"{k}_std"] = std
    return out


def build_eval_loss_stats(eval_losses: Dict[str, list]) -> Dict[str, Dict[str, float]]:
    out = {}
    for name, vals in eval_losses.items():
        mean, std = ms(vals)
        out[name] = {"mean": mean, "std": std}
    return out


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _build_test_payload(
    *,
    status: str,
    instance_stats: list,
    summary_lists: Dict[str, Dict[str, list]],
    eval_losses: Dict[str, list],
    n_ss: list,
) -> Dict[str, Any]:
    n_ss_mean, n_ss_std = ms(n_ss)
    return {
        "summary": {
            "count": len(instance_stats),
            "status": status,
            "methods": {
                "greedy_pred": summarize_method(summary_lists["greedy_pred"]),
                "entropy_greedy": summarize_method(summary_lists["entropy_greedy"]),
                "guided_decimation": summarize_method(summary_lists["guided_decimation"]),
            },
            "eval_losses": build_eval_loss_stats(eval_losses),
            "num_ss_mean": n_ss_mean,
            "num_ss_std": n_ss_std,
        },
        "instances": instance_stats,
    }


def _run_best_of_n_decimation(
    *,
    pb,
    backend,
    padapter,
    n_samples: int,
    maximize: bool,
) -> np.ndarray:
    best_val = None
    best_sol = None
    for _ in range(max(1, int(n_samples))):
        sol = greedy_decimation(pb, backend, padapter)
        sol_arr = _solution_array(sol)
        val = objective_value(pb, sol_arr)
        if best_val is None or (maximize and val > best_val) or ((not maximize) and val < best_val):
            best_val = val
            best_sol = sol_arr
    return np.asarray(best_sol)


def final_test_eval(
    cfg: Dict[str, Any],
    adapter: MethodAdapter,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    log_dir: str,
    output_json: Optional[str] = None,
) -> str:
    """Final test evaluation with per-instance JSON logging.

    Logged solve methods:
      1. greedy_pred: greedy projection guided by one forward pass scores.
      2. entropy_greedy: greedy_decimation with min_entropy selection and
         run_every = infinity, so inference is performed only once.
      3. guided_decimation: greedy_decimation with min_entropy selection and
         run_every = 1, so marginals/scores are refreshed after each fixing step.

    The JSON schema mirrors the flow test evaluator: it writes a running JSON
    after each instance and a completed JSON at the end.
    """
    model.eval()

    eval_loss_names = [e["name"] for e in cfg.get("eval", {}).get("losses", [])]
    eval_losses = {n: [] for n in eval_loss_names}
    n_ss = []
    maximize = bool(cfg.get("problem", {}).get("maximize", True))
    test_limit = int(cfg.get("eval", {}).get("test_limit", 20000))

    ensure_seeded_run_name(cfg)

    if output_json is not None:
        out_path = output_json
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    else:
        paths = make_run_paths(cfg)
        log_dir = paths.log_dir
        os.makedirs(log_dir, exist_ok=True)
        out_path = os.path.join(log_dir, "test_eval.json")

    print("IN TEST logdir = ", os.path.dirname(out_path) or log_dir)

    _ensure_greedy_decimation_cfg(cfg)
    gd_cfg = cfg["eval"]["greedy_decimation"]
    n_samples = int(gd_cfg.get("adapter", {}).get("n_samples", 1))

    instance_stats = []
    summary_lists = {
        "greedy_pred": {"obj": [], "gap": [], "forward_t": [], "decode_t": [], "total_t": []},
        "entropy_greedy": {"obj": [], "gap": [], "forward_t": [], "decode_t": [], "total_t": []},
        "guided_decimation": {"obj": [], "gap": [], "forward_t": [], "decode_t": [], "total_t": []},
    }

    with torch.no_grad():
        bar = Bar("[Test]          ", max=len(loader))

        for idx, (pb, data, y) in enumerate(CUDAPrefetcher(loader, device)):
            pb0 = pb[0] if isinstance(pb, (list, tuple)) else pb
            stem = None
            if hasattr(loader, "dataset") and hasattr(loader.dataset, "stems"):
                try:
                    stem = Path(loader.dataset.stems[idx]).name
                except Exception:
                    stem = None

            pre_t = float(getattr(data, "preprocess_time_s", 0.0))
            ref_val = objective_value(pb0, y)

            # ----- shared forward pass -----
            _sync(device)
            t0 = time.perf_counter()
            pred = adapter.forward_fn(model, data)
            _sync(device)
            t1 = time.perf_counter()
            shared_forward_t = t1 - t0

            scores = adapter.greedy_scores_fn(pb, data, pred)
            scores_np = np.asarray(scores)

            # ----- greedy_pred -----
            _sync(device)
            t2 = time.perf_counter()
            greedy_pred_sol = pb0.greedy(scores=scores_np).x
            _sync(device)
            t3 = time.perf_counter()
            greedy_pred_decode_t = t3 - t2
            greedy_pred_total_t = pre_t + shared_forward_t + greedy_pred_decode_t

            # ----- entropy_greedy: min_entropy, run_every = infinity -----
            gd_cfg["args"]["run_every"] = 1000000
            gd_cfg["adapter"]["pick_mode"] = "min_entropy"
            gd_cfg["adapter"]["value_mode"] = "argmax"
            gd_cfg["args"]["max_iter"] = 10
            gd_cfg["args"]["damping"] = 0.5
            gd_cfg["args"]["neg_inf"] = -100.

            _sync(device)
            t4 = time.perf_counter()
            backend = adapter.decimation_backend_fn(pb0, data, pred, cfg, device=device)
            padapter = adapter.decimation_adapter_fn(pb0, cfg)
            entropy_greedy_sol = _run_best_of_n_decimation(
                pb=pb0,
                backend=backend,
                padapter=padapter,
                n_samples=n_samples,
                maximize=maximize,
            )
            _sync(device)
            t5 = time.perf_counter()
            entropy_greedy_decode_t = t5 - t4
            entropy_greedy_total_t = pre_t + shared_forward_t + entropy_greedy_decode_t

            # ----- guided_decimation: min_entropy, run_every = 1 -----
            gd_cfg["args"]["run_every"] = 1
            gd_cfg["adapter"]["pick_mode"] = "min_entropy"
            gd_cfg["adapter"]["value_mode"] = "argmax"
            gd_cfg["args"]["max_iter"] = 10
            gd_cfg["args"]["damping"] = 0.5
            gd_cfg["args"]["neg_inf"] = -100.

            _sync(device)
            t6 = time.perf_counter()
            backend = adapter.decimation_backend_fn(pb0, data, pred, cfg, device=device)
            padapter = adapter.decimation_adapter_fn(pb0, cfg)
            guided_decimation_sol = _run_best_of_n_decimation(
                pb=pb0,
                backend=backend,
                padapter=padapter,
                n_samples=n_samples,
                maximize=maximize,
            )
            _sync(device)
            t7 = time.perf_counter()
            guided_decimation_decode_t = t7 - t6
            guided_decimation_total_t = pre_t + shared_forward_t + guided_decimation_decode_t

            # ----- loss values from shared forward -----
            spec = adapter.loss_spec_fn(adapter, data, pred, y, cfg)
            loss = compute_loss_from_spec(spec, loss_registry=LOSS_REGISTRY)

            for eval_loss_cfg in cfg.get("eval", {}).get("losses", []):
                name = eval_loss_cfg["name"]
                args = eval_loss_cfg.get("args", {})
                val = LOSS_REGISTRY[name](pred, y, data=data, **args)
                eval_losses[name].append(float(val.detach().cpu().item()))

            if hasattr(data, "lbp_edges"):
                n_ss += data.lbp_edges.n_ss.detach().cpu().tolist()
            if hasattr(data, "nmf_edges"):
                n_ss += data.nmf_edges.n_ss.detach().cpu().tolist()

            greedy_pred_val = objective_value(pb0, greedy_pred_sol)
            entropy_greedy_val = objective_value(pb0, entropy_greedy_sol)
            guided_decimation_val = objective_value(pb0, guided_decimation_sol)

            greedy_pred_gap = relative_objective_gap(ref_val, greedy_pred_val, maximize=maximize)
            entropy_greedy_gap = relative_objective_gap(ref_val, entropy_greedy_val, maximize=maximize)
            guided_decimation_gap = relative_objective_gap(ref_val, guided_decimation_val, maximize=maximize)

            instance_record = {
                "instance_index": idx,
                "instance_stem": stem,
                "reference_objective": float(ref_val),
                "loss": float(loss.detach().cpu().item()),
                "preprocess_time_s": float(pre_t),
                "greedy_pred": {
                    "objective": float(greedy_pred_val),
                    "gap": float(greedy_pred_gap),
                    "forward_time_s": float(shared_forward_t),
                    "decode_time_s": float(greedy_pred_decode_t),
                    "total_time_s": float(greedy_pred_total_t),
                },
                "entropy_greedy": {
                    "objective": float(entropy_greedy_val),
                    "gap": float(entropy_greedy_gap),
                    "forward_time_s": float(shared_forward_t),
                    "decode_time_s": float(entropy_greedy_decode_t),
                    "total_time_s": float(entropy_greedy_total_t),
                    "run_every": 1000000,
                    "pick_mode": "min_entropy",
                    "value_mode": "argmax",
                },
                "guided_decimation": {
                    "objective": float(guided_decimation_val),
                    "gap": float(guided_decimation_gap),
                    "forward_time_s": float(shared_forward_t),
                    "decode_time_s": float(guided_decimation_decode_t),
                    "total_time_s": float(guided_decimation_total_t),
                    "run_every": 1,
                    "pick_mode": "min_entropy",
                    "value_mode": "argmax",
                },
            }
            instance_stats.append(instance_record)

            for name, val, gap, dec_t, total_t in [
                ("greedy_pred", greedy_pred_val, greedy_pred_gap, greedy_pred_decode_t, greedy_pred_total_t),
                ("entropy_greedy", entropy_greedy_val, entropy_greedy_gap, entropy_greedy_decode_t, entropy_greedy_total_t),
                ("guided_decimation", guided_decimation_val, guided_decimation_gap, guided_decimation_decode_t, guided_decimation_total_t),
            ]:
                summary_lists[name]["obj"].append(float(val))
                summary_lists[name]["gap"].append(float(gap))
                summary_lists[name]["forward_t"].append(float(shared_forward_t))
                summary_lists[name]["decode_t"].append(float(dec_t))
                summary_lists[name]["total_t"].append(float(total_t))

            partial_payload = _build_test_payload(
                status="running",
                instance_stats=instance_stats,
                summary_lists=summary_lists,
                eval_losses=eval_losses,
                n_ss=n_ss,
            )
            atomic_write_json(out_path, partial_payload)

            print()
            print(
                f"[test {idx:04d}] ref={ref_val:.4f} | "
                f"GRD gap={greedy_pred_gap:.4f} t={greedy_pred_total_t:.4f}s | "
                f"EGG gap={entropy_greedy_gap:.4f} t={entropy_greedy_total_t:.4f}s | "
                f"GD gap={guided_decimation_gap:.4f} t={guided_decimation_total_t:.4f}s"
            )

            bar.next()
            if len(instance_stats) >= test_limit:
                break

        bar.finish()

    final_payload = _build_test_payload(
        status="completed",
        instance_stats=instance_stats,
        summary_lists=summary_lists,
        eval_losses=eval_losses,
        n_ss=n_ss,
    )
    atomic_write_json(out_path, final_payload)
    return out_path


def run_training(cfg: Dict[str, Any]) -> None:
    # seeds
    torch.manual_seed(int(cfg.get("seed", 0)))
    np.random.seed(int(cfg.get("seed", 0)))

    print("SEED value = ", cfg.get("seed", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run dirs + logger
    ensure_seeded_run_name(cfg)

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
    use_amp = bool(cfg.get("train", {}).get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    for epoch in range(st.epoch, epochs + 1):
        tr, tr_steps = train_one_epoch(cfg, adapter_train, model, optimizer, loaders.train_loader, device, epoch, global_step, scaler, use_amp)
        global_step += tr_steps
        va, gap, eloss = validate_one_epoch(cfg, adapter_train, model, loaders.val_loader, device, epoch, use_amp=use_amp)

        logger.add_scalars("loss", {"train": tr, "val": va}, epoch)
        logger.add_scalars("lr", {f"pg{k}":pg["lr"] for k,pg in enumerate(optimizer.param_groups)}, epoch)
        logger.add_scalars("gap", {"val": gap}, epoch)
        epoch_out = f"Epoch {epoch:03d} | train {tr:.4f} | val {va:.4f} | gap {gap:.4f}"
        if len(eloss):
            eloss_out = " | ".join(f"{n} {eloss[n]:.4f}" for n in eloss.keys())
            epoch_out += " | " + eloss_out
            print("LR = ", str(list(optimizer.param_groups)[0]["lr"]))
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

