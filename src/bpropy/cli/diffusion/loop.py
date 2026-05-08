from __future__ import annotations
import json
import copy
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple, TypedDict
from pathlib import Path

import numpy as np
import torch
from progress.bar import Bar

from bpropy.datasets.prefetcher import CUDAPrefetcher
from bpropy.gd.greedy_decimation import greedy_decimation

from .checkpoint import EpochState, load_checkpoint, save_checkpoint
from .data import make_loaders
from .logging import TrainLogger, make_run_paths
from .metrics import EvalStats, _sync, gap_from_solution, write_test_eval_json
from .schedulers import build_scheduler, _apply_lr_schedulers

from bpropy.models import MODEL_REGISTRY
from bpropy.losses import LOSS_REGISTRY
from bpropy.cli.train.method_registry import METHOD_REGISTRY

class NoiseSchedule(TypedDict):
    betas: torch.Tensor
    alpha: torch.Tensor
    alpha_bar: torch.Tensor


# ============================================================
# Existing helpers kept as-is
# ============================================================

def build_model(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    name = cfg["model"]["name"]
    args = cfg["model"].get("args", {})
    Model = MODEL_REGISTRY[name]
    model = Model(**args).to(device)

    init_from = cfg["model"].get("init_from")
    if init_from:
        print("Loading from", init_from)
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
    frac = (global_step + 1) / warmup_steps
    factor = start_factor + frac * (1.0 - start_factor)

    for pg, base_lr in zip(optimizer.param_groups, optimizer._base_lrs):
        pg["lr"] = base_lr * factor


# ============================================================
# Diffusion helpers
# ============================================================

def linear_noise_schedule(
    num_steps: int,
    beta_start: float = 0.02,
    beta_end: float = 0.50,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    beta_t = probability of corrupting a variable at step t.
    We use the cumulative corruption probability directly.
    """
    betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
    return betas.clamp(0.0, 1.0)


def cosine_noise_schedule(
    num_steps: int,
    beta_min: float = 0.01,
    beta_max: float = 0.50,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Smooth schedule from low to high corruption.
    """
    t = torch.linspace(0.0, 1.0, num_steps, device=device)
    w = 0.5 * (1.0 - torch.cos(torch.pi * t))
    betas = beta_min + (beta_max - beta_min) * w
    return betas.clamp(0.0, 1.0)


def build_noise_schedule(cfg: Dict[str, Any], device: torch.device) -> NoiseSchedule:
    dcfg = cfg.get("diffusion", {})
    num_steps = int(dcfg.get("num_steps", 16))
    name = dcfg.get("schedule", "linear").lower()

    if name == "linear":
        betas = linear_noise_schedule(
            num_steps=num_steps,
            beta_start=float(dcfg.get("beta_start", 0.02)),
            beta_end=float(dcfg.get("beta_end", 0.50)),
            device=device,
        )
    elif name == "cosine":
        betas = cosine_noise_schedule(
            num_steps=num_steps,
            beta_min=float(dcfg.get("beta_start", 0.01)),
            beta_max=float(dcfg.get("beta_end", 0.50)),
            device=device,
        )
    else:
        raise ValueError(f"Unknown diffusion.schedule '{name}'")

    alpha = 1.0 - betas
    alpha_bar = torch.cumprod(alpha, dim=0)

    return {
        "betas": betas,
        "alpha": alpha,
        "alpha_bar": alpha_bar,
    }

def _num_variable_nodes(data: Any) -> int:
    if hasattr(data, "node_types") and ("variable" in getattr(data, "node_types", [])):
        return int(data["variable"].num_nodes)
    return int((data.bipartite == 0).sum().item())


def _variable_mask(data: Any) -> torch.Tensor:
    if hasattr(data, "node_types") and ("variable" in getattr(data, "node_types", [])):
        raise NotImplementedError(
            "This simple implementation assumes homogeneous data with data.x "
            "and data.bipartite. For hetero data, append the column to "
            "data['variable'].x instead."
        )
    return (data.bipartite == 0)


def sample_timesteps_for_variables(
    y: torch.Tensor,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample one timestep per variable.
    Shape: [num_variables]
    """
    return torch.randint(0, num_steps, (y.numel(),), device=device)

def apply_binary_noise(
    y: torch.Tensor,
    t: torch.Tensor,
    noise_schedule: NoiseSchedule,
    mode: str = "replace",
    ) -> torch.Tensor:
    """
    Pseudo-diffusion corruption for binary labels.

    y: binary tensor in {0,1}, shape [N]
    t: timestep tensor, shape [N]

    replace mode:
    keep clean value with probability alpha_bar[t],
    otherwise replace by Bernoulli(0.5)

    flip mode:
    flip bit with probability betas[t]
    """
    y = y.float()

    if mode == "replace":
        keep_prob = noise_schedule["alpha_bar"][t]
        keep = torch.bernoulli(keep_prob).bool()
        noise = torch.bernoulli(0.5 * torch.ones_like(y))
        y_t = torch.where(keep, y, noise)

    elif mode == "flip":
        flip_prob = noise_schedule["betas"][t]
        flip = torch.bernoulli(flip_prob).bool()
        y_t = torch.where(flip, 1.0 - y, y)

    else:
        raise ValueError(f"Unknown diffusion corruption mode '{mode}'")

    return y_t

def append_noisy_label_feature(
    data: Any,
    y_t: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    num_steps: Optional[int] = None,
) -> Any:
    """
    Appends y_t as one additional feature column to variable nodes only.
    Non-variable nodes get 0 in that column.

    Also optionally appends normalized timestep t / (num_steps-1) as another column.
    """
    data = copy.copy(data)

    var_mask = _variable_mask(data)
    x = data.x
    N = x.size(0)
    device = x.device
    dtype = x.dtype

    y_col = torch.zeros((N, 1), device=device, dtype=dtype)
    y_col[var_mask, 0] = y_t.to(dtype)

    cols = [x, y_col]

    if t is not None and num_steps is not None:
        t_col = torch.zeros((N, 1), device=device, dtype=dtype)
        denom = max(num_steps - 1, 1)
        t_col[var_mask, 0] = (t.float() / denom).to(dtype)
        cols.append(t_col)

    data.x = torch.cat(cols, dim=1)
    data.y_t = y_t
    if t is not None:
        data.t_var = t
    return data


def make_noisy_batch(
    data: Any,
    y: torch.Tensor,
    noise_schedule: NoiseSchedule,
    cfg: Dict[str, Any],
    *,
    fixed_t: Optional[int] = None,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor]:
    """
    Returns:
    noisy_data: copy of data with extra feature column(s)
    y_t:        noisy labels
    t:          timestep per variable
    """
    dcfg = cfg.get("diffusion", {})
    num_steps = int(dcfg.get("num_steps", noise_schedule["betas"].numel()))
    mode = dcfg.get("corruption", "replace")
    append_t = bool(dcfg.get("append_t_feature", True))

    if fixed_t is None:
        t = sample_timesteps_for_variables(y, num_steps, y.device)
    else:
        t = torch.full_like(y, fill_value=int(fixed_t), dtype=torch.long, device=y.device)

    y_t = apply_binary_noise(y, t, noise_schedule, mode=mode)
    noisy_data = append_noisy_label_feature(
        data,
        y_t=y_t,
        t=t if append_t else None,
        num_steps=num_steps if append_t else None,
    )
    return noisy_data, y_t, t

# ============================================================
# Updated loops
# ============================================================

def train_one_epoch(
    cfg: Dict[str, Any],
    adapter,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    device: torch.device,
    epoch: int,
    global_step: int,
    noise_schedule: NoiseSchedule,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    clip: Optional[float],
) -> Tuple[float, int]:
    model.train()
    losses = []
    steps = 0
    max_steps = int(cfg["train"]["steps_per_epoch"])

    grad_accum_steps = int(cfg.get("train", {}).get("grad_accum_steps", 1))
    grad_accum_steps = max(1, grad_accum_steps)
    
    update_steps = 0
    optimizer.zero_grad(set_to_none=True)

    bar = Bar(f"[Train] epoch {epoch}", max=len(loader))

    for data, y in CUDAPrefetcher(loader, device):

        noisy_data, y_t, t = make_noisy_batch(data, y, noise_schedule, cfg)

        with torch.amp.autocast(device.type, enabled=use_amp):
            pred = adapter.forward_fn(model, noisy_data)
            spec = adapter.loss_spec_fn(adapter, noisy_data, pred, y, cfg)
    
        if spec.backward_fn is not None:
            raise RuntimeError(
                "Gradient accumulation is not implemented for LossSpec.backward_fn paths yet."
            )
        
        loss = compute_loss_from_spec(spec, loss_registry=LOSS_REGISTRY)
        loss_scalar = float(loss.detach().cpu().item())
        
        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()
        
        is_update_step = ((steps + 1) % grad_accum_steps == 0)
        
        if is_update_step:
            if clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
            _warmup_lr(cfg, optimizer, global_step + update_steps)
        
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
            update_steps += 1
        losses.append(loss_scalar)

        bar.next()
        bar.suffix = f" loss: {np.mean(losses):.4f}"
        steps += 1

        if steps >= max_steps:
            break
    if steps > 0 and steps % grad_accum_steps != 0:
        if clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    
        _warmup_lr(cfg, optimizer, global_step + update_steps)
    
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    
        update_steps += 1
    bar.finish()
    return float(np.mean(losses)) if losses else 0.0, update_steps

def get_validation_timesteps(cfg: Dict[str, Any], noise_schedule: NoiseSchedule) -> list[int]:
    if "val_timesteps" in cfg.get("eval", {}):
        return [int(t) for t in cfg["eval"]["val_timesteps"]]

    num_steps = int(cfg.get("diffusion", {}).get("num_steps", noise_schedule["betas"].numel()))
    return [0, num_steps // 4, num_steps // 2, num_steps - 1]

def validate_one_epoch(
    cfg: Dict[str, Any],
    adapter,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    epoch: int,
    noise_schedule: NoiseSchedule,
    use_amp: bool = False,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Validation for the pseudo-diffusion denoiser.

    Each clean solution is corrupted at one or more fixed timesteps, and the model
    is evaluated on one-shot denoising from the corrupted assignment. The reported
    gap is obtained by decoding greedy scores from the denoiser output; this is not
    a reverse diffusion rollout.
    """
    model.eval()
    losses = []
    eval_loss_names = [e["name"] for e in cfg.get("eval", {}).get("losses", [])]
    eval_losses = {n: [] for n in eval_loss_names}
    gaps = []
    score_deltas = []

    max_steps = int(cfg["train"]["val_steps"])
    maximize = bool(cfg.get("problem", {}).get("maximize", True))
    steps = 0

    val_timesteps = get_validation_timesteps(cfg, noise_schedule)

    with torch.no_grad():
        bar = Bar(f"[Val]   epoch {epoch}", max=len(loader))

        for batch in CUDAPrefetcher(loader, device):
            pbs, data, y = batch  # keep if this is your actual val format

            batch_losses = []
            batch_gaps = []
            batch_score_deltas = []

            for fixed_t in val_timesteps:
                noisy_data, y_t, t = make_noisy_batch(
                    data, y, noise_schedule, cfg, fixed_t=fixed_t
                )

                with torch.amp.autocast(device.type, enabled=use_amp):
                    pred = adapter.forward_fn(model, noisy_data)
                    spec = adapter.loss_spec_fn(adapter, noisy_data, pred, y, cfg)
                loss = compute_loss_from_spec(spec, loss_registry=LOSS_REGISTRY)

                batch_losses.append(float(loss.detach().cpu().item()))

                scores = adapter.greedy_scores_fn(pbs, noisy_data, pred)

                y_np = y.detach().cpu().numpy()
                _0_idx, _1_idx = (y_np == 0), (y_np == 1)
                if _0_idx.any() and _1_idx.any():
                    batch_score_deltas.append(float(np.mean(scores[_1_idx]) - np.mean(scores[_0_idx])))

                nv_offset = 0
                per_t_gaps = []
                for pb in pbs:
                    nv = pb.nv()
                    grd_sol = pb.greedy(scores=scores[nv_offset:nv_offset + nv]).x
                    gap = gap_from_solution(
                        #y_true_sum=float(y[nv_offset:nv_offset + nv].sum().item()),
                        y_true_sum=np.dot(y[nv_offset:nv_offset + nv].detach().cpu().numpy(), pb.w),
                        #y_pred_sum=float(grd_sol.sum()),
                        y_pred_sum=np.dot(grd_sol, pb.w),

                        maximize=maximize,
                        eps=1e-8,
                    )
                    per_t_gaps.append(float(gap))
                    nv_offset += nv

                batch_gaps.extend(per_t_gaps)

            losses.append(float(np.mean(batch_losses)))
            gaps.append(float(np.mean(batch_gaps)))
            if batch_score_deltas:
                score_deltas.append(float(np.mean(batch_score_deltas)))
            bar.next()
            steps += 1
            if steps >= max_steps:
                break

        bar.finish()

    eld = {n: float(np.mean(v)) for n, v in eval_losses.items() if len(v)}
    eld["score_delta"] = float(np.mean(score_deltas)) if score_deltas else 0.0

    return (
        float(np.mean(losses)) if losses else 0.0,
        float(np.mean(gaps)) if gaps else 0.0,
        eld,
    )

def objective_value(pb, sol: np.ndarray) -> float:
    if isinstance(sol, torch.Tensor):
        s = sol.detach().cpu().numpy()
    else:
        s = np.asarray(sol, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)

    if hasattr(pb, "w"):
        return float(np.dot(pb.w, s))
    if hasattr(pb, "objective"):
        return float(pb.objective(s))
    raise AttributeError("Problem object has no recognized objective accessor.")

def relative_objective_gap(
    obj_true: float,
    obj_pred: float,
    maximize: bool,
    eps: float = 1e-8,
    ) -> float:
    denom = max(abs(obj_true), eps)
    if maximize:
        return (obj_true - obj_pred) / denom
    else:
        return (obj_pred - obj_true) / denom

def _ensure_greedy_decimation_cfg(cfg: Dict[str, Any]) -> None:
    cfg.setdefault("eval", {})
    cfg["eval"].setdefault("greedy_decimation", {})
    cfg["eval"]["greedy_decimation"].setdefault("args", {})
    cfg["eval"]["greedy_decimation"].setdefault("adapter",{"pick_mode":"min_entropy","value_mode":"argmax"})

def diffusion_decoding(
    model,
    y_init,
    noise_schedule,
    eval_t,
    cfg,
    adapter,
    data,
    pb,
    device,
    sample: bool = True,
    return_pred: bool = False
    ):
    """
    Iterative pseudo-diffusion decoding.

    Starts from a single initial state y_init and repeatedly denoises from t=eval_t
    down to t=0. No additional corruption is applied inside the reverse loop.

    IMPORTANT:
    adapter.greedy_scores_fn must return probabilities in [0, 1] for stochastic
    sampling to be meaningful. If it returns logits or generic scores, convert
    them explicitly before sampling.
    """
    y_cur = y_init.clone().float().to(device)
    num_steps = int(cfg.get("diffusion", {}).get("num_steps", noise_schedule["betas"].numel()))
    append_t = bool(cfg.get("diffusion", {}).get("append_t_feature", True))

    for diff_step in range(int(eval_t), -1, -1):
        t_var = torch.full_like(y_cur, diff_step, dtype=torch.long, device=device)

        noisy_data = append_noisy_label_feature(
            data,
            y_t=y_cur,
            t=t_var if append_t else None,
            num_steps=num_steps if append_t else None,
        )

        pred = adapter.forward_fn(model, noisy_data)
        scores = adapter.greedy_scores_fn(pb, noisy_data, pred)
        scores = torch.as_tensor(scores, device=device, dtype=torch.float32)

        # Safer handling:
        # - if already probabilities, keep them
        # - otherwise map logits/scores to probabilities with sigmoid
        if torch.all((scores >= 0.0) & (scores <= 1.0)):
            probs = scores
        else:
            probs = torch.sigmoid(scores)

        probs = probs.clamp(0.0, 1.0)

        if sample:
            y_cur = torch.bernoulli(probs)
        else:
            y_cur = (probs >= 0.5).float()
    if return_pred:
        return y_cur, pred

    return y_cur

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

def build_partial_eval_payload(
    eval_t: int,
    instance_stats: list,
    summary_lists: Dict[str, Dict[str, list]],
    eval_losses: Dict[str, list],
    extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    payload = {
        "summary": {
            "count": len(instance_stats),
            "eval_t": int(eval_t),
            "methods": {
                "grd": summarize_method(summary_lists["grd"]),
                "gdinf": summarize_method(summary_lists["gdinf"]),
                "gd1": summarize_method(summary_lists["gd1"]),
                "diff": summarize_method(summary_lists["diff"]),
                "diff_rand": summarize_method(summary_lists["diff_rand"]),
            },
            "eval_losses": build_eval_loss_stats(eval_losses),
            },
            "instances": instance_stats,
        }
    if extra:
        payload["summary"].update(extra)
    return payload

def final_test_eval(
    cfg: Dict[str, Any],
    adapter,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    log_dir: str,
    noise_schedule: torch.Tensor,
    output_json: Optional[str] = None,
) -> str:
    """
    Test-time: use a fixed timestep so all instances are evaluated at the same noise level.
    This is a one-shot denoising test, not a full reverse diffusion chain.
    """
    model.eval()

    losses, gaps, dec_gaps = [], [], []
    eval_loss_names = [e["name"] for e in cfg.get("eval", {}).get("losses", [])]
    eval_losses = {n: [] for n in eval_loss_names}


    maximize = bool(cfg.get("problem", {}).get("maximize", True))
    num_steps = int(cfg.get("diffusion", {}).get("num_steps", len(noise_schedule)))
    eval_t = int(cfg.get("eval", {}).get("diffusion_t", num_steps - 1))
    eval_t = max(0, min(eval_t, len(noise_schedule["betas"])-1))

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
    c = 0
    test_limit = int(cfg.get("eval", {}).get("test_limit", 20000))

    _ensure_greedy_decimation_cfg(cfg)

    ############ Statistics #############
    n_ss = []
    instance_stats = []

    summary_lists = {
            "grd": {"obj": [], "gap": [], "forward_t": [], "decode_t": [], "total_t": []},
            "gdinf": {"obj": [], "gap": [], "forward_t": [], "decode_t": [], "total_t": []},
            "gd1": {"obj": [], "gap": [], "forward_t": [], "decode_t": [], "total_t": []},
            "diff": {"obj": [], "gap": [], "decode_t": [], "project_t": [], "total_t": [], "sample_feasible": [], "sample_val": [], "sample_gap": []},
            "diff_rand": {"obj": [], "gap": [], "decode_t": [], "project_t": [], "total_t": [], "sample_feasible": [], "sample_val": [], "sample_gap": []},
            "gd1_pure_noise": {"obj": [], "gap": [], "forward_t": [],"decode_t": [],"total_t": []},
            "diff_egg": {"obj": [],"gap": [], "forward_t": [], "decode_t": [], "total_t": [] },
            "diff_gd": {"obj": [],"gap": [], "forward_t": [], "decode_t": [], "total_t": [] },

            }

    with torch.no_grad():
        bar = Bar("[Test]          ", max=len(loader))

        diff_gaps = []

        for idx, (pb, data, y) in enumerate(CUDAPrefetcher(loader, device)):
            stem = Path(loader.dataset.stems[idx]).name

            ############## Shared corrupted input for all "from corrupted reference" methods #################
            t_init = torch.full_like(y, eval_t, dtype=torch.long, device=device)
            y_init_corr = apply_binary_noise(
                y.float(),
                t_init,
                noise_schedule,
                mode=cfg.get("diffusion", {}).get("corruption", "replace"),
            )

            noisy_data_shared = append_noisy_label_feature(
                data,
                y_t=y_init_corr,
                t=t_init if cfg.get("diffusion", {}).get("append_t_feature", True) else None,
                num_steps=num_steps if cfg.get("diffusion", {}).get("append_t_feature", True) else None,
            )
            pre_t = float(getattr(noisy_data_shared, "preprocess_time_s", 0.0))

            ############## Single shared forward pass #################
            _sync(device)
            t0 = time.perf_counter()
            pred_shared = adapter.forward_fn(model, noisy_data_shared)
            _sync(device)
            t1 = time.perf_counter()
            shared_forward_t = t1 - t0

            ############## Simple score driven greedy #################
            scores = adapter.greedy_scores_fn(pb, noisy_data_shared, pred_shared)
            scores_np = np.asarray(scores)
            grd_sol = pb[0].greedy(scores=scores_np).x
            _sync(device)
            t2 = time.perf_counter()

            forward_grd_t = shared_forward_t
            decode_grd_t = t2 - t1
            total_grd_t = pre_t + forward_grd_t + decode_grd_t

            ############## Entropy-guided greedy without reinference #################
            cfg["eval"]["greedy_decimation"]["args"]["run_every"] = 1000000000
            cfg["eval"]["greedy_decimation"]["adapter"]["pick_mode"] = "min_entropy"
            cfg["eval"]["greedy_decimation"]["adapter"]["value_mode"] = "argmax"

            _sync(device)
            t0 = time.perf_counter()
            backend = adapter.decimation_backend_fn(pb[0], noisy_data_shared, pred_shared, cfg, device=device)
            padapter = adapter.decimation_adapter_fn(pb[0], cfg)
            gdinf_sol = greedy_decimation(pb[0], backend, padapter)
            _sync(device)
            t1 = time.perf_counter()

            forward_gdinf_t = shared_forward_t
            decode_gdinf_t = t1 - t0
            total_gdinf_t = pre_t + forward_gdinf_t + decode_gdinf_t

            ############## Entropy-guided greedy with reinference #################
            cfg["eval"]["greedy_decimation"]["args"]["run_every"] = 1
            cfg["eval"]["greedy_decimation"]["adapter"]["pick_mode"] = "min_entropy"
            cfg["eval"]["greedy_decimation"]["adapter"]["value_mode"] = "argmax"

            _sync(device)
            t0 = time.perf_counter()
            backend = adapter.decimation_backend_fn(pb[0], noisy_data_shared, pred_shared, cfg, device=device)
            padapter = adapter.decimation_adapter_fn(pb[0], cfg)
            gd1_sol = greedy_decimation(pb[0], backend, padapter)
            _sync(device)
            t1 = time.perf_counter()

            # If reinference happens inside greedy_decimation, this timing is fair:
            # decode time includes all iterative re-calls.
            forward_gd1_t = shared_forward_t
            decode_gd1_t = t1 - t0
            total_gd1_t = pre_t + forward_gd1_t + decode_gd1_t

            ############## Pseudo-diffusion decoding from the SAME corrupted reference #################
            _sync(device)
            t0 = time.perf_counter()
            diff_sample = diffusion_decoding(
                model=model,
                y_init=y_init_corr,
                noise_schedule=noise_schedule,
                eval_t=eval_t,
                cfg=cfg,
                adapter=adapter,
                data=data,
                pb=pb,
                device=device,
                sample=True,
            )
            _sync(device)
            t1 = time.perf_counter()

            # Repair / projection heuristic
            diff_sol = pb[0].greedy(scores=diff_sample.detach().cpu().numpy()).x
            _sync(device)
            t2 = time.perf_counter()

            decode_diff_t = t1 - t0
            correct_diff_t = t2 - t1
            total_diff_t = pre_t + decode_diff_t + correct_diff_t

            ############## Pseudo-diffusion decoding from random noise #################
            _sync(device)
            t0 = time.perf_counter()
            y_pure_noise = torch.randint(0, 2, (y.numel(),), device=device).float()

            diff_rand_sample = diffusion_decoding(
                model=model,
                y_init=y_pure_noise,
                noise_schedule=noise_schedule,
                eval_t=eval_t,
                cfg=cfg,
                adapter=adapter,
                data=data,
                pb=pb,
                device=device,
                sample=True,
            )
            _sync(device)
            t1 = time.perf_counter()

            diff_rand_sol = pb[0].greedy(scores=diff_rand_sample.detach().cpu().numpy()).x
            _sync(device)
            t2 = time.perf_counter()

            decode_diff_rand_t = t1 - t0
            correct_diff_rand_t = t2 - t1
            total_diff_rand_t = pre_t + decode_diff_rand_t + correct_diff_rand_t

            ############ Guided decimation from pure noise ###############
            pure_noisy_data_shared = append_noisy_label_feature(
                data,
                y_t=y_pure_noise,
                t=t_init if cfg.get("diffusion", {}).get("append_t_feature", True) else None,
                num_steps=num_steps if cfg.get("diffusion", {}).get("append_t_feature", True) else None,
            )
            pure_noise_pred_shared = adapter.forward_fn(model, pure_noisy_data_shared)

            cfg["eval"]["greedy_decimation"]["args"]["run_every"] = 1
            cfg["eval"]["greedy_decimation"]["adapter"]["pick_mode"] = "min_entropy"
            cfg["eval"]["greedy_decimation"]["adapter"]["value_mode"] = "argmax"

            _sync(device)
            t0 = time.perf_counter()
            backend = adapter.decimation_backend_fn(pb[0], pure_noisy_data_shared, pure_noise_pred_shared, cfg, device=device)
            padapter = adapter.decimation_adapter_fn(pb[0], cfg)
            pn_gd1_sol = greedy_decimation(pb[0], backend, padapter)
            _sync(device)
            t1 = time.perf_counter()

            # If reinference happens inside greedy_decimation, this timing is fair:
            # decode time includes all iterative re-calls.
            forward_pn_gd1_t = shared_forward_t
            decode_pn_gd1_t = t1 - t0
            total_pn_gd1_t = pre_t + forward_pn_gd1_t + decode_pn_gd1_t

            ############ Diffusion marginal prediction + Entropy Guided Greedy Variable fixing (corrupted reference) ######################
            cfg["eval"]["greedy_decimation"]["args"]["run_every"] = 1000000
            cfg["eval"]["greedy_decimation"]["adapter"]["pick_mode"] = "min_entropy"
            cfg["eval"]["greedy_decimation"]["adapter"]["value_mode"] = "argmax"
            corrupted_data_shared = append_noisy_label_feature(
                data,
                y_t=y_init_corr,
                t=t_init if cfg.get("diffusion", {}).get("append_t_feature", True) else None,
                num_steps=num_steps if cfg.get("diffusion", {}).get("append_t_feature", True) else None,
            )

            _sync(device)
            t0 = time.perf_counter()
            diff_egg_sample, diff_egg_pred = diffusion_decoding(
                model=model,
                y_init=y_init_corr,
                noise_schedule=noise_schedule,
                eval_t=eval_t,
                cfg=cfg,
                adapter=adapter,
                data=data,
                pb=pb,
                device=device,
                sample=True,
                return_pred=True
            )
            _sync(device)
            t1 = time.perf_counter()

            backend = adapter.decimation_backend_fn(pb[0], corrupted_data_shared, diff_egg_pred, cfg, device=device)
            padapter = adapter.decimation_adapter_fn(pb[0], cfg)
            diff_egg_sol = greedy_decimation(pb[0], backend, padapter)
            _sync(device)
            t2 = time.perf_counter()
            diffusion_diff_egg_t = t1 - t0
            decode_diff_egg_t = t2 - t1
            total_diff_egg_t = pre_t + diffusion_diff_egg_t + decode_diff_egg_t


            ############ Diffusion marginal prediction + Entropy Guided Greedy Variable fixing (pure noise) ######################


            ############ Diffusion marginal prediction + Guided Decimation with the final marginals (corrupted reference) #####################
            cfg["eval"]["greedy_decimation"]["args"]["run_every"] = 1
            cfg["eval"]["greedy_decimation"]["adapter"]["pick_mode"] = "min_entropy"
            cfg["eval"]["greedy_decimation"]["adapter"]["value_mode"] = "argmax"
            corrupted_data_shared = append_noisy_label_feature(
                data,
                y_t=y_init_corr,
                t=t_init if cfg.get("diffusion", {}).get("append_t_feature", True) else None,
                num_steps=num_steps if cfg.get("diffusion", {}).get("append_t_feature", True) else None,
            )

            _sync(device)
            t0 = time.perf_counter()
            diff_gd_sample, diff_gd_pred = diffusion_decoding(
                model=model,
                y_init=y_init_corr,
                noise_schedule=noise_schedule,
                eval_t=eval_t,
                cfg=cfg,
                adapter=adapter,
                data=data,
                pb=pb,
                device=device,
                sample=True,
                return_pred=True
            )
            _sync(device)
            t1 = time.perf_counter()

            backend = adapter.decimation_backend_fn(pb[0], corrupted_data_shared, diff_gd_pred, cfg, device=device)
            padapter = adapter.decimation_adapter_fn(pb[0], cfg)
            diff_gd_sol = greedy_decimation(pb[0], backend, padapter)
            _sync(device)
            t2 = time.perf_counter()
            diffusion_diff_gd_t = t1 - t0
            decode_diff_gd_t = t2 - t1
            total_diff_gd_t = pre_t + diffusion_diff_gd_t + decode_diff_gd_t

            ############ Diffusion marginal prediction + Guided Decimation with the final marginals (pure noise) #####################



            ############ Loss values ##############
            spec = adapter.loss_spec_fn(adapter, noisy_data_shared, pred_shared, y, cfg)
            loss = compute_loss_from_spec(spec, loss_registry=LOSS_REGISTRY)
            losses.append(float(loss.detach().cpu().item()))

            for eval_loss_cfg in cfg.get("eval", {}).get("losses", []):
                name = eval_loss_cfg["name"]
                args = eval_loss_cfg.get("args", {})
                val = LOSS_REGISTRY[name](pred_shared, y, data=noisy_data_shared, **args)
                eval_losses[name].append(float(val.detach().cpu().item()))

            if hasattr(noisy_data_shared, "lbp_edges"):
                n_ss += noisy_data_shared.lbp_edges.n_ss.detach().cpu().tolist()
            if hasattr(noisy_data_shared, "nmf_edges"):
                n_ss += noisy_data_shared.nmf_edges.n_ss.detach().cpu().tolist()

            ############ Relative gap values #############
            ref_val = objective_value(pb[0], y)
            grd_sol_val = objective_value(pb[0], grd_sol)
            gdinf_sol_val = objective_value(pb[0], gdinf_sol)
            gd1_sol_val = objective_value(pb[0], gd1_sol)
            diff_sol_val = objective_value(pb[0], diff_sol)
            diff_sample_val = objective_value(pb[0], diff_sample)
            diff_rand_sol_val = objective_value(pb[0], diff_rand_sol)
            diff_rand_sample_val = objective_value(pb[0], diff_rand_sample)
            pn_gd1_sol_val = objective_value(pb[0], pn_gd1_sol)
            diff_egg_sol_val = objective_value(pb[0], diff_egg_sol)
            diff_gd_sol_val = objective_value(pb[0], diff_gd_sol)


            grd_rel_gap = relative_objective_gap(ref_val, grd_sol_val, maximize=maximize, eps=1e-8)
            gdinf_rel_gap = relative_objective_gap(ref_val, gdinf_sol_val, maximize=maximize, eps=1e-8)
            gd1_rel_gap = relative_objective_gap(ref_val, gd1_sol_val, maximize=maximize, eps=1e-8)
            diff_rel_gap = relative_objective_gap(ref_val, diff_sol_val, maximize=maximize, eps=1e-8)
            diff_sample_rel_gap = relative_objective_gap(ref_val, diff_sample_val, maximize=maximize, eps=1e-8)
            diff_rand_rel_gap = relative_objective_gap(ref_val, diff_rand_sol_val, maximize=maximize, eps=1e-8)
            diff_rand_sample_rel_gap = relative_objective_gap(ref_val, diff_rand_sample_val, maximize=maximize, eps=1e-8)
            pn_gd1_rel_gap = relative_objective_gap(ref_val, pn_gd1_sol_val, maximize=maximize, eps=1e-8)
            diff_egg_rel_gap = relative_objective_gap(ref_val, diff_egg_sol_val, maximize=maximize, eps= 1e-8)
            diff_gd_rel_gap = relative_objective_gap(ref_val, diff_gd_sol_val, maximize=maximize, eps= 1e-8)


            ########### Log instance stats ####################
            instance_record = {
                "instance_index": c,
                "instance_stem": stem,
                "reference_objective": float(ref_val),
                "preprocess_time_s": float(pre_t),
                "grd": {
                    "objective": float(grd_sol_val),
                    "gap": float(grd_rel_gap),
                    "forward_time_s": float(forward_grd_t),
                    "decode_time_s": float(decode_grd_t),
                    "total_time_s": float(total_grd_t),
                },
                "gdinf": {
                    "objective": float(gdinf_sol_val),
                    "gap": float(gdinf_rel_gap),
                    "forward_time_s": float(forward_gdinf_t),
                    "decode_time_s": float(decode_gdinf_t),
                    "total_time_s": float(total_gdinf_t),
                },
                "gd1": {
                    "objective": float(gd1_sol_val),
                    "gap": float(gd1_rel_gap),
                    "forward_time_s": float(forward_gd1_t),
                    "decode_time_s": float(decode_gd1_t),
                    "total_time_s": float(total_gd1_t),
                },
                "gd1_pure_noise": {
                    "objective": float(pn_gd1_sol_val),
                    "gap": float(pn_gd1_rel_gap),
                    "forward_time_s": float(forward_pn_gd1_t),
                    "decode_time_s": float(decode_pn_gd1_t),
                    "total_time_s": float(total_pn_gd1_t),
                },
                "diff_egg": {
                    "objective": float(diff_egg_sol_val),
                    "gap": float(diff_egg_rel_gap),
                    "diffusion_time_s": float(diffusion_diff_egg_t),
                    "decode_time_s": float(decode_diff_egg_t),
                    "total_time_s": float(total_diff_egg_t),
                },
                "diff_gd": {
                    "objective": float(diff_gd_sol_val),
                    "gap": float(diff_gd_rel_gap),
                    "diffusion_time_s": float(diffusion_diff_gd_t),
                    "decode_time_s": float(decode_diff_gd_t),
                    "total_time_s": float(total_diff_gd_t),
                },


                "diff": {
                    "objective": float(diff_sol_val),
                    "gap": float(diff_rel_gap),
                    "decode_time_s": float(decode_diff_t),
                    "projection_time_s": float(correct_diff_t),
                    "total_time_s": float(total_diff_t),
                    "sample_feasible": float(pb[0].check_solution(diff_sample.detach().cpu().numpy())),
                    "sample_val": float(diff_sample_val),
                    "sample_gap": float(diff_sample_rel_gap),
                },
                "diff_rand": {
                    "objective": float(diff_rand_sol_val),
                    "gap": float(diff_rand_rel_gap),
                    "decode_time_s": float(decode_diff_rand_t),
                    "projection_time_s": float(correct_diff_rand_t),
                    "total_time_s": float(total_diff_rand_t),
                    "sample_feasible": float(pb[0].check_solution(diff_rand_sample.detach().cpu().numpy())),
                    "sample_val": float(diff_rand_sample_val),
                    "sample_gap": float(diff_rand_sample_rel_gap),

                },
            }
            instance_stats.append(instance_record)


            ########### Log average/std stats ################
            summary_lists["grd"]["obj"].append(float(grd_sol_val))
            summary_lists["grd"]["gap"].append(float(grd_rel_gap))
            summary_lists["grd"]["forward_t"].append(float(forward_grd_t))
            summary_lists["grd"]["decode_t"].append(float(decode_grd_t))
            summary_lists["grd"]["total_t"].append(float(total_grd_t))

            summary_lists["gdinf"]["obj"].append(float(gdinf_sol_val))
            summary_lists["gdinf"]["gap"].append(float(gdinf_rel_gap))
            summary_lists["gdinf"]["forward_t"].append(float(forward_gdinf_t))
            summary_lists["gdinf"]["decode_t"].append(float(decode_gdinf_t))
            summary_lists["gdinf"]["total_t"].append(float(total_gdinf_t))

            summary_lists["gd1"]["obj"].append(float(gd1_sol_val))
            summary_lists["gd1"]["gap"].append(float(gd1_rel_gap))
            summary_lists["gd1"]["forward_t"].append(float(forward_gd1_t))
            summary_lists["gd1"]["decode_t"].append(float(decode_gd1_t))
            summary_lists["gd1"]["total_t"].append(float(total_gd1_t))

            summary_lists["gd1_pure_noise"]["obj"].append(float(pn_gd1_sol_val))
            summary_lists["gd1_pure_noise"]["gap"].append(float(pn_gd1_rel_gap))
            summary_lists["gd1_pure_noise"]["forward_t"].append(float(forward_pn_gd1_t))
            summary_lists["gd1_pure_noise"]["decode_t"].append(float(decode_pn_gd1_t))
            summary_lists["gd1_pure_noise"]["total_t"].append(float(total_pn_gd1_t))

            summary_lists["diff"]["obj"].append(float(diff_sol_val))
            summary_lists["diff"]["gap"].append(float(diff_rel_gap))
            summary_lists["diff"]["decode_t"].append(float(decode_diff_t))
            summary_lists["diff"]["project_t"].append(float(correct_diff_t))
            summary_lists["diff"]["total_t"].append(float(total_diff_t))
            summary_lists["diff"]["sample_feasible"].append(
                float(pb[0].check_solution(diff_sample.detach().cpu().numpy()))
            )
            summary_lists["diff"]["sample_val"].append(float(diff_sample_val))
            summary_lists["diff"]["sample_gap"].append(float(diff_sample_rel_gap))

            summary_lists["diff_egg"]["obj"].append(float(diff_egg_sol_val))
            summary_lists["diff_egg"]["gap"].append(float(diff_egg_rel_gap))
            summary_lists["diff_egg"]["decode_t"].append(float(decode_diff_egg_t))
            summary_lists["diff_egg"]["forward_t"].append(float(diffusion_diff_egg_t))
            summary_lists["diff_egg"]["total_t"].append(float(total_diff_egg_t))

            summary_lists["diff_gd"]["obj"].append(float(diff_gd_sol_val))
            summary_lists["diff_gd"]["gap"].append(float(diff_gd_rel_gap))
            summary_lists["diff_gd"]["decode_t"].append(float(decode_diff_gd_t))
            summary_lists["diff_gd"]["forward_t"].append(float(diffusion_diff_gd_t))
            summary_lists["diff_gd"]["total_t"].append(float(total_diff_gd_t))

            
            summary_lists["diff_rand"]["obj"].append(float(diff_rand_sol_val))
            summary_lists["diff_rand"]["gap"].append(float(diff_rand_rel_gap))
            summary_lists["diff_rand"]["decode_t"].append(float(decode_diff_rand_t))
            summary_lists["diff_rand"]["project_t"].append(float(correct_diff_rand_t))
            summary_lists["diff_rand"]["total_t"].append(float(total_diff_rand_t))
            summary_lists["diff_rand"]["sample_feasible"].append(
                float(pb[0].check_solution(diff_rand_sample.detach().cpu().numpy()))
            )
            summary_lists["diff_rand"]["sample_val"].append(float(diff_rand_sample_val))
            summary_lists["diff_rand"]["sample_gap"].append(float(diff_rand_sample_rel_gap))
            
            ########### Save JSON output ###############
            partial_payload = build_partial_eval_payload(
                eval_t=eval_t,
                instance_stats=instance_stats,
                summary_lists=summary_lists,
                eval_losses=eval_losses,
                extra={
                    "last_completed_instance_index": c,
                    "status": "running",
                },
            )
            atomic_write_json(out_path, partial_payload)
            print()
            print(
                f"[test {c:04d}] "
                f"ref={ref_val:.4f} | "
                f"GRD gap={grd_rel_gap:.4f} t={total_grd_t:.4f}s | "
                f"GDinf gap={gdinf_rel_gap:.4f} t={total_gdinf_t:.4f}s | "
                f"GD1 gap={gd1_rel_gap:.4f} t={total_gd1_t:.4f}s | "
                f"PN GD1 gap={pn_gd1_rel_gap:.4f} t={total_pn_gd1_t:.4f}s | "
                f"DIFF gap={diff_rel_gap:.4f} t={total_diff_t:.4f}s | "
                f"DIFF-R gap={diff_rand_rel_gap:.4f} t={total_diff_rand_t:.4f}s | "
                f"DIFF-EGG gap={diff_egg_rel_gap:.4f} t={total_diff_egg_t:.4f}s | "
                f"DIFF-GD gap={diff_gd_rel_gap:.4f} t={total_diff_gd_t:.4f}s | "

            )
            bar.next()
            c += 1
            if c >= test_limit:
                break

        bar.finish()

    final_payload = build_partial_eval_payload(
       eval_t=eval_t,
       instance_stats=instance_stats,
       summary_lists=summary_lists,
       eval_losses=eval_losses,
       extra={
           "last_completed_instance_index": c - 1,
           "status": "completed",
       },
    )
    atomic_write_json(out_path, final_payload)
    return out_path

def run_training(cfg: Dict[str, Any]) -> None:
    torch.manual_seed(int(cfg.get("seed", 0)))
    np.random.seed(int(cfg.get("seed", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_seeded_run_name(cfg)
    paths = make_run_paths(cfg)
    logger = TrainLogger(paths, cfg)
    logger.write_hparams(extra={"device": str(device)})

    loaders = make_loaders(cfg)

    method_name = cfg.get("train", {}).get("method", "var_logits")
    if method_name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown train.method '{method_name}'")
    adapter_train = METHOD_REGISTRY[method_name]

    eval_method = cfg.get("eval", {}).get("method") or method_name
    adapter_eval = METHOD_REGISTRY[eval_method]
    adapter_eval.loss_name = cfg.get("eval", {}).get("loss-name") or cfg["loss"]["name"]

    model = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)
    optimizer._base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    scheduler = build_scheduler(cfg, optimizer)

    noise_schedule = build_noise_schedule(cfg, device)

    st = EpochState()
    resume_path = cfg.get("run", {}).get("resume")
    if resume_path:
        st, _ = load_checkpoint(resume_path, model, optimizer, scheduler, map_location=device)

    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"].get("early_stop", {}).get("patience", 20))
    global_step = 0
    warmup_steps = int(cfg.get("train", {}).get("warmup", {}).get("steps", 0))

    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    clip_cfg = cfg["train"].get("grad_clip_norm")
    clip = float(clip_cfg) if clip_cfg is not None else None

    for epoch in range(st.epoch, epochs + 1):
        tr, tr_steps = train_one_epoch(
            cfg=cfg,
            adapter=adapter_train,
            model=model,
            optimizer=optimizer,
            loader=loaders.train_loader,
            device=device,
            epoch=epoch,
            global_step=global_step,
            noise_schedule=noise_schedule,
            scaler=scaler,
            use_amp=use_amp,
            clip=clip,
        )
        global_step += tr_steps

        va, gap, eloss = validate_one_epoch(
            cfg=cfg,
            adapter=adapter_eval,
            model=model,
            loader=loaders.val_loader,
            device=device,
            epoch=epoch,
            noise_schedule=noise_schedule,
            use_amp=use_amp,
        )

        logger.add_scalars("loss", {"train": tr, "val": va}, epoch)
        logger.add_scalars("lr", {f"pg{k}": pg["lr"] for k, pg in enumerate(optimizer.param_groups)}, epoch)
        logger.add_scalars("gap", {"val": gap}, epoch)
        if len(eloss):
            logger.add_scalars("extra_loss", eloss, epoch)

        print(
            f"Epoch {epoch:03d} | train {tr:.4f} | val {va:.4f} | gap {gap:.4f}"
            + (" | " + " | ".join(f"{k} {v:.4f}" for k, v in eloss.items()) if len(eloss) else "")
        )

        metrics_dict = {
            "train/loss": tr,
            "val/loss": va,
            "val/gap": gap,
            **eloss,
        }

        if global_step >= warmup_steps:
            _apply_lr_schedulers(cfg, scheduler, epoch, metrics_dict)

        st.epoch = epoch
        metric_name = cfg["train"].get("early_stop", {}).get("metric", "val/gap")
        metric_value = metrics_dict[metric_name]
        improved = metric_value < st.best_gap

        if improved:
            st.best_gap = metric_value
            st.best_epoch = epoch
            st.epochs_bad = 0
        else:
            st.epochs_bad += 1

        last_path = os.path.join(paths.ckpt_dir, "last.pt")
        save_checkpoint(last_path, st, model, optimizer, scheduler)

        if improved and cfg["train"]["checkpoint"].get("save_best", True):
            best_path = os.path.join(paths.ckpt_dir, "best.pt")
            save_checkpoint(best_path, st, model, optimizer, scheduler)

        if st.epochs_bad >= patience:
            print(f"[early-stop] No improvement for {st.epochs_bad} epochs.")
            break

    best_path = os.path.join(paths.ckpt_dir, "best.pt")
    if os.path.exists(best_path):
        load_checkpoint(best_path, model, optimizer=None, map_location=device)

    if cfg.get("eval", {}).get("enabled", True):
        out_path = final_test_eval(
            cfg=cfg,
            adapter=adapter_eval,
            model=model,
            loader=loaders.test_loader,
            device=device,
            log_dir=paths.log_dir,
            noise_schedule=noise_schedule,
        )
        print(f"[test-eval] saved stats to {out_path}")

    logger.close()
