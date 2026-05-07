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


def append_flow_features(
    data: Any,
    p_t: torch.Tensor,
    t: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    ) -> Any:
    data = copy.copy(data)

    var_mask = _variable_mask(data)
    x = data.x
    N = x.size(0)
    device = x.device
    dtype = x.dtype

    p_col = torch.zeros((N, 1), device=device, dtype=dtype)
    p_col[var_mask, 0] = p_t.to(dtype)

    cols = [x, p_col]

    if t is not None:
        t_col = torch.zeros((N, 1), device=device, dtype=dtype)
        t_col[var_mask, 0] = t.to(dtype)
        cols.append(t_col)

    data.x = torch.cat(cols, dim=1)
    #print("AAAA ",torch.mean(t), torch.mean(data.x[var_mask,-1]))

    data.p_t = p_t
    data.t_var = t
    data.z_var = z

    return data

def make_flow_batch(
    data: Any,
    y: torch.Tensor,
    source_prob: float = 0.5,
    append_t_feature: bool = True,
    t_policy: str = "per_problem",
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
    flow_data : graph with appended p_t and optionally t
    p_t       : interpolated state
    t         : interpolation time
    z         : source sample
    """
    t = sample_flow_times(data, y, y.device, policy=t_policy)
    z = sample_flow_source(y, y.device, source_prob=source_prob)
    p_t = make_probability_path(y, z, t)

    flow_data = append_flow_features(
        data=data,
        p_t=p_t,
        t=t if append_t_feature else None,
        z=z,
    )
    return flow_data, p_t, t, z

########## Flow utilities #############
def _variable_batch_index(data: Any) -> torch.Tensor:
    var_mask = _variable_mask(data)

    if hasattr(data, "batch") and data.batch is not None:
        return data.batch[var_mask].to(dtype=torch.long)

    # fallback: single graph / single subproblem
    return torch.zeros(int(var_mask.sum().item()), device=data.x.device, dtype=torch.long)

#def sample_flow_times(
#    data: Any,
#    y: torch.Tensor,
#    device: torch.device,
#    ) -> torch.Tensor:
#    """
#    Sample one scalar t per subproblem in the minibatch, constant across all
#    variables of that subproblem, but different across subproblems.
#    Returns shape [num_variables].
#    """
#    var_batch = _variable_batch_index(data).to(device)
#    batch_size = int(var_batch.max().item()) + 1 if var_batch.numel() > 0 else 1
#
#    t_per_pb = torch.rand(batch_size, device=device)
#    return t_per_pb[var_batch].float()

def sample_flow_times(
    data: Any,
    y: torch.Tensor,
    device: torch.device,
    policy: str = "per_problem",
) -> torch.Tensor:
    """
    policy:
      - "per_problem": one scalar t per problem/subproblem
      - "per_variable": one independent t per variable
    """
    if policy == "per_variable":
        return torch.rand_like(y, dtype=torch.float32, device=device)

    if policy == "per_problem":
        var_batch = _variable_batch_index(data).to(device)
        batch_size = int(var_batch.max().item()) + 1 if var_batch.numel() > 0 else 1
        t_per_pb = torch.rand(batch_size, device=device)
        return t_per_pb[var_batch].float()

    raise ValueError(f"Unknown flow.t_policy={policy}")

def sample_flow_source(
    y: torch.Tensor,
    device: torch.device,
    source_prob: float = 0.5,
    ) -> torch.Tensor:
    """
    Binary source z ~ Bernoulli(source_prob)^n.
    """
    return torch.rand(torch.full_like(y.float), device=device)
    return torch.bernoulli(torch.full_like(y.float(), source_prob, device=device))

def make_probability_path(
    y: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    ) -> torch.Tensor:
    """
    p_t = (1 - t) z + t y
    """
    y = y.float()
    z = z.float()
    t = t.float()
    return (1.0 - t) * z + t * y

def get_validation_times(cfg: Dict[str, Any]) -> list[float]:
    n = int(cfg.get("flow", {}).get("num_validation_times", 5))
    if n <= 0:
        raise ValueError("flow.num_validation_times must be >= 1")

    # interior points only: exclude 0 and 1
    grid = torch.linspace(0.0, 1.0, steps=n + 2)
    return [float(x.item()) for x in grid[1:-1]]

def get_test_times(cfg: Dict[str, Any]) -> list[float]:
    n = int(cfg.get("flow", {}).get("num_test_times", 8))
    if n <= 0:
        raise ValueError("flow.num_test_times must be >= 1")

    grid = torch.linspace(0.0, 1.0, steps=n + 2)
    return [float(x.item()) for x in grid[1:-1]]
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
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    clip: Optional[float],
    ) -> Tuple[float, int]:
    model.train()
    losses = []
    steps = 0
    max_steps = int(cfg["train"]["steps_per_epoch"])

    source_prob = float(cfg.get("flow", {}).get("source_prob", 0.5))
    append_t_feature = bool(cfg.get("flow", {}).get("append_t_feature", True))

    t_policy = cfg.get("flow", {}).get("t_policy", "per_problem")

    bar = Bar(f"[Train] epoch {epoch}", max=len(loader))

    for data, y in CUDAPrefetcher(loader, device):
        optimizer.zero_grad(set_to_none=True)

        flow_data, p_t, t, z = make_flow_batch(
            data,
            y,
            source_prob=source_prob,
            append_t_feature=append_t_feature,
            t_policy=t_policy,
        )

        with torch.amp.autocast(device.type, enabled=use_amp):
            pred = adapter.forward_fn(model, flow_data)
            spec = adapter.loss_spec_fn(adapter, flow_data, pred, y, cfg)

        if spec.backward_fn is not None:
            logged_loss = spec.backward_fn()
            loss_scalar = (
                float(logged_loss.detach().cpu().item())
                if torch.is_tensor(logged_loss)
                else float(logged_loss)
            )
        else:
            loss = compute_loss_from_spec(spec, loss_registry=LOSS_REGISTRY)
            scaler.scale(loss).backward()
            loss_scalar = float(loss.detach().cpu().item())

        if clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        _warmup_lr(cfg, optimizer, global_step + steps)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss_scalar)

        bar.next()
        bar.suffix = f" loss: {np.mean(losses):.4f}"
        steps += 1

        if steps >= max_steps:
            break

    bar.finish()
    return float(np.mean(losses)) if losses else 0.0, steps

def validate_one_epoch(
    cfg,
    adapter,
    model,
    loader,
    device,
    epoch,
    use_amp: bool = False,
):
    model.eval()
    losses = []
    gaps = []
    score_deltas = []

    eval_loss_names = [e["name"] for e in cfg.get("eval", {}).get("losses", [])]
    eval_losses = {n: [] for n in eval_loss_names}

    max_steps = int(cfg["train"]["val_steps"])
    maximize = bool(cfg.get("problem", {}).get("maximize", True))
    source_prob = float(cfg.get("flow", {}).get("source_prob", 0.5))
    append_t_feature = bool(cfg.get("flow", {}).get("append_t_feature", True))

    t_policy = cfg.get("flow", {}).get("t_policy", "per_problem")

    steps = 0

    with torch.no_grad():
        bar = Bar(f"[Val]   epoch {epoch}", max=len(loader))

        for pbs, data, y in CUDAPrefetcher(loader, device):
            # -------------------------------------------------
            # 1. Validation loss: supervised flow loss at random interior t
            # -------------------------------------------------
            flow_data, p_t, t, z_loss = make_flow_batch(
                data,
                y,
                source_prob=source_prob,
                append_t_feature=append_t_feature,
                t_policy=t_policy
            )

            with torch.amp.autocast(device.type, enabled=use_amp):
                pred = adapter.forward_fn(model, flow_data)
                spec = adapter.loss_spec_fn(adapter, flow_data, pred, y, cfg)

            loss = compute_loss_from_spec(spec, loss_registry=LOSS_REGISTRY)
            losses.append(float(loss.detach().cpu().item()))

            for eval_loss_cfg in cfg.get("eval", {}).get("losses", []):
                name = eval_loss_cfg["name"]
                args = eval_loss_cfg.get("args", {})
                val = LOSS_REGISTRY[name](pred, y, data=flow_data, **args)
                eval_losses[name].append(float(val.detach().cpu().item()))

            scores = adapter.greedy_scores_fn(pbs, flow_data, pred)
            scores_np = np.asarray(scores)
            y_np = y.detach().cpu().numpy()

            #zero_idx = y_np == 0
            #one_idx = y_np == 1
            #if zero_idx.any() and one_idx.any():
            #    score_deltas.append(
            #        float(np.mean(scores_np[one_idx]) - np.mean(scores_np[zero_idx]))
            #    )

            # -------------------------------------------------
            # 2. Validation gap: actual test-time flow rollout
            # -------------------------------------------------
            z = sample_flow_source(y, y.device, source_prob=source_prob)
            p_final = flow_decoding(
                model=model,
                p_init=z,
                adapter=adapter,
                data=data,
                pb=pbs,
                device=device,
                cfg=cfg,
            )



            p_final_np = p_final.detach().cpu().numpy()

            y_np = y.detach().cpu().numpy()

            zero_idx = y_np == 0
            one_idx = y_np == 1
            if zero_idx.any() and one_idx.any():
                score_deltas.append(
                    #float(np.mean(scores_np[one_idx]) - np.mean(scores_np[zero_idx]))
                    float(np.mean(p_final_np[one_idx]) - np.mean(p_final_np[zero_idx]))

                )


            nv_offset = 0
            batch_gaps = []


            for pb in pbs:
                nv = pb.nv()
                scores_i = p_final_np[nv_offset:nv_offset + nv]
                y_i = y_np[nv_offset:nv_offset + nv]

                sol_i = pb.greedy(scores=scores_i).x

                ref_val = objective_value(pb, y_i)
                pred_val = objective_value(pb, sol_i)

                gap_i = relative_objective_gap(
                    ref_val,
                    pred_val,
                    maximize=maximize,
                    eps=1e-8,
                )
                batch_gaps.append(float(gap_i))

                nv_offset += nv

            gaps.append(float(np.mean(batch_gaps)))

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

#def flow_decoding(
#    model,
#    p_init,
#    adapter,
#    data,
#    pb,
#    device,
#    cfg: Dict[str, Any],
#    return_pred: bool = False,
#    update_mode = "step",
#):
#    """
#    Stable flow rollout using convex-combination updates:
#        p_{t+dt} = (1 - dt) * p_t + dt * q1
#    where q1 is the model's estimate of the terminal probabilities.
#
#    Returns the final probability vector in [0,1].
#    """
#    p_cur = p_init.clone().float().to(device)
#    z_src = p_init.clone().float().to(device)
#    pred = None
#
#    n_steps = int(cfg.get("flow", {}).get("num_test_times", 8))
#    if n_steps <= 0:
#        raise ValueError("flow.num_test_times must be >= 1")
#
#    # grid: 0, 1/n, 2/n, ..., 1
#    grid = torch.linspace(0.0, 1.0, steps=n_steps + 1, device=device)
#
#    append_t_feature = bool(cfg.get("flow", {}).get("append_t_feature", True))
#
#    for k in range(n_steps):
#        t_scalar = float(grid[k].item())
#        dt = float((grid[k + 1] - grid[k]).item())
#
#        t_var = torch.full_like(
#            p_cur,
#            fill_value=t_scalar,
#            dtype=torch.float32,
#            device=device,
#        )
#
#        flow_data = append_flow_features(
#            data=data,
#            p_t=p_cur,
#            t=t_var if append_t_feature else None,
#            z=z_src,
#        )
#
#        pred = adapter.forward_fn(model, flow_data)
#        scores = adapter.greedy_scores_fn(pb, flow_data, pred)
#        scores = torch.as_tensor(scores, device=device, dtype=torch.float32)
#
#        if torch.all((scores >= 0.0) & (scores <= 1.0)):
#            q1 = scores.clamp(0.0, 1.0)
#        else:
#            q1 = torch.sigmoid(scores)
#
#        # Stable convex-combination step toward q1
#        if update_mode == "convex":
#            alpha = dt
#            p_cur = (1.0 - alpha) * p_cur + alpha * q1
#            p_cur = p_cur.clamp(0.0, 1.0)
#        if update_mode == "step":
#            alpha = dt
#            p_cur = p_cur + alpha * (q1 - p_cur) / ( 1.0 - t_scalar)
#            p_cur = p_cur.clamp(0.0, 1.0)
#
#
#    if return_pred:
#        return p_cur, pred
#    return p_cur

def flow_decoding(
    model,
    p_init,
    adapter,
    data,
    pb,
    device,
    cfg: Dict[str, Any],
    return_pred: bool = False,
    update_mode: str = "endpoint_velocity",
):
    """
    Flow rollout for an endpoint-predicting model.

    The model is interpreted as predicting q1_hat = f_theta(p_t, t),
    i.e. an estimate of the terminal probabilities p_1.

    We then construct the instantaneous velocity

        v_hat_t = (q1_hat - p_t) / max(1 - t, eps)

    and update with explicit Euler:

        p_{t+dt} = p_t + dt * v_hat_t.

    This is equivalent to the previous "step" update, but makes the
    endpoint-to-velocity conversion explicit.
    """
    p_cur = p_init.clone().float().to(device)
    z_src = p_init.clone().float().to(device)
    pred = None

    flow_cfg = cfg.get("flow", {})
    n_steps = int(flow_cfg.get("num_test_times", 8))
    if n_steps <= 0:
        raise ValueError("flow.num_test_times must be >= 1")

    append_t_feature = bool(flow_cfg.get("append_t_feature", True))

    # Avoid numerical explosion near t = 1.
    eps = float(flow_cfg.get("velocity_eps", 1e-6))

    # Optional: smaller integration steps than the model time grid.
    # Usually 1 is enough. Try 2, 4, etc. if the rollout is unstable.
    substeps = int(flow_cfg.get("decode_substeps", 1))
    if substeps <= 0:
        raise ValueError("flow.decode_substeps must be >= 1")

    # Main grid: 0, 1 / n_steps, ..., 1.
    grid = torch.linspace(0.0, 1.0, steps=n_steps + 1, device=device)

    for k in range(n_steps):
        t_start = float(grid[k].item())
        t_end = float(grid[k + 1].item())
        dt_total = t_end - t_start
        dt = dt_total / substeps

        for s in range(substeps):
            t_scalar = t_start + s * dt

            t_var = torch.full_like(
                p_cur,
                fill_value=t_scalar,
                dtype=torch.float32,
                device=device,
            )

            flow_data = append_flow_features(
                data=data,
                p_t=p_cur,
                t=t_var if append_t_feature else None,
                z=z_src,
            )

            pred = adapter.forward_fn(model, flow_data)
            scores = adapter.greedy_scores_fn(pb, flow_data, pred)
            q1_hat = torch.as_tensor(scores, device=device, dtype=torch.float32)

            # Make sure q1_hat is a probability vector.
            # For var_logits adapters, greedy_scores_fn already applies sigmoid.
            # This fallback keeps the function robust for raw-score adapters.
            if torch.any(q1_hat < 0.0) or torch.any(q1_hat > 1.0):
                q1_hat = torch.sigmoid(q1_hat)

            q1_hat = q1_hat.clamp(0.0, 1.0)

            if update_mode == "endpoint_velocity":
                denom = max(1.0 - t_scalar, eps)
                v_hat = (q1_hat - p_cur) / denom
                p_cur = p_cur + dt * v_hat
                p_cur = p_cur.clamp(0.0, 1.0)

            elif update_mode == "convex":
                # Diagnostic/stable alternative:
                # p <- (1-dt) p + dt q1_hat
                # Not the same as rectified-flow endpoint dynamics.
                p_cur = (1.0 - dt) * p_cur + dt * q1_hat
                p_cur = p_cur.clamp(0.0, 1.0)

            else:
                raise ValueError(f"Unknown flow decoding update_mode={update_mode}")

    if return_pred:
        return p_cur, pred
    return p_cur

def final_test_eval(
    cfg: Dict[str, Any],
    adapter,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    log_dir: str,
    output_json: Optional[str] = None,
) -> str:
    """
    Final test evaluation for the flow model.

    Test protocol:
      - always start from t = 0
      - initialize p_0 = z, where z ~ Bernoulli(source_prob)^n
      - evaluate:
          * random_grd : one forward pass + greedy projection
          * random_egg : greedy_decimation with run_every = 1000000
          * random_gd  : greedy_decimation with run_every = 1
          * random_flow: rollout from p_0 using convex updates toward q1,
                         then sample once at the end and repair/project

    For random_flow, logs:
      - sample_feasible before correction
      - sample_val before correction
      - sample_gap before correction
      - repaired objective/gap after greedy correction
    """
    model.eval()

    maximize = bool(cfg.get("problem", {}).get("maximize", True))
    source_prob = float(cfg.get("flow", {}).get("source_prob", 0.5))
    append_t_feature = bool(cfg.get("flow", {}).get("append_t_feature", True))
    test_limit = int(cfg.get("eval", {}).get("test_limit", 20000))

    eval_loss_names = [e["name"] for e in cfg.get("eval", {}).get("losses", [])]
    eval_losses = {n: [] for n in eval_loss_names}

    run_cfg = cfg.setdefault("run", {})
    base = run_cfg.get("run_name") or run_cfg.get("name") or cfg.get("run_name") or "run"
    seed = int(cfg.get("seed", 0))
    unique_name = f"{base}_seed{seed}"
    run_cfg["run_name"] = unique_name
    run_cfg["name"] = unique_name
    cfg["run_name"] = unique_name

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

    _ensure_greedy_decimation_cfg(cfg)

    instance_stats = []
    n_ss = []

    summary_lists = {
        "random_grd": {"obj": [], "gap": [], "forward_t": [], "decode_t": [], "total_t": []},
        "random_egg": {"obj": [], "gap": [], "forward_t": [], "decode_t": [], "total_t": []},
        "random_gd":  {"obj": [], "gap": [], "forward_t": [], "decode_t": [], "total_t": []},
        "random_flow": {
            "obj": [], "gap": [],
            "decode_t": [], "project_t": [], "total_t": [],
            "sample_feasible": [], "sample_val": [], "sample_gap": [],
        },
        "random_flow_cvx": {
            "obj": [], "gap": [],
            "decode_t": [], "project_t": [], "total_t": [],
            "sample_feasible": [], "sample_val": [], "sample_gap": [],
        },

    }

    with torch.no_grad():
        bar = Bar("[Test]          ", max=len(loader))

        for idx, (pb, data, y) in enumerate(CUDAPrefetcher(loader, device)):
            stem = Path(loader.dataset.stems[idx]).name
            print(len(pb))
            pb0 = pb[0]

            # Reference objective from ground-truth label
            ref_val = objective_value(pb0, y)

            # Shared random-source initialization for all test methods
            z = sample_flow_source(y, y.device, source_prob=source_prob)
            t0_var = torch.zeros_like(y, dtype=torch.float32, device=device)
            p_t = make_probability_path(y, z, t0_var)


            flow_data_shared = append_flow_features(
                data=data,
                p_t=p_t,
                t=t0_var if append_t_feature else None,
                z=z,
            )
            #print(torch.mean(flow_data_shared.x[:,-1]))
            pre_t = float(getattr(flow_data_shared, "preprocess_time_s", 0.0))

            # ----- shared forward pass for one-shot methods -----
            _sync(device)
            t0 = time.perf_counter()
            pred_shared = adapter.forward_fn(model, flow_data_shared)
            _sync(device)
            t1 = time.perf_counter()
            shared_forward_t = t1 - t0

            scores = adapter.greedy_scores_fn(pb, flow_data_shared, pred_shared)
            #print("Greedy scores : ", scores)
            scores_np = np.asarray(scores)

            # ----- random_grd -----
            _sync(device)
            t2 = time.perf_counter()
            random_grd_sol = pb0.greedy(scores=scores_np).x
            _sync(device)
            t3 = time.perf_counter()

            forward_random_grd_t = shared_forward_t
            decode_random_grd_t = t3 - t2
            total_random_grd_t = pre_t + forward_random_grd_t + decode_random_grd_t

            # ----- random_egg -----
            cfg["eval"]["greedy_decimation"]["args"]["run_every"] = 1000000
            cfg["eval"]["greedy_decimation"]["adapter"]["pick_mode"] = "min_entropy"
            cfg["eval"]["greedy_decimation"]["adapter"]["value_mode"] = "argmax"

            _sync(device)
            t4 = time.perf_counter()
            backend = adapter.decimation_backend_fn(pb0, flow_data_shared, pred_shared, cfg, device=device)
            padapter = adapter.decimation_adapter_fn(pb0, cfg)
            random_egg_sol = greedy_decimation(pb0, backend, padapter)
            _sync(device)
            t5 = time.perf_counter()

            forward_random_egg_t = shared_forward_t
            decode_random_egg_t = t5 - t4
            total_random_egg_t = pre_t + forward_random_egg_t + decode_random_egg_t

            # ----- random_gd -----
            cfg["eval"]["greedy_decimation"]["args"]["run_every"] = 1
            cfg["eval"]["greedy_decimation"]["adapter"]["pick_mode"] = "min_entropy"
            cfg["eval"]["greedy_decimation"]["adapter"]["value_mode"] = "argmax"

            _sync(device)
            t6 = time.perf_counter()
            backend = adapter.decimation_backend_fn(pb0, flow_data_shared, pred_shared, cfg, device=device)
            padapter = adapter.decimation_adapter_fn(pb0, cfg)
            random_gd_sol = greedy_decimation(pb0, backend, padapter)
            _sync(device)
            t7 = time.perf_counter()

            forward_random_gd_t = shared_forward_t
            decode_random_gd_t = t7 - t6
            total_random_gd_t = pre_t + forward_random_gd_t + decode_random_gd_t

            # ----- random_flow rollout -----
            _sync(device)
            t8 = time.perf_counter()
            random_flow_probs = flow_decoding(
                model=model,
                p_init=z,
                adapter=adapter,
                data=data,
                pb=pb,
                device=device,
                cfg=cfg,
            )
            _sync(device)
            t9 = time.perf_counter()

            # Sample once at the end for pre-correction feasibility logging
            random_flow_sample = torch.bernoulli(random_flow_probs.clamp(0.0, 1.0))

            # Repair / project sampled state using greedy heuristic
            random_flow_sol = pb0.greedy(
                scores=random_flow_probs.detach().cpu().numpy()
            ).x
            _sync(device)
            t10 = time.perf_counter()

            decode_random_flow_t = t9 - t8
            project_random_flow_t = t10 - t9
            total_random_flow_t = pre_t + decode_random_flow_t + project_random_flow_t

            # ----- random_flow rollout -----
            _sync(device)
            t8 = time.perf_counter()
            random_flow_cvx_probs = flow_decoding(
                model=model,
                p_init=z,
                adapter=adapter,
                data=data,
                pb=pb,
                device=device,
                cfg=cfg,
                update_mode="convex",
            )
            _sync(device)
            t9 = time.perf_counter()

            # Sample once at the end for pre-correction feasibility logging
            random_flow_cvx_sample = torch.bernoulli(random_flow_cvx_probs.clamp(0.0, 1.0))

            # Repair / project sampled state using greedy heuristic
            random_flow_cvx_sol = pb0.greedy(
                scores=random_flow_cvx_probs.detach().cpu().numpy()
            ).x
            _sync(device)
            t10 = time.perf_counter()

            decode_random_flow_cvx_t = t9 - t8
            project_random_flow_cvx_t = t10 - t9
            total_random_flow_cvx_t = pre_t + decode_random_flow_cvx_t + project_random_flow_cvx_t


            # ----- loss values from shared one-shot forward -----
            spec = adapter.loss_spec_fn(adapter, flow_data_shared, pred_shared, y, cfg)
            loss = compute_loss_from_spec(spec, loss_registry=LOSS_REGISTRY)

            for eval_loss_cfg in cfg.get("eval", {}).get("losses", []):
                name = eval_loss_cfg["name"]
                args = eval_loss_cfg.get("args", {})
                val = LOSS_REGISTRY[name](pred_shared, y, data=flow_data_shared, **args)
                eval_losses[name].append(float(val.detach().cpu().item()))

            if hasattr(flow_data_shared, "lbp_edges"):
                n_ss += flow_data_shared.lbp_edges.n_ss.detach().cpu().tolist()
            if hasattr(flow_data_shared, "nmf_edges"):
                n_ss += flow_data_shared.nmf_edges.n_ss.detach().cpu().tolist()

            # ----- objectives / gaps -----
            random_grd_val = objective_value(pb0, random_grd_sol)
            random_egg_val = objective_value(pb0, random_egg_sol)
            random_gd_val = objective_value(pb0, random_gd_sol)

            random_flow_sample_val = objective_value(pb0, random_flow_sample)
            random_flow_val = objective_value(pb0, random_flow_sol)

            random_flow_cvx_sample_val = objective_value(pb0, random_flow_cvx_sample)
            random_flow_cvx_val = objective_value(pb0, random_flow_cvx_sol)

            random_grd_gap = relative_objective_gap(ref_val, random_grd_val, maximize=maximize, eps=1e-8)
            random_egg_gap = relative_objective_gap(ref_val, random_egg_val, maximize=maximize, eps=1e-8)
            random_gd_gap  = relative_objective_gap(ref_val, random_gd_val,  maximize=maximize, eps=1e-8)

            random_flow_sample_gap = relative_objective_gap(
                ref_val, random_flow_sample_val, maximize=maximize, eps=1e-8
            )
            random_flow_gap = relative_objective_gap(
                ref_val, random_flow_val, maximize=maximize, eps=1e-8
            )

            random_flow_cvx_sample_gap = relative_objective_gap(
                ref_val, random_flow_cvx_sample_val, maximize=maximize, eps=1e-8
            )
            random_flow_cvx_gap = relative_objective_gap(
                ref_val, random_flow_cvx_val, maximize=maximize, eps=1e-8
            )


            # ----- instance record -----
            instance_record = {
                "instance_index": idx,
                "instance_stem": stem,
                "reference_objective": float(ref_val),
                "preprocess_time_s": float(pre_t),
                "random_grd": {
                    "objective": float(random_grd_val),
                    "gap": float(random_grd_gap),
                    "forward_time_s": float(forward_random_grd_t),
                    "decode_time_s": float(decode_random_grd_t),
                    "total_time_s": float(total_random_grd_t),
                },
                "random_egg": {
                    "objective": float(random_egg_val),
                    "gap": float(random_egg_gap),
                    "forward_time_s": float(forward_random_egg_t),
                    "decode_time_s": float(decode_random_egg_t),
                    "total_time_s": float(total_random_egg_t),
                },
                "random_gd": {
                    "objective": float(random_gd_val),
                    "gap": float(random_gd_gap),
                    "forward_time_s": float(forward_random_gd_t),
                    "decode_time_s": float(decode_random_gd_t),
                    "total_time_s": float(total_random_gd_t),
                },
                "random_flow": {
                    "objective": float(random_flow_val),
                    "gap": float(random_flow_gap),
                    "decode_time_s": float(decode_random_flow_t),
                    "projection_time_s": float(project_random_flow_t),
                    "total_time_s": float(total_random_flow_t),
                    "sample_feasible": float(pb0.check_solution(random_flow_sample.detach().cpu().numpy())),
                    "sample_val": float(random_flow_sample_val),
                    "sample_gap": float(random_flow_sample_gap),
                },
                "random_flow_cvx": {
                    "objective": float(random_flow_cvx_val),
                    "gap": float(random_flow_cvx_gap),
                    "decode_time_s": float(decode_random_flow_cvx_t),
                    "projection_time_s": float(project_random_flow_cvx_t),
                    "total_time_s": float(total_random_flow_cvx_t),
                    "sample_feasible": float(pb0.check_solution(random_flow_cvx_sample.detach().cpu().numpy())),
                    "sample_val": float(random_flow_cvx_sample_val),
                    "sample_gap": float(random_flow_cvx_sample_gap),
                },

            }
            instance_stats.append(instance_record)

            # ----- accumulate summary stats -----
            summary_lists["random_grd"]["obj"].append(float(random_grd_val))
            summary_lists["random_grd"]["gap"].append(float(random_grd_gap))
            summary_lists["random_grd"]["forward_t"].append(float(forward_random_grd_t))
            summary_lists["random_grd"]["decode_t"].append(float(decode_random_grd_t))
            summary_lists["random_grd"]["total_t"].append(float(total_random_grd_t))

            summary_lists["random_egg"]["obj"].append(float(random_egg_val))
            summary_lists["random_egg"]["gap"].append(float(random_egg_gap))
            summary_lists["random_egg"]["forward_t"].append(float(forward_random_egg_t))
            summary_lists["random_egg"]["decode_t"].append(float(decode_random_egg_t))
            summary_lists["random_egg"]["total_t"].append(float(total_random_egg_t))

            summary_lists["random_gd"]["obj"].append(float(random_gd_val))
            summary_lists["random_gd"]["gap"].append(float(random_gd_gap))
            summary_lists["random_gd"]["forward_t"].append(float(forward_random_gd_t))
            summary_lists["random_gd"]["decode_t"].append(float(decode_random_gd_t))
            summary_lists["random_gd"]["total_t"].append(float(total_random_gd_t))

            summary_lists["random_flow"]["obj"].append(float(random_flow_val))
            summary_lists["random_flow"]["gap"].append(float(random_flow_gap))
            summary_lists["random_flow"]["decode_t"].append(float(decode_random_flow_t))
            summary_lists["random_flow"]["project_t"].append(float(project_random_flow_t))
            summary_lists["random_flow"]["total_t"].append(float(total_random_flow_t))
            summary_lists["random_flow"]["sample_feasible"].append(
                float(pb0.check_solution(random_flow_sample.detach().cpu().numpy()))
            )
            summary_lists["random_flow"]["sample_val"].append(float(random_flow_sample_val))
            summary_lists["random_flow"]["sample_gap"].append(float(random_flow_sample_gap))

            summary_lists["random_flow_cvx"]["obj"].append(float(random_flow_cvx_val))
            summary_lists["random_flow_cvx"]["gap"].append(float(random_flow_cvx_gap))
            summary_lists["random_flow_cvx"]["decode_t"].append(float(decode_random_flow_cvx_t))
            summary_lists["random_flow_cvx"]["project_t"].append(float(project_random_flow_cvx_t))
            summary_lists["random_flow_cvx"]["total_t"].append(float(total_random_flow_cvx_t))
            summary_lists["random_flow_cvx"]["sample_feasible"].append(
                float(pb0.check_solution(random_flow_cvx_sample.detach().cpu().numpy()))
            )
            summary_lists["random_flow_cvx"]["sample_val"].append(float(random_flow_cvx_sample_val))
            summary_lists["random_flow_cvx"]["sample_gap"].append(float(random_flow_cvx_sample_gap))


            # ----- checkpoint JSON after each instance -----
            partial_payload = {
                "summary": {
                    "count": len(instance_stats),
                    "status": "running",
                    "methods": {
                        "random_grd": summarize_method(summary_lists["random_grd"]),
                        "random_egg": summarize_method(summary_lists["random_egg"]),
                        "random_gd": summarize_method(summary_lists["random_gd"]),
                        "random_flow": summarize_method(summary_lists["random_flow"]),
                        "random_flow_cvx": summarize_method(summary_lists["random_flow_cvx"]),

                    },
                    "eval_losses": build_eval_loss_stats(eval_losses),
                },
                "instances": instance_stats,
            }
            atomic_write_json(out_path, partial_payload)

            print()
            print(
                f"[test {idx:04d}] "
                f"ref={ref_val:.4f} | "
                f"R-GRD gap={random_grd_gap:.4f} t={total_random_grd_t:.4f}s | "
                f"R-EGG gap={random_egg_gap:.4f} t={total_random_egg_t:.4f}s | "
                f"R-GD gap={random_gd_gap:.4f} t={total_random_gd_t:.4f}s | "
                f"R-FLOW gap={random_flow_gap:.4f} t={total_random_flow_t:.4f}s "
                f"(feas={pb0.check_solution(random_flow_sample.detach().cpu().numpy())}) | "
                f"R-FLOW-CVX gap={random_flow_cvx_gap:.4f} t={total_random_flow_cvx_t:.4f}s "
                f"(feas={pb0.check_solution(random_flow_cvx_sample.detach().cpu().numpy())})"

            )

            bar.next()
            if len(instance_stats) >= test_limit:
                break

        bar.finish()

    final_payload = {
        "summary": {
            "count": len(instance_stats),
            "status": "completed",
            "methods": {
                "random_grd": summarize_method(summary_lists["random_grd"]),
                "random_egg": summarize_method(summary_lists["random_egg"]),
                "random_gd": summarize_method(summary_lists["random_gd"]),
                "random_flow": summarize_method(summary_lists["random_flow"]),
                "random_flow_cvx": summarize_method(summary_lists["random_flow_cvx"]),

            },
            "eval_losses": build_eval_loss_stats(eval_losses),
        },
        "instances": instance_stats,
    }
    atomic_write_json(out_path, final_payload)
    return out_path

def run_training(cfg: Dict[str, Any]) -> None:
    torch.manual_seed(int(cfg.get("seed", 0)))
    np.random.seed(int(cfg.get("seed", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_cfg = cfg.setdefault("run", {})
    base = run_cfg.get("run_name") or run_cfg.get("name") or cfg.get("run_name") or "run"
    seed = int(cfg.get("seed", 0))
    unique_name = f"{base}_seed{seed}"
    run_cfg["run_name"] = unique_name
    run_cfg["name"] = unique_name
    cfg["run_name"] = unique_name

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

    clip = float(cfg["train"].get("grad_clip_norm"))

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
        )
        print(f"[test-eval] saved stats to {out_path}")

    logger.close()
