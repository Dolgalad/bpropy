from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

from bpropy.cli.train.method_adapter import MethodAdapter

import torch


def num_variable_nodes(data: Any) -> int:
    if hasattr(data, "node_types") and ("variable" in getattr(data, "node_types", [])):
        return int(data["variable"].num_nodes)

    return int((data.bipartite == 0).sum().item())


def variable_mask(data: Any) -> torch.Tensor:
    """
    Return a boolean mask selecting variable nodes.

    Current implementation assumes homogeneous PyG data with:
        data.x
        data.bipartite

    For hetero data, append flow features to data["variable"].x instead.
    """
    if hasattr(data, "node_types") and ("variable" in getattr(data, "node_types", [])):
        raise NotImplementedError(
            "Homogeneous flow feature appending assumes data.x and data.bipartite. "
            "For hetero data, append flow columns to data['variable'].x."
        )

    return data.bipartite == 0


def variable_batch_index(data: Any) -> torch.Tensor:
    """
    Batch index for variable nodes.

    Returns shape [num_variables].
    """
    var_mask = variable_mask(data)

    if hasattr(data, "batch") and data.batch is not None:
        return data.batch[var_mask].to(dtype=torch.long)

    return torch.zeros(
        int(var_mask.sum().item()),
        device=data.x.device,
        dtype=torch.long,
    )


def append_flow_features(
    data: Any,
    p_t: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
) -> Any:
    """
    Return a shallow-copied graph with flow features appended.

    Appends to variable nodes:
      - p_t
      - optionally t

    Non-variable nodes receive zeros for appended columns.

    Also stores:
      - data.p_t
      - data.t_var
      - data.z_var
    """
    data = copy.copy(data)

    var_mask = variable_mask(data)
    x = data.x
    n_nodes = x.size(0)
    device = x.device
    dtype = x.dtype

    p_col = torch.zeros((n_nodes, 1), device=device, dtype=dtype)
    p_col[var_mask, 0] = p_t.to(device=device, dtype=dtype)

    cols = [x, p_col]

    if t is not None:
        t_col = torch.zeros((n_nodes, 1), device=device, dtype=dtype)
        t_col[var_mask, 0] = t.to(device=device, dtype=dtype)
        cols.append(t_col)

    data.x = torch.cat(cols, dim=1)
    data.p_t = p_t
    data.t_var = t
    data.z_var = z

    return data


def sample_flow_times(
    data: Any,
    y: torch.Tensor,
    device: torch.device,
    policy: str = "per_problem",
) -> torch.Tensor:
    """
    Sample interpolation times.

    policy:
      - "per_problem": one scalar t per problem/subproblem
      - "per_variable": independent t per variable
    """
    if policy == "per_variable":
        return torch.rand_like(y, dtype=torch.float32, device=device)

    if policy == "per_problem":
        var_batch = variable_batch_index(data).to(device)
        batch_size = int(var_batch.max().item()) + 1 if var_batch.numel() > 0 else 1
        t_per_pb = torch.rand(batch_size, device=device)
        return t_per_pb[var_batch].float()

    raise ValueError(f"Unknown flow.t_policy={policy!r}")


def sample_flow_source(
    y: torch.Tensor,
    device: torch.device,
    source_prob: float = 0.5,
    source: str = "uniform",
) -> torch.Tensor:
    """
    Sample p_0 source.

    source:
      - "uniform": p_0 ~ Uniform(0, 1)^n
      - "bernoulli": p_0 ~ Bernoulli(source_prob)^n

    Note: loop.py currently says Bernoulli in the docstring but returns
    torch.rand(...) before the Bernoulli line. This function makes the
    source choice explicit.
    """
    if source == "uniform":
        return torch.rand(y.shape, device=device, dtype=torch.float32)

    if source == "bernoulli":
        return torch.bernoulli(
            torch.full_like(
                y.float(),
                fill_value=float(source_prob),
                device=device,
            )
        )

    raise ValueError(f"Unknown flow.source={source!r}")


def sample_evidence_aware_flow_source(
    num_variables: int,
    evidence: Optional[Dict[int, int]],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    source_prob: float = 0.5,
    source: str = "uniform",
) -> torch.Tensor:
    """
    Sample p_0 while respecting evidence:

        p_0[i] = 0 if evidence fixes x_i = 0
        p_0[i] = 1 if evidence fixes x_i = 1
        p_0[i] random otherwise
    """
    if source == "uniform":
        p0 = torch.rand(num_variables, device=device, dtype=dtype)
    elif source == "bernoulli":
        p0 = torch.bernoulli(
            torch.full(
                (num_variables,),
                fill_value=float(source_prob),
                device=device,
                dtype=dtype,
            )
        )
    else:
        raise ValueError(f"Unknown flow.source={source!r}")

    if evidence:
        for i, value in evidence.items():
            p0[int(i)] = float(value)

    return p0


def make_probability_path(
    y: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Linear probability path:

        p_t = (1 - t) z + t y
    """
    y = y.float()
    z = z.float()
    t = t.float()
    return (1.0 - t) * z + t * y


def make_flow_batch(
    data: Any,
    y: torch.Tensor,
    source_prob: float = 0.5,
    append_t_feature: bool = True,
    t_policy: str = "per_problem",
    source: str = "uniform",
) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a supervised flow-training batch.

    Returns:
        flow_data : graph with appended p_t and optional t
        p_t       : interpolated state
        t         : interpolation time
        z         : source sample
    """
    t = sample_flow_times(data, y, y.device, policy=t_policy)
    z = sample_flow_source(
        y,
        y.device,
        source_prob=source_prob,
        source=source,
    )
    p_t = make_probability_path(y, z, t)

    flow_data = append_flow_features(
        data=data,
        p_t=p_t,
        t=t if append_t_feature else None,
        z=z,
    )

    return flow_data, p_t, t, z


def get_validation_times(cfg: Dict[str, Any]) -> list[float]:
    n = int(cfg.get("flow", {}).get("num_validation_times", 5))
    if n <= 0:
        raise ValueError("flow.num_validation_times must be >= 1")

    grid = torch.linspace(0.0, 1.0, steps=n + 2)
    return [float(x.item()) for x in grid[1:-1]]


def get_test_times(cfg: Dict[str, Any]) -> list[float]:
    n = int(cfg.get("flow", {}).get("num_test_times", 8))
    if n <= 0:
        raise ValueError("flow.num_test_times must be >= 1")

    grid = torch.linspace(0.0, 1.0, steps=n + 2)
    return [float(x.item()) for x in grid[1:-1]]


def as_probability_vector(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure model scores are probabilities in [0, 1].
    """
    if torch.any(x < 0.0) or torch.any(x > 1.0):
        x = torch.sigmoid(x)

    return x.clamp(0.0, 1.0)


@dataclass
class FlowDecodeResult:
    p_final: torch.Tensor
    pred: Optional[Any] = None
    theta: Optional[torch.Tensor] = None


def _extract_theta_from_pred(pred: Any) -> torch.Tensor:
    """
    Extract theta from a model/adapter prediction.

    Supports:
      - pred is theta
      - pred is (..., theta), e.g. (inference_result, theta)
    """
    if isinstance(pred, (tuple, list)) and len(pred) >= 2:
        theta = pred[1]
    else:
        theta = pred

    if not torch.is_tensor(theta):
        raise TypeError(
            "Could not extract theta from prediction. Provide theta_fn or "
            "make the model return theta directly."
        )

    return theta


def default_theta_fn(
    model: torch.nn.Module,
    flow_data: Any,
    adapter: Optional[Any] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Default theta accessor for MRF models.

    Preferred behavior:
      model(flow_data, inference='theta')

    Fallback:
      adapter.forward_fn(model, flow_data), then extract theta.
    """
    try:
        theta = model(flow_data, inference="theta")
        if isinstance(theta, (tuple, list)):
            theta = theta[-1]
        return theta
    except TypeError:
        pass

    if adapter is None:
        raise TypeError(
            "default_theta_fn could not call model(..., inference='theta'), "
            "and no adapter was provided."
        )

    pred = adapter.forward_fn(model, flow_data)
    return _extract_theta_from_pred(pred)


def flow_decoding(
    model: torch.nn.Module,
    p_init: torch.Tensor,
    adapter: MethodAdapter,
    data: Any,
    pb: Any,
    device: torch.device,
    cfg: Dict[str, Any],
    return_pred: bool = False,
    update_mode: str = "endpoint_velocity",
    n_steps: Optional[int] = None,
) -> torch.Tensor | Tuple[torch.Tensor, Any]:
    """
    Full flow rollout for an endpoint-predicting model.

    Model prediction is interpreted as q1_hat = f_theta(p_t, t), an estimate
    of terminal probabilities p_1.

    endpoint_velocity update:

        v_hat_t = (q1_hat - p_t) / max(1 - t, eps)
        p_{t+dt} = p_t + dt * v_hat_t
    """
    result = flow_rollout(
        model=model,
        p_init=p_init,
        adapter=adapter,
        data=data,
        pb=pb,
        device=device,
        cfg=cfg,
        update_mode=update_mode,
        return_final_theta=False,
        n_steps=n_steps,
    )

    if return_pred:
        return result.p_final, result.pred

    return result.p_final


def flow_rollout(
    model: torch.nn.Module,
    p_init: torch.Tensor,
    adapter: MethodAdapter,
    data: Any,
    pb: Any,
    device: torch.device,
    cfg: Dict[str, Any],
    update_mode: str = "endpoint_velocity",
    return_final_theta: bool = False,
    theta_fn: Optional[Any] = None,
    n_steps: Optional[int] = None,
) -> FlowDecodeResult:
    """
    Shared flow rollout used by:
      - validation/test decoding
      - FlowLBPBackend
      - FlowNMFBackend

    If return_final_theta=True, performs one final model/theta call at t=1
    after the rollout and returns it in FlowDecodeResult.theta.
    """
    p_cur = p_init.clone().float().to(device)
    z_src = p_init.clone().float().to(device)
    pred = None

    flow_cfg = cfg.get("flow", {})

    if n_steps is None:
        n_steps = int(flow_cfg.get("num_test_times", 8))
    else:
        n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError("flow.num_test_times must be >= 1")

    append_t_feature = bool(flow_cfg.get("append_t_feature", True))
    eps = float(flow_cfg.get("velocity_eps", 1e-6))

    substeps = int(flow_cfg.get("decode_substeps", 1))
    if substeps <= 0:
        raise ValueError("flow.decode_substeps must be >= 1")

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
            q1_hat = as_probability_vector(q1_hat)

            if update_mode == "endpoint_velocity":
                denom = max(1.0 - t_scalar, eps)
                v_hat = (q1_hat - p_cur) / denom
                p_cur = (p_cur + dt * v_hat).clamp(0.0, 1.0)

            elif update_mode == "convex":
                p_cur = ((1.0 - dt) * p_cur + dt * q1_hat).clamp(0.0, 1.0)

            else:
                raise ValueError(f"Unknown flow decoding update_mode={update_mode!r}")

    theta = None
    if return_final_theta:
        t_final = torch.ones_like(p_cur, dtype=torch.float32, device=device)

        final_flow_data = append_flow_features(
            data=data,
            p_t=p_cur,
            t=t_final if append_t_feature else None,
            z=z_src,
        )

        if theta_fn is not None:
            theta = theta_fn(model, final_flow_data, adapter, cfg)
        elif hasattr(adapter, "theta_fn"):
            theta = adapter.theta_fn(model, final_flow_data, cfg)
            if isinstance(theta, (tuple, list)):
                theta = theta[-1]
        else:
            theta = default_theta_fn(
                model=model,
                flow_data=final_flow_data,
                adapter=adapter,
                cfg=cfg,
            )

    return FlowDecodeResult(
        p_final=p_cur,
        pred=pred,
        theta=theta,
    )
