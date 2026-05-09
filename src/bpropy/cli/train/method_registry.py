from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from bpropy.losses.regularizers import REGULARIZER_REGISTRY
from bpropy.co.base_lp import LPProblem


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class LossSpec:
    # Standard loss path:
    pred: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None
    loss_name: Optional[str] = None
    loss_kwargs: dict = field(default_factory=dict)
    extra_terms: dict = field(default_factory=dict)

    # Gradient-only path:
    backward_fn: Optional[Callable[[], torch.Tensor]] = None


@dataclass
class MethodAdapter:
    """
    Wrap model forward + how to compute:
      - supervision loss input (prediction tensor)
      - scores to pass to pb.greedy(scores=...)
    """
    forward_fn: Callable[[torch.nn.Module, Any], Any]
    loss_spec_fn: Callable[[Any, Any, torch.Tensor, Any, Optional[str]], LossSpec]
    greedy_scores_fn: Callable[[Any, Any, Any], np.ndarray]
    decimation_backend_fn: Callable[LPProblem, Any, torch.Tensor, Any]
    decimation_adapter_fn: Callable[LPProblem, Any]

    flow_decimation_backend_fn: Optional[Callable[...,Any]] = None
    theta_fn: Optional[Callable[..., torch.Tensor]] = None

    loss_name: Optional[str] = None  # optional adapter-level loss override


# ----------------------------
# Small helpers (homo + hetero safe)
# ----------------------------

def _num_vars(data: Any) -> int:
    # HeteroDataBatch: variable store exists
    if hasattr(data, "node_types") and ("variable" in getattr(data, "node_types", [])):
        return int(data["variable"].num_nodes)
    # Homo Data: bipartite mask, variables are bipartite==0
    return int((data.bipartite == 0).sum().item())


def _as_theta(pred: Any) -> torch.Tensor:
    """
    Normalize factor-logits prediction into a single flat tensor theta.
    Supports:
      - theta tensor
      - dict with key "theta" or "variable" (fallback)
    """
    if isinstance(pred, dict):
        if "theta" in pred:
            return pred["theta"]
        if "variable" in pred:
            return pred["variable"]
        raise KeyError(f"Cannot extract theta from dict keys={list(pred.keys())}")
    if not torch.is_tensor(pred):
        raise TypeError(f"Expected theta tensor/dict, got {type(pred)}")
    return pred


def _as_inference_tuple(pred: Any):
    """
    Normalize inference prediction into (inference_res, theta).
    Supports:
      - (inference_res, theta)
      - dict with keys ("inference_res","theta") if you ever add that later
    """
    if isinstance(pred, tuple) and len(pred) == 2:
        return pred[0], pred[1]
    if isinstance(pred, dict):
        if "inference_res" in pred and "theta" in pred:
            return pred["inference_res"], pred["theta"]
    raise TypeError(f"Expected (inference_res, theta), got {type(pred)} / {pred}")


def _unary_scores_from_flat(vec: torch.Tensor, nv: int) -> np.ndarray:
    """
    vec is a flat vector whose first 2*nv entries correspond to unary factors
    in (0/1) order; returns score for state=1.
    """
    return vec[: 2 * nv].reshape(-1, 2)[:, 1].detach().cpu().numpy()


def _pick_loss_name(adapter: MethodAdapter, cfg: Dict[str, Any], loss_name_override: Optional[str]) -> str:
    if loss_name_override is not None:
        return loss_name_override
    if adapter.loss_name is not None:
        return adapter.loss_name
    return cfg.get("loss", {}).get("name", "bce_logits")


# ----------------------------
# Forward functions
# ----------------------------

def _forward_var_logits(model: torch.nn.Module, data: Any) -> torch.Tensor:
    out = model(data)
    if isinstance(out, dict):  # hetero models
        return out["variable"]
    # If hetero data but tensor output, assume it's already variable logits
    if hasattr(data, "node_types") and ("variable" in getattr(data, "node_types", [])):
        return out
    # homo: filter variable nodes
    return out[data.bipartite == 0]


def _forward_factor_logits(model: torch.nn.Module, data: Any) -> torch.Tensor:
    # Model returns flat theta
    return model(data)


def _forward_factor_inference(model: torch.nn.Module, data: Any):
    # Model returns (inference_res, theta)
    return model(data)


def _forward_hybrid_theta(model: torch.nn.Module, data: Any) -> torch.Tensor:
    # Train-time: no inference, just return theta
    return model(data, inference="no_inference")

def _theta_fn_var_logits(
    model: torch.nn.Module,
    data: Any,
    cfg: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Variable-logit models do not expose MRF theta.
    This is intentionally unsupported.
    """
    raise NotImplementedError(
        "theta_fn is not defined for variable-logit models."
    )


def _theta_fn_factor_logits(
    model: torch.nn.Module,
    data: Any,
    cfg: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Return raw factor theta.
    """
    return model(data)


def _theta_fn_factor_no_inference(
    model: torch.nn.Module,
    data: Any,
    cfg: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Return raw factor theta from MRF models that support inference='theta'
    or inference='no_inference'.
    """
    try:
        return model(data, inference="theta")
    except TypeError:
        return model(data, inference="no_inference")


# ----------------------------
# Loss spec functions
# ----------------------------
def _apply_regularizers(spec: LossSpec, 
                        adapter: MethodAdapter,
                        batch: Any,
                        pred: Any,
                        labels: torch.Tensor,
                        cfg: Dict[str,Any],
                        ):
    for r in cfg.get("loss", {}).get("regularizers", []):
        w = float(r.get("weight", 0.0))
        if w == 0.0:
            continue
        r_name = r["name"]
        r_args = dict(r.get("args", {}))
        term = REGULARIZER_REGISTRY[r_name](adapter, batch, pred, labels, cfg, **r_args)
        spec.extra_terms[f"reg/{r_name}"] = w * term 
    return spec


def _loss_spec_var_logits(
    adapter: MethodAdapter,
    batch: Any,
    pred: torch.Tensor,
    labels: torch.Tensor,
    cfg: Dict[str, Any],
    loss_name_override: Optional[str] = None,
) -> LossSpec:
    loss_name = _pick_loss_name(adapter, cfg, loss_name_override)
    spec = LossSpec(
        pred=pred,
        target=labels.to(pred.device, non_blocking=True).float(),
        loss_name=loss_name,
    )
    _apply_regularizers(spec, adapter, batch, pred, labels, cfg)
    return spec


def _loss_spec_factor_logits(
    adapter: MethodAdapter,
    batch: Any,
    pred_theta: Any,
    labels: torch.Tensor,
    cfg: Dict[str, Any],
    loss_name_override: Optional[str] = None,
) -> LossSpec:
    loss_name = _pick_loss_name(adapter, cfg, loss_name_override)
    theta = _as_theta(pred_theta)

    target = batch.y_ss
    if loss_name in ("mse", "l1"):
        target = target.float()

    spec = LossSpec(
        pred=theta,
        target=target,
        loss_name=loss_name,
        loss_kwargs={"data": batch, **cfg.get("loss", {}).get("args", {})},
    )
    _apply_regularizers(spec, adapter, batch, pred_theta, labels, cfg)
    return spec


def _loss_spec_factor_inference(
    adapter: MethodAdapter,
    batch: Any,
    pred: Any,
    labels: torch.Tensor,
    cfg: Dict[str, Any],
    loss_name_override: Optional[str] = None,
) -> LossSpec:
    """
    pred is (inference_res, theta). We decide which tensor to feed into the loss:
      - if loss_name endswith "_logits": use theta
      - else: use inference_res.mu
    """
    loss_name = _pick_loss_name(adapter, cfg, loss_name_override)
    inference_res, theta = _as_inference_tuple(pred)

    target = batch.y_ss
    if loss_name in ("mse", "l1"):
        target = target.float()

    pred_used = theta if loss_name.endswith("_logits") else (inference_res, theta)

    spec = LossSpec(
        pred=pred_used,
        target=target,
        loss_name=loss_name,
        loss_kwargs={"data": batch, **cfg.get("loss", {}).get("args", {})},
    )
    _apply_regularizers(spec, adapter, batch, pred, labels, cfg)
    return spec

def _loss_spec_variable_inference(
    adapter: MethodAdapter,
    batch: Any,
    pred: Any,
    labels: torch.Tensor,
    cfg: Dict[str, Any],
    loss_name_override: Optional[str] = None,
) -> LossSpec:
    """
    pred is (inference_res, theta). We decide which tensor to feed into the loss:
      - if loss_name endswith "_logits": use theta
      - else: use inference_res.mu
    """
    loss_name = _pick_loss_name(adapter, cfg, loss_name_override)
    inference_res, theta = _as_inference_tuple(pred)

    target = labels
    if loss_name in ("mse", "l1"):
        target = target.float()

    pred_used = theta if loss_name.endswith("_logits") else (inference_res, theta)

    spec = LossSpec(
        pred=pred_used,
        target=target,
        loss_name=loss_name,
        loss_kwargs={"data": batch, **cfg.get("loss", {}).get("args", {})},
    )
    _apply_regularizers(spec, adapter, batch, pred, labels, cfg)
    return spec

def _diffusion_target_from_cfg(
    batch: Any,
    pred: Any,
    labels: torch.Tensor,
    cfg: Dict[str, Any],
    ):
    diff_cfg = cfg.get("diffusion", {})
    training_mode = diff_cfg.get("training_mode", "denoise")
    target_mode = diff_cfg.get("target_mode", "x0")
    target_space = diff_cfg.get("target_space", "prob")

    loss_name = cfg.get("loss", {}).get("name", "bce_logits")
    loss_kwargs = dict(cfg.get("loss", {}).get("args", {}))

    diff_targets = getattr(batch, "diff_targets", None)

    if training_mode == "denoise":
        # current behavior
        y = labels.to(pred.device if torch.is_tensor(pred) else pred[1].device).float()

        if target_space == "logit":
            return pred, y, loss_name, loss_kwargs
        elif target_space == "prob":
            if isinstance(pred, tuple):
                if loss_name == "vce":
                    return pred, y, loss_name, {"data": batch, **loss_kwargs}
                elif loss_name in ("mse", "l1"):
                    pred_used = pred[0].mu_v.reshape(-1, 2)[:, 1]
                    return pred_used, y, loss_name, loss_kwargs
                else:
                    raise ValueError(f"Unsupported loss {loss_name}")
            else:
                pred_used = torch.sigmoid(pred)
                return pred_used, y, loss_name, loss_kwargs
        else:
            raise ValueError(f"Unsupported target space {target_space}")

    elif training_mode == "faithful":
        if diff_targets is None:
            raise ValueError("Faithful diffusion training requires batch.diff_targets")

        reverse_prob = diff_targets["reverse_prob"]

        if target_mode == "reverse_prob":
            # predict Bernoulli parameter for y_{t-1}
            if target_space == "logit":
                return pred, reverse_prob, "bce_logits", loss_kwargs
            elif target_space == "prob":
                pred_used = torch.sigmoid(pred) if torch.is_tensor(pred) else pred[0].mu_v.reshape(-1, 2)[:, 1]
                return pred_used, reverse_prob, "mse", loss_kwargs
            else:
                raise ValueError(f"Unsupported target space {target_space}")

        elif target_mode == "x0":
            # x0-prediction with faithful reverse loss would need
            # special loss that builds p_theta(y_{t-1}|y_t,\hat x0)
            return pred, reverse_prob, "diffusion_reverse_kl", {"data": batch, **loss_kwargs}

        else:
            raise ValueError(f"Unsupported target mode {target_mode}")

def _flow_target_from_cfg(
    batch: Any,
    pred: Any,
    labels: torch.Tensor,
    cfg: Dict[str, Any],
    ):
    flow_cfg = cfg.get("flow", {})
    target_kind = flow_cfg.get("loss_target", "q1")
    target_space = flow_cfg.get("target_space", "prob")

    if isinstance(pred, tuple):
        device = pred[1].device
    else:
        device = pred.device

    y = labels.to(device, non_blocking=True).float()

    z = getattr(batch, "z_var", None)
    if z is not None:
        z = z.to(device).float()

    loss_name = cfg.get("loss", {}).get("name", "bce_logits")
    loss_kwargs = dict(cfg.get("loss", {}).get("args", {}))

    if target_kind == "q1":
        if target_space == "prob":
            if isinstance(pred, tuple):
                # variable-marginal path
                if loss_name == "vce":
                    # keep full pred so the vce loss can unpack it
                    return pred, y, loss_name, {"data": batch, **loss_kwargs}
                elif loss_name in ("mse", "l1"):
                    pred_used = pred[0].mu_v.reshape(-1, 2)[:, 1]
                    return pred_used, y, loss_name, loss_kwargs
                else:
                    raise ValueError(
                        f"Unsupported loss '{loss_name}' for flow q1 prob with marginal model."
                    )
            else:
                pred_used = torch.sigmoid(pred)
                return pred_used, y, loss_name, loss_kwargs

        elif target_space == "logit":
            return pred, y, loss_name, loss_kwargs

        else:
            raise ValueError(f"Unknown flow.target_space={target_space}")

    elif target_kind == "velocity":
        if z is None:
            raise ValueError("Flow velocity target requires batch.z_var.")

        if isinstance(pred, tuple):
            q1_hat = pred[0].mu_v.reshape(-1, 2)[:, 1]
        else:
            q1_hat = torch.sigmoid(pred)

        pred_used = q1_hat - z
        target = y - z

        return pred_used, target, loss_name, loss_kwargs
    elif target_kind == "instant_velocity":
        if z is None:
            raise ValueError("Flow instant_velocity target requires batch.z_var.")
        p_t = getattr(batch, "p_t", None)
        t = getattr(batch, "t_var", None)
        if p_t is None or t is None:
            raise ValueError("Flow instant_velocity target requires batch.p_t and batch.t_var.")

        eps = float(flow_cfg.get("velocity_eps", 1e-2))

        if isinstance(pred, tuple):
            q1_hat = pred[0].mu_v.reshape(-1, 2)[:, 1]
        else:
            q1_hat = torch.sigmoid(pred)

        z = z.to(device).float()
        p_t = p_t.to(device).float()
        t = t.to(device).float()
        y = labels.to(device, non_blocking=True).float()

        denom = (1.0 - t).clamp_min(eps)

        pred_used = (q1_hat - p_t) / denom
        target = y - z

        return pred_used, target, loss_name, loss_kwargs

    else:
        raise ValueError(f"Unknown flow.loss_target={target_kind}")

def _loss_spec_flow_q1_or_velocity(
    adapter: MethodAdapter,
    batch: Any,
    pred: torch.Tensor,
    labels: torch.Tensor,
    cfg: Dict[str, Any],
    loss_name_override: Optional[str] = None,
    ) -> LossSpec:
    pred_used, target, loss_name, loss_kwargs = _flow_target_from_cfg(
        batch=batch,
        pred=pred,
        labels=labels,
        cfg=cfg,
    )

    if loss_name_override is not None:
        loss_name = loss_name_override

    spec = LossSpec(
        pred=pred_used,
        target=target,
        loss_name=loss_name,
        loss_kwargs=loss_kwargs,
    )
    _apply_regularizers(spec, adapter, batch, pred, labels, cfg)
    return spec

def _loss_spec_diffusion(
    adapter: MethodAdapter,
    batch: Any,
    pred: Any,
    labels: torch.Tensor,
    cfg: Dict[str, Any],
    loss_name_override: Optional[str] = None,
    ) -> LossSpec:
    pred_used, target, loss_name, loss_kwargs = _diffusion_target_from_cfg(
        batch=batch,
        pred=pred,
        labels=labels,
        cfg=cfg,
    )

    if loss_name_override is not None:
        loss_name = loss_name_override

    spec = LossSpec(
        pred=pred_used,
        target=target,
        loss_name=loss_name,
        loss_kwargs=loss_kwargs,
    )
    _apply_regularizers(spec, adapter, batch, pred, labels, cfg)
    return spec

# ----------------------------
# Greedy scoring functions
# ----------------------------

def _greedy_scores_var_logits(pb: Any, data: Any, pred: torch.Tensor) -> np.ndarray:
    return torch.sigmoid(pred).cpu().numpy()
    return pred.detach().cpu().numpy()


def _greedy_scores_factor_logits(pb: Any, data: Any, pred: Any) -> np.ndarray:
    theta = _as_theta(pred)
    nv = _num_vars(data)
    return _unary_scores_from_flat(theta, nv)


def _greedy_scores_factor_inference(pb: Any, data: Any, pred: Any) -> np.ndarray:
    inference_res, _theta = _as_inference_tuple(pred)
    nv = _num_vars(data)
    return inference_res.mu_v.reshape(-1,2)[:,1].detach().cpu().numpy()
    return _unary_scores_from_flat(inference_res.mu, nv)

# ----------------------------
# Greedy decimation backends
# ----------------------------
from bpropy.gd.score_backend import ScoreBackend
def _decimation_backend_var_logits(pb, data, pred, cfg, device=None):
    backend = ScoreBackend()
    backend.reset(pb)
    backend.set_logits(pred)
    return backend

from bpropy.mrf.factor_graph import FactorGraph
from bpropy.gd.factor_logit_backend import FactorLogitAggregationBackend
from bpropy.gd.lbp_backend import LBPBackend
from bpropy.gd.nmf_backend import NMFBackend
from bpropy.gd.flow_backend import FlowLBPBackend, FlowNMFBackend, FlowScoreBackend

def _flow_decimation_backend_var_logits(
    pb,
    data,
    model,
    cfg,
    device=None,
    rollout_steps: Optional[int] = None,
):
    gd = cfg.get("eval", {}).get("greedy_decimation", {})
    mode_args = dict(gd.get("args", {}))

    # This backend accepts rollout_steps separately.
    mode_args.pop("rollout_steps", None)

    return FlowScoreBackend(
        model=model,
        adapter=_METHOD_FLOW_VAR,
        data=data,
        cfg=cfg,
        device=device,
        dtype=torch.float32,
        rollout_steps=rollout_steps,
        **mode_args,
    )

def _decimation_backend_factor_logits(pb, data, pred_theta, cfg, device=None):
    gd = cfg.get("eval", {}).get("greedy_decimation", {})
    mode = gd.get("mode", "aggregate")
    mode_args = gd.get("args", {})

    # IMPORTANT: build fg from pb
    #fg_builder = lambda pb: FactorGraph(pb)

    # IMPORTANT: “freeze” predicted theta for this instance
    def theta_builder(_pb, _fg, _device, _dtype):
        return pred_theta.to(device=_device, dtype=_dtype)

    def fg_builder(pb, mode):
        if mode == "lbp":
            return data.lbp_edges
        if mode == "nmf":
            return data.nmf_edges
        return FactorGraph(pb)

    if mode == "unary":
        return _decimation_backend_var_logits(pb, data, pred_theta, cfg, device=device)

    if mode == "aggregate":
        return FactorLogitAggregationBackend(
            fg_builder=lambda pb: fg_builder(pb, "other"),
            theta=pred_theta.to(device),
            neg_inf=gd.get("neg_inf", None),
            data=data,
        )

    if mode == "loopy_belief_propagation":
        return LBPBackend(
            fg_builder=lambda pb: fg_builder(pb,"lbp"),
            theta_builder=theta_builder,
            device=device,
            dtype=pred_theta.dtype,
            **mode_args,
        )

    if mode == "naive_mean_field":
        return NMFBackend(
            fg_builder=lambda pb: fg_builder(pb, "nmf"),
            theta_builder=theta_builder,
            device=device,
            dtype=pred_theta.dtype,
            **mode_args,
        )

    raise ValueError(f"Unknown decimation mode: {mode}")

def _decimation_backend_factor_marginals(pb, data, pred, cfg, device=None):
    model_cfg = cfg.get("model", {})
    model_args = model_cfg.get("args", {})
    gd = cfg.get("eval", {}).get("greedy_decimation", {})
    mode = gd.get("mode", model_args.get("inference", "aggregate"))
    mode_args = gd.get("args", {})

    inference, pred_theta = pred

    # IMPORTANT: build fg from pb
    def fg_builder(pb, cfg):
        fg_approx_cfg = cfg.get("data",{}).get("precompute", {}).get("fg_approx", None)
        if fg_approx_cfg is not None:
            from bpropy.mrf.factor_graph_approx import APPROX_BUILDERS
            from bpropy.graphs.vc_graph import VCGraph
            vcg = VCGraph(pb)  # builds from pb.constraints(...) which now reflect mutations. :contentReference[oaicite:2]{index=2}
            fg_true = FactorGraph(pb)
            fg_approx,_,_ = APPROX_BUILDERS[fg_approx_cfg.get("method")](pb=pb, vcg=vcg, fg_true=fg_true, approx=fg_approx_cfg)
            return fg_approx
        return FactorGraph(pb)

    # IMPORTANT: “freeze” predicted theta for this instance
    def theta_builder(_pb, _fg, _device, _dtype):
        return pred_theta.to(device=_device, dtype=_dtype)

    if mode == "unary":
        return _decimation_backend_var_logits(pb, data, pred_theta, cfg, device=device)

    if mode == "aggregate":
        return FactorLogitAggregationBackend(
            fg_builder=fg_builder,
            theta=pred_theta.to(device),
            neg_inf=gd.get("neg_inf", None),
            data=data,
        )

    if mode == "loopy_belief_propagation":
        return LBPBackend(
            fg_builder=lambda pb: fg_builder(pb,cfg),
            theta_builder=theta_builder,
            device=device,
            dtype=pred_theta.dtype,
            **mode_args,
        )

    if mode == "naive_mean_field":
        return NMFBackend(
            fg_builder=lambda pb: fg_builder(pb, cfg),
            theta_builder=theta_builder,
            device=device,
            dtype=pred_theta.dtype,
            **mode_args,
        )

    raise ValueError(f"Unknown decimation mode: {mode}")

def _flow_decimation_backend_factor_marginals(
    pb,
    data,
    model,
    cfg,
    device=None,
    rollout_steps: Optional[int] = None,
):
    """
    Flow-conditioned decimation backend.

    Unlike _decimation_backend_factor_marginals, this does not freeze one
    predicted theta. Instead, each scheduled observe() call performs a full
    flow rollout under the current evidence, extracts final theta, applies
    evidence masking, and then runs LBP/NMF.

    Supported modes:
      - loopy_belief_propagation
      - naive_mean_field
    """
    model_cfg = cfg.get("model", {})
    model_args = model_cfg.get("args", {})

    gd = cfg.get("eval", {}).get("greedy_decimation", {})
    mode = gd.get("mode", model_args.get("inference", "loopy_belief_propagation"))
    mode_args = dict(gd.get("args", {}))

    # run_every belongs to the backend scheduling.
    # rollout_steps is controlled separately.
    mode_args.pop("rollout_steps", None)

    def fg_builder(pb, cfg):
        fg_approx_cfg = cfg.get("data", {}).get("precompute", {}).get("fg_approx", None)
        if fg_approx_cfg is not None:
            from bpropy.mrf.factor_graph_approx import APPROX_BUILDERS
            from bpropy.graphs.vc_graph import VCGraph

            vcg = VCGraph(pb)
            fg_true = FactorGraph(pb)
            fg_approx, _, _ = APPROX_BUILDERS[fg_approx_cfg.get("method")](
                pb=pb,
                vcg=vcg,
                fg_true=fg_true,
                approx=fg_approx_cfg,
            )
            return fg_approx

        return FactorGraph(pb)

    def theta_builder(_pb, _fg, _device, _dtype):
        """
        Only used to initialize EvidenceMaskFactory shape/state.
        The actual theta used by observe() comes from the flow rollout.
        """
        with torch.no_grad():
            return model(data, inference="theta").to(device=_device, dtype=_dtype)

    if mode == "loopy_belief_propagation":
        return FlowLBPBackend(
            fg_builder=lambda pb: fg_builder(pb, cfg),
            theta_builder=theta_builder,
            model=model,
            adapter=_METHOD_FLOW_VARIABLE_MARGINALS,
            data=data,
            cfg=cfg,
            device=device,
            dtype=torch.float32,
            rollout_steps=rollout_steps,
            **mode_args,
        )

    if mode == "naive_mean_field":
        return FlowNMFBackend(
            fg_builder=lambda pb: fg_builder(pb, cfg),
            theta_builder=theta_builder,
            model=model,
            adapter=_METHOD_FLOW_VARIABLE_MARGINALS,
            data=data,
            cfg=cfg,
            device=device,
            dtype=torch.float32,
            rollout_steps=rollout_steps,
            **mode_args,
        )

    raise ValueError(
        "Flow decimation backend only supports "
        "'loopy_belief_propagation' and 'naive_mean_field'. "
        f"Got mode={mode!r}."
    )

# ----------------------------
# Greedy decimation problem adapters
# ----------------------------
from bpropy.co.mis import MIS
from bpropy.co.scp import SCP
from bpropy.gd.mis_adapter import MISAdapter
from bpropy.gd.scp_adapter import SCPAdapter
def _decimation_adapter_default(pb: LPProblem, cfg: Any, fix_prop: Optional[float] = None):
    gd_cfg = cfg.get("eval", {}).get("greedy_decimation", {}).get("adapter", {})
    if isinstance(pb, MIS):
        #return MISAdapter()
        return MISAdapter(pick_mode=gd_cfg.get("pick_mode", "min_entropy"),
                          value_mode=gd_cfg.get("value_mode", "argmax"),
                          j_temperature=gd_cfg.get("j_temperature", 1.0),
                          seed=gd_cfg.get("seed", None),
                          fix_prop=fix_prop,
                          )

    if isinstance(pb, SCP):
        return SCPAdapter(pick_mode=gd_cfg.get("pick_mode", "min_entropy"),
                          value_mode=gd_cfg.get("value_mode", "argmax"),
                          j_temperature=gd_cfg.get("j_temperature", 1.0),
                          seed=gd_cfg.get("seed", None),
                          fix_prop=fix_prop,
                          )
    raise Exception(f"unknown problem type :  {type(pb)}")


# ----------------------------
# Registry
# ----------------------------

_METHOD_VAR_LOGITS = MethodAdapter(
    forward_fn=_forward_var_logits,
    loss_spec_fn=_loss_spec_var_logits,
    greedy_scores_fn=_greedy_scores_var_logits,
    decimation_backend_fn=_decimation_backend_var_logits,
    decimation_adapter_fn=_decimation_adapter_default,
)

_METHOD_FACTOR_LOGITS = MethodAdapter(
    forward_fn=_forward_factor_logits,
    loss_spec_fn=_loss_spec_factor_logits,
    greedy_scores_fn=_greedy_scores_factor_logits,
    decimation_backend_fn=_decimation_backend_factor_logits,
    decimation_adapter_fn=_decimation_adapter_default,
)

_METHOD_FACTOR_MARGINALS = MethodAdapter(
    forward_fn=_forward_factor_inference,
    loss_spec_fn=_loss_spec_factor_inference,
    greedy_scores_fn=_greedy_scores_factor_inference,
    decimation_backend_fn=_decimation_backend_factor_marginals,
    decimation_adapter_fn=_decimation_adapter_default,
)

_METHOD_VARIABLE_MARGINALS = MethodAdapter(
    forward_fn=_forward_factor_inference,
    loss_spec_fn=_loss_spec_variable_inference,
    greedy_scores_fn=_greedy_scores_factor_inference,
    decimation_backend_fn=_decimation_backend_factor_marginals,
    decimation_adapter_fn=_decimation_adapter_default,
)


_METHOD_HYBRID_FACTOR_LOGITS = MethodAdapter(
    forward_fn=_forward_hybrid_theta,
    loss_spec_fn=_loss_spec_factor_logits,
    greedy_scores_fn=_greedy_scores_factor_logits,
    decimation_backend_fn=_decimation_backend_factor_marginals,
    decimation_adapter_fn=_decimation_adapter_default,
)

_METHOD_FLOW_VAR = MethodAdapter(
    forward_fn=_forward_var_logits,
    loss_spec_fn=_loss_spec_flow_q1_or_velocity,
    greedy_scores_fn=_greedy_scores_var_logits,
    decimation_backend_fn=_decimation_backend_var_logits,
    decimation_adapter_fn=_decimation_adapter_default,
    flow_decimation_backend_fn=_flow_decimation_backend_var_logits,
    theta_fn=None,
)

_METHOD_FLOW_VARIABLE_MARGINALS = MethodAdapter(
    forward_fn=_forward_factor_inference,
    loss_spec_fn=_loss_spec_flow_q1_or_velocity,
    greedy_scores_fn=_greedy_scores_factor_inference,
    decimation_backend_fn=_decimation_backend_factor_marginals,
    decimation_adapter_fn=_decimation_adapter_default,
    theta_fn=_theta_fn_factor_no_inference,
    flow_decimation_backend_fn=_flow_decimation_backend_flow_variable_marginals,
)

_METHOD_DIFFUSION_VAR = MethodAdapter(
    forward_fn=_forward_var_logits,
    loss_spec_fn=_loss_spec_diffusion,
    greedy_scores_fn=_greedy_scores_var_logits,
    decimation_backend_fn=_decimation_backend_var_logits,
    decimation_adapter_fn=_decimation_adapter_default,
)

_METHOD_DIFFUSION_VARIABLE_MARGINALS = MethodAdapter(
    forward_fn=_forward_factor_inference,
    loss_spec_fn=_loss_spec_diffusion,
    greedy_scores_fn=_greedy_scores_factor_inference,
    decimation_backend_fn=_decimation_backend_factor_marginals,
    decimation_adapter_fn=_decimation_adapter_default,
)


# Keep old names as aliases for backwards compatibility.
METHOD_REGISTRY: Dict[str, MethodAdapter] = {
    "var_logits": _METHOD_VAR_LOGITS,
    "hetero_var_logits": _METHOD_VAR_LOGITS,

    "factor_logits": _METHOD_FACTOR_LOGITS,
    "hetero_factor_logits": _METHOD_FACTOR_LOGITS,

    "factor_marginals": _METHOD_FACTOR_MARGINALS,
    "variable_marginals": _METHOD_VARIABLE_MARGINALS,

    "hetero_factor_marginals": _METHOD_FACTOR_MARGINALS,
    "hetero_factor_marginals": _METHOD_FACTOR_MARGINALS,  # (typo-safe)

    "hybrid_factor_logits": _METHOD_HYBRID_FACTOR_LOGITS,

    # rectified flow
    "flow_variable_logits": _METHOD_FLOW_VAR,
    "flow_variable_marginals": _METHOD_FLOW_VARIABLE_MARGINALS,

    # diffusion
    "diffusion_variable_logits": _METHOD_DIFFUSION_VAR,
    "diffusion_variable_marginals": _METHOD_DIFFUSION_VARIABLE_MARGINALS,

}

