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
    print("in _forward_var_logits: using .bipartite")
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

# ----------------------------
# Greedy decimation problem adapters
# ----------------------------
from bpropy.co.mis import MIS
from bpropy.co.scp import SCP
from bpropy.gd.mis_adapter import MISAdapter
from bpropy.gd.scp_adapter import SCPAdapter
def _decimation_adapter_default(pb: LPProblem, cfg: Any):
    gd_cfg = cfg.get("eval", {}).get("greedy_decimation", {}).get("adapter", {})
    print(gd_cfg)
    if isinstance(pb, MIS):
        return MISAdapter()
    if isinstance(pb, SCP):
        return SCPAdapter(pick_mode=gd_cfg.get("pick_mode", "min_entropy"),
                          value_mode=gd_cfg.get("value_mode", "argmax"),
                          j_temperature=gd_cfg.get("j_temperature", 1.0),
                          seed=gd_cfg.get("seed", None)
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

_METHOD_HYBRID_FACTOR_LOGITS = MethodAdapter(
    forward_fn=_forward_hybrid_theta,
    loss_spec_fn=_loss_spec_factor_logits,
    greedy_scores_fn=_greedy_scores_factor_logits,
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
    "hetero_factor_marginals": _METHOD_FACTOR_MARGINALS,
    "hetero_factor_marginals": _METHOD_FACTOR_MARGINALS,  # (typo-safe)

    "hybrid_factor_logits": _METHOD_HYBRID_FACTOR_LOGITS,
}

