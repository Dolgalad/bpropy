from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from bpropy.losses.regularizers import REGULARIZER_REGISTRY


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
        print("term = ", term, w * term)
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
    _apply_regularizers(spec, adapter, pred, labels, cfg)
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
    return pred.detach().cpu().numpy()


def _greedy_scores_factor_logits(pb: Any, data: Any, pred: Any) -> np.ndarray:
    theta = _as_theta(pred)
    nv = _num_vars(data)
    return _unary_scores_from_flat(theta, nv)


def _greedy_scores_factor_inference(pb: Any, data: Any, pred: Any) -> np.ndarray:
    inference_res, _theta = _as_inference_tuple(pred)
    nv = _num_vars(data)
    return _unary_scores_from_flat(inference_res.mu, nv)


# ----------------------------
# Registry
# ----------------------------

_METHOD_VAR_LOGITS = MethodAdapter(
    forward_fn=_forward_var_logits,
    loss_spec_fn=_loss_spec_var_logits,
    greedy_scores_fn=_greedy_scores_var_logits,
)

_METHOD_FACTOR_LOGITS = MethodAdapter(
    forward_fn=_forward_factor_logits,
    loss_spec_fn=_loss_spec_factor_logits,
    greedy_scores_fn=_greedy_scores_factor_logits,
)

_METHOD_FACTOR_MARGINALS = MethodAdapter(
    forward_fn=_forward_factor_inference,
    loss_spec_fn=_loss_spec_factor_inference,
    greedy_scores_fn=_greedy_scores_factor_inference,
)

_METHOD_HYBRID_FACTOR_LOGITS = MethodAdapter(
    forward_fn=_forward_hybrid_theta,
    loss_spec_fn=_loss_spec_factor_logits,
    greedy_scores_fn=_greedy_scores_factor_logits,
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

