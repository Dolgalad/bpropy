from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import numpy as np

@dataclass
class LossSpec:
    # Standard loss path:
    pred: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None
    loss_name: Optional[str] = None          # e.g. "bce_logits", "ce", ...
    loss_kwargs: dict = field(default_factory=dict)
    # Extra scalar terms (regularizers, Bethe, etc.)
    extra_terms: dict = field(default_factory=dict)  # name->scalar tensor

    # Gradient-only path:
    # If provided, training loop calls this and skips loss_fn(pred,target)
    backward_fn: Optional[Callable[[], torch.Tensor]] = None
    # backward_fn should do .backward() internally and return a logging scalar

@dataclass
class MethodAdapter:
    """
    Wrap model forward + how to compute:
      - supervision loss input (prediction tensor)
      - scores to pass to pb.greedy(scores=...)
    """
    forward_fn: Callable[[torch.nn.Module, Any], torch.Tensor]
    loss_spec_fn: Callable[[Any,torch.Tensor,torch.Tensor,Any], LossSpec]
    greedy_scores_fn: Callable[[Any, Any, torch.Tensor], np.ndarray]  # (pb, prediction)->scores


# Variable Logits method (non-hetero graph) : for each variable node of the graph predict a logit value and compute loss wrt ground truth assignment
def _forward_var_logits(model: torch.nn.Module, data: Any) -> torch.Tensor:
    return model(data)[data.bipartite==0]

def _loss_spec_var_logits(batch, pred, labels, cfg, phase="train"):
    return LossSpec(pred=pred, 
                    target=labels.to(pred.device, non_blocking=True).float(), 
                    loss_name=cfg.get("loss", {}).get("name", "bce_logits")
                    )

def _greedy_scores_var_logits(pb: Any, data: Any, pred: torch.Tensor) -> np.ndarray:
    return pred.detach().cpu().numpy()

# Variable Logits method (hetero graph) : for each variable node of the graph predict a logit value and compute loss wrt ground truth assignment
def _forward_hetero_var_logits(model: torch.nn.Module, data: Any) -> torch.Tensor:
    out = model(data)
    if isinstance(out, dict):
        return out["variable"]
    return out

# Factor Logits method (non-hetero graph) : for each variable (factor of size 1) and constraint nodes (factor of size > 1) predict 2**k log-probabilities
def _forward_factor_logits(model: torch.nn.Module, data: Any) -> torch.Tensor:
    theta = model(data)
    return theta
def _loss_spec_factor_logits(batch, pred_theta, labels, cfg):
    loss_name = cfg["loss"]["name"]
    target = batch.y_ss
    #target = labels.to(pred_theta.device, non_blocking=True)
    # optionally cast depending on loss
    if loss_name in ("mse", "l1"):
        target = target.float()
    return LossSpec(pred=pred_theta, target=target, loss_name=loss_name, loss_kwargs={"data": batch, **cfg["loss"].get("args",{})})
def _greedy_scores_factor_logits(pb: Any, data: Any, pred: torch.Tensor):
    # Hetero model might return dicts; we only need the flattened theta vector
    if isinstance(pred, dict):
        # If your hetero factor-logits model outputs a flat vector, keep that.
        # Otherwise, you must define how to flatten; here we assume it provides "theta".
        pred = pred.get("theta", pred.get("variable"))

    # HeteroDataBatch: nv from variable store
    if hasattr(data, "node_types") and ("variable" in data.node_types):
        nv = int(data["variable"].num_nodes)
    else:
        # Homo Data: nv from bipartite mask (variable nodes are bipartite==0)
        nv = int((data.bipartite == 0).sum().item())

    #nv = (data.bipartite==0).sum().item()
    return pred[:2*nv].reshape(-1, 2)[:,1].cpu().numpy()

# Factor Marginal inference method (non-hetero graph) : for each variable (factor of size 1) and constraint nodes (factor of size > 1) predict 2**k probabilities
def _forward_factor_inference(model: torch.nn.Module, data: Any) -> torch.Tensor:
    inference_res, theta = model(data)
    return inference_res, theta
def _loss_spec_factor_inference(batch, pred, labels, cfg):
    loss_name = cfg["loss"]["name"]
    target = batch.y_ss
    #target = labels.to(pred_theta.device, non_blocking=True)
    # optionally cast depending on loss
    if loss_name in ("mse", "l1"):
        target = target.float()
    return LossSpec(pred=pred, target=target, loss_name=loss_name, loss_kwargs={"data": batch, **cfg["loss"].get("args",{})})
def _greedy_scores_factor_inference(pb: Any, data: Any, pred: torch.Tensor):
    # Hetero model might return dicts; we only need the flattened theta vector
    if isinstance(pred, dict):
        # If your hetero factor-logits model outputs a flat vector, keep that.
        # Otherwise, you must define how to flatten; here we assume it provides "theta".
        pred = pred.get("theta", pred.get("variable"))

    # HeteroDataBatch: nv from variable store
    if hasattr(data, "node_types") and ("variable" in data.node_types):
        nv = int(data["variable"].num_nodes)
    else:
        # Homo Data: nv from bipartite mask (variable nodes are bipartite==0)
        nv = int((data.bipartite == 0).sum().item())

    inference, theta = pred
    mu = inference.mu

    #nv = (data.bipartite==0).sum().item()
    return mu[:2*nv].reshape(-1, 2)[:,1].cpu().numpy()


METHOD_REGISTRY: Dict[str, MethodAdapter] = {
    "var_logits": MethodAdapter(
        forward_fn=_forward_var_logits,
        loss_spec_fn=_loss_spec_var_logits,
        greedy_scores_fn=_greedy_scores_var_logits,
    ),
    "hetero_var_logits": MethodAdapter(
        forward_fn=_forward_hetero_var_logits,
        loss_spec_fn=_loss_spec_var_logits,
        greedy_scores_fn=_greedy_scores_var_logits,
    ),
    "factor_logits": MethodAdapter(
        forward_fn=_forward_factor_logits,
        loss_spec_fn=_loss_spec_factor_logits,
        greedy_scores_fn=_greedy_scores_factor_logits,

    ),
    "hetero_factor_logits": MethodAdapter(
        forward_fn=_forward_factor_logits,
        loss_spec_fn=_loss_spec_factor_logits,
        greedy_scores_fn=_greedy_scores_factor_logits,

    ),
    "factor_marginals": MethodAdapter(
        forward_fn=_forward_factor_inference,
        loss_spec_fn=_loss_spec_factor_inference,
        greedy_scores_fn=_greedy_scores_factor_inference,

    ),
    "hetero_factor_marginals": MethodAdapter(
        forward_fn=_forward_factor_inference,
        loss_spec_fn=_loss_spec_factor_inference,
        greedy_scores_fn=_greedy_scores_factor_inference,

    ),

    # Later:
    # "lbp": forward returns lbp_res, theta; greedy scores = mu_v[:,1] :contentReference[oaicite:8]{index=8}
    # "factor_logits": greedy scores = logits[:2*nv].reshape(-1,2)[:,1].exp() :contentReference[oaicite:9]{index=9}
}

