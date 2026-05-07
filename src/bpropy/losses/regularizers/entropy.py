from __future__ import annotations
from typing import Any, Dict
import torch
import torch.nn.functional as F

from bpropy.losses.regularizers.registry import register_regularizer

@register_regularizer("entropy_bernoulli")
def entropy_bernoulli(
    adapter: Any,
    data: Any,
    pred: Any,
    labels: torch.Tensor,
    cfg: Dict[str, Any],
    **kwargs,
) -> torch.Tensor:
    """
    Encourages high-entropy Bernoulli predictions for var-logits.
    Returns NEGATIVE entropy so that adding +lambda*reg maximizes entropy.

    kwargs:
      eps: float = 1e-12
      reduction: 'mean'|'sum'|'none' = 'mean'
    """
    #print("in entropy_bernoulli ", data)
    #print("\t", pred)
    #print("\t", labels.shape)
    eps = float(kwargs.pop("eps", 1e-12))
    reduction = kwargs.pop("reduction", "mean")
    if kwargs:
        raise TypeError(f"Unexpected kwargs: {list(kwargs.keys())}")

    #if not torch.is_tensor(pred):
    #    raise TypeError(f"entropy_bernoulli expects pred to be a Tensor of logits, got {type(pred)}")

    if isinstance(pred, tuple):
        if len(pred) >= 2:
            p = pred[0].mu_v.reshape(-1,2)[:,1]
    else:
        logits = pred
        p = torch.sigmoid(logits)
    ent = -(p * torch.log(p.clamp_min(eps)) + (1 - p) * torch.log((1 - p).clamp_min(eps)))

    if reduction == "mean":
        ent = ent.mean()
    elif reduction == "sum":
        ent = ent.sum()
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    print(" ent = ", ent)
    return -ent

