# bpropy/losses/regularizers/local_consistency_regs.py

from __future__ import annotations
from typing import Any, Dict, Tuple

import torch

from bpropy.losses.regularizers.registry import register_regularizer
from bpropy.losses.local_consistency import (
    loss_local_consistency_l2,
    loss_local_consistency_kl,
)
from bpropy.losses.bethe import factor_logits_to_inference_like


def _infer_mu_from_pred(
    pred: Any,
    *,
    data: Any,
    cfg: Dict[str, Any],
    eps_softmax: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (mu_f, mu_v) as flat tensors.

    Supports:
      1) pred = (inference_res, theta)  with inference_res.mu (+ inference_res.mu_v optional)
      2) pred = theta (factor logits)   -> use factor_logits_to_inference_like(theta, ...)
    """
    # Case 1: inference output
    if isinstance(pred, tuple) and len(pred) == 2:
        inference_res, _theta = pred
        if not hasattr(inference_res, "mu"):
            raise ValueError("Expected inference_res.mu for local-consistency regularizer.")
        mu_f = inference_res.mu
        mu_v = getattr(inference_res, "mu_v", None)
        if mu_v is None:
            raise ValueError("Expected inference_res.mu_v for local-consistency regularizer.")
        return mu_f, mu_v

    # Case 2: factor-logits path (pred is theta/logits)
    if torch.is_tensor(pred):
        mu_v_mode = cfg.get("loss", {}).get("args", {}).get("mu_v_mode", "unary")
        inf = factor_logits_to_inference_like(
            pred,
            data=data,
            mu_v_mode=mu_v_mode,
            eps_softmax=eps_softmax,
        )
        return inf.mu, inf.mu_v

    raise TypeError(f"Unsupported pred type for local-consistency regularizer: {type(pred)}")


@register_regularizer("local_consistency_l2")
def local_consistency_l2(
    adapter: Any,
    data: Any,
    pred: Any,
    labels: torch.Tensor,
    cfg: Dict[str, Any],
    **kwargs,
) -> torch.Tensor:
    """
    kwargs:
      eps: float = 1e-12          (numerical stability inside penalty)
      eps_softmax: float = 1e-20  (only used if pred is logits and we build beliefs via softmax)
      reduction: 'mean'|'sum'|'none' = 'mean'
    """
    eps = float(kwargs.pop("eps", 1e-12))
    eps_softmax = float(kwargs.pop("eps_softmax", 1e-20))
    reduction = kwargs.pop("reduction", "mean")
    if kwargs:
        raise TypeError(f"Unexpected kwargs: {list(kwargs.keys())}")

    mu_f, mu_v = _infer_mu_from_pred(pred, data=data, cfg=cfg, eps_softmax=eps_softmax)
    return loss_local_consistency_l2(mu_f=mu_f, mu_v=mu_v, data=data, eps=eps, reduction=reduction)


@register_regularizer("local_consistency_kl")
def local_consistency_kl(
    adapter: Any,
    data: Any,
    pred: Any,
    labels: torch.Tensor,
    cfg: Dict[str, Any],
    **kwargs,
) -> torch.Tensor:
    """
    kwargs:
      eps: float = 1e-12
      eps_softmax: float = 1e-20
      reduction: 'mean'|'sum'|'none' = 'mean'
    """
    eps = float(kwargs.pop("eps", 1e-12))
    eps_softmax = float(kwargs.pop("eps_softmax", 1e-20))
    reduction = kwargs.pop("reduction", "mean")
    if kwargs:
        raise TypeError(f"Unexpected kwargs: {list(kwargs.keys())}")

    mu_f, mu_v = _infer_mu_from_pred(pred, data=data, cfg=cfg, eps_softmax=eps_softmax)
    return loss_local_consistency_kl(mu_f=mu_f, mu_v=mu_v, data=data, eps=eps, reduction=reduction)

