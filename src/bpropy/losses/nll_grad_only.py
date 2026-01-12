import torch
import torch_scatter

from .registry import register_loss
from .bethe import factor_logits_to_inference_like, _factor_strides_hetero_or_homo  # adjust import if inside bethe.py


@register_loss("nll_grad_only")
def loss_nll_grad_only(
    pred,
    target,
    *,
    data,
    reduction="mean",
    eps=1e-12,
    mu_v_mode: str = "unary",
    eps_softmax: float = 1e-20,
):
    """
    Gradient-only NLL surrogate.

    Supports:
      - factor_marginals method: pred = (inference_like, theta)
            uses inference_like.mu as mu_ss (flat, same shape as theta)
      - factor_logits method: pred = theta
            builds inference_like via factor_logits_to_inference_like(theta, ...)

    The returned scalar has gradient: d/dtheta = mu_ss - y_ss.
    Value is NOT the true NLL (since logZ is not computed).

    Args:
      reduction: "mean" | "sum" | "none" (per-graph)
      mu_v_mode: passed to factor_logits_to_inference_like when pred is theta-only
      eps_softmax: stability for softmax when building beliefs from logits
    """
    # --- unpack theta and mu_ss ---
    if isinstance(pred, tuple) and len(pred) == 2:
        inference, theta = pred
        mu_ss = inference.mu
    else:
        theta = pred
        inf = factor_logits_to_inference_like(theta, data=data, mu_v_mode=mu_v_mode, eps_softmax=eps_softmax)
        mu_ss = inf.mu

    device = theta.device
    theta = theta.to(device=device)
    mu_ss = mu_ss.to(device=device)

    # --- target sufficient statistics (flat one-hot) ---
    y_ss = data.y_ss.to(device=device)
    if y_ss.numel() != theta.numel():
        raise ValueError(f"y_ss length {y_ss.numel()} must match theta length {theta.numel()}")
    if mu_ss.numel() != theta.numel():
        raise ValueError(f"mu_ss length {mu_ss.numel()} must match theta length {theta.numel()}")

    # --- gradient-only trick: detach mu_ss ---
    delta = mu_ss.detach() - y_ss  # ensures d/dtheta = mu_ss - y_ss

    if hasattr(data, "fg_map"):
        val = (theta * delta).sum()
        if reduction == "none":
            # optional: per-graph via data.ptr or data.batch if theta is node-aligned (often it isn't)
            return val.unsqueeze(0)
        if reduction == "sum":
            return val
        if reduction == "mean":
            # mean over graphs if available, else leave as sum
            B = int(getattr(data, "num_graphs", 1))
            return val / max(B, 1)

    # --- produce per-graph losses (like your Bethe NLL does) ---
    strides, f_batch = _factor_strides_hetero_or_homo(data, device=device)
    nF = int(strides.numel())
    if nF == 0:
        out = theta.new_zeros((1,))
        return out if reduction == "none" else out.sum() * 0.0

    # assignment -> factor index
    m_idx = torch.arange(nF, device=device).repeat_interleave(strides)

    # assignment -> graph index via factor batch
    g_idx = f_batch[m_idx]

    # per-graph scalar = sum over assignments belonging to factors in that graph
    B = int(g_idx.max().item()) + 1 if g_idx.numel() else 1
    per_graph = torch.zeros(B, device=device, dtype=theta.dtype).scatter_add_(0, g_idx, theta * delta)

    if reduction == "none":
        return per_graph
    if reduction == "sum":
        return per_graph.sum()
    if reduction == "mean":
        return per_graph.mean()
    raise ValueError(f"Unknown reduction={reduction!r}")

