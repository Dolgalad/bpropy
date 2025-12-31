from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

import torch_scatter

from .registry import LOSS_REGISTRY, register_loss


def _flat_factor_sizes_in_ss_order(data, device):
    # Homo: already a single authoritative tensor
    if hasattr(data, "factor_sizes") and torch.is_tensor(data.factor_sizes):
        return data.factor_sizes.to(device=device, dtype=torch.long)

    # Hetero: build factor list = unary variables + constraint factors,
    # then sort by (batch_id, factor_size) to match your SS convention.
    if hasattr(data, "node_types") and ("variable" in data.node_types) and ("constraint" in data.node_types):
        nv = int(data["variable"].num_nodes)
        nc = int(data["constraint"].num_nodes)

        fs_var = torch.ones(nv, device=device, dtype=torch.long)
        fs_con = data["constraint"].factor_sizes.to(device=device, dtype=torch.long)
        fs = torch.cat([fs_var, fs_con], dim=0)

        b_var = data["variable"].batch.to(device=device, dtype=torch.long)  # [nv]
        b_con = data["constraint"].batch.to(device=device, dtype=torch.long)  # [nc]
        b = torch.cat([b_var, b_con], dim=0)

        # Stable lex sort by (batch, factor_size).
        # Do stable sort by secondary key first, then primary.
        idx = torch.arange(fs.numel(), device=device)
        idx = idx[torch.argsort(fs[idx], stable=True)]
        idx = idx[torch.argsort(b[idx], stable=True)]

        return fs[idx]

    raise AttributeError("Cannot infer factor sizes order from data.")


# ---- Simple Binary Cross-Entropy with logits ----
@register_loss("bce_logits")
def loss_bce_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, y.float())

# ---- Factor Cross-Entropy with logits ----
def _factor_strides_from_factor_sizes(data, scores):
    """
    Returns strides (2**|F|) for *factor nodes only* in the same order as data.factor_sizes.
    """
    device = scores.device
    # HeteroData / HeteroDataBatch case
    if hasattr(data, "node_types") and ("variable" in data.node_types) and ("constraint" in data.node_types):
        fs = _flat_factor_sizes_in_ss_order(data, scores.device)
        return (1 << fs)
    fs = data.factor_sizes.to(device=device, dtype=torch.long)  # [n_nodes]
    strides = (1 << fs)
    return strides

def _check_layout(strides, scores, y_ss=None):
    total = int(strides.sum().item())
    if scores.numel() != total:
        raise ValueError(f"scores has length {scores.numel()} but sum(strides)={total}. "
                         "Your flattening order or included factors differ.")
    if y_ss is not None and y_ss.numel() != total:
        raise ValueError(f"y_ss has length {y_ss.numel()} but sum(strides)={total}.")
def _make_m_idx(strides: torch.Tensor) -> torch.Tensor:
    """Map each flattened assignment entry -> factor index."""
    nF = int(strides.numel())
    device = strides.device
    return torch.arange(nF, device=device).repeat_interleave(strides)


def _global_labels_from_y_ss(y_ss: torch.Tensor, m_idx: torch.Tensor, nF: int) -> torch.Tensor:
    """One-hot y_ss -> global index of the true assignment per factor."""
    _, global_labels = torch_scatter.scatter_max(y_ss, m_idx, dim=0, dim_size=nF)
    return global_labels.to(dtype=torch.long, device=y_ss.device)

def _extract_unary_y_ss(data, device: torch.device) -> torch.Tensor:
    """
    Extract unary-factor one-hot targets (2 entries per variable) from data.y_ss,
    without Python loops. Safe for batched y_ss = cat(y_ss_i).
    """
    y = data.y_ss.to(device=device)

    # Factor sizes in the same order as y_ss blocks
    if hasattr(data, "node_types") and ("variable" in data.node_types) and ("constraint" in data.node_types):
        fs = _flat_factor_sizes_in_ss_order(data, device)
    else:
        fs = data.factor_sizes.to(device=device, dtype=torch.long)

    strides = (1 << fs)  # block sizes in y_ss (2**|F|)
    # starts[k] = starting offset of factor k's block in y_ss
    starts = torch.zeros_like(strides)
    starts[1:] = torch.cumsum(strides, dim=0)[:-1]

    # sanity (optional but nice in debug)
    total = int(strides.sum().item())
    if y.numel() != total:
        raise ValueError(f"factor_sizes imply total {total} ss entries but y_ss has {y.numel()}")

    unary_mask = (fs == 1)
    unary_starts = starts[unary_mask]  # [nv_total]

    # Gather 2 entries per unary factor: start and start+1
    #idx = unary_starts[:, None] + torch.tensor([0, 1], device=device)
    idx = unary_starts[:, None] + unary_starts.new_tensor([0, 1])

    return y[idx.reshape(-1)]


def _select_factor_layout_for_marginals(mu: torch.Tensor, theta: torch.Tensor, data) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For marginal-based losses:
      - LBP: mu contains full factor marginals (len == sum(2**|F|) over all factor nodes)
      - NMF: mu contains only unary marginals (len == 2*nv)
    Returns: (strides, y_ss) aligned with mu.
    """
    device = mu.device
    strides_full = _factor_strides_from_factor_sizes(data, theta)
    total_full = int(strides_full.sum().item())

    # infer nv robustly, fall back to mu for NMF
    try:
        nv = _num_vars_from_data(data)
    except AttributeError:
        if mu.numel() % 2 != 0:
            raise
        nv = mu.numel() // 2

    total_unary = 2 * nv

    if mu.numel() == total_full:
        return strides_full.to(device=device), data.y_ss.to(device=device)

    if mu.numel() == total_unary:
        strides = torch.full((nv,), 2, device=device, dtype=torch.long)
        y_ss = _extract_unary_y_ss(data, device=device)
        if y_ss.numel() != total_unary:
            raise ValueError(f"unary y_ss has length {y_ss.numel()} but expected {total_unary}")
        return strides, y_ss
    raise ValueError(
        f"mu has length {mu.numel()} but expected full={total_full} or unary={total_unary}."
    )

def _logp_true_from_logits(scores: torch.Tensor, strides: torch.Tensor, y_ss: torch.Tensor) -> torch.Tensor:
    """
    Returns log p(true assignment | factor) for each factor given unnormalized scores.
    """
    _check_layout(strides, scores, y_ss)

    nF = int(strides.numel())
    m_idx = _make_m_idx(strides)

    global_labels = _global_labels_from_y_ss(y_ss, m_idx, nF)

    # stable logsumexp per factor block
    maxv, _ = torch_scatter.scatter_max(scores, m_idx, dim=0, dim_size=nF)
    sumexp = torch_scatter.scatter_add((scores - maxv[m_idx]).exp(), m_idx, dim=0, dim_size=nF)
    logZ = maxv + sumexp.clamp_min(1e-20).log()

    label_scores = scores[global_labels]
    return label_scores - logZ  # [nF]

@register_loss("fce_logits")
def loss_factor_ce_from_logits_factor_sizes(scores, target, *, data, reduction="mean"):
    strides = _factor_strides_from_factor_sizes(data, scores)
    logp_t = _logp_true_from_logits(scores, strides, data.y_ss.to(device=scores.device))
    nll = -logp_t
    return nll.mean() if reduction == "mean" else nll.sum() if reduction == "sum" else nll

@register_loss("ffce_logits")
def loss_factor_focal_ce_from_logits_factor_sizes(
    scores, target, *, data, reduction="mean", alpha=1.0, gamma=2.0
):
    strides = _factor_strides_from_factor_sizes(data, scores)
    logp_t = _logp_true_from_logits(scores, strides, data.y_ss.to(device=scores.device))
    p_t = logp_t.exp()
    l = -alpha * ((1.0 - p_t).clamp_min(0.0) ** gamma) * logp_t
    return l.mean() if reduction == "mean" else l.sum() if reduction == "sum" else l

# losses for marginal inference
def _num_vars_from_data(data) -> int:
    # HeteroData / HeteroDataBatch: node store accessible both ways sometimes
    if hasattr(data, "node_types") and ("variable" in data.node_types):
        try:
            return int(data["variable"].num_nodes)
        except Exception:
            pass

    # Some PyG batches expose a `.variable` store
    if hasattr(data, "variable"):
        var_store = getattr(data, "variable")
        if hasattr(var_store, "num_nodes") and var_store.num_nodes is not None:
            return int(var_store.num_nodes)
        # fall back to x shape
        if hasattr(var_store, "x") and var_store.x is not None:
            return int(var_store.x.size(0))

    # Common explicit fields (recommended but optional)
    for name in ("num_vars", "n_vars", "num_variables", "n_variables"):
        if hasattr(data, name):
            return int(getattr(data, name))

    # Heuristic: if you store unary factor nodes somewhere
    if hasattr(data, "factor_sizes"):
        fs = data.factor_sizes
        if torch.is_tensor(fs):
            return int((fs == 1).sum().item())

    raise AttributeError(
        "Cannot infer #variables from data (no variable store / num_vars / factor_sizes==1)."
    )
def _p_true_from_marginals(mu: torch.Tensor, strides: torch.Tensor, y_ss: torch.Tensor, eps: float) -> torch.Tensor:
    _check_layout(strides, mu, y_ss)
    nF = int(strides.numel())
    m_idx = _make_m_idx(strides)
    global_labels = _global_labels_from_y_ss(y_ss, m_idx, nF)
    return mu.clamp_min(eps)[global_labels]  # [nF]

@register_loss("fce")
def loss_factor_ce_from_marginals_factor_sizes(pred, target, *, data, reduction="mean", eps=1e-8):
    inference, theta = pred
    mu = inference.mu
    strides, y_ss = _select_factor_layout_for_marginals(mu, theta, data)

    p_t = _p_true_from_marginals(mu, strides, y_ss, eps)
    ce = -p_t.log()
    return ce.mean() if reduction == "mean" else ce.sum() if reduction == "sum" else ce

@register_loss("ffce")
def loss_factor_focal_ce_from_marginals_factor_sizes(
    pred, target, *, data, reduction="mean", eps=1e-8, alpha=1.0, gamma=2.0
):
    inference, theta = pred
    mu = inference.mu
    strides, y_ss = _select_factor_layout_for_marginals(mu, theta, data)

    p_t = _p_true_from_marginals(mu, strides, y_ss, eps)
    l = -alpha * ((1.0 - p_t).clamp_min(0.0) ** gamma) * p_t.log()
    return l.mean() if reduction == "mean" else l.sum() if reduction == "sum" else l



