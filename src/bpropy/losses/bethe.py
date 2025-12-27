from __future__ import annotations
from dataclasses import dataclass
import torch
import torch_scatter

@dataclass
class _InferenceLike:
    mu: torch.Tensor
    mu_v: torch.Tensor

def _factor_softmax_flat(logits: torch.Tensor, strides: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """
    Convert flattened per-assignment logits to flattened per-assignment probabilities
    by doing a softmax independently in each factor block.

    logits: [sum(strides)]
    strides: [nF] where each stride = 2**k_f
    returns: mu: [sum(strides)] with blocks summing to 1
    """
    nF = int(strides.numel())
    device = logits.device

    m_idx = torch.arange(nF, device=device).repeat_interleave(strides)  # assignment -> factor

    maxv, _ = torch_scatter.scatter_max(logits, m_idx, dim=0, dim_size=nF)
    expv = (logits - maxv[m_idx]).exp()
    denom = torch_scatter.scatter_add(expv, m_idx, dim=0, dim_size=nF).clamp_min(eps)
    mu = expv / denom[m_idx]
    return mu


def _extract_unary_mu_v_from_mu_f(
    mu_f: torch.Tensor,
    strides: torch.Tensor,
    data,
) -> torch.Tensor:
    """
    mu_v_mode='unary': take the unary factor marginals (size==2 blocks)
    and concatenate them into a flat (2*nv,) vector.

    Assumes the factor-node order matches `_factor_strides_hetero_or_homo`:
      hetero: variables first (all unary), then constraints
      homo: data.factor_sizes order (unary factors are those with factor_size==1)
    """
    device = mu_f.device

    # Build fs in same order as strides
    if hasattr(data, "node_types") and ("variable" in data.node_types) and ("constraint" in data.node_types):
        nv = int(data["variable"].num_nodes)
        # variables are unary factors by construction in this layout
        unary_mask = torch.zeros_like(strides, dtype=torch.bool, device=device)
        unary_mask[:nv] = True
    else:
        fs = data.factor_sizes.to(device=device, dtype=torch.long)
        unary_mask = (fs == 1)

    # offsets of each block start in the flattened mu_f
    starts = torch.zeros_like(strides, device=device)
    starts[1:] = torch.cumsum(strides, dim=0)[:-1]

    unary_starts = starts[unary_mask]  # [#unary_factors]
    # gather the 2 entries per unary factor: start, start+1
    idx = unary_starts[:, None] + unary_starts.new_tensor([0, 1])
    mu_v = mu_f[idx.reshape(-1)]
    return mu_v

def aggregate_mu_v_from_mu_f_using_edges_hetero(
    mu_f: torch.Tensor,
    data,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Aggregate variable marginals mu_v from factor beliefs mu_f using only hetero edge_index.

    Assumptions:
      - mu_f is flat beliefs for *all factors* in the order:
          [unary factors for variables 0..nv-1] then [constraint factors 0..nc-1]
        i.e. first 2*nv entries correspond to unary beliefs for each variable.
      - For each constraint factor, the assignment indexing / bit positions correspond to
        variables sorted in increasing variable index.
      - data is a PyG HeteroData or HeteroDataBatch with:
          data["variable"].num_nodes
          data["constraint"].factor_sizes   (LongTensor [nc])
          data["variable","v2c","constraint"].edge_index  (LongTensor [2,E]) with rows (var, con)

    Returns:
      mu_v as flat tensor of shape [2*nv].
    """
    device = mu_f.device
    dtype = mu_f.dtype

    # ---- fetch required fields from data ----
    if not (hasattr(data, "node_types") and ("variable" in data.node_types) and ("constraint" in data.node_types)):
        raise ValueError("Expected hetero data with node types 'variable' and 'constraint'.")

    nv = int(data["variable"].num_nodes)
    con_factor_sizes = data["constraint"].factor_sizes.to(device=device, dtype=torch.long)
    nc = int(con_factor_sizes.numel())

    edge_store = data["variable", "v2c", "constraint"]
    edge_index_v2c = edge_store.edge_index.to(device=device, dtype=torch.long)
    if edge_index_v2c.dim() != 2 or edge_index_v2c.size(0) != 2:
        raise ValueError("Expected v2c edge_index of shape [2, E].")

    var = edge_index_v2c[0]
    con = edge_index_v2c[1]

    # ---- 1) unary contribution (first nv unary factors) ----
    if mu_f.numel() < 2 * nv:
        raise ValueError(f"mu_f too short: needs at least 2*nv={2*nv} entries for unary beliefs.")
    mu_unary = mu_f[: 2 * nv].view(nv, 2)

    acc = mu_unary.clone()
    cnt = torch.ones((nv,), device=device, dtype=dtype)

    # ---- 2) constraints: marginalize each constraint belief down to its incident vars ----

    # sort edges by (con, var) to enforce increasing var id within each constraint
    key = con * (nv + 1) + var
    perm = torch.argsort(key)
    con_s = con[perm]
    var_s = var[perm]

    # degrees per constraint from edges; must match factor_sizes
    ones = torch.ones_like(con_s, dtype=dtype)
    deg_c = torch_scatter.scatter_add(ones, con_s, dim=0, dim_size=nc).to(torch.long)

    if deg_c.numel() != nc:
        raise ValueError("Unexpected deg_c size.")
    if not torch.all(deg_c == con_factor_sizes):
        raise ValueError("edge_index degrees per constraint do not match constraint.factor_sizes")

    # start offsets of each constraint in sorted edge list
    starts_edges = torch.cumsum(deg_c, dim=0) - deg_c  # [nc]

    # constraint block starts in mu_f: unary blocks first, then packed constraint blocks
    con_strides = (1 << con_factor_sizes)  # [nc]
    starts_mu = 2 * nv + (torch.cumsum(con_strides, dim=0) - con_strides)  # [nc]

    # loop only over distinct arities (typically small)
    for k in torch.unique(con_factor_sizes).tolist():
        k = int(k)
        c_idx = (con_factor_sizes == k).nonzero(as_tuple=False).view(-1)
        if c_idx.numel() == 0:
            continue

        S = 1 << k

        # neighbors for each constraint: V [n_ck, k]
        V = var_s[starts_edges[c_idx][:, None] + torch.arange(k, device=device)[None, :]]

        # belief blocks for each constraint: mu_blk [n_ck, S]
        blk_idx = starts_mu[c_idx][:, None] + torch.arange(S, device=device)[None, :]
        mu_blk = mu_f[blk_idx]

        # bits matrix: [k, S]
        a = torch.arange(S, device=device)
        bits = ((a[None, :] >> torch.arange(k, device=device)[:, None]) & 1).to(dtype)

        # unary marginals: m1 [n_ck, k], m0 = 1 - m1
        m1 = (mu_blk[:, None, :] * bits[None, :, :]).sum(dim=2)
        m0 = 1.0 - m1

        # scatter-add into variable accumulators
        vid = V.reshape(-1)
        acc0 = m0.reshape(-1)
        acc1 = m1.reshape(-1)

        acc.index_put_((vid, torch.zeros_like(vid)), acc0, accumulate=True)
        acc.index_put_((vid, torch.ones_like(vid)),  acc1, accumulate=True)
        cnt.index_put_((vid,), torch.ones_like(acc0), accumulate=True)

    # average + renormalize
    mu_v = acc / cnt.clamp_min(1.0)[:, None]
    mu_v = mu_v.clamp_min(eps)
    mu_v = mu_v / mu_v.sum(dim=1, keepdim=True).clamp_min(eps)

    return mu_v.reshape(-1)

def aggregate_mu_v_from_mu_f_using_edges_homo(mu_f: torch.Tensor, data, eps: float = 1e-12) -> torch.Tensor:
    device = mu_f.device
    dtype = mu_f.dtype

    fs = data.factor_sizes.to(device=device, dtype=torch.long)  # [n_nodes]
    n_nodes = int(fs.numel())

    e = data.edge_index.to(device=device, dtype=torch.long)     # [2,E] endpoints are NODE ids
    n0 = e[0].to(torch.long)  # node id (should be variable node for var->factor edges)
    n1 = e[1].to(torch.long)  # node id (factor endpoint; includes constraints)

    # variable nodes are those with fs==1
    var_nodes = (fs == 1).nonzero(as_tuple=False).view(-1)      # [n_vars]
    n_vars = int(var_nodes.numel())

    # map node id -> compact variable id
    node2vid = torch.full((n_nodes,), -1, device=device, dtype=torch.long)
    node2vid[var_nodes] = torch.arange(n_vars, device=device, dtype=torch.long)

    vid0 = node2vid[n0]   # [-1 for non-var nodes]
    # keep only edges from variable nodes to constraint/factor nodes (fs>1)
    fac_mask = (fs[n1] > 1) & (vid0 >= 0)
    vid = vid0[fac_mask]           # compact var ids in [0,n_vars)
    fac_node = n1[fac_mask]        # global factor NODE ids
    if fac_node.numel() == 0:
        # fallback: just use unary beliefs as mu_v
        # unary blocks in mu_f are in fs order; gather them:
        strides = (1 << fs)
        starts = torch.zeros_like(strides)
        starts[1:] = torch.cumsum(strides, dim=0)[:-1]
        unary_starts = starts[fs == 1]
        mu_v = mu_f[(unary_starts[:, None] + unary_starts.new_tensor([0, 1])).reshape(-1)]
        return mu_v

    # baseline acc/cnt from unary beliefs (in the SAME order as var_nodes)
    # unary blocks in mu_f are ordered by node-id order among fs==1 nodes
    strides = (1 << fs)
    starts_mu = torch.zeros_like(strides)
    starts_mu[1:] = torch.cumsum(strides, dim=0)[:-1]
    unary_starts = starts_mu[fs == 1]                              # [n_vars]
    mu_unary = mu_f[(unary_starts[:, None] + unary_starts.new_tensor([0, 1])).reshape(-1)].view(n_vars, 2)

    acc = mu_unary.clone()
    cnt = torch.ones((n_vars,), device=device, dtype=dtype)

    # sort edges by (fac_node, vid) so within each factor variables are in increasing vid
    key = fac_node * (n_vars + 1) + vid
    perm = torch.argsort(key)
    fac_s = fac_node[perm]
    vid_s = vid[perm]

    # degrees per factor NODE id (global)
    ones = torch.ones_like(fac_s, dtype=dtype)
    deg_all = torch_scatter.scatter_add(ones, fac_s, dim=0, dim_size=n_nodes).to(torch.long)

    fac_nodes = (fs > 1).nonzero(as_tuple=False).view(-1)
    if not torch.all(deg_all[fac_nodes] == fs[fac_nodes]):
        bad = fac_nodes[(deg_all[fac_nodes] != fs[fac_nodes]).nonzero(as_tuple=False).view(-1)]
        raise ValueError(f"homo: degrees mismatch for factor nodes. Sample bad nodes: {bad[:10].tolist()}")

    # start offsets into sorted edge list for each factor NODE id
    prefix = torch.cumsum(deg_all, dim=0)
    starts_edges = prefix - deg_all

    # Now marginalize per arity bucket
    fac_sizes = fs[fac_nodes]
    for k in torch.unique(fac_sizes).tolist():
        k = int(k)
        nodes_k = fac_nodes[(fs[fac_nodes] == k).nonzero(as_tuple=False).view(-1)]
        if nodes_k.numel() == 0:
            continue
        S = 1 << k

        # variable ids per factor: [n_k, k]
        V = vid_s[starts_edges[nodes_k][:, None] + torch.arange(k, device=device)[None, :]]

        # factor belief blocks for these factor nodes: [n_k, S]
        blk_idx = starts_mu[nodes_k][:, None] + torch.arange(S, device=device)[None, :]
        mu_blk = mu_f[blk_idx]

        a = torch.arange(S, device=device)
        bits = ((a[None, :] >> torch.arange(k, device=device)[:, None]) & 1).to(dtype)

        m1 = (mu_blk[:, None, :] * bits[None, :, :]).sum(dim=2)
        m0 = 1.0 - m1

        vv = V.reshape(-1)
        acc.index_put_((vv, torch.zeros_like(vv)), m0.reshape(-1), accumulate=True)
        acc.index_put_((vv, torch.ones_like(vv)),  m1.reshape(-1), accumulate=True)
        cnt.index_put_((vv,), torch.ones_like(vv, dtype=dtype), accumulate=True)

    mu_v = acc / cnt.clamp_min(1.0)[:, None]
    mu_v = mu_v.clamp_min(eps)
    mu_v = mu_v / mu_v.sum(dim=1, keepdim=True).clamp_min(eps)
    return mu_v.reshape(-1)

def aggregate_mu_v_from_mu_f_using_edges(mu_f: torch.Tensor, data, eps: float = 1e-12) -> torch.Tensor:
    if hasattr(data, "node_types") and ("variable" in data.node_types) and ("constraint" in data.node_types):
        return aggregate_mu_v_from_mu_f_using_edges_hetero(mu_f, data, eps=eps)
    return aggregate_mu_v_from_mu_f_using_edges_homo(mu_f, data, eps=eps)


def _factor_strides_hetero_or_homo(data, device):
    # Same logic as _factor_strides_from_factor_sizes but returns also a factor->graph batch if available.
    if hasattr(data, "node_types") and ("variable" in data.node_types) and ("constraint" in data.node_types):
        nv = int(data["variable"].num_nodes)
        var_sizes = torch.ones(nv, device=device, dtype=torch.long)
        con_sizes = data["constraint"].factor_sizes.to(device=device, dtype=torch.long)
        fs = torch.cat([var_sizes, con_sizes], dim=0)  # variables first, then constraints

        # batching: constraint store has batch for constraints; variables store has batch for variables
        if hasattr(data["variable"], "batch") and data["variable"].batch is not None:
            vb = data["variable"].batch.to(device=device)
        else:
            vb = torch.zeros(nv, device=device, dtype=torch.long)

        if hasattr(data["constraint"], "batch") and data["constraint"].batch is not None:
            cb = data["constraint"].batch.to(device=device)
        else:
            cb = torch.zeros(int(con_sizes.numel()), device=device, dtype=torch.long)

        f_batch = torch.cat([vb, cb], dim=0)
        return (1 << fs), f_batch

    # homo case
    fs = data.factor_sizes.to(device=device, dtype=torch.long)
    strides = (1 << fs)
    if hasattr(data, "batch") and data.batch is not None:
        f_batch = data.batch.to(device=device, dtype=torch.long)
    elif hasattr(data, "factor_batch") and data.factor_batch is not None:
        f_batch = data.factor_batch.to(device=device, dtype=torch.long)
    else:
        f_batch = torch.zeros(int(fs.numel()), device=device, dtype=torch.long)
    return strides, f_batch


def bethe_free_energy_from_beliefs(
    *,
    mu_f: torch.Tensor,
    mu_v: torch.Tensor,
    theta: torch.Tensor,
    strides: torch.Tensor,
    f_batch: torch.Tensor,
    deg: torch.Tensor | None = None,
    v_batch: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Returns per-graph Bethe free energy F_B (shape [B]).
    Sign convention: F_B = -sum_f <theta_f> + sum_f sum mu log mu - sum_i (d_i-1) sum mu_i log mu_i
    """
    device = theta.device
    dtype = theta.dtype

    if theta.shape != mu_f.shape:
        raise ValueError(f"theta.shape {tuple(theta.shape)} must match mu_f.shape {tuple(mu_f.shape)}")

    nF = int(strides.numel())
    if nF == 0:
        return torch.zeros((1,), device=device, dtype=dtype)

    # assignment -> factor index
    m_idx = torch.arange(nF, device=device).repeat_interleave(strides)

    # num graphs from f_batch
    B = int(f_batch.max().item()) + 1 if f_batch.numel() else 1

    # per-factor <theta> and entropy
    mu_f_cl = mu_f.clamp_min(eps)
    e_per_f = torch.zeros(nF, device=device, dtype=dtype).scatter_add_(0, m_idx, mu_f * theta)
    h_per_f = torch.zeros(nF, device=device, dtype=dtype).scatter_add_(0, m_idx, mu_f * mu_f_cl.log())

    e_g = torch.zeros(B, device=device, dtype=dtype).scatter_add_(0, f_batch, e_per_f)
    h_g = torch.zeros(B, device=device, dtype=dtype).scatter_add_(0, f_batch, h_per_f)

    F_g = (-e_g) + h_g

    # variable correction term
    if mu_v is None:
        return F_g

    if mu_v.numel() % 2 != 0:
        raise ValueError(f"mu_v must be length 2*n_vars, got {mu_v.numel()}")
    nv = mu_v.numel() // 2

    mu_v2 = mu_v.view(nv, 2)
    mu_v2_cl = mu_v2.clamp_min(eps)
    h_v = (mu_v2 * mu_v2_cl.log()).sum(dim=1)  # sum_x mu log mu  (negative)

    if deg is None:
        deg = torch.ones(nv, device=device, dtype=dtype)

    if v_batch is None:
        v_batch = torch.zeros(nv, device=device, dtype=torch.long)
    else:
        v_batch = v_batch.to(device=device)

    Bv = int(v_batch.max().item()) + 1 if v_batch.numel() else 1
    if Bv > B:
        F_new = torch.zeros(Bv, device=device, dtype=dtype)
        F_new[:B] = F_g
        F_g = F_new
        B = Bv

    v_term = -(deg - 1.0) * h_v  # - (d_i-1) * sum mu log mu
    v_term_g = torch.zeros(B, device=device, dtype=dtype).scatter_add_(0, v_batch, v_term)

    return F_g + v_term_g


def loss_nll_bethe(pred, target, *, data, reduction="mean", eps=1e-12):
    """
    Bethe-approximated negative log-likelihood:
        NLL_B = - <theta, y_ss> + logZ_B
             ≈ - <theta, y_ss> - F_B(mu; theta)
    Requires LBP-like inference returning full factor marginals (mu over ALL factors).
    """
    inference, theta = pred
    device = theta.device

    mu_f = inference.mu
    mu_v = getattr(inference, "mu_v", None)

    strides, f_batch = _factor_strides_hetero_or_homo(data, device=device)
    total = int(strides.sum().item())
    if mu_f.numel() != total:
        raise ValueError(
            f"loss_bethe_nll requires full factor marginals from LBP. "
            f"Got mu length {mu_f.numel()} but expected sum(strides)={total}."
        )

    y_ss = data.y_ss.to(device=device)
    if y_ss.numel() != theta.numel():
        raise ValueError(f"y_ss length {y_ss.numel()} must match theta length {theta.numel()}")

    # ground-truth unnormalized log score per *graph*
    m_idx = torch.arange(int(strides.numel()), device=device).repeat_interleave(strides)
    # pick the true assignment entry per factor (one-hot -> argmax index)
    _, global_labels = torch_scatter.scatter_max(y_ss, m_idx, dim=0, dim_size=int(strides.numel()))
    true_scores_per_factor = theta[global_labels]  # theta_f(x*_f)

    E_true_g = torch.zeros(
        int(f_batch.max().item()) + 1 if f_batch.numel() else 1,
        device=device,
        dtype=theta.dtype,
    ).scatter_add_(0, f_batch, true_scores_per_factor)

    # degrees/batching for variable correction term (optional but recommended)
    deg = None
    v_batch = None
    if hasattr(data, "variable") and hasattr(data.variable, "neighbors") and data.variable.neighbors is not None:
        deg = data.variable.neighbors.to(device=device, dtype=theta.dtype)
    if hasattr(data, "variable") and hasattr(data.variable, "batch") and data.variable.batch is not None:
        v_batch = data.variable.batch.to(device=device)

    F_g = bethe_free_energy_from_beliefs(
        mu_f=mu_f,
        mu_v=mu_v,
        theta=theta,
        strides=strides,
        f_batch=f_batch,
        deg=deg,
        v_batch=v_batch,
        eps=eps,
    )
    logZ_g = -F_g
    nll_g = -E_true_g + logZ_g

    if reduction == "none":
        return nll_g
    if reduction == "sum":
        return nll_g.sum()
    return nll_g.mean()

def factor_logits_to_inference_like(
    factor_logits: torch.Tensor,
    *,
    data,
    mu_v_mode: str = "unary",
    eps_softmax: float = 1e-20,
) -> _InferenceLike:
    """
    Turn flattened factor logits into an object with .mu (factor marginals) and .mu_v (variable marginals).
    Currently supports mu_v_mode='unary' (extract variable marginals from unary factors).
    """
    device = factor_logits.device
    strides, _ = _factor_strides_hetero_or_homo(data, device=device)

    total = int(strides.sum().item())
    if factor_logits.numel() != total:
        raise ValueError(
            f"factor_logits has length {factor_logits.numel()} but expected sum(strides)={total}."
        )

    mu_f = _factor_softmax_flat(factor_logits, strides, eps=eps_softmax)

    if mu_v_mode == "unary":
        mu_v = _extract_unary_mu_v_from_mu_f(mu_f, strides, data)
        return _InferenceLike(mu=mu_f, mu_v=mu_v)
    if mu_v_mode == "aggregate":
        #mu_v = aggregate_mu_v_from_mu_f_using_edges_hetero(mu_f=mu_f, data=data, eps=eps_softmax)
        mu_v = aggregate_mu_v_from_mu_f_using_edges(mu_f=mu_f, data=data, eps=eps_softmax)
        return _InferenceLike(mu=mu_f, mu_v=mu_v)

    raise ValueError(f"Unknown mu_v_mode={mu_v_mode!r}. Supported: 'unary'.")

#def loss_nll_bethe_from_logits(
#    pred, target, *, data, reduction="mean", eps=1e-12, mu_v_mode: str = "unary"
#):
#    """
#    Bethe NLL where pred is (factor_logits, theta) instead of (inference, theta).
#    Converts logits -> marginals, fabricates an inference-like object, then reuses loss_nll_bethe.
#    """
#    print(pred)
#    factor_logits, theta = pred
#    inf = factor_logits_to_inference_like(factor_logits, data=data, mu_v_mode=mu_v_mode)
#    return loss_nll_bethe((inf, theta), target, data=data, reduction=reduction, eps=eps)

def loss_nll_bethe_from_logits(
    pred, target, *, data, reduction="mean", eps=1e-12, mu_v_mode: str = "unary"
):
    """
    Factor-logits path: pred is theta (flat log-potentials / logits over factor assignments).
    We convert theta -> beliefs (mu, mu_v) and reuse loss_nll_bethe((inf, theta), ...).
    """
    theta = pred  # <-- IMPORTANT: pred is a single tensor in factor_logits mode
    inf = factor_logits_to_inference_like(theta, data=data, mu_v_mode=mu_v_mode)
    return loss_nll_bethe((inf, theta), target, data=data, reduction=reduction, eps=eps)

