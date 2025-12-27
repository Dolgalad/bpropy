import torch
import torch_scatter

from .registry import register_loss

def _edge_unary_marginals_from_factor_beliefs_hetero(mu_f: torch.Tensor, data, eps: float = 1e-12):
    """
    Returns:
      var_s: [E] variable ids (sorted within each factor)
      edge_marg: [E,2] unary marginals from the incident constraint factor
    Only uses constraint factors (not unary factors).
    """
    device = mu_f.device
    dtype = mu_f.dtype

    nv = int(data["variable"].num_nodes)
    con_sizes = data["constraint"].factor_sizes.to(device=device, dtype=torch.long)
    nc = int(con_sizes.numel())

    e = data["variable", "v2c", "constraint"].edge_index.to(device=device, dtype=torch.long)
    var = e[0]
    con = e[1]

    # sort edges by (con, var) so var order is increasing within each constraint
    key = con * (nv + 1) + var
    perm = torch.argsort(key)
    con_s = con[perm]
    var_s = var[perm]

    # degree per constraint
    ones = torch.ones_like(con_s, dtype=dtype)
    deg_c = torch_scatter.scatter_add(ones, con_s, dim=0, dim_size=nc).to(torch.long)
    if not torch.all(deg_c == con_sizes):
        raise ValueError("hetero: edge degrees per constraint != constraint.factor_sizes")

    starts_edges = torch.cumsum(deg_c, dim=0) - deg_c  # [nc]

    # constraint blocks in mu_f start after unary variable blocks
    con_strides = (1 << con_sizes)  # [nc]
    starts_mu = 2 * nv + (torch.cumsum(con_strides, dim=0) - con_strides)  # [nc]

    # Build [E,2] edge unary marginals by arity buckets (small loop over unique k)
    edge_marg = torch.empty((var_s.numel(), 2), device=device, dtype=dtype)

    # We need, for each constraint, the list of its incident edges in sorted order.
    # Within that list, position p = 0..k-1 corresponds to bit position p in the factor assignment index.
    for k in torch.unique(con_sizes).tolist():
        k = int(k)
        c_idx = (con_sizes == k).nonzero(as_tuple=False).view(-1)
        if c_idx.numel() == 0:
            continue
        S = 1 << k

        # edges belonging to these constraints are contiguous in the sorted list, but we need their indices
        # We'll gather per-constraint edge rows and then place results back.
        edge_rows = starts_edges[c_idx][:, None] + torch.arange(k, device=device)[None, :]  # [n_ck,k]
        edge_rows_flat = edge_rows.reshape(-1)  # indices into var_s / edge_marg

        # factor belief blocks: [n_ck, S]
        blk = starts_mu[c_idx][:, None] + torch.arange(S, device=device)[None, :]
        mu_blk = mu_f[blk]  # [n_ck,S]

        # bits matrix: [k, S]
        a = torch.arange(S, device=device)
        bits = ((a[None, :] >> torch.arange(k, device=device)[:, None]) & 1).to(dtype)

        # m1: [n_ck,k] = sum_a mu_blk[a] * bits[p,a]
        m1 = (mu_blk[:, None, :] * bits[None, :, :]).sum(dim=2)
        m0 = 1.0 - m1

        # write into edge_marg for the corresponding edges
        edge_marg[edge_rows_flat, 0] = m0.reshape(-1)
        edge_marg[edge_rows_flat, 1] = m1.reshape(-1)

    # numerical cleanup
    edge_marg = edge_marg.clamp_min(eps)
    edge_marg = edge_marg / edge_marg.sum(dim=1, keepdim=True).clamp_min(eps)
    return var_s, edge_marg

def _edge_unary_marginals_from_factor_beliefs_homo(mu_f: torch.Tensor, data, eps: float = 1e-12):
    """
    Returns:
      vid_s: [E_fac] compact variable ids (0..n_vars-1), sorted within each factor
      edge_marg: [E_fac,2] unary marginals from the incident higher-order factor
    Uses only factor nodes with fs>1.
    """
    device = mu_f.device
    dtype = mu_f.dtype

    fs = data.factor_sizes.to(device=device, dtype=torch.long)  # [n_nodes]
    n_nodes = int(fs.numel())

    e = data.edge_index.to(device=device, dtype=torch.long)     # [2,E] node ids
    n0 = e[0].to(torch.long)  # variable node id
    n1 = e[1].to(torch.long)  # factor node id

    # variable nodes are those with fs==1; map to compact variable id
    var_nodes = (fs == 1).nonzero(as_tuple=False).view(-1)
    n_vars = int(var_nodes.numel())
    node2vid = torch.full((n_nodes,), -1, device=device, dtype=torch.long)
    node2vid[var_nodes] = torch.arange(n_vars, device=device, dtype=torch.long)

    vid0 = node2vid[n0]
    mask = (vid0 >= 0) & (fs[n1] > 1)
    vid = vid0[mask]       # compact variable id
    fac = n1[mask]         # factor node id (global)

    if fac.numel() == 0:
        return vid, torch.empty((0, 2), device=device, dtype=dtype)

    # block starts for each node in mu_f (fs-order)
    strides = (1 << fs)
    starts_mu = torch.zeros_like(strides)
    starts_mu[1:] = torch.cumsum(strides, dim=0)[:-1]  # [n_nodes]

    # sort edges by (fac, vid) so vid order is increasing within each factor
    key = fac * (n_vars + 1) + vid
    perm = torch.argsort(key)
    fac_s = fac[perm]
    vid_s = vid[perm]

    # degrees per factor node id
    ones = torch.ones_like(fac_s, dtype=dtype)
    deg_all = torch_scatter.scatter_add(ones, fac_s, dim=0, dim_size=n_nodes).to(torch.long)

    fac_nodes = (fs > 1).nonzero(as_tuple=False).view(-1)
    if not torch.all(deg_all[fac_nodes] == fs[fac_nodes]):
        raise ValueError("homo: edge degrees per factor node != factor_sizes for fs>1 nodes")

    prefix = torch.cumsum(deg_all, dim=0)
    starts_edges = prefix - deg_all  # [n_nodes], valid for factor nodes

    # fill per-edge marginals
    edge_marg = torch.empty((vid_s.numel(), 2), device=device, dtype=dtype)

    # bucket by arity (small loop)
    fac_sizes = fs[fac_nodes]
    for k in torch.unique(fac_sizes).tolist():
        k = int(k)
        nodes_k = fac_nodes[(fs[fac_nodes] == k).nonzero(as_tuple=False).view(-1)]
        if nodes_k.numel() == 0:
            continue
        S = 1 << k

        edge_rows = starts_edges[nodes_k][:, None] + torch.arange(k, device=device)[None, :]
        edge_rows_flat = edge_rows.reshape(-1)

        blk = starts_mu[nodes_k][:, None] + torch.arange(S, device=device)[None, :]
        mu_blk = mu_f[blk]  # [n_k,S]

        a = torch.arange(S, device=device)
        bits = ((a[None, :] >> torch.arange(k, device=device)[:, None]) & 1).to(dtype)

        m1 = (mu_blk[:, None, :] * bits[None, :, :]).sum(dim=2)
        m0 = 1.0 - m1

        edge_marg[edge_rows_flat, 0] = m0.reshape(-1)
        edge_marg[edge_rows_flat, 1] = m1.reshape(-1)

    edge_marg = edge_marg.clamp_min(eps)
    edge_marg = edge_marg / edge_marg.sum(dim=1, keepdim=True).clamp_min(eps)
    return vid_s, edge_marg

@register_loss("local_consistency_l2")
def loss_local_consistency_l2(mu_f: torch.Tensor, mu_v: torch.Tensor, data, eps: float = 1e-12, reduction="mean"):
    """
    L2 local-consistency penalty between variable marginals mu_v and factor-implied unary marginals.
    mu_v: flat [2*n_vars]
    mu_f: flat over all factors (same as in your bethe.py layout)
    """
    device = mu_f.device
    mu_v2 = mu_v.view(-1, 2).to(device=device)

    if hasattr(data, "node_types") and ("variable" in data.node_types) and ("constraint" in data.node_types):
        vid, edge_marg = _edge_unary_marginals_from_factor_beliefs_hetero(mu_f, data, eps=eps)
    else:
        vid, edge_marg = _edge_unary_marginals_from_factor_beliefs_homo(mu_f, data, eps=eps)

    if edge_marg.numel() == 0:
        return torch.zeros((), device=device, dtype=mu_f.dtype)

    diff = edge_marg - mu_v2[vid]                 # [E,2]
    per_edge = (diff * diff).sum(dim=1)           # [E]

    if reduction == "none":
        return per_edge
    if reduction == "sum":
        return per_edge.sum()
    return per_edge.mean()

@register_loss("local_consistency_kl")
def loss_local_consistency_kl(mu_f: torch.Tensor, mu_v: torch.Tensor, data, eps: float = 1e-12, reduction="mean"):
    """
    KL local-consistency penalty: KL(edge_marg || mu_v[var]).
    edge_marg is factor-implied unary marginal for that variable.
    """
    device = mu_f.device
    mu_v2 = mu_v.view(-1, 2).to(device=device).clamp_min(eps)
    mu_v2 = mu_v2 / mu_v2.sum(dim=1, keepdim=True).clamp_min(eps)

    if hasattr(data, "node_types") and ("variable" in data.node_types) and ("constraint" in data.node_types):
        vid, edge_marg = _edge_unary_marginals_from_factor_beliefs_hetero(mu_f, data, eps=eps)
    else:
        vid, edge_marg = _edge_unary_marginals_from_factor_beliefs_homo(mu_f, data, eps=eps)

    if edge_marg.numel() == 0:
        return torch.zeros((), device=device, dtype=mu_f.dtype)

    p = edge_marg.clamp_min(eps)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
    q = mu_v2[vid]  # [E,2]

    per_edge = (p * (p.log() - q.log())).sum(dim=1)  # [E]

    if reduction == "none":
        return per_edge
    if reduction == "sum":
        return per_edge.sum()
    return per_edge.mean()

