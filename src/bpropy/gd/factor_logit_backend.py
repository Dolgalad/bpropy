from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F
import torch_scatter
from torch_geometric.data import HeteroData

Evidence = Dict[int, int]


@dataclass
class FactorLogitAggregationBackend:
    fg_builder: Callable[[Any], Any]
    theta: torch.Tensor
    neg_inf: Optional[float] = None
    data: Optional[Any] = None
    eps: float = 1e-12

    _fg: Any = None
    _evidence_factory: Any = None
    #_idx: Optional[Any] = None
    _cache: Optional[Any] = None
    _mu_v: Optional[torch.Tensor] = None

    def reset(self, pb: Any) -> None:
        from bpropy.mrf.evidence_factory import EvidenceMaskFactory  # you already use this elsewhere

        self._fg = self.fg_builder(pb)
        self._evidence_factory = EvidenceMaskFactory(
            self._fg, theta=self.theta, device=self.theta.device, neg_inf=self.neg_inf
        )
        #self._cache = build_factor_neighbor_index_homo(self.data, device=self.theta.device)

        if isinstance(self.data, HeteroData) or hasattr(self.data, "node_types"):
            self._cache = build_factor_neighbor_index_hetero(self.data, device=self.theta.device)
        else:
            self._cache = build_factor_neighbor_index_homo(self.data, device=self.theta.device)


        self._mu_v = None

    def update(self, evidence: Evidence) -> None:
        if evidence:
            self._evidence_factory.add_evidence(evidence, strict=True)
            self._mu_v = None

    def observe(self) -> torch.Tensor:
        if self._mu_v is not None:
            return self._mu_v

        th = self._evidence_factory.apply(self.theta)
        mu_v_flat = aggregate_mu_v_scatter(th.clone(), self._cache, eps=self.eps)
        self._mu_v = mu_v_flat.view(-1,2)

        return self._mu_v

def build_assignment_index_from_fg(fg, device: torch.device) -> dict:
    """
    Precompute scatter indices for fast aggregation of variable marginals from factor beliefs.

    Returns a dict with:
      - offsets: (nF,) start offset of each factor block in theta
      - strides: (nF,) block size (=2**k)
      - m_idx:   (M,)  assignment -> factor id
      - v_idx:   (M*K,) expanded mapping assignment->variable (one entry per (assignment, local var pos))
      - bit:     (M*K,) 0/1 whether that assignment sets that variable to 1
      - cnt:     (nV,) number of factor-contributions per variable (for averaging)
      - nV: int
      - nF: int
    Assumes factor assignment indexing uses local variable order fg.factor_variables(Fn)
    with bit p = (a >> p) & 1 (same convention you were using).
    """
    # number of variables
    if hasattr(fg, "nv"):
        nV = int(fg.nv())
    else:
        nV = sum(1 for _ in fg.iter_variables())

    # factor sizes and variable lists
    factor_vars = []
    strides = []
    for Fn in fg.iter_factors():
        vs = list(fg.factor_variables(Fn))
        k = len(vs)  # or fg.factor_size(Fn) if that exists
        factor_vars.append(torch.tensor(vs, device=device, dtype=torch.long))
        strides.append(1 << k)

    strides = torch.tensor(strides, device=device, dtype=torch.long)
    nF = int(strides.numel())
    offsets = torch.zeros(nF, device=device, dtype=torch.long)
    if nF > 0:
        offsets[1:] = torch.cumsum(strides[:-1], dim=0)

    # assignment -> factor id
    m_idx = torch.arange(nF, device=device, dtype=torch.long).repeat_interleave(strides)

    # Expanded per-(assignment, local position) arrays
    # We'll build v_idx and bit by concatenating each factor's contribution.
    v_idx_parts = []
    bit_parts = []

    for iF in range(nF):
        twk = int(strides[iF].item())
        vs = factor_vars[iF]          # (k,)
        k = int(vs.numel())

        a = torch.arange(twk, device=device, dtype=torch.long)  # assignment index within block

        # For each local pos p, build length twk vector of bits (a>>p)&1
        # and map to global variable id vs[p].
        for p in range(k):
            v_idx_parts.append(vs[p].expand(twk))              # (twk,)
            bit_parts.append(((a >> p) & 1).to(torch.long))    # (twk,)

    v_idx = torch.cat(v_idx_parts, dim=0) if v_idx_parts else torch.empty(0, device=device, dtype=torch.long)
    bit = torch.cat(bit_parts, dim=0) if bit_parts else torch.empty(0, device=device, dtype=torch.long)

    # counts: how many factor-contributions each variable receives
    cnt = torch.zeros(nV, device=device, dtype=torch.float32)
    if v_idx.numel() > 0:
        cnt.scatter_add_(0, v_idx, torch.ones_like(v_idx, dtype=torch.float32))

    return dict(
        nV=nV, nF=nF,
        offsets=offsets, strides=strides,
        m_idx=m_idx,
        v_idx=v_idx,
        bit=bit,
        cnt=cnt,
    )

def _aggregate_mu_v_from_theta_fg(fg, theta: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """
    Aggregates variable marginals from per-factor softmax beliefs under (possibly clamped) theta.
    Returns mu_v with shape [n_vars, 2].
    """
    device, dtype = theta.device, theta.dtype

    # number of vars
    if hasattr(fg, "nv"):
        nV = int(fg.nv())
    else:
        nV = sum(1 for _ in fg.iter_variables())

    mu_acc = torch.zeros((nV, 2), device=device, dtype=dtype)
    mu_cnt = torch.zeros((nV, 1), device=device, dtype=dtype)

    # iterate factors; assume theta is concatenated per-factor blocks in this same order
    offset = 0
    for Fn in fg.iter_factors():
        k = int(fg.factor_size(Fn))
        twk = 1 << k
        blk = theta[offset : offset + twk]
        offset += twk

        mu_f = F.softmax(blk, dim=0)

        vars_in_F = list(fg.factor_variables(Fn))  # local order used by assignment encoding
        a = torch.arange(twk, device=device)
        for p, v in enumerate(vars_in_F):
            bit = (a >> p) & 1
            mu_acc[int(v), 0] += mu_f[bit == 0].sum()
            mu_acc[int(v), 1] += mu_f[bit == 1].sum()
            mu_cnt[int(v), 0] += 1.0

    mu_v = mu_acc / (mu_cnt + eps)
    mu_v = torch.nan_to_num(mu_v, nan=0.0)
    mu_v = mu_v / mu_v.sum(dim=1, keepdim=True).clamp_min(eps)
    return mu_v

def build_factor_neighbor_index_homo(data, *, device=None):
    """
    Returns a dict of tensors you can cache in reset(pb):

      fac_nodes: (n_fac,) global node ids where factor_sizes>1
      fs_fac:    (n_fac,) arities k for each factor node in fac_nodes
      starts_mu: (n_nodes,) start offset of each node's belief block in mu_f (node-order packed)
      V_flat:    (sum_k n_fac_k * k,) flattened neighbor var-ids in the *sorted* (factor,var) order
      starts_edges: (n_nodes,) start offset into V_flat for each node id (0 where deg=0)
      deg_all:   (n_nodes,) degree of each node in the sorted edge list

    Assumes homo encoding where:
      - data.factor_sizes is per NODE id
      - data.edge_index are NODE ids
      - variable nodes are those with fs==1, factor nodes fs>1
    """
    if device is None:
        device = data.edge_index.device

    fs = data.factor_sizes.to(device=device, dtype=torch.long)   # [n_nodes]
    n_nodes = int(fs.numel())

    strides = (1 << fs)  # (n_nodes,) block size per node in packed theta/mu_f
    m_idx = torch.arange(n_nodes, device=device, dtype=torch.long).repeat_interleave(strides)

    e = data.edge_index.to(device=device, dtype=torch.long)      # [2,E]
    n0, n1 = e[0], e[1]

    # variable nodes are fs==1
    var_nodes = (fs == 1).nonzero(as_tuple=False).view(-1)
    n_vars = int(var_nodes.numel())

    # map node id -> compact var id in [0,n_vars)
    node2vid = torch.full((n_nodes,), -1, device=device, dtype=torch.long)
    node2vid[var_nodes] = torch.arange(n_vars, device=device, dtype=torch.long)

    vid0 = node2vid[n0]
    fac_mask = (fs[n1] > 1) & (vid0 >= 0)  # keep var -> factor edges
    vid = vid0[fac_mask]                  # compact var ids
    fac_node = n1[fac_mask]               # global factor node ids

    # Sort by (factor_node, vid) so neighbor positions align with bit positions.
    key = fac_node * (n_vars + 1) + vid
    perm = torch.argsort(key)
    fac_s = fac_node[perm]
    vid_s = vid[perm]

    # degrees per *node id* in sorted edge list
    ones = torch.ones_like(fac_s, dtype=torch.float32)
    deg_all = torch_scatter.scatter_add(ones, fac_s, dim=0, dim_size=n_nodes).to(torch.long)

    # sanity: factor node degree should match factor_sizes
    fac_nodes = (fs > 1).nonzero(as_tuple=False).view(-1)
    if fac_nodes.numel() and not torch.all(deg_all[fac_nodes] == fs[fac_nodes]):
        bad = fac_nodes[(deg_all[fac_nodes] != fs[fac_nodes]).nonzero(as_tuple=False).view(-1)]
        raise ValueError(f"degree != factor_sizes for some factor nodes; sample bad: {bad[:10].tolist()}")

    # start offsets into sorted edge list for each node id
    prefix = torch.cumsum(deg_all, dim=0)
    starts_edges = prefix - deg_all

    # start offsets into mu_f block packing by node id (node-order packed by 2^fs)
    strides = (1 << fs)  # [n_nodes]
    starts_mu = torch.zeros_like(strides)
    starts_mu[1:] = torch.cumsum(strides, dim=0)[:-1]

    return dict(
        fs=fs,
        n_vars=n_vars,
        fac_nodes=fac_nodes,
        starts_mu=starts_mu,
        vid_s=vid_s,
        fac_s=fac_s,
        deg_all=deg_all,
        starts_edges=starts_edges,
        m_idx=m_idx,
        strides=strides,
    )
def build_factor_neighbor_index_hetero(data, *, device=None, var_type="variable", fac_type="constraint",
                                       v2f=("variable", "v2c", "constraint")):
    """
    Build the same cache as build_factor_neighbor_index_homo, but for PyG HeteroDataBatch.

    Assumptions consistent with your training batches:
      - factor sizes stored at data[fac_type].factor_sizes (per factor node)
      - variable nodes are binary => fs=1 for all variable nodes
      - edges var->fac in data[v2f].edge_index with shape [2,E] mapping
            edge_index[0] = var_id in [0,n_vars)
            edge_index[1] = fac_id in [0,n_factors)
    """
    if device is None:
        device = data[v2f].edge_index.device

    fs_fac = data[fac_type].factor_sizes.to(device=device, dtype=torch.long)  # [n_factors]
    n_fac = int(fs_fac.numel())
    n_var = int(data[var_type].num_nodes)

    # Build a unified "node id space" where variables come first, then factors:
    # node 0..n_var-1 are variables, node n_var..n_var+n_fac-1 are factors.
    n_nodes = n_var + n_fac
    fs = torch.empty(n_nodes, device=device, dtype=torch.long)
    fs[:n_var] = 1
    fs[n_var:] = fs_fac

    strides = (1 << fs)  # block size per node
    m_idx = torch.arange(n_nodes, device=device, dtype=torch.long).repeat_interleave(strides)

    # edges
    e = data[v2f].edge_index.to(device=device, dtype=torch.long)
    vid = e[0]                          # [E] in 0..n_var-1
    fac_local = e[1]                    # [E] in 0..n_fac-1
    fac_node = fac_local + n_var        # shift into unified node space

    # sort by (factor_node, vid) to match the bit-position convention used in your reference aggregation
    key = fac_node * (n_var + 1) + vid
    perm = torch.argsort(key)
    fac_s = fac_node[perm]
    vid_s = vid[perm]  # NOTE: already compact var ids 0..n_var-1

    # degree per node in unified space
    ones = torch.ones_like(fac_s, dtype=torch.float32)
    deg_all = torch_scatter.scatter_add(ones, fac_s, dim=0, dim_size=n_nodes).to(torch.long)

    # sanity: factor degree must match factor_sizes
    fac_nodes = torch.arange(n_var, n_nodes, device=device, dtype=torch.long)
    if fac_nodes.numel() and not torch.all(deg_all[fac_nodes] == fs[fac_nodes]):
        bad = fac_nodes[(deg_all[fac_nodes] != fs[fac_nodes]).nonzero(as_tuple=False).view(-1)]
        raise ValueError(f"degree != factor_sizes for some factor nodes; sample bad: {bad[:10].tolist()}")

    prefix = torch.cumsum(deg_all, dim=0)
    starts_edges = prefix - deg_all

    starts_mu = torch.zeros_like(strides)
    starts_mu[1:] = torch.cumsum(strides, dim=0)[:-1]

    return dict(
        fs=fs,
        n_vars=n_var,
        fac_nodes=fac_nodes,
        starts_mu=starts_mu,
        vid_s=vid_s,
        fac_s=fac_s,
        deg_all=deg_all,
        starts_edges=starts_edges,
        m_idx=m_idx,
        strides=strides,
        n_var=n_var,
        n_fac=n_fac,
    )

def softmax_per_factor_flat(theta: torch.Tensor, cache: dict, *, eps: float = 1e-12) -> torch.Tensor:
    """
    theta: flat, packed by node-id order with block size 2**fs[node]
    cache must contain:
      - nF: number of nodes
      - strides: (n_nodes,) = 2**fs
      - m_idx: (M,) assignment -> node id (or factor id)
    Returns:
      mu_f: flat, same shape as theta
    """
    m_idx = cache["m_idx"]
    n_nodes = int(cache["strides"].numel())

    # per-node max for stability
    max_per = torch.full((n_nodes,), -float("inf"), device=theta.device, dtype=theta.dtype)
    max_per = max_per.scatter_reduce(0, m_idx, theta, reduce="amax", include_self=True)

    exp = torch.exp(theta - max_per[m_idx])
    sum_per = torch.zeros((n_nodes,), device=theta.device, dtype=theta.dtype)
    sum_per.scatter_add_(0, m_idx, exp)

    return exp / sum_per[m_idx].clamp_min(eps)


def aggregate_mu_v_scatter(theta: torch.Tensor, cache: dict, *, eps: float = 1e-12) -> torch.Tensor:
    """
    Your desired signature: (theta, idx/cache, eps) -> mu_v_flat.

    theta is expected to already be clamped by evidence (i.e., -inf for incompatible assignments).
    cache is built once in reset(pb) via build_factor_neighbor_index_homo(...).
    """
    mu_f = softmax_per_factor_flat(theta, cache, eps=eps)
    return aggregate_mu_v_scatter_from_mu_f_homo(mu_f, cache=cache, eps=eps)


def aggregate_mu_v_scatter_from_mu_f_homo(mu_f: torch.Tensor, cache: dict, *, eps: float = 1e-12) -> torch.Tensor:
    """
    mu_f -> mu_v_flat using only cached indices (no data needed at runtime).
    Returns flat mu_v of shape (2*n_vars,).
    """
    device = mu_f.device
    dtype = mu_f.dtype

    fs = cache["fs"]                      # (n_nodes,)
    n_vars = int(cache["n_vars"])
    fac_nodes = cache["fac_nodes"]        # factor node ids (fs>1)
    starts_mu = cache["starts_mu"]        # start offset into mu_f per node id
    vid_s = cache["vid_s"]                # sorted var-ids for each (var->factor) edge
    starts_edges = cache["starts_edges"]  # start offset into vid_s per node id

    # unary baseline from unary nodes
    unary_nodes = (fs == 1).nonzero(as_tuple=False).view(-1)
    unary_starts = starts_mu[unary_nodes]                      # (n_vars,)
    mu_unary = mu_f[(unary_starts[:, None] + torch.tensor([0, 1], device=device)).reshape(-1)].view(n_vars, 2)

    acc0 = mu_unary[:, 0].clone()
    acc1 = mu_unary[:, 1].clone()
    cnt  = torch.ones((n_vars,), device=device, dtype=dtype)

    fac_sizes = fs[fac_nodes]
    for k in torch.unique(fac_sizes).tolist():
        k = int(k)
        nodes_k = fac_nodes[(fac_sizes == k).nonzero(as_tuple=False).view(-1)]
        if nodes_k.numel() == 0:
            continue

        S = 1 << k

        # neighbor var ids per factor: (n_k, k)
        V = vid_s[starts_edges[nodes_k][:, None] + torch.arange(k, device=device)[None, :]]

        # factor belief blocks: (n_k, S)
        blk_idx = starts_mu[nodes_k][:, None] + torch.arange(S, device=device)[None, :]
        mu_blk = mu_f[blk_idx]

        a = torch.arange(S, device=device)
        bits = ((a[None, :] >> torch.arange(k, device=device)[:, None]) & 1).to(dtype)

        m1 = (mu_blk[:, None, :] * bits[None, :, :]).sum(dim=2)  # (n_k,k)
        m0 = 1.0 - m1

        vv = V.reshape(-1)
        acc0 = acc0.scatter_add(0, vv, m0.reshape(-1))
        acc1 = acc1.scatter_add(0, vv, m1.reshape(-1))
        cnt  = cnt.scatter_add(0, vv, torch.ones_like(m0.reshape(-1), dtype=dtype))

    mu0 = acc0 / cnt.clamp_min(1.0)
    mu1 = acc1 / cnt.clamp_min(1.0)
    mu = torch.stack([mu0, mu1], dim=1).clamp_min(eps)
    mu = mu / mu.sum(dim=1, keepdim=True).clamp_min(eps)
    return mu.reshape(-1)
