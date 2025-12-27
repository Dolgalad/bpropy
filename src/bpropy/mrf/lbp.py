import torch
import networkx as nx
import itertools
import numpy as np
import time
import torch_scatter

class BPSolution:
    def __init__(self, mu=None, u=None, n_iter=None, mu_v=None):
        self.mu = mu
        self.u = u
        self.n_iter = n_iter
        self.mu_v = mu_v

def loopy_belief_propagation(theta, v2f, edges, max_iter=100, tol=1e-6, damping=0.0):
    # strides
    dev = theta.device
    edges = edges.to(dev)
    q_stride = torch.full((int(v2f.shape[0]/2),), 2, device=dev)
    q_idx = torch.arange(q_stride.size(0), device=dev).repeat_interleave(2)
    #m_stride = torch.tensor([2**len(list(g.neighbors(F))) for F in iter_factor_nodes(g)])
    repeats = edges.m_stride.to(device=theta.device, dtype=torch.long)
    m_idx = torch.arange(repeats.size(0), device=theta.device).repeat_interleave(repeats)
    mu = torch.zeros(int(edges.n_ss.sum().item()), device=dev)
    
    nv = edges.n_vars.sum() #number_of_variables(g)
    vmar = torch.zeros(2 * nv, dtype=theta.dtype, device=dev)
    vmar_log = torch.zeros_like(vmar)

    stack = torch.empty(theta.shape[0] + v2f.shape[0], dtype=theta.dtype, device=theta.device)
    v2f_subsum = torch.empty(edges.n_f2v_ss.sum(), dtype=theta.dtype, device=theta.device)
    qbar = torch.empty_like(v2f)
    mufbar = torch.empty_like(theta)

    for T in range(max_iter):
        #t0 = time.time()

        #stack = torch.concatenate((theta,v2f), 0)
        stack[:theta.shape[0]].copy_(theta)
        stack[theta.shape[0]:].copy_(v2f)
        # subsum vect
        #v2f_subsum = torch.zeros(edges.n_f2v_ss.sum(), dtype=theta.dtype, device=theta.device)
        v2f_subsum.zero_()
        v2f_subsum.scatter_add_(0, edges.f2v_edges[:,0], stack[edges.f2v_edges[:,1]])
        
        r = torch_scatter.scatter_logsumexp(v2f_subsum[edges.r_edges[:,1]], edges.r_edges[:,0])
        
        # variable to factor messages
        qbar.zero_()
        qbar.scatter_add_(0, edges.q_edges[:,0], r[edges.q_edges[:,1]])
        #qbar = torch.zeros_like(v2f).scatter_add_(0, edges.q_edges[:,0], r[edges.q_edges[:,1]])
        delta = torch_scatter.scatter_logsumexp(qbar, q_idx).repeat_interleave(2)
        new_v2f = qbar - delta

        # damping on messages
        v2f = (1-damping)*new_v2f + damping*v2f
        
        # marginals
        mufbar.copy_(theta).scatter_add_(0, edges.m_edges[:,0], v2f[edges.m_edges[:,1]])
        #mufbar = theta.clone().scatter_add_(0, edges.m_edges[:,0], v2f[edges.m_edges[:,1]])
        z = torch_scatter.scatter_logsumexp(mufbar, m_idx).repeat_interleave(edges.m_stride, dim=0)
    
        mup = (mufbar - z).exp()

        u = (mup - mu).abs().max()
        if torch.any(torch.isinf(mup)) or torch.any(torch.isnan(mup)):
            break

        mu = mup
        #mu = (1 - damping) * mup + damping * mu

        # marginals
        vmar.zero_().scatter_add_(0, edges.vm_edges[:,0], r[edges.vm_edges[:,1]])
        # normalize per variable (2 states)
        zv = torch.logsumexp(vmar.view(-1,2), dim=1, keepdim=True)
        vmar_log_norm = (vmar.view(-1,2) - zv).view(-1)
        # probabilities if/when needed:
        vmar_p = vmar_log_norm.exp()

        #vmar = torch.zeros_like(vmar).scatter_add_(0, edges.vm_edges[:,0], r[edges.vm_edges[:,1]])

        if u < tol:
            break
    return BPSolution(mu, u, T, mu_v=vmar_p)

def loopy_belief_propagation_with_graph(
    theta: torch.Tensor,
    v2f: torch.Tensor,
    edges,
    max_iter: int = 50,
    tol: float = 1e-6,
    damping: float = 0.0,              # message damping (mix old/new v2f)
    capture_iters: int = 10,           # how many iterations to capture in the graph
):
    """
    CUDA-Graphs accelerated variant of loopy_belief_propagation with the same signature.
    Falls back to the baseline implementation when CUDA or torch_scatter is unavailable.
    """
    # ---- Fallbacks ---------------------------------------------------------
    device = theta.device
    want_cuda = (device.type == "cuda")
    if (not want_cuda) or (scatter_logsumexp is None) or (not torch.cuda.is_available()):
        # No CUDA (or no torch_scatter): use your baseline solver
        return loopy_belief_propagation(theta, v2f, edges, max_iter, tol, damping)

    # Ensure edges are on the right device
    edges = edges.to(device) if hasattr(edges, "to") else edges

    # ---- Sizes & static buffers -------------------------------------------
    SS_total  = int(edges.n_ss.sum().item())
    V2F_total = int(edges.n_f2v_msg.sum().item())

    # Number of r-accumulator slots (pre–logsumexp buckets)
    # (use explicit field if you expose it; else compute from f2v_edges)
    r_slots = int(edges.f2v_edges[:, 0].max().item()) + 1

    dtype = theta.dtype

    # Static inputs/outputs that keep their addresses between replays
    static_theta = torch.empty(SS_total,  device=device, dtype=dtype)
    static_v2f   = torch.empty(V2F_total, device=device, dtype=dtype)
    static_mu    = torch.empty(SS_total,  device=device, dtype=dtype)             # factor beliefs (probs)
    static_vmar  = torch.empty(2 * int(edges.n_vars.sum().item()), device=device, dtype=dtype)  # variable beliefs (probs)

    # Temps reused inside captured iterations
    stack   = torch.empty(SS_total + V2F_total, device=device, dtype=dtype)
    r_acc   = torch.empty(r_slots, device=device, dtype=dtype)     # sums before logsumexp
    r_msg   = torch.empty(V2F_total, device=device, dtype=dtype)   # r_{F->i} (log-domain)
    qbar    = torch.empty(V2F_total, device=device, dtype=dtype)   # sum of incoming r's except self
    mufbar  = torch.empty(SS_total,  device=device, dtype=dtype)   # θ + sum incoming to each factor cfg
    z_fac   = torch.empty(int(edges.n_factors.sum().item()) if hasattr(edges, "n_factors") else int(edges.m_idx.max().item()+1), device=device, dtype=dtype)

    # Precompute a per-message normalization index (one id per message, repeated for 2 states)
    if hasattr(edges, "q_idx"):
        q_idx = edges.q_idx
    else:
        # assume pairs [0,1],[2,3],...
        q_idx = torch.arange(V2F_total // 2, device=device).repeat_interleave(2)

    # ---- Warm-up to materialize kernels -----------------------------------
    static_theta.copy_(theta)      # initial inputs
    static_v2f.copy_(v2f)

    # Initialize buffers
    stack.zero_(); r_acc.zero_(); r_msg.zero_(); qbar.zero_(); mufbar.zero_(); static_mu.zero_(); static_vmar.zero_()

    # ---- Capture one chunk of iterations ----------------------------------
    g = torch.cuda.CUDAGraph()

    def one_iteration_sumproduct():
        # Build [ SS | V2F ] stack view
        stack[:SS_total].copy_(static_theta)
        stack[SS_total:].copy_(static_v2f)

        # r-accumulation: sum contributions per r-slot
        r_acc.zero_()
        # gather sources from stack by f2v_edges[:,1], reduce by f2v_edges[:,0]
        src = stack.index_select(0, edges.f2v_edges[:, 1])
        r_acc.scatter_add_(0, edges.f2v_edges[:, 0], src)

        # r messages: logsumexp over r-slots grouped by r_edges[:,0]
        # r_edges: [dst=v2f_id, src=r_slot_id]
        r_msg[:] = scatter_logsumexp(r_acc.index_select(0, edges.r_edges[:, 1]),
                                     edges.r_edges[:, 0], dim=0, dim_size=V2F_total)

        # qbar: sum incoming r into each v2f (excluding the destination factor via prewired q_edges)
        qbar.zero_()
        qbar.scatter_add_(0, edges.q_edges[:, 0], r_msg.index_select(0, edges.q_edges[:, 1]))

        # Normalize per message (2 states): v2f = qbar - logsumexp_pair
        delta = scatter_logsumexp(qbar, q_idx, dim=0, dim_size=(V2F_total // 2)).repeat_interleave(2)
        new_v2f = qbar - delta

        # Message damping
        if damping > 0.0:
            static_v2f.lerp_(new_v2f, damping)
        else:
            static_v2f.copy_(new_v2f)

        # Factor beliefs μ_F(y_F) ∝ exp(θ + sum incoming v→f)
        mufbar.copy_(static_theta)
        mufbar.scatter_add_(0, edges.m_edges[:, 0], static_v2f.index_select(0, edges.m_edges[:, 1]))
        # per-factor normalization
        z = scatter_logsumexp(mufbar, edges.m_idx, dim=0)  # one per factor
        # expand z to all cfgs via stride/repeat_interleave
        z_exp = z.repeat_interleave(edges.m_stride)
        static_mu.copy_((mufbar - z_exp).exp())

        # Variable beliefs (optional every iter; cheap anyway)
        static_vmar.zero_()
        static_vmar.scatter_add_(0, edges.vm_edges[:, 0], r_msg.index_select(0, edges.vm_edges[:, 1]))
        # normalize per variable (2 states)
        static_vmar.view(-1, 2).sub_(torch.logsumexp(static_vmar.view(-1, 2), dim=1, keepdim=True)).exp_()

    # Warm up a couple times (outside capture) to populate kernels/caches
    for _ in range(2):
        for _ in range(min(2, capture_iters)):
            one_iteration_sumproduct()
    torch.cuda.synchronize()

    with torch.cuda.graph(g):
        for _ in range(capture_iters):
            one_iteration_sumproduct()

    # ---- Replay until max_iter or tolerance -------------------------------
    iters = 0
    last_u = torch.inf

    prev_mu = torch.empty_like(static_mu)

    # prime prev_mu
    prev_mu.copy_(static_mu)

    remaining = max_iter
    while remaining > 0:
        step = min(capture_iters, remaining)
        # we replay full 'capture_iters' each time; if step < capture_iters, it simply does extra
        g.replay()
        iters += capture_iters
        remaining -= capture_iters

        # check convergence (∞-norm on factor beliefs, like your baseline)
        last_u = (static_mu - prev_mu).abs().max()
        prev_mu.copy_(static_mu)
        if float(last_u) <= tol:
            break

    # ---- Return a BPSolution-like object ----------------------------------
    # If your project exports BPSolution, return that; otherwise use a tiny shim.
    try:
        sol = BPSolution(mu=static_mu.clone(), u=float(last_u), n_iter=iters, mu_v=static_vmar.clone())
    except NameError:
        class _BPSolution:
            def __init__(self, mu, u, n_iter, mu_v): self.mu, self.u, self.n_iter, self.mu_v = mu, u, n_iter, mu_v
        sol = _BPSolution(mu=static_mu.clone(), u=float(last_u), n_iter=iters, mu_v=static_vmar.clone())

    return sol

def loopy_belief_propagation_bak(theta, v2f, g, f2v_edges, f2v_subsum_idx, r_edges, q_edges, m_edges, max_iter=100, tol=1e-6, damping=0.0):
    # strides
    q_stride = torch.full((int(v2f.shape[0]/2),), 2)
    q_idx = torch.arange(q_stride.size(0)).repeat_interleave(2)
    m_stride = torch.tensor([2**len(list(g.neighbors(F))) for F in iter_factor_nodes(g)])
    m_idx = torch.arange(m_stride.size(0)).repeat_interleave(m_stride)
    mu = torch.zeros(sufficient_statistics_dim(g))
    
    for T in range(max_iter):
        t0 = time.time()
        
        stack = torch.concatenate((theta,v2f), 0)
        # subsum vect
        v2f_subsum = torch.zeros(len(f2v_subsum_idx), dtype=theta.dtype)
        v2f_subsum.scatter_add_(0, f2v_edges[:,0], stack[f2v_edges[:,1]])
        
        r = torch_scatter.scatter_logsumexp(v2f_subsum[r_edges[:,1]], r_edges[:,0])
        
        # variable to factor messages
        qbar = torch.zeros_like(v2f).scatter_add_(0, q_edges[:,0], r[q_edges[:,1]])
        delta = torch_scatter.scatter_logsumexp(qbar, q_idx).repeat_interleave(2)
        v2f = qbar - delta
        
        # marginals
        mufbar = theta.clone().scatter_add_(0, m_edges[:,0], v2f[m_edges[:,1]])
        z = torch_scatter.scatter_logsumexp(mufbar, m_idx).repeat_interleave(m_stride, dim=0)
    
        mup = (mufbar - z).exp()

        u = (mup - mu).abs().max()

        if torch.any(torch.isinf(mup)) or torch.any(torch.isnan(mup)):
            break
        mu = (1 - damping) * mup + damping * mu
        if u < tol:
            break
    return mu

def residual_loopy_belief_propagation(theta, v2f, g, f2v_edges, f2v_subsum_idx, r_edges, q_edges, m_edges, max_iter=100, tol=1e-6):
    device = theta.device
    # strides
    q_stride = torch.full((int(v2f.shape[0]/2),), 2, device=device)
    q_idx = torch.arange(q_stride.size(0),device=device).repeat_interleave(2)
    m_stride = torch.tensor([2**len(list(g.neighbors(F))) for F in iter_factor_nodes(g)]).to(theta.device)
    m_idx = torch.arange(m_stride.size(0),device=device).repeat_interleave(m_stride)
    mu = torch.zeros(sufficient_statistics_dim(g), device=device)

    v2f_residuals = torch.zeros_like(v2f)
    n_msg = int(v2f_residuals.shape[0]/2)
    
    for T in range(max_iter):
        t0 = time.time()
        
        stack = torch.concatenate((theta,v2f), 0)
        # subsum vect
        v2f_subsum = torch.zeros(len(f2v_subsum_idx), dtype=theta.dtype, device=device)
        v2f_subsum.scatter_add_(0, f2v_edges[:,0], stack[f2v_edges[:,1]])
        
        r = torch_scatter.scatter_logsumexp(v2f_subsum[r_edges[:,1]], r_edges[:,0])
        
        # variable to factor messages
        qbar = torch.zeros_like(v2f).scatter_add_(0, q_edges[:,0], r[q_edges[:,1]])
        delta = torch_scatter.scatter_logsumexp(qbar, q_idx).repeat_interleave(2)
        tmp_v2f = qbar - delta

        # update residuals
        residuals = torch.abs(tmp_v2f - v2f)
        # find the max residual
        max_residuals,_ = torch.max(residuals.reshape(n_msg,2), dim=1)
        max_idx = torch.argmax(max_residuals)
        v2f_mask = torch.zeros_like(v2f, dtype=torch.bool)
        v2f_mask[2*max_idx:2*max_idx+2] = 1

        v2f[v2f_mask] = tmp_v2f[v2f_mask]
        
        # marginals
        mufbar = theta.clone().scatter_add_(0, m_edges[:,0], v2f[m_edges[:,1]])
        z = torch_scatter.scatter_logsumexp(mufbar, m_idx).repeat_interleave(m_stride, dim=0)
    
        mup = (mufbar - z).exp()

        u = (mup - mu).abs().max()

        if torch.any(torch.isinf(mup)) or torch.any(torch.isnan(mup)):
            break
        mu = mup
        if u < tol:
            break
    return mu


def max_sum_loopy_belief_propagation(theta, v2f, edges, max_iter=100, tol=1e-6, damping=0.):
    dev = theta.device
    # strides
    q_stride = torch.full((int(v2f.shape[0]/2),), 2, device=dev)
    q_idx = torch.arange(q_stride.size(0), device=dev).repeat_interleave(2)
    #m_stride = torch.tensor([2**len(list(g.neighbors(F))) for F in iter_factor_nodes(g)])
    m_idx = torch.arange(edges.m_stride.size(0), device=dev).repeat_interleave(edges.m_stride)
    #mu = torch.zeros(sufficient_statistics_dim(g))
    #mu = torch.zeros(int(edges.n_ss.sum().item()))


    nv = edges.n_vars.sum() #number_of_variables(g)
    mu = torch.zeros(2 * nv, device=dev)

    q_n = int(v2f.size(0) / 2)
   
    for T in range(max_iter):
        t0 = time.time()
        
        stack = torch.concatenate((theta,v2f), 0)
        # subsum vect
        #v2f_subsum = torch.zeros(len(f2v_subsum_idx))
        v2f_subsum = torch.zeros(edges.n_f2v_ss.sum(), dtype=theta.dtype, device=dev)
        v2f_subsum.scatter_add_(0, edges.f2v_edges[:,0], stack[edges.f2v_edges[:,1]])
        r = torch.zeros_like(v2f).scatter_reduce_(0, edges.r_edges[:,0], v2f_subsum[edges.r_edges[:,1]], "amax", include_self=False)

        # variable to factor messages
        qbar = torch.zeros_like(v2f).scatter_add_(0, edges.q_edges[:,0], r[edges.q_edges[:,1]])
        #delta = torch.repeat_interleave(
        #            torch.zeros(int(q_n), device=dev).scatter_add_(0, q_idx, qbar) / 2,
        #            2
        #        )
        delta = qbar.reshape(-1,2).amax(dim=1).repeat_interleave(2)
        v2f = qbar - delta
        
        # marginals
        mup = torch.zeros_like(mu).scatter_add_(0, edges.vm_edges[:,0], r[edges.vm_edges[:,1]])

        u = (mup - mu).abs().max()

        if torch.any(torch.isinf(mup)) or torch.any(torch.isnan(mup)):
            break
        mu = mup
        if u < tol:
            break
    return BPSolution(mu, u, T)
    return mu

def max_sum_loopy_belief_propagation_bak(theta, v2f, g, f2v_edges, f2v_subsum_idx, r_edges, q_edges, m_edges, vm_edges, max_iter=100, tol=1e-6, damping=0.):
    # strides
    q_stride = torch.full((int(v2f.shape[0]/2),), 2)
    q_idx = torch.arange(q_stride.size(0)).repeat_interleave(2)
    m_stride = torch.tensor([2**len(list(g.neighbors(F))) for F in iter_factor_nodes(g)])
    m_idx = torch.arange(m_stride.size(0)).repeat_interleave(m_stride)
    mu = torch.zeros(sufficient_statistics_dim(g))

    nv = number_of_variables(g)
    mu = torch.zeros(2 * nv)

    q_n = int(v2f.size(0) / 2)
   
    for T in range(max_iter):
        t0 = time.time()
        
        stack = torch.concatenate((theta,v2f), 0)
        # subsum vect
        v2f_subsum = torch.zeros(len(f2v_subsum_idx))
        v2f_subsum.scatter_add_(0, f2v_edges[:,0], stack[f2v_edges[:,1]])
        r = torch.zeros_like(v2f).scatter_reduce_(0, r_edges[:,0], v2f_subsum[r_edges[:,1]], "amax", include_self=False)

        # variable to factor messages
        qbar = torch.zeros_like(v2f).scatter_add_(0, q_edges[:,0], r[q_edges[:,1]])
        delta = torch.repeat_interleave(
                    torch.zeros(int(q_n)).scatter_add_(0, q_idx, qbar) / 2,
                    2
                )
        v2f = qbar - delta
        
        # marginals
        mup = torch.zeros_like(mu).scatter_add_(0, vm_edges[:,0], r[vm_edges[:,1]])

        u = (mup - mu).abs().max()

        if torch.any(torch.isinf(mup)) or torch.any(torch.isnan(mup)):
            break
        mu = mup
        if u < tol:
            break
    return mu


def _loopy_belief_propagation(edges, theta, v2f, max_iter=100, tol=1e-6, damping=0.5, mu0 = None):
    device = theta.device
    # strides
    q_stride = torch.full((int(v2f.shape[0]/2),), 2, device=device)
    q_idx = torch.arange(q_stride.size(0), device=device).repeat_interleave(2)
    m_stride = edges.m_stride
    m_idx = torch.arange(m_stride.size(0), device=device).repeat_interleave(m_stride)
    if mu0 is None:
        mu = torch.zeros(edges.n_ss.sum(), device=device)
    else:
        mu = mu0.clone().to(device)

    for T in range(max_iter):
        t0 = time.time()
        
        stack = torch.concatenate((theta,v2f), 0)
        # subsum vect
        #v2f_subsum = torch.zeros(len(f2v_subsum_idx), dtype=theta.dtype)
        v2f_subsum = torch.zeros(edges.n_f2v_ss.sum(), dtype=theta.dtype, device=device)

        v2f_subsum.scatter_add_(0, edges.f2v_edges[:,0], stack[edges.f2v_edges[:,1]])
        
        r = torch_scatter.scatter_logsumexp(v2f_subsum[edges.r_edges[:,1]], edges.r_edges[:,0])
        
        # variable to factor messages
        qbar = torch.zeros_like(v2f).scatter_add_(0, edges.q_edges[:,0], r[edges.q_edges[:,1]])
        delta = torch_scatter.scatter_logsumexp(qbar, q_idx).repeat_interleave(2)
        v2f = qbar - delta
        
        # marginals
        mufbar = theta.clone().scatter_add_(0, edges.m_edges[:,0], v2f[edges.m_edges[:,1]])
        z = torch_scatter.scatter_logsumexp(mufbar, m_idx).repeat_interleave(m_stride, dim=0)
    
        mup = (mufbar - z).exp()

        u = (mup - mu).abs().max()

        if torch.any(torch.isinf(mup)) or torch.any(torch.isnan(mup)):
            break
        mu = (1 - damping) * mup + damping * mu
        if u < tol:
            break
    return mu

def _loopy_belief_propagation_with_evidence(data, theta, v2f, f2v_edges, r_edges, q_edges, m_edges, max_iter=100, tol=1e-6, damping=0.5, mu0 = None, evidence=None):
    device = theta.device
    # strides
    q_stride = torch.full((int(v2f.shape[0]/2),), 2, device=device)
    q_idx = torch.arange(q_stride.size(0), device=device).repeat_interleave(2)
    m_stride = data.m_stride
    m_idx = torch.arange(m_stride.size(0), device=device).repeat_interleave(m_stride)
    if mu0 is None:
        mu = torch.zeros(data.ss_dim, device=device)
    else:
        mu = mu0.clone().to(device)

    for T in range(max_iter):
        t0 = time.time()
        
        stack = torch.concatenate((theta,v2f), 0)
        # subsum vect
        #v2f_subsum = torch.zeros(len(f2v_subsum_idx), dtype=theta.dtype)
        v2f_subsum = torch.zeros(data.f2v_ss_dim, dtype=theta.dtype, device=device)

        v2f_subsum.scatter_add_(0, f2v_edges[:,0], stack[f2v_edges[:,1]])
        
        r = torch_scatter.scatter_logsumexp(v2f_subsum[r_edges[:,1]], r_edges[:,0])
        
        # variable to factor messages
        qbar = torch.zeros_like(v2f).scatter_add_(0, q_edges[:,0], r[q_edges[:,1]])
        if not evidence is None:
            print(qbar[evidence])
        delta = torch_scatter.scatter_logsumexp(qbar, q_idx).repeat_interleave(2)
        v2f = qbar - delta
       
        # marginals
        mufbar = theta.clone().scatter_add_(0, m_edges[:,0], v2f[m_edges[:,1]])
        #if not evidence is None:
        #    mufbar[evidence] = -1000
        z = torch_scatter.scatter_logsumexp(mufbar, m_idx).repeat_interleave(m_stride, dim=0)
    
        mup = (mufbar - z).exp()

        u = (mup - mu).abs().max()

        if torch.any(torch.isinf(mup)) or torch.any(torch.isnan(mup)):
            break
        mu = (1 - damping) * mup + damping * mu
        if u < tol:
            break
    return mu

