import torch
import networkx as nx
import itertools
import numpy as np
import time
import torch_scatter

class BPSolution:
    def __init__(self, mu=None, u=None, n_iter=None):
        self.mu = mu
        self.u = u
        self.n_iter = None


def loopy_belief_propagation(theta, v2f, edges, max_iter=100, tol=1e-6, damping=0.0):
    # strides
    dev = theta.device
    q_stride = torch.full((int(v2f.shape[0]/2),), 2, device=dev)
    q_idx = torch.arange(q_stride.size(0), device=dev).repeat_interleave(2)
    #m_stride = torch.tensor([2**len(list(g.neighbors(F))) for F in iter_factor_nodes(g)])
    m_idx = torch.arange(edges.m_stride.size(0), device=theta.device).repeat_interleave(edges.m_stride)
    mu = torch.zeros(int(edges.n_ss.sum().item()), device=dev)
    
    for T in range(max_iter):
        t0 = time.time()
        
        stack = torch.concatenate((theta,v2f), 0)
        # subsum vect
        v2f_subsum = torch.zeros(edges.n_f2v_ss.sum(), dtype=theta.dtype, device=theta.device)
        v2f_subsum.scatter_add_(0, edges.f2v_edges[:,0], stack[edges.f2v_edges[:,1]])
        
        r = torch_scatter.scatter_logsumexp(v2f_subsum[edges.r_edges[:,1]], edges.r_edges[:,0])
        
        # variable to factor messages
        qbar = torch.zeros_like(v2f).scatter_add_(0, edges.q_edges[:,0], r[edges.q_edges[:,1]])
        delta = torch_scatter.scatter_logsumexp(qbar, q_idx).repeat_interleave(2)
        v2f = qbar - delta
        
        # marginals
        mufbar = theta.clone().scatter_add_(0, edges.m_edges[:,0], v2f[edges.m_edges[:,1]])
        z = torch_scatter.scatter_logsumexp(mufbar, m_idx).repeat_interleave(edges.m_stride, dim=0)
    
        mup = (mufbar - z).exp()

        u = (mup - mu).abs().max()
        if torch.any(torch.isinf(mup)) or torch.any(torch.isnan(mup)):
            break
        mu = (1 - damping) * mup + damping * mu

        if u < tol:
            break
    return BPSolution(mu, u, T)
    return mu

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
        delta = torch.repeat_interleave(
                    torch.zeros(int(q_n), device=dev).scatter_add_(0, q_idx, qbar) / 2,
                    2
                )
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

