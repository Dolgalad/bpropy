import torch
import networkx as nx
import itertools
import numpy as np
import time
import torch_scatter

def sufficient_statistics_dim(g):
    return sum([2**len(list(g.neighbors(n))) for n,d in g.nodes(data=True) if d["bipartite"]==1])
def number_vm_edges(g):
    return sum([2*len(list(g.neighbors(n))) for n,d in g.nodes(data=True) if d["bipartite"]==0])
def number_of_variables(g):
    return len([n for n,d in g.nodes(data=True) if d["bipartite"]==0])
def f2v_messages_dim(g):
    return sum([2*len(list(g.neighbors(n))) for n,d in g.nodes(data=True) if d["bipartite"]==1])
def iter_factor_nodes(g):
    for n,d in g.nodes(data=True):
        if d["bipartite"]==1:
            yield n
def f2v_index_map(g):
    m,n,ss = {}, {}, {}
    c,d,e = 0,0,0
    for Fc,F in enumerate(iter_factor_nodes(g)):
        neighbors = list(g.neighbors(F))
        nN = len(neighbors)
        for vals in range(2**nN):
            yF = np.unpackbits(np.array([vals],dtype=np.uint8), count=nN, bitorder="little")
            ss[F,vals] = e
            e += 1
            for i in neighbors:
                iidx = neighbors.index(i)
                for s,v in enumerate([0,1]):
                    #if not fi_value_set[i][s]:
                    if not (F,i,v) in m.keys():
                        m[F,i,v] = c
                        c += 1
                    # f2v subsum indexes
                    if yF[iidx] == v:
                        n[F,vals,i,v] = d
                        d += 1
    return m,n,ss

def f2v_subsum_indexes(g, f2v_idx,ss_idx):
    factor_sizes = np.array([len(list(g.neighbors(F))) for F in iter_factor_nodes(g)])
    nSS = len(ss_idx)
    edges = []
    redges = []
    c = 0
    for c_F,F in enumerate(iter_factor_nodes(g)):
        neighbors = list(g.neighbors(F))
        nN = len(neighbors)
        for vals in range(2**nN):
            yF = np.unpackbits(np.array([vals],dtype=np.uint8), count=nN, bitorder="little")
            for c_i,i in enumerate(neighbors):
                iidx = neighbors.index(i)
                for c_v,v in enumerate([0,1]):
                    if yF[iidx] == v:
                        # add edge f2v -> rFyi
                        redges.append([f2v_idx[F,i,v], c])
                        # add edge ss -> current pos
                        edges.append([c, ss_idx[F,vals]])
                        # add one edge qyif -> current pos for each yF[j] for j!=i
                        for c_j,(j,yFj) in enumerate(zip(neighbors,yF)):
                            if j!=i:
                                edges.append([c, nSS+f2v_idx[F,j,yFj]])
                        c += 1
    return np.array(edges), np.array(redges)

def f2v_subsum_indexes2(g):
    factor_sizes = np.array([len(list(g.neighbors(F))) for F in iter_factor_nodes(g)])
    factor_value_sizes = (2**factor_sizes).cumsum() - 2**factor_sizes
    nSS = sufficient_statistics_dim(g)

    f2v_edges_n = (2**(factor_sizes) * factor_sizes**2 ).sum()
    f2v_edges = np.empty((f2v_edges_n,2))
    f2v_edges_c = 0

    redges_n = (2**factor_sizes * factor_sizes ).sum()
    redges = np.empty((redges_n,2))
    redges_c = 0
    c = 0
    for c_F,F in enumerate(iter_factor_nodes(g)):
        neighbors = list(g.neighbors(F))
        nN = len(neighbors)
        #for c_yF,yF in enumerate(itertools.product(*[[0,1] for _ in range(nN)])):
        for vals in range(2**nN):
            yF = np.unpackbits(np.array([vals],dtype=np.uint8), count=nN, bitorder="little")

            for c_i,i in enumerate(neighbors):
                iidx = neighbors.index(i)
                for c_v,v in enumerate([0,1]):
                    if yF[iidx] == v:
                        # add edge f2v -> rFyi
                        redges[redges_c] = [factor_value_sizes[c_F]+2*c_i+c_v, redges_c]

                        # add edge ss -> current pos
                        f2v_edges[f2v_edges_c] = [redges_c, factor_value_sizes[c_F]+vals]
                        f2v_edges_c += 1

                        # add one edge qyif -> current pos for each yF[j] for j!=i
                        for c_j,(j,yFj) in enumerate(zip(neighbors,yF)):
                            if j!=i:
                                f2v_edges[f2v_edges_c] = [redges_c, nSS+factor_value_sizes[c_F]+2*c_j+yFj]
                                f2v_edges_c += 1
                        redges_c += 1
    return f2v_edges, redges

def v2f_indexes(g, f2v_idx):
    t0 = time.time()
    #f2v_idx,f2v_subsum_idx,ss_idx = f2v_index_map(g)
    factor_sizes = np.array([len(list(g.neighbors(F))) for F in iter_factor_nodes(g)])
    factor_offset_index = (2*factor_sizes).cumsum() - 2*factor_sizes[0]
    factor_indices = {F:i for i,F in enumerate(iter_factor_nodes(g))}
    factor_neighbor_indices = {F:{i:idx for idx,i in enumerate(g.neighbors(F))} for F in iter_factor_nodes(g)}
    edges = []
    for c_F,F in enumerate(iter_factor_nodes(g)):
        neighbors = list(g.neighbors(F))
        nN = len(neighbors)
        for c_i,i in enumerate(neighbors):
            for v in [0,1]:
                for Fp in g.neighbors(i):
                    if Fp!=F:
                        edges.append([f2v_idx[F,i,v], f2v_idx[Fp,i,v]])

    rr = np.array(edges)
    return rr
def v2f_indexes2(g):
    f2v_dim = f2v_messages_dim(g)
    factor_indices = {F:i for i,F in enumerate(iter_factor_nodes(g))}
    factor_neighbor_indices = {F:{i:idx for idx,i in enumerate(g.neighbors(F))} for F in iter_factor_nodes(g)}

    jj=np.array([sum([len(list(g.neighbors(i)))-1 for i in g.neighbors(F)]) for F in iter_factor_nodes(g)])
    jjj = 2*jj.cumsum() - 2*jj

    factor_neighborhood_sizes = {F:np.array([len(list(g.neighbors(i)))-1 for i in g.neighbors(F)]) for F in iter_factor_nodes(g)}
    factor_neighborhood_cumsize = {k:v.cumsum()-v for k,v in factor_neighborhood_sizes.items()}


    dd = np.array([len(list(g.neighbors(i)))-1 for F in iter_factor_nodes(g) for i in g.neighbors(F)])
    # edge sources
    src = np.arange(f2v_dim).repeat(dd.repeat(2)) # VALID
    # edge destinations
    t0 = time.time()
    dst_idx= np.array([
            jjj[factor_indices[Fp]]+2*factor_neighborhood_cumsize[Fp][factor_neighbor_indices[Fp][i]]+v*factor_neighborhood_sizes[Fp][factor_neighbor_indices[Fp][i]]
            for F in iter_factor_nodes(g)
            for i in g.neighbors(F)
            for v in [0,1]
            for Fp in g.neighbors(i) if Fp!=F
        ])
    dst = src[dst_idx]
    return np.stack((src,dst),axis=-1)


def marginal_indexes(g, f2v_idx, ss_idx):
    edges = []
    for F in iter_factor_nodes(g):
        neighbors = list(g.neighbors(F))
        nN = len(neighbors)
        for vals in range(2**nN):
            yF = np.unpackbits(np.array([vals],dtype=np.uint8), count=nN, bitorder="little")
            for (j,yFj) in zip(neighbors, yF):
                edges.append([ss_idx[F,vals], f2v_idx[F,j,yFj]])
    return np.array(edges)
def marginal_indexes2(g):
    factor_sizes = np.array([len(list(g.neighbors(F))) for F in iter_factor_nodes(g)])
    factor_cumsum = (2**factor_sizes).cumsum() - 2**factor_sizes

    nM = (factor_sizes * 2**factor_sizes).sum()

    dst = np.empty(nM, dtype=np.long)
    c = 0
    for c_F,F in enumerate(iter_factor_nodes(g)):
        neighbors = list(g.neighbors(F))
        nN = len(neighbors)
        for vals in range(2**nN):
            yF = np.unpackbits(np.array([vals],dtype=np.uint8), count=nN, bitorder="little")
            for c_i,(j,yFj) in enumerate(zip(neighbors, yF)):
                dst[c] =factor_cumsum[c_F]+2*c_i+yFj
                c+=1

    factors = list(iter_factor_nodes(g))
    factor_neighbor_sizes = np.array([len(list(g.neighbors(factors[Fidx]))) for Fidx,Fs in enumerate(factor_sizes) for _ in range(2**Fs)])
    aa = np.arange(np.sum(2**factor_sizes))
    src = aa.repeat(factor_neighbor_sizes) # VALID
    return np.stack((src, dst), axis=-1)

def variable_marginal_indexes(g, f2v_idx, ss_idx):
    vm_edges = []
    for n,d in g.nodes(data=True):
        if d["bipartite"]==0:
            for f in g.neighbors(n):
                for v in range(2):
                    vme = [2*n+v,f2v_idx[f,n,v]]
                    vm_edges.append(vme)
    return np.array(vm_edges)



def create_factor_graph(g):
    fg = nx.Graph()
    fg.add_nodes_from(g.nodes, bipartite=0)
    c = g.number_of_nodes()
    for n in g.nodes:
        fg.add_node(c, bipartite=1)
        fg.add_edge(n,c)
        c += 1
    for e in g.edges:
        fg.add_node(c, bipartite=1)
        fg.add_edge(e[0], c)
        fg.add_edge(e[1], c)
        c += 1
    return fg

def loopy_belief_propagation(theta, v2f, g, f2v_edges, f2v_subsum_idx, r_edges, q_edges, m_edges, max_iter=100, tol=1e-6):
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
        mu = mup
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
        #print("max_id = ", max_idx)

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


def max_sum_loopy_belief_propagation(theta, v2f, g, f2v_edges, f2v_subsum_idx, r_edges, q_edges, m_edges, vm_edges, max_iter=100, tol=1e-6):
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

def _loopy_belief_propagation(data, theta, v2f, f2v_edges, r_edges, q_edges, m_edges, max_iter=100, tol=1e-6):
    device = theta.device
    # strides
    q_stride = torch.full((int(v2f.shape[0]/2),), 2, device=device)
    q_idx = torch.arange(q_stride.size(0), device=device).repeat_interleave(2)
    m_stride = data.m_stride
    m_idx = torch.arange(m_stride.size(0), device=device).repeat_interleave(m_stride)
    mu = torch.zeros(data.ss_dim, device=device)

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
        delta = torch_scatter.scatter_logsumexp(qbar, q_idx).repeat_interleave(2)
        v2f = qbar - delta
        
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

