"""
Naive Mean Field implementation
"""
import numpy as np
import networkx as nx
import torch
import torch_scatter

from bpropy.mrf.nmf_edges import NMFEdges

class NMFSolution:
    def __init__(self, mu, u, n_iter):
        self.mu = mu
        self.u = u
        self.n_iter = n_iter

def naive_mean_field(theta, edges, max_iter=1000, tol=1e-6):
    dev = theta.device
    nV = int(edges.n_vars.sum().item())
    mu = torch.full((2*nV,), 0.5, dtype=torch.float, device=dev)  # neutral start

    # repeat θ so that each factor-assignment is used once per center variable u
    th = torch.repeat_interleave(theta, edges.theta_rep)

    for it in range(max_iter):
        # product over neighbors per slot d
        prod = torch_scatter.scatter_mul(mu[edges.pmprod_edges[:, 1]], edges.pmprod_edges[:, 0])

        # weight by θ and sum into variable-states
        x1 = th * prod
        y1 = torch_scatter.scatter_add(x1[edges.pmprodsum_edges[:, 1]], edges.pmprodsum_edges[:, 0])

        # normalize per variable
        mup = torch.flatten(torch.nn.functional.softmax(y1.reshape(nV, 2), dim=1))

        delta = (mup - mu).abs().max()
        mu = mup
        if delta < tol:
            return NMFSolution(mu, delta, it + 1)

    return NMFSolution(mu, (mup - mu).abs().max(), max_iter)

def naive_mean_field_old(theta, edges, max_iter=1000, tol=1e-6):
    dev = theta.device
    nV = edges.n_vars.sum()
    mu = torch.zeros(2*nV, dtype=torch.float, device=dev)

    th = torch.repeat_interleave(theta, edges.theta_rep)

    for it in range(max_iter):
        x1 = th * torch_scatter.scatter_mul(mu[edges.pmprod_edges[:,1]], edges.pmprod_edges[:,0])
        y1 = torch_scatter.scatter_add(x1[edges.pmprodsum_edges[:,1]], edges.pmprodsum_edges[:,0])

        # softmax result
        mup = torch.flatten(torch.nn.functional.softmax(y1.reshape(nV,2), dim=1))

        u = (mup - mu).abs().max()
        if u < 1e-6:
            mu = mup
            break
        mu = mup
    return NMFSolution(mu, u, it)
    return mu

def _naive_mean_field(data, theta, max_iter=1000, tol=1e-6, mu0=None):
    nV = int((data.bipartite==0).sum().item())
    if mu0 is None:
        mu = torch.zeros(2*nV, dtype=torch.float, device=theta.device)
    else:
        mu = mu0

    theta_rep = data.factor_size.clone().to(torch.long)#torch.tensor([factor_size(fg,F) for F in iter_factor_nodes(fg)], dtype=torch.long)
    theta_rep = torch.repeat_interleave(theta_rep, 2**theta_rep)
    th = torch.repeat_interleave(theta, theta_rep)

    for it in range(max_iter):
        x1 = th * torch_scatter.scatter_mul(mu[data.pmprod_edges[:,1]], data.pmprod_edges[:,0])
        y1 = torch_scatter.scatter_add(x1[data.pmprodsum_edges[:,1]], data.pmprodsum_edges[:,0])

        # softmax result
        mup = torch.flatten(torch.nn.functional.softmax(y1.reshape(nV,2), dim=1))

        u = (mup - mu).abs().max()
        if u < 1e-6:
            mu = mup
            break
        mu = mup
    return mu


if __name__=="__main__":
    g = nx.generators.random_graphs.erdos_renyi_graph(3, 1)

    fg = create_factor_graph(g)
    tF = fg.number_of_nodes()
    fg.add_node(tF, bipartite=1)
    fg.add_edge(0,tF)
    fg.add_edge(1,tF)
    fg.add_edge(2,tF)

    nss = sufficient_statistics_dim(fg)
    nV = number_of_variables(fg)

    pmprod_edges,pmprodsum_edges = naive_mean_field_edges(fg)

    
    print("Number sufficient statistics = ", nss)
    theta = torch.randn(nss)
    print("theta = ")
    print(theta)

    print(naive_mean_field(fg, theta, pmprod_edges, pmprodsum_edges))
    exit()
          

    print(edges)
    pmprod_n = sum([2**factor_size(fg,F)*factor_size(fg,F) for F in iter_factor_nodes(fg)])
    print("pmprod_n = ", pmprod_n)
    print(edges[:,0].min(), edges[:,0].max())
    print(edges[:,1].min(), edges[:,1].max())

    mu = torch.zeros(2*nV, dtype=torch.float)


    theta_rep = torch.tensor([factor_size(fg,F) for F in iter_factor_nodes(fg)], dtype=torch.long)
    theta_rep = torch.repeat_interleave(theta_rep, 2**theta_rep)
    print("theta_rep.shape = ", theta_rep.shape)
    th = torch.repeat_interleave(theta, theta_rep)

    print("th.shape = ",th.shape)
    print(th)

    for it in range(1000):
        print("iteration ", it)
        print(mu[edges[:,1]])
        x1 = th * torch_scatter.scatter_mul(mu[edges[:,1]], edges[:,0])
        print("x1.shape = ", x1.shape)
        print(x1)

        y1 = torch_scatter.scatter_add(x1[sum_edges[:,1]], sum_edges[:,0])
        print("y1.shape = ", y1.shape)
        print(y1)

        # softmax result
        mup = torch.flatten(torch.nn.functional.softmax(y1.reshape(nV,2), dim=1))

        u = (mup - mu).abs().max()
        print("u = ", u)
        if u < 1e-6:
            mu = mup
            print("Tol reached")
            break
        mu = mup

    print("MU")
    print(mu.reshape(nV,2))




