import numpy as np
import networkx as nx
import torch
import random
random.seed(123)
np.random.seed(123)

from bpropy.lbp import *
from dopy.maxcut import *


g = nx.generators.erdos_renyi_graph(10, 0.15)
pb = MaxCut(g)

print(g)
print("\tnv = ", nx.number_of_nodes(g))
print("\tne = ", nx.number_of_edges(g))

# node and edges index maps
node_idx_map = {n:i for i,n in enumerate(g.nodes)}
edge_idx_map = {tuple(sorted(e)):i for i,e in enumerate(g.edges)}

fg = create_factor_graph(g)
n_ss = sufficient_statistics_dim(fg)

theta = torch.zeros(n_ss, requires_grad=False)
c = 0
for i in range(pb.nv()):
    theta[c+1] = 1
    c += 2
for e in pb.g.edges:
    theta[c+1] = 1. / len(list(g.neighbors(e[0])))
    theta[c+2] = 1. / len(list(g.neighbors(e[1])))
    theta[c+3] = -1
    c += 4
theta = theta - theta.min() + 1

def edge_based_reparameterization(theta, g, nV, max_iter=100):
    # initialize pseudo-max-marginals
    v = torch.zeros_like(theta)
    v[:2*nV] = theta[:2*nV].exp()
    for i in range(nV):
        v[2*i:2*i+2] /= v[2*i:2*i+2].sum()
    for ei,e in enumerate(g.edges):
        v[2*nV+4*ei] = (theta[2*nV+4*ei] + theta[2*e[0]] + theta[2*e[1]]).exp()
        v[2*nV+4*ei+1] = (theta[2*nV+4*ei+1] + theta[2*e[0]+1] + theta[2*e[1]]).exp()
        v[2*nV+4*ei+2] = (theta[2*nV+4*ei+2] + theta[2*e[0]] + theta[2*e[1]+1]).exp()
        v[2*nV+4*ei+3] = (theta[2*nV+4*ei+3] + theta[2*e[0]+1] + theta[2*e[1]+1]).exp()
        v[2*nV+4*ei:2*nV+4*ei+4] /= v[2*nV+4*ei:2*nV+4*ei+4].sum()
    nv = torch.zeros_like(v)
    for it in range(max_iter):
        for i in range(nV):
            nv[2*i:2*i+2] = v[2*i:2*i+2]
            vs0,vs1 = [], []
            for ei,e in enumerate(g.edges):
                if e[0] == i:
                    # x_s == 0
                    vs0.append(np.max([v[2*nV+4*ei], nv[2*nV+4*ei+2]])/v[2*i])
                    # x_s == 1
                    vs1.append(np.max([v[2*nV+4*ei+1], nv[2*nV+4*ei+3]])/v[2*i+1])
                if e[1] == i:
                    # x_s == 0
                    vs0.append(np.max([v[2*nV+4*ei], nv[2*nV+4*ei+1]])/v[2*i])
                    # x_s == 1
                    vs1.append(np.max([v[2*nV+4*ei+2], nv[2*nV+4*ei+3]])/v[2*i+1])
            nv[2*i] *= np.prod(vs0)
            nv[2*i+1] *= np.prod(vs1)
            nv[2*i:2*i+2] /= nv[2*i:2*i+2].sum()
        for ei,e in enumerate(g.edges):
            nv[2*nV+4*ei] = nv[2*e[0]] * nv[2*e[1]] * v[2*nV+4*ei] / (np.max([v[2*nV+4*ei], nv[2*nV+4*ei+2]]) * np.max([v[2*nV+4*ei], nv[2*nV+4*ei+1]]))
            nv[2*nV+4*ei+1] = nv[2*e[0]+1] * nv[2*e[1]] * v[2*nV+4*ei+1] / (np.max([v[2*nV+4*ei+1], nv[2*nV+4*ei+3]]) * np.max([v[2*nV+4*ei], nv[2*nV+4*ei+1]]))
            nv[2*nV+4*ei+2] = nv[2*e[0]] * nv[2*e[1]+1] * v[2*nV+4*ei+2] / (np.max([v[2*nV+4*ei], nv[2*nV+4*ei+2]]) * np.max([v[2*nV+4*ei+2], nv[2*nV+4*ei+3]]))
            nv[2*nV+4*ei+3] = nv[2*e[0]+1] * nv[2*e[1]+1] * v[2*nV+4*ei+3] / (np.max([v[2*nV+4*ei+1], nv[2*nV+4*ei+3]]) * np.max([v[2*nV+4*ei+2], nv[2*nV+4*ei+3]]))
            nv[2*nV+4*ei:2*nV+4*ei+4] /= nv[2*nV+4*ei:2*nV+4*ei+4].sum()

        if (v - nv).abs().max() < 1e-6:
            return nv
        v = nv
    return v
    

pvm = edge_based_reparameterization(theta, g, g.number_of_nodes())
print(pvm)
