import os
import torch
from torch_geometric.data import Data
import networkx as nx
from dopy.mis import *
from bipartite_graph import *
from bpropy.lbp import *
from bpropy.lbp import _loopy_belief_propagation

def networkx_to_tgdata(g):
    g = g.to_directed()
    edge_index = torch.empty((2, g.number_of_edges()), dtype=torch.long)
    bipartite = torch.empty(g.number_of_nodes())
    for i,(src,dst) in enumerate(g.edges):
        edge_index[0,i] = src
        edge_index[1,i] = dst
    for n,d in g.nodes(data=True):
        bipartite[n] = d["bipartite"]
    return Data(edge_index=edge_index, bipartite=bipartite)

def solution_to_indicators(x, fg):
    n_ss = sufficient_statistics_dim(fg)
    ind = np.zeros(n_ss)
    fact_offset = 0
    for i,F in enumerate(iter_factor_nodes(fg)):
        neighbors = list(fg.neighbors(F))
        nF = len(neighbors)
        for v in range(2**nF):
            yF = np.unpackbits(np.array([v],dtype=np.uint8), count=nF, bitorder="little")
            xF = np.array([int(x[j] > 0.5) for j in neighbors])
            if np.all(yF == xF):
                ind[fact_offset + v] = 1
        fact_offset += 2**nF
    return ind


# instance path
instance_path = "datasets/mis_er_n100-100_p0.15_tl1000.0_pooltl1000.0_eb/train/1"

# load graph
g = nx.read_adjlist(os.path.join(instance_path, "graph.adjlist"))
nmap = {str(i):i for i in range(g.number_of_nodes())}
g = nx.relabel_nodes(g, nmap, copy=False)

#g = nx.generators.random_graphs.erdos_renyi_graph(100, 0.2)

nV = nx.number_of_nodes(g)

print("Graph")
print(g)

pb = MIS(g)

# bipartite graph
bpg,_ = bipartite_graph(pb)

# factor graph
fg = create_factor_graph(pb.g)
for F in iter_factor_nodes(fg):
    if factor_size(fg, F)==2:
        neighbors = list(fg.neighbors(F))
        [s,t] = neighbors
        if not g.has_edge(s,t):
            print("Factor graph invalid")
            print("\t", F, neighbors)
            print("\t", list(g.neighbors(s)), list(g.neighbors(t)))
            exit()

# simple solution
sol = mis_greedy(pb, np.ones(nV))

# indicators
ind_sol = solution_to_indicators(sol, fg)
# recover solution
rsol = ind_sol[:2*nV].reshape(nV,2)[:,1]

print("Solution valid = ", solution_validity(pb, sol))
print("Reconstructed solution valid = ", solution_validity(pb, rsol))

if not np.all(sol == rsol):
    print("Solution and reconstructed solution are not identical")
    exit()

for F in iter_factor_nodes(fg):
    neighbors = list(fg.neighbors(F))
    factor_val = rsol[neighbors]
    print(F, factor_size(fg, F), factor_val)
    print("\t", neighbors)
    if factor_size(fg,F)==2:
        print("\t", g.has_edge(*neighbors))
        if not g.has_edge(*neighbors):
            print("Error in factor graph")
            exit()

print(g.is_directed())
print(fg.is_directed())
