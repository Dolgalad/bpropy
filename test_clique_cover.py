"""
Test that the clique covering formulation represetnation of the problem does not lead to 
issues in the ordering of nodes
"""
import itertools
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from dopy.mis import *
from clique_covering import *
from bipartite_graph import *

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

def get_neighbors(data, u):
    r=data.edge_index[1, data.edge_index[0,:]==u].numpy()
    return r

def tgdata_to_networkx(data):
    nv = int(torch.sum(data.bipartite==0).item())
    nc = int(torch.sum(data.bipartite).item())
    g = nx.Graph()
    for i in range(nv):
        g.add_node(i)
    for c in range(nc):
        neighbors = get_neighbors(data, c+nv)
        for e in itertools.product(neighbors, neighbors):
            if e[0]!=e[1]:
                g.add_edge(e[0], e[1])
    return g

g = nx.generators.random_graphs.erdos_renyi_graph(50, 0.5)

pb = MIS(g)

# cover the graph with cliques
C = clique_covering_3(g, randomize=True)
print("clique cover")
for c in C:
    print(c)
#C = [list(e) for e in g.edges]


# bipartite graph
bpg,bpg_feat = bipartite_graph(pb, cliques=C)

# convert to torch_geometric Data object
data = networkx_to_tgdata(bpg)

# convert it back to MIS instance
g1 = tgdata_to_networkx(data)

print("original graph")
print(g)
print(g1)
for e in g.edges:
    print(e, g1.has_edge(e[0], e[1]))

print("reconstructed")
for e in g1.edges:
    print(e, g.has_edge(e[0], e[1]))

print((g.number_of_nodes()==g1.number_of_nodes()) and (g.number_of_edges() == g1.number_of_edges()) and (all(g.has_edge(*e) for e in g1.edges)))
