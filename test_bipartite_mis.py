import networkx as nx

from dopy.mis import *
from bipartite_graph import *
from clique_covering import *
g = nx.generators.random_graphs.erdos_renyi_graph(10,0.15)

pb = MIS(g)

bpg,_ = bipartite_graph(pb)
print("Edges")
for n,d in bpg.nodes(data=True):
    print(n,d)
print()

C = clique_covering_3(g)

bpg,_ = bipartite_graph(pb, cliques=C)
print("Cliques")
for n,d in bpg.nodes(data=True):
    print(n,d)


