import os
import networkx as nx

from dopy.mis import *
from bipartite_graph import bipartite_graph

workdir = "test_mis_encoding"

g = nx.generators.random_graphs.erdos_renyi_graph(10, 0.15)

print("Node order before reading")
for n in g.nodes:
    print(n)
for e in g.edges:
    print(e)

os.makedirs("test_mis_encoding", exist_ok=True)
# save the graph
graph_path = os.path.join(workdir, "graph.adjlist")
nx.write_adjlist(g, graph_path)
g = nx.read_adjlist(graph_path)
nmap = {str(i): i for i in range(g.number_of_nodes())}
g = nx.relabel_nodes(g, nmap, copy=False)
print("Node order after reading")
for n in g.nodes:
    print(n)
for e in g.edges:
    print(e)

pb = MIS(g)

# variable features
C = [list(e) for e in g.edges]
md = cplex_model(pb, continuous_relaxation=True)
varlist = [md.get_var_by_name(f"x{i}") for i in range(g.number_of_nodes())]
sol = md.solve()
x = np.array([sol.get_value(var) for var in varlist])
for e in g.edges:
    print(e, x[e[0]], x[e[1]])


# bipartite graph
bpg,bpg_feat = bipartite_graph(pb, cliques=C)
bipartite_graph_file_path = os.path.join(instance_path, "bipartite.adjlist")
nx.write_adjlist(bpg, bipartite_graph_file_path)

