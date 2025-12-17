import networkx as nx

from bpropy.co.mis import MIS, random_mis

def very_simple_mis():
    g = nx.Graph()
    g.add_nodes_from([0,1])
    g.add_edges_from([[0,1]])
    return MIS(g)

def simple_mis():
    g = nx.Graph()
    g.add_nodes_from([0,1,2])
    g.add_edges_from([[0,1],[1,2],[0,2]])
    return MIS(g)

