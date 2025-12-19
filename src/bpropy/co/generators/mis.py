import networkx as nx

from bpropy.co.mis import MIS
from bpropy.clique_cover import clique_cover

def random_mis(n=100, p=0.15, cliques=False):
    g = nx.generators.random_graphs.erdos_renyi_graph(n, p)
    if cliques:
        cc = clique_cover(g)
        pb = MIS(g, cliques=cc)
    else:
        pb = MIS(g)
    while pb.g.number_of_edges() == 0:
        g = nx.generators.random_graphs.erdos_renyi_graph(n, p)
        pb = MIS(g, cliques=clique_cover(g)) if cliques else MIS(g)
    return pb

