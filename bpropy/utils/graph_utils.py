import numpy as np
import networkx as nx
import torch

from bpropy.utils.set_sort import lenlex_sort

def ordered_edges_old(g):
    """Returns edges in increasing lexicographical order (old implementation based on adjacency matrix)
    """
    nV = nx.number_of_nodes(g)
    adj = nx.to_numpy_array(g, nodelist=range(nV))
    if g.is_directed():
        for (s,t) in zip(*np.where(adj > 0)):
            yield [s,t]
    else:
        for (s,t) in zip(*np.where(adj > 0)):
            if s<t:
                yield [s,t]

def ordered_edges(g):
    """Returns edges in increasing lexicographical order (old implementation based on adjacency matrix)
    """
    return lenlex_sort([sorted(e) for e in g.edges])

def get_edge_index(g):
    if not g.is_directed():
        g = g.to_directed()
    edge_index = torch.empty((2, g.number_of_edges()), dtype=torch.long)
    for i,(src,dst) in enumerate(g.edges):
        edge_index[0,i] = src
        edge_index[1,i] = dst
    return edge_index

if __name__=="__main__":
    import time
    import sys
    
    ts = []
    for _ in range(10):
        sys.stdout.write(".")
        sys.stdout.flush()
        g = nx.generators.random_graphs.erdos_renyi_graph(1000, 0.15)

        t1 = time.time()
        oedges1 = list(ordered_edges_old(g))
        t1 = time.time() - t1

        t2 = time.time()
        oedges2 = ordered_edges(g)
        t2 = time.time() - t2

        ts.append([t1,t2])

        if len(oedges1) != len(oedges2):
            print("List lengths are not equal")
            break
        if oedges1 != oedges2:
            print("Lists are not equal")
            break
    sys.stdout.write("\n")
    sys.stdout.flush()

    print(np.mean(ts, axis=0))
