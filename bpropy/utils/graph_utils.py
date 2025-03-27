import numpy as np
import networkx as nx

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
