"""
Routines for covering graphs with cliques
"""

import networkx as nx
import numpy as np
import time

def find_neighbor_of_all_3(g, c):
    if len(c)==1:
        neighbors = list(g.neighbors(c[0]))
        if len(neighbors)==0:
            return None
        return neighbors[0]
    set0 = set(g.neighbors(c[0]))
    for u in c[1:]:
        set0 = set0.intersection(set(g.neighbors(u)))
        if len(set0)==0:
            return None
    return next(iter(set0))

def find_neighbor_of_all_2(g, c):
    node_idx_map = {n:i for i,n in enumerate(g.nodes)}
    node_idx = np.arange(g.number_of_nodes())
    node_list = np.array(list(g.nodes))
    c_idx = np.array([node_idx_map[u] for u in c])
    adj = nx.to_numpy_array(g)
    noa_idx = np.argwhere(adj[c_idx,:].sum(axis=0) == len(c))
    if len(noa_idx) == 0:
        return None
    return node_list[noa_idx[0][0]]

def find_neighbor_of_all(g, c):
    for n in g.nodes:
        if not n in c:
            if all([g.has_edge(n,u) for u in c]):
                return n
    return None

def subgraph_edges(g, c):
    return [e for e in g.edges if not ((e[0] in c) and (e[1] in c))]

def clique_covering(g):
    _g = g.copy()
    C = []

    while _g.number_of_edges() > 0:
        # select first edge
        e = next(iter(_g.edges))
        _g.remove_edge(*e)
        c = [*e]
        t0 = time.time()
        while True:
            u = find_neighbor_of_all_3(_g, c)
            if u is None:
                break
            # remove edges
            for v in c:
                _g.remove_edge(u,v)
            c.append(u)
        C.append(c)
        t0 = time.time() - t0
    return C

def invert_permutation(p):
    """Return an array s with which np.array_equal(arr[p][s], arr) is True.
    The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
    """
    p = np.asanyarray(p) # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s

def clique_cmp(c1, c2):
    if len(c1)==len(c2):
        if c1 < c2:
            return -1
        else:
            return 1
    else:
        return len(c1) - len(c2)
def sorted_cliques(cliques):
    from functools import cmp_to_key
    cliques = sorted(cliques, key=cmp_to_key(clique_cmp))
    return cliques

def clique_covering_3(g, randomize=False):
    _g = g.copy()
    n = _g.number_of_nodes()
    if randomize:
        perm = np.random.permutation(n)
        nmap = dict(enumerate(perm))
        _g = nx.relabel_nodes(_g, nmap)
    else:
        perm = np.arange(n)

    perm = invert_permutation(perm)

    # mark all edges as not covered
    uncovered_edges = set(list(_g.edges))

    C = []
    while len(uncovered_edges):
        # select an uncovered edge
        c = list(uncovered_edges.pop())
        W = set(_g.neighbors(c[0])).intersection(_g.neighbors(c[1])) - set(c)
        while len(W):
            w = W.pop()
            for u in c:
                e = (min(u,w),max(u,w))
                uncovered_edges.discard((min(u,w),max(u,w)))

            c.append(w)
            W = W.intersection(_g.neighbors(w))
        A = [perm[i] for i in c]
        A.sort()
        C.append(A)
    return sorted_cliques(C)


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from dopy.mis import *

    graph_sizes = np.arange(400, 1000, 10)
    runtimes = []
    number_of_edges = []
    number_of_nodes = []
    mean_clique_sizes = []
    number_of_cliques = []
    sufficient_statistics_dims = []
    scores_boxplots = [[],[]]
    reps = 1
    for gz in graph_sizes:
        print("Graph size ", gz)

        for i in range(reps):
            g = nx.generators.random_graphs.erdos_renyi_graph(gz, 0.15)
            #pb  = MIS(g)

            #x = np.array(pb.solve().x, dtype=int)
            #print("x = ", x)
            #print(x.shape)
            ## solve continuous relaxation
            #mdl = cplex_model(pb, continuous_relaxation=True)
            #sol = mdl.solve()
            ##x1 = sol.as_df()["value"].to_numpy()
            #x1 = np.array([sol.get_value(mdl.get_var_by_name(f"x{i}")) for i in range(pb.nv())])

            #print("x1 = ", x1)
            #print(x1.shape)

            #print(g.number_of_nodes())

            t0 = time.time()
            C = clique_covering(g)
            t0 = time.time() - t0
            print("clique_covering")
            print("\tclique covering size = ", len(C))
            print("\tclique covering time = ", t0)
            clique_sizes = np.array([len(c) for c in C])
            print(np.unique(clique_sizes,return_counts=True))


            t0 = time.time()
            C = clique_covering_2(g)
            t0 = time.time() - t0
            print("clique_covering_2")
            print("\tclique covering size = ", len(C))
            print("\tclique covering time = ", t0)
            clique_sizes = np.array([len(c) for c in C])
            print(np.unique(clique_sizes,return_counts=True))

            t0 = time.time()
            C = clique_covering_3(g)
            t0 = time.time() - t0
            print("clique_covering_3")
            print("\tclique covering size = ", len(C))
            print("\tclique covering time = ", t0)
            clique_sizes = np.array([len(c) for c in C])
            print(np.unique(clique_sizes,return_counts=True))

            t0 = time.time()
            C = clique_covering_3(g, randomize=True)
            t0 = time.time() - t0
            print("clique_covering_3 randomized")
            print("\tclique covering size = ", len(C))
            print("\tclique covering time = ", t0)
            clique_sizes = np.array([len(c) for c in C])
            print(np.unique(clique_sizes,return_counts=True))

            exit()

            # solve with clique covering
            pb = MIS(g)
            mdl2 = cplex_model(pb, continuous_relaxation=True, cliques=C)
            sol2 = mdl2.solve()
            #x2 = sol2.as_df()["value"].to_numpy()
            x2 = np.array([sol2.get_value(mdl2.get_var_by_name(f"x{i}")) for i in range(pb.nv())])
            #print([sol2.get_value(mdl2.get_var_by_name(f"x{i}")) for i in range(pb.nv())])
            print("x2 = ", x2)
            print(x2.shape)
            scores_boxplots[0] += list(x2[x == 0])
            scores_boxplots[1] += list(x2[x == 1])

            print(np.vstack((x,x1,x2)))

            runtimes.append(t0)
            number_of_edges.append(g.number_of_edges())
            number_of_nodes.append(g.number_of_nodes())
            mean_clique_sizes.append(np.mean([len(c) for c in C]))
            number_of_cliques.append(np.mean(len(C)))
            sufficient_statistics_dims.append(2*g.number_of_nodes()+np.sum([2**len(c) for c in C]))

        print("t = ", runtimes[-1])

    plt.figure()
    pf = np.polyfit(number_of_edges, runtimes, 3)
    xe = np.linspace(np.min(number_of_edges), np.max(number_of_edges), 1000)
    plt.suptitle("Clique covering runtime vs. graph size")
    plt.scatter(number_of_edges, runtimes, marker="+")
    plt.plot(xe, pf[0] * xe**3 + pf[1] * xe**2 + pf[2] * xe + pf[3], "r--", label=f"y=poly3(x)")
    plt.ylabel("Runtime (s)")
    plt.xlabel("|E|")
    plt.legend()
    print(pf)

    plt.figure()
    plt.suptitle("Mean clique size vs. graph size")
    plt.scatter(number_of_edges, mean_clique_sizes, marker="+")
    plt.ylabel("Clique size")
    plt.xlabel("|E|")

    plt.figure()
    plt.suptitle("Number of cliques size vs. graph size")
    plt.plot(number_of_edges, number_of_edges, "r--")
    plt.scatter(number_of_edges, number_of_cliques, marker="+")
    plt.ylabel("Number of cliques")
    plt.xlabel("|E|")

    plt.figure()
    plt.suptitle("Number of cliques / number of edges")
    plt.scatter(number_of_edges, np.array(number_of_cliques)/np.array(number_of_edges), marker="+")
    plt.ylabel("Number of cliques / |E|")
    plt.xlabel("|E|")

    plt.figure()
    y = np.array(sufficient_statistics_dims)/(2*np.array(number_of_nodes)+4*np.array(number_of_edges))
    plt.suptitle("Number of MRF parameters in clique covering / edge covering")
    plt.scatter(number_of_edges, y, marker="+")
    plt.ylabel("Number of cliques / |E|")
    plt.xlabel("|E|")

    plt.figure()
    plt.suptitle("Fractional solution value per class")
    plt.boxplot(scores_boxplots)
    plt.ylabel("x value")
    plt.xlabel("fractional value")

    plt.show()



