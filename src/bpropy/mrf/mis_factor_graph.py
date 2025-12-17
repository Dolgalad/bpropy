import numpy as np
import networkx as nx

from bpropy.mrf.factor_graph import FactorGraph
from bpropy.co.mis import MIS
from bpropy.utils.set_sort import lenlex_sort

class MISFactorGraph(FactorGraph):
    def __init__(self, pb=None):
        super().__init__()
        self.num_vars = np.array([], dtype=int)
        self.num_factors = np.array([], dtype=int)
        self.num_graphs = 0

        if not pb is None:
            #import time
            #t_pb = 0
            #t_av = 0
            #t0 = time.time()
            for i in range(pb.num_graphs):
                #_t_pb = time.time()
                _pb = pb.get_problem(i)
                #_t_pb = time.time() - _t_pb
                #t_pb += _t_pb

                #_t_av = time.time()
                self.add_variables_factors_from_mis(_pb)
                #_t_av = time.time() - _t_av
                #t_av += _t_av
            #t0 = time.time() - t0
            #print(t0, t_pb, t_av)
            #if pb.num_graphs==1:
            #    self.add_nodes_from(pb.g.nodes, bipartite=0)
            #    nV = pb.g.number_of_nodes()
            #    c = nV

            #    # unary factors
            #    for n in range(nV):
            #        self.add_node(c, bipartite=1)
            #        self.add_edge(n,c)
            #        c += 1
            #    for fn in pb.C:
            #        self.add_node(c, bipartite=1)
            #        for n in fn:
            #            self.add_edge(n, c)
            #        c += 1
            #else:
            #    pb0 = pb.get_problem(0)
            #    print(0, "pb ", pb0.nv(), pb0.ne(), len(pb0.C))

            #    fg1 = MISFactorGraph(pb0)
            #    for i in range(1, pb.num_graphs):
            #        _pb = pb.get_problem(i)
            #        print(i, "pb ", _pb.nv(), _pb.ne(), len(_pb.C))
            #        ag = nx.Graph(MISFactorGraph(_pb))
            #        nV1 = fg1.number_of_nodes()
            #        fg1.add_nodes_from([n+nV1 for n in ag.nodes])
            #        fg1.add_edges_from([[e[0]+nV1,e[1]+nV1] for e in ag.edges])
            #        nx.set_node_attributes(fg1, {n+nV1: d["bipartite"] for n,d in ag.nodes(data=True)}, "bipartite")


            #    self.add_nodes_from(sorted(list(fg1.nodes)))
            #    nx.set_node_attributes(self, {n:d["bipartite"] for n,d in fg1.nodes(data=True)}, "bipartite")
            #    self.add_edges_from(fg1.edges)

            #self.num_graphs = pb.num_graphs
            #num_vars = []
            #num_factors = []
            #for i in range(pb.num_graphs):
            #    pb_ = pb.get_problem(i)
            #    num_vars.append(pb_.nv())
            #    num_factors.append(len(pb_.C)+pb_.nv())
            #self.num_vars = np.array(num_vars)
            #self.num_factors = np.array(num_factors)

            #print("num_vars ", self.num_vars)
            #print("num_factors ", self.num_factors)

    def add_variables_factors_from_mis(self, pb):
        c0 = self.number_of_nodes()
        c = self.number_of_nodes()
        nV = pb.g.number_of_nodes()
        #self.add_nodes_from(pb.g.nodes, bipartite=0)
        self.add_nodes_from(range(c, c+nV), bipartite=0)
        c += nV

        # unary factors
        for n in range(nV):
            self.add_node(c, bipartite=1)
            self.add_edge(n+c0,c)
            c += 1
        for fn in pb.C:
            self.add_node(c, bipartite=1)
            for n in fn:
                self.add_edge(n+c0, c)
            c += 1
        self.num_graphs += 1
        self.num_vars = np.append(self.num_vars, pb.nv())
        self.num_factors = np.append(self.num_factors, len(pb.C)+pb.nv())

    def get_subgraph(self, idx):
        """Get sub factor graph by index
        """
        offset = np.roll((self.num_vars+self.num_factors).cumsum(),1)
        offset[0] = 0
        nV = (self.num_vars+self.num_factors)[idx]
        g = nx.subgraph(self, [n for n in range(offset[idx], offset[idx]+nV)]).copy()
        nmap = {n+offset[idx]:n for n in range(nV)}
        g = nx.relabel_nodes(g, nmap)

        fg = MISFactorGraph()
        fg.add_nodes_from(sorted(list(g.nodes)))
        nx.set_node_attributes(fg, {n:d["bipartite"] for n,d in g.nodes(data=True)}, "bipartite")
        fg.add_edges_from(lenlex_sort(g.edges))
        fg.num_graphs = 1
        fg.num_vars = np.array([self.num_vars[idx]])
        fg.num_factors = np.array([self.num_factors[idx]])
        return fg






