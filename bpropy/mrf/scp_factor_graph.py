import numpy as np
import networkx as nx

from bpropy.mrf.factor_graph import FactorGraph
from bpropy.co.scp import SCP
from bpropy.utils.set_sort import lenlex_sort

class SCPFactorGraph(FactorGraph):
    def __init__(self, pb=None):
        super().__init__()
        self.num_vars = np.array([], dtype=int)
        self.num_factors = np.array([], dtype=int)
        self.num_graphs = 0

        self.add_variables_factors_from_scp(pb)

        #if not pb is None:
        #    for i in range(pb.num_graphs):
        #        _pb = pb.get_problem(i)
        #        self.add_variables_factors_from_mis(_pb)

    def add_variables_factors_from_scp(self, pb):
        c0 = self.number_of_nodes()
        c = self.number_of_nodes()
        nV = pb.number_of_variables()
        self.add_nodes_from(range(c, c+nV), bipartite=0)
        c += nV

        # unary factors
        for n in range(nV):
            self.add_node(c, bipartite=1)
            self.add_edge(n+c0,c)
            c += 1
        # constraints
        constraints = pb.constraints()
        for fn in range(len(constraints)):
            C = constraints[fn]
            self.add_node(c, bipartite=1)
            for n in np.argwhere(C.A.flatten()!=0).flatten():
                self.add_edge(n+c0, c)

            c += 1
        self.num_graphs += 1
        self.num_vars = np.append(self.num_vars, pb.nv())
        self.num_factors = np.append(self.num_factors, len(constraints)+pb.nv())

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






