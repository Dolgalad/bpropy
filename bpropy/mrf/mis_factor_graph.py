from bpropy.mrf.factor_graph import FactorGraph
from bpropy.co.mis import MIS

class MISFactorGraph(FactorGraph):
    def __init__(self, pb):
        super().__init__()
        self.add_nodes_from(pb.g.nodes, bipartite=0)
        nV = pb.g.number_of_nodes()
        c = nV
        # unary factors
        for n in range(nV):
            self.add_node(c, bipartite=1)
            self.add_edge(n,c)
            c += 1
        for fn in pb.C:
            self.add_node(c, bipartite=1)
            for n in fn:
                self.add_edge(n, c)
            c += 1
