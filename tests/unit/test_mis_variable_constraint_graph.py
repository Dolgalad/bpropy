import unittest
import networkx as nx
import numpy as np

from bpropy.graphs.vc_graph import VCGraph
from bpropy.co.mis import MIS
from bpropy.co.generators.mis import random_mis
from bpropy.clique_cover import clique_cover

from tests.helpers.mis import very_simple_mis, simple_mis

class TestMISVCGraph(unittest.TestCase):
    def test_variable_constraint_graph(self):
        pb = random_mis(100)
        vcg = VCGraph(pb)

        self.assertTrue(pb.nv() == vcg.nv())
        self.assertTrue(pb.ne() == vcg.nc())

        for C in vcg.iter_constraints():
            var_l = vcg.constraint_variables(C)
            self.assertTrue(pb.g.has_edge(*var_l))

        pb1 = MIS.from_vcgraph(vcg)
        self.assertTrue(pb.nv() == pb1.nv())
        self.assertTrue(pb.ne() == pb1.ne())
        for e in pb.g.edges:
            self.assertTrue(pb1.g.has_edge(*e))
    def test_variable_constraint_clique_graph(self):
        g = nx.generators.random_graphs.erdos_renyi_graph(100, 0.15)
        cc = clique_cover(g)
        pb = MIS(g, cliques=cc)

        vcg = VCGraph(pb)
        self.assertTrue(vcg.nc() == len(cc))

        pb1 = MIS.from_vcgraph(vcg)
        self.assertTrue(np.all([np.all(c1==c2) for c1,c2 in zip(pb.C,pb1.C)]))
        self.assertTrue(pb.nv() == pb1.nv())
        self.assertTrue(pb.ne() == pb1.ne())

        for i in range(len(cc)):
            self.assertTrue(np.all(pb.C[i] == np.array(sorted(list(vcg.neighbors(i+pb.nv()))), dtype=int)))
            self.assertTrue(np.all(pb.C[i] == pb1.C[i]))


if __name__=="__main__":
    unittest.main()





