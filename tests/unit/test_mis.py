import unittest
import time
import importlib.util

import numpy as np
import networkx as nx

from bpropy.co.mis import MIS, random_mis
from bpropy.clique_cover import clique_cover

from tests.helpers.mis import very_simple_mis, simple_mis

#def very_simple_mis():
#    g = nx.Graph()
#    g.add_nodes_from([0,1])
#    g.add_edges_from([[0,1]])
#    return MIS(g)
#
#def simple_mis():
#    g = nx.Graph()
#    g.add_nodes_from([0,1,2])
#    g.add_edges_from([[0,1],[1,2],[0,2]])
#    return MIS(g)

class TestMIS(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simple_pb = simple_mis()
        self.random_pb = random_mis()
        self.random_pb_cc = random_mis()
        self.random_pb_cc.C = clique_cover(self.random_pb_cc.g)

    def test_mis_init(self):
        self.assertTrue(self.simple_pb.nv() == 3)
        self.assertTrue(self.simple_pb.ne() == 3)

    def test_mis_solution(self):
        #pb = simple_mis()
        x = np.array([0,1,0])
        x_bad = np.array([1,1,0])
        self.assertTrue(self.simple_pb.check_solution(x))
        self.assertFalse(self.simple_pb.check_solution(x_bad))

    def test_greedy(self):
        #pb = random_mis()
        x = self.random_pb.greedy()
        self.assertTrue(self.random_pb.check_solution(x))

    def test_ilp_solve(self):
        #pb = simple_mis()
        res = self.simple_pb.solve()
        self.assertTrue(self.simple_pb.check_solution(res.x))

    def test_lprelax_solve(self):
        t_cbc = time.time()
        res_cbc = self.random_pb_cc.solve(relax_integrality=True)
        self.assertTrue(self.random_pb_cc.check_solution(res_cbc.x))
        t_cbc = time.time() - t_cbc

        docplex_spec = importlib.util.find_spec("docplex")
        if docplex_spec is not None:
            t_cplex = time.time()
            res_cplex = self.random_pb_cc.solve(solver="cplex", relax_integrality=True)
            self.assertTrue(self.random_pb_cc.check_solution(res_cplex.x))
            t_cplex = time.time() - t_cplex

        t_scipy = time.time()
        res_scipy = self.random_pb_cc.solve(solver="scipy", relax_integrality=True)
        self.assertTrue(self.random_pb_cc.check_solution(res_scipy.x))
        t_scipy = time.time() - t_scipy

    def test_mis_constraints(self):
        #pb = simple_mis()
        constr = self.simple_pb.constraints()
        self.assertTrue(len(constr) == self.simple_pb.ne())
        res1 = self.simple_pb.solve()

        constr = self.simple_pb.constraints(cliques = [[0,1,2]])
        self.assertTrue(len(constr) == 1)
        res2 = self.simple_pb.solve()

        self.assertTrue(self.simple_pb.check_solution(res1.x))
        self.assertTrue(self.simple_pb.check_solution(res2.x))


if __name__=="__main__":
    unittest.main()
        


