import unittest
import time
import importlib.util

import numpy as np
import networkx as nx

from bpropy.co.mis import MIS
from bpropy.co.generators.mis import random_mis
from bpropy.clique_cover import clique_cover

from tests.helpers.mis import very_simple_mis, simple_mis

class TestMIS(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simple_pb = simple_mis()
        self.random_pb = random_mis(seed=2026)
        self.random_pb_cc = random_mis(seed=2027)
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
        self.assertTrue(self.random_pb.check_solution(x.x))

    def test_ilp_solve(self):
        #pb = simple_mis()
        res = self.simple_pb.solve()
        self.assertTrue(self.simple_pb.check_solution(res.x))

    def _assert_relaxation_feasible(self, result, solver):
        # Numerical LP solvers can return small constraint/bound violations.
        # Keep a strict, explicit feasibility tolerance rather than relying on
        # LPProblem.check_solution's 1e-9 exact-ish default. Never ignore NaNs.
        x = np.asarray(result.x, dtype=float)
        self.assertTrue(np.isfinite(x).all(), f"{solver} returned non-finite LP values")
        feasible, slacks = self.random_pb_cc.check_solution(
            x, tol=1e-6, return_slack=True
        )
        finite_slacks = [s[np.isfinite(s)] for s in slacks if s is not None]
        finite_slacks = [s for s in finite_slacks if s.size]
        minimum = min((float(s.min()) for s in finite_slacks), default=float("inf"))
        self.assertTrue(
            feasible,
            f"{solver} LP relaxation violates feasibility (worst slack {minimum:.3e})",
        )

    def test_lprelax_solve(self):
        t_cbc = time.time()
        res_cbc = self.random_pb_cc.solve(relax_integrality=True)
        self._assert_relaxation_feasible(res_cbc, "CBC")
        t_cbc = time.time() - t_cbc

        docplex_spec = importlib.util.find_spec("docplex")
        if docplex_spec is not None:
            t_cplex = time.time()
            res_cplex = self.random_pb_cc.solve(solver="cplex", relax_integrality=True)
            self._assert_relaxation_feasible(res_cplex, "CPLEX")
            t_cplex = time.time() - t_cplex

        t_scipy = time.time()
        res_scipy = self.random_pb_cc.solve(solver="scipy", relax_integrality=True)
        self._assert_relaxation_feasible(res_scipy, "SciPy")
        t_scipy = time.time() - t_scipy

    def test_mis_constraints(self):
        #pb = simple_mis()
        A,_,_ = self.simple_pb.linear_constraints_matrix()
        self.assertTrue(A.shape[0] == self.simple_pb.ne())
        res1 = self.simple_pb.solve()

        A,_,_ = self.simple_pb.linear_constraints_matrix(cliques = [[0,1,2]])
        self.assertTrue(A.shape[0] == 1)
        res2 = self.simple_pb.solve()

        self.assertTrue(self.simple_pb.check_solution(res1.x))
        self.assertTrue(self.simple_pb.check_solution(res2.x))


if __name__=="__main__":
    unittest.main()
        


