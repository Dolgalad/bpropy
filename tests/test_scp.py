# bpropy/test/test_scp.py
import unittest
import time
import importlib.util
import numpy as np

from scipy.optimize import LinearConstraint

from bpropy.co.scp import (
    SCP,
    random_scp_items_k,
    random_scp_items_k_balanced,
)
# The MIS tests use this style/structure; we mirror that here. :contentReference[oaicite:2]{index=2}


def very_simple_scp():
    # 2 items, 2 sets; each set covers a distinct item → solution is any single set per item
    # A = {0,1}, F0={0}, F1={1}
    A = [0, 1]
    F = [[0], [1]]
    return SCP(A, F, weights=np.array([1.0, 2.0]))  # weighted to let objective checks bite


def simple_scp():
    # 3 items, 3 sets with overlap:
    # F0={0,1}, F1={1,2}, F2={0,2}; min cover with unit weights costs 2
    A = [0, 1, 2]
    F = [[0, 1], [1, 2], [0, 2]]
    return SCP(A, F, weights=np.ones(3))


class TestSCP(unittest.TestCase):
    def setUp(self):
        self.pb_simple = simple_scp()
        # Medium instance via K-regular (on items) generator
        self.pb_rand = random_scp_items_k(n=60, m=25, k=3, seed=0)

    # ---------- basic structure ----------
    def test_scp_init(self):
        self.assertEqual(self.pb_simple.ne(), 3)   # items
        self.assertEqual(self.pb_simple.nv(), 3)   # sets
        self.assertEqual(self.pb_rand.ne(), 60)
        self.assertEqual(self.pb_rand.nv(), 25)

    def test_constraints_shape_and_count(self):
        cons = self.pb_simple.constraints()
        self.assertEqual(len(cons), self.pb_simple.ne())
        for c in cons:
            self.assertIsInstance(c, LinearConstraint)
            # A single row over nv variables
            self.assertEqual(c.A.shape, (1, self.pb_simple.nv()))

    # ---------- feasibility & greedy ----------
    def test_scp_solution_check(self):
        pb = very_simple_scp()
        x_ok  = np.array([1, 1], dtype=int)  # both sets chosen → feasible
        x_bad = np.array([1, 0], dtype=int)  # item 1 uncovered → infeasible
        self.assertTrue(pb.check_solution(x_ok))
        self.assertFalse(pb.check_solution(x_bad))

    def test_greedy_feasible(self):
        x = self.pb_rand.greedy()
        self.assertTrue(self.pb_rand.check_solution(x))

    # ---------- ILP & LP relax ----------
    def test_ilp_solve_cbc(self):
        res = self.pb_simple.solve(solver="cbc", relax_integrality=False)
        self.assertTrue(self.pb_simple.check_solution(res.x))

    def test_lp_relax_scipy(self):
        # LP relaxation should never be worse than the integer optimum
        t0 = time.time()
        res_int = self.pb_rand.solve(solver="cbc", relax_integrality=False)
        t_cbc = time.time() - t0

        self.assertTrue(self.pb_rand.check_solution(res_int.x))

        t1 = time.time()
        res_lp = self.pb_rand.solve(solver="scipy", relax_integrality=True)
        t_sp = time.time() - t1


        self.assertTrue(self.pb_rand.check_solution(res_lp.x))

        # Objective values (SciPy minimizes true costs; CBC too): LP ≤ IP
        cost_ip = float(np.dot(self.pb_rand.w, res_int.x))
        cost_lp = float(np.dot(self.pb_rand.w, res_lp.x))
        self.assertLessEqual(cost_lp + 1e-7, cost_ip)

        # Just smoke-print timings so failures are easier to inspect locally
        _ = (t_cbc, t_sp)  # keep variables live; no asserts on timings

    # ---------- weighted objective consistency ----------
    def test_weighted_objective_agreement_small(self):
        # Tiny random instance where both solvers should land at the same optimum
        rng = np.random.default_rng(123)
        pb = random_scp_items_k(n=6, m=4, k=2, seed=1, weights=1.0 + rng.random(4))
        res_cbc  = pb.solve(solver="cbc",  relax_integrality=False)
        res_scip = pb.solve(solver="scipy", relax_integrality=False)

        cost_cbc  = float(np.dot(pb.w, res_cbc.x))
        cost_scip = float(np.dot(pb.w, res_scip.x))
        self.assertAlmostEqual(cost_cbc, cost_scip, places=6)

        # Ensure feasibility as well
        self.assertTrue(pb.check_solution(res_cbc.x))
        self.assertTrue(pb.check_solution(res_scip.x))

    # ---------- generator sanity ----------
    def test_random_k_generator_regular_on_items(self):
        n, m, k = 50, 20, 3
        pb = random_scp_items_k(n=n, m=m, k=k, seed=42)
        # Each item appears in exactly k sets:
        deg = np.zeros(n, dtype=int)
        for j, f in enumerate(pb.F):
            deg[np.array(f, dtype=int)] += 1
        self.assertTrue(np.all(deg == k))
        # Adjacency matrix shape & boolean semantics
        adj = pb.adj_matrix()
        self.assertEqual(adj.shape, (n, m))
        for j, f in enumerate(pb.F):
            covered = np.where(adj[:, j] == 1)[0].tolist()
            self.assertEqual(sorted(covered), sorted(f))

    def test_balanced_generator_no_empty_sets(self):
        n, m, k = 80, 30, 2
        pb = random_scp_items_k_balanced(n=n, m=m, k=k, seed=7)
        sizes = [len(f) for f in pb.F]
        self.assertTrue(min(sizes) >= 1)  # no empty sets after repair

    # ---------- edge cases ----------
    def test_empty_is_trivially_feasible(self):
        pb = SCP(A=[], F=[], weights=np.array([]))
        self.assertTrue(pb.check_solution(np.array([], dtype=int)))

    def test_greedy_handles_zero_gain(self):
        # Construct a case where some set covers no uncovered items mid-run
        A = [0, 1, 2]
        F = [[0], [0, 1], [1, 2]]
        w = np.array([1.0, 1.0, 1.0])
        pb = SCP(A, F, weights=w)
        x = pb.greedy()
        self.assertTrue(pb.check_solution(x))


if __name__ == "__main__":
    unittest.main()

