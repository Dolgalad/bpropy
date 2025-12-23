import unittest
import numpy as np
import networkx as nx

from bpropy.co.mis import MIS, mis_batch, mis_equal, mis_greedy
from bpropy.clique_cover import clique_cover
from tests.helpers.mis import very_simple_mis, simple_mis


class TestMISCore(unittest.TestCase):
    def setUp(self):
        self.pb_simple = simple_mis()
        self.pb_very_simple = very_simple_mis()

    # ---------- construction / invariants ----------
    def test_init_counts(self):
        self.assertEqual(self.pb_simple.nv(), 3)
        # triangle has 3 edges
        self.assertEqual(self.pb_simple.ne(), 3)

    def test_init_weights_length_mismatch_raises(self):
        g = nx.Graph()
        g.add_nodes_from([0, 1, 2])
        g.add_edges_from([(0, 1), (1, 2)])
        with self.assertRaises(ValueError):
            MIS(g, weights=np.array([1.0, 2.0]))  # wrong length

    def test_init_cliques_must_cover_edges(self):
        # triangle graph but only one edge clique => not edge cover
        g = nx.Graph()
        g.add_nodes_from([0, 1, 2])
        g.add_edges_from([(0, 1), (1, 2), (0, 2)])
        with self.assertRaises(ValueError):
            MIS(g, cliques=[[0, 1]])  # missing edges (1,2) and (0,2)

    # ---------- LP view ----------
    def test_variable_bounds(self):
        lb, ub = self.pb_simple.variable_bounds()
        self.assertEqual(lb.shape, (self.pb_simple.nv(),))
        self.assertEqual(ub.shape, (self.pb_simple.nv(),))
        np.testing.assert_allclose(lb, np.zeros(self.pb_simple.nv()))
        np.testing.assert_allclose(ub, np.ones(self.pb_simple.nv()))

    def test_linear_constraints_matrix_edges_default(self):
        A, lb, ub = self.pb_simple.linear_constraints_matrix()
        # default cliques are edges; #constraints == #edges
        self.assertEqual(A.shape[0], self.pb_simple.ne())
        self.assertEqual(A.shape[1], self.pb_simple.nv())
        self.assertEqual(lb.shape, (self.pb_simple.ne(),))
        self.assertEqual(ub.shape, (self.pb_simple.ne(),))
        self.assertTrue(np.all(np.isneginf(lb)))
        self.assertTrue(np.allclose(ub, 1.0))

    def test_linear_constraints_matrix_custom_cliques(self):
        A, lb, ub = self.pb_simple.linear_constraints_matrix(cliques=[[0, 1, 2]])
        self.assertEqual(A.shape[0], 1)
        self.assertEqual(A.shape[1], self.pb_simple.nv())
        self.assertTrue(np.isneginf(lb[0]))
        self.assertAlmostEqual(float(ub[0]), 1.0)

    # ---------- feasibility ----------
    def test_check_solution_feasible_and_infeasible(self):
        # For triangle, any single 1 is feasible; any two 1s violate an edge constraint.
        x_ok = np.array([1, 0, 0], dtype=float)
        x_bad = np.array([1, 1, 0], dtype=float)
        self.assertTrue(self.pb_simple.check_solution(x_ok))
        self.assertFalse(self.pb_simple.check_solution(x_bad))

    def test_check_solution_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            self.pb_simple.check_solution(np.array([1.0, 0.0]))

    # ---------- greedy heuristic ----------
    def test_greedy_returns_feasible(self):
        sol = self.pb_simple.greedy()
        self.assertTrue(self.pb_simple.check_solution(sol.x))
        self.assertEqual(sol.solver, "greedy")

    def test_mis_greedy_argmax_policy(self):
        # deterministic on a path graph of 3 nodes:
        # scores choose node 1 highest => solution {1}
        g = nx.path_graph(3)
        pb = MIS(g)
        scores = np.array([0.0, 10.0, 0.0])
        x = mis_greedy(pb, scores, policy="argmax")
        np.testing.assert_allclose(x, np.array([0.0, 1.0, 0.0]))

    def test_mis_greedy_unknown_policy_raises(self):
        g = nx.path_graph(3)
        pb = MIS(g)
        scores = np.zeros(pb.nv())
        with self.assertRaises(ValueError):
            mis_greedy(pb, scores, policy="nope")

    # ---------- batching core ----------
    def test_batch_and_get_problem_roundtrip(self):
        pb1 = self.pb_simple
        pb2 = MIS(nx.path_graph(4))
        pb2.C = clique_cover(pb2.g)
        pb_batch = mis_batch([pb1, pb2])

        self.assertEqual(pb_batch.num_subproblems, 2)
        self.assertEqual(pb_batch.var_sizes.sum(), pb_batch.nv())

        p0 = pb_batch.get_problem(0)
        p1 = pb_batch.get_problem(1)

        self.assertTrue(mis_equal(p0, pb1))
        self.assertTrue(mis_equal(p1, pb2))

    def test_batch_greedy_splits_feasible(self):
        pb1 = MIS(nx.cycle_graph(5))
        pb1.C = clique_cover(pb1.g)
        pb2 = MIS(nx.path_graph(6))
        pb2.C = clique_cover(pb2.g)

        pb_batch = mis_batch([pb1, pb2])
        sol = pb_batch.greedy()
        x = np.asarray(sol.x, dtype=float)

        offsets = np.roll(pb_batch.var_sizes.cumsum(), 1)
        offsets[0] = 0

        x1 = x[offsets[0]: offsets[0] + pb_batch.var_sizes[0]]
        x2 = x[offsets[1]: offsets[1] + pb_batch.var_sizes[1]]

        self.assertTrue(pb1.check_solution(x1))
        self.assertTrue(pb2.check_solution(x2))

    def test_get_problem_out_of_range_raises(self):
        pb_batch = mis_batch([self.pb_simple])
        with self.assertRaises(ValueError):
            pb_batch.get_problem(1)


if __name__ == "__main__":
    unittest.main()

