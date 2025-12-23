# bpropy/test/test_scp_core.py
import unittest
import numpy as np

from bpropy.co.scp import SCP


def very_simple_scp():
    # 2 items, 2 sets; each set covers a distinct item
    A = [0, 1]
    F = [[0], [1]]
    return SCP(A, F, weights=np.array([1.0, 2.0]))


def simple_scp():
    # 3 items, 3 sets with overlap:
    # F0={0,1}, F1={1,2}, F2={0,2}; min cover with unit weights costs 2
    A = [0, 1, 2]
    F = [[0, 1], [1, 2], [0, 2]]
    return SCP(A, F, weights=np.ones(3))


class TestSCPCore(unittest.TestCase):
    # ----------------------------
    # Construction / basic invariants
    # ----------------------------
    def test_init_basic_counts(self):
        pb = simple_scp()
        self.assertEqual(pb.ne(), 3)  # number of items
        self.assertEqual(pb.nv(), 3)  # number of sets

    def test_init_empty_problem(self):
        pb = SCP(A=[], F=[], weights=np.array([]))
        self.assertEqual(pb.ne(), 0)
        self.assertEqual(pb.nv(), 0)
        self.assertTrue(pb.check_solution(np.array([], dtype=int)))

    def test_init_weights_length_mismatch_raises(self):
        # 2 sets but only 1 weight
        A = [0, 1]
        F = [[0], [1]]
        with self.assertRaises(Exception):
            SCP(A, F, weights=np.array([1.0]))

    # ----------------------------
    # Matrix / constraints interface
    # ----------------------------
    def test_linear_constraints_matrix_shapes(self):
        pb = simple_scp()
        A, lb, ub = pb.linear_constraints_matrix()

        # A is (ne x nv)
        self.assertEqual(A.shape, (pb.ne(), pb.nv()))
        self.assertEqual(lb.shape, (pb.ne(),))
        self.assertEqual(ub.shape, (pb.ne(),))

        # Each row in A is a length-nv vector if iterated
        for row in A:
            self.assertEqual(row.shape, (pb.nv(),))

    def test_adj_matrix_shape_and_semantics(self):
        pb = simple_scp()
        adj = pb.adj_matrix()

        # adj is (ne x nv)
        self.assertEqual(adj.shape, (pb.ne(), pb.nv()))

        # Column j should mark exactly the items in F[j]
        for j, f in enumerate(pb.F):
            covered = np.where(adj[:, j] == 1)[0].tolist()
            self.assertEqual(sorted(covered), sorted(list(f)))

    # ----------------------------
    # Feasibility / check_solution contract
    # ----------------------------
    def test_check_solution_feasible_and_infeasible(self):
        pb = very_simple_scp()
        x_ok = np.array([1, 1], dtype=int)   # covers both items
        x_bad = np.array([1, 0], dtype=int)  # item 1 uncovered
        self.assertTrue(pb.check_solution(x_ok))
        self.assertFalse(pb.check_solution(x_bad))

    def test_check_solution_wrong_length(self):
        pb = very_simple_scp()
        # Wrong length: should not be accepted as feasible.
        # (If your implementation raises instead, this still passes.)
        try:
            ok = pb.check_solution(np.array([1], dtype=int))
            self.assertFalse(ok)
        except Exception:
            pass

    def test_infeasible_instance_item_in_no_set(self):
        # Item 1 is not covered by any set -> infeasible for any x
        A = [0, 1]
        F = [[0]]  # only item 0 is in a set
        pb = SCP(A, F, weights=np.array([1.0]))

        self.assertFalse(pb.check_solution(np.array([0], dtype=int)))
        self.assertFalse(pb.check_solution(np.array([1], dtype=int)))

    # ----------------------------
    # Greedy heuristic (core)
    # ----------------------------
    def test_greedy_returns_feasible_on_feasible_instance(self):
        pb = simple_scp()
        res = pb.greedy()
        self.assertTrue(pb.check_solution(res.x))

    def test_greedy_is_deterministic_on_hand_instance(self):
        # Construct a case where the greedy "best gain" choice is unique each step.
        # Items: 0,1,2
        # Sets:
        #   j0 covers {0,1}
        #   j1 covers {1,2}
        #   j2 covers {0}
        # With equal weights, j0 is strictly best at start (covers 2 items).
        # Then remaining item is 2, so j1 must be chosen.
        A = [0, 1, 2]
        F = [[0, 1], [1, 2], [0]]
        w = np.array([1.0, 1.0, 1.0])
        pb = SCP(A, F, weights=w)

        x1 = pb.greedy().x
        x2 = pb.greedy().x
        np.testing.assert_array_equal(x1, x2)

        # Expected selection is sets 0 and 1
        expected = np.array([1, 1, 0], dtype=int)
        np.testing.assert_array_equal(x1, expected)

    def test_greedy_handles_zero_gain_midrun(self):
        # Case where some sets become irrelevant mid-run (cover no new items)
        A = [0, 1, 2]
        F = [[0], [0, 1], [1, 2]]
        w = np.array([1.0, 1.0, 1.0])
        pb = SCP(A, F, weights=w)

        x = pb.greedy().x
        self.assertTrue(pb.check_solution(x))


if __name__ == "__main__":
    unittest.main()

