# tests/unit/co/test_lpproblem.py
import unittest
import numpy as np

from bpropy.co.base_lp import LPProblem, LPSolveConfig
from bpropy.co.batched_lp import BatchedLP
from bpropy.co.solution import Solution

try:
    import scipy.sparse as sp
except Exception:
    sp = None


class _ToyMinLP(LPProblem):
    """
    Minimize c^T x
    s.t. x0 + x1 >= 1
         0 <= x <= 1
    Unique optimum: x = [1, 0] if c=[1,2]
    """
    def __init__(self, c=(1.0, 2.0), *, sparse=False):
        self._c = np.asarray(c, dtype=float)
        self._sparse = bool(sparse)

    def nv(self) -> int:
        return 2

    @property
    def objective_sense(self):
        return "min"

    def objective_coeffs(self) -> np.ndarray:
        return self._c.copy()

    def variable_bounds(self):
        lb = np.zeros(2, dtype=float)
        ub = np.ones(2, dtype=float)
        return lb, ub

    def linear_constraints_matrix(self):
        A = np.array([[1.0, 1.0]], dtype=float)
        if self._sparse and sp is not None:
            A = sp.csr_matrix(A)
        lb = np.array([1.0], dtype=float)
        ub = np.array([np.inf], dtype=float)
        return A, lb, ub


class _ToyMaxLP(_ToyMinLP):
    @property
    def objective_sense(self):
        return "max"


class TestLPProblemCore(unittest.TestCase):
    def test_constraints_wrapper(self):
        pb = _ToyMinLP()
        cons = pb.constraints()
        self.assertEqual(len(cons), 1)
        # SciPy LinearConstraint has attributes .lb/.ub but A stored inside .A
        self.assertTrue(hasattr(cons[0], "A"))
        self.assertTrue(hasattr(cons[0], "lb"))
        self.assertTrue(hasattr(cons[0], "ub"))

    def test_weights_sign_convention(self):
        pb_min = _ToyMinLP(c=(3.0, 4.0))
        pb_max = _ToyMaxLP(c=(3.0, 4.0))
        np.testing.assert_allclose(pb_min.weights(), np.array([-3.0, -4.0]))
        np.testing.assert_allclose(pb_max.weights(), np.array([3.0, 4.0]))

    def test_check_solution_feasible_dense(self):
        pb = _ToyMinLP()
        self.assertTrue(pb.check_solution(np.array([1.0, 0.0])))
        self.assertTrue(pb.check_solution(np.array([0.0, 1.0])))
        self.assertFalse(pb.check_solution(np.array([0.0, 0.0])))

    def test_check_solution_with_slacks(self):
        pb = _ToyMinLP()
        feas, (lb_slack, ub_slack, v_lb_slack, v_ub_slack) = pb.check_solution(
            np.array([1.0, 0.0]),
            return_slack=True,
            check_var_bounds=True,
        )
        self.assertTrue(feas)
        self.assertEqual(lb_slack.shape, (1,))
        self.assertEqual(ub_slack.shape, (1,))
        self.assertEqual(v_lb_slack.shape, (2,))
        self.assertEqual(v_ub_slack.shape, (2,))

    def test_check_solution_wrong_length_raises(self):
        pb = _ToyMinLP()
        with self.assertRaises(ValueError):
            pb.check_solution(np.array([1.0]))

    def test_check_solution_var_bounds_enforced(self):
        pb = _ToyMinLP()
        # violates ub bound
        self.assertFalse(pb.check_solution(np.array([2.0, 0.0])))
        # bounds ignored if check_var_bounds=False
        self.assertTrue(pb.check_solution(np.array([2.0, 0.0]), check_var_bounds=False))

    @unittest.skipUnless(sp is not None, "scipy.sparse not available")
    def test_check_solution_sparse_A(self):
        pb = _ToyMinLP(sparse=True)
        self.assertTrue(pb.check_solution(np.array([1.0, 0.0])))
        self.assertFalse(pb.check_solution(np.array([0.0, 0.0])))

    def test_solve_scipy_integral(self):
        pb = _ToyMinLP(c=(1.0, 2.0))
        sol = pb.solve(solver="scipy", relax_integrality=False)
        self.assertIsInstance(sol, Solution)
        self.assertEqual(sol.solver, "scipy")
        # should be integral
        x = np.asarray(sol.x, dtype=float)
        self.assertTrue(np.all(np.abs(x - np.round(x)) <= 1e-8))
        # optimum is [1,0]
        np.testing.assert_allclose(np.round(x), np.array([1.0, 0.0]))

    def test_solve_scipy_relaxed(self):
        pb = _ToyMinLP(c=(1.0, 2.0))
        sol = pb.solve(solver="scipy", relax_integrality=True)
        self.assertEqual(sol.solver, "scipy")
        x = np.asarray(sol.x, dtype=float)
        self.assertTrue(np.all(x >= -1e-8))
        self.assertTrue(np.all(x <= 1.0 + 1e-8))
        # still should satisfy the constraint
        self.assertTrue(pb.check_solution(x))

    def test_solve_override_constraints(self):
        pb = _ToyMinLP(c=(1.0, 2.0))
        # Override with x0 >= 1 (forces x0=1)
        A = np.array([[1.0, 0.0]])
        lb = np.array([1.0])
        ub = np.array([np.inf])
        sol = pb.solve(
            solver="scipy",
            relax_integrality=False,
            override_linear_constraints=(A, lb, ub),
        )
        x = np.round(np.asarray(sol.x, dtype=float))
        np.testing.assert_allclose(x, np.array([1.0, 0.0]))

    def test_solve_unknown_solver_raises(self):
        pb = _ToyMinLP()
        with self.assertRaises(ValueError):
            pb.solve(solver="not-a-solver")

    def test_solve_config_overrides(self):
        pb = _ToyMinLP()
        cfg = LPSolveConfig(solver="scipy", relax_integrality=True, verbose=False)
        sol = pb.solve(config=cfg, relax_integrality=False)  # explicit arg should override config
        x = np.asarray(sol.x, dtype=float)
        # should be integral since relax_integrality=False
        self.assertTrue(np.all(np.abs(x - np.round(x)) <= 1e-8))


class TestBatchedLP(unittest.TestCase):
    def test_batchedlp_basic_metadata(self):
        p1 = _ToyMinLP(c=(1.0, 2.0))
        p2 = _ToyMinLP(c=(3.0, 1.0))
        bat = BatchedLP([p1, p2])

        self.assertEqual(bat.num_subproblems, 2)
        np.testing.assert_array_equal(bat.var_sizes, np.array([2, 2]))
        np.testing.assert_array_equal(bat.var_offsets, np.array([0, 2]))
        self.assertEqual(bat.nv(), 4)
        self.assertEqual(bat.objective_sense, "min")

    def test_batchedlp_objective_coeffs_concat(self):
        p1 = _ToyMinLP(c=(1.0, 2.0))
        p2 = _ToyMinLP(c=(3.0, 1.0))
        bat = BatchedLP([p1, p2])
        np.testing.assert_allclose(bat.objective_coeffs(), np.array([1.0, 2.0, 3.0, 1.0]))

    def test_batchedlp_bounds_concat(self):
        p1 = _ToyMinLP()
        p2 = _ToyMinLP()
        bat = BatchedLP([p1, p2])
        lb, ub = bat.variable_bounds()
        self.assertEqual(lb.shape, (4,))
        self.assertEqual(ub.shape, (4,))
        np.testing.assert_allclose(lb, np.zeros(4))
        np.testing.assert_allclose(ub, np.ones(4))

    def test_batchedlp_constraints_block_diagonal(self):
        p1 = _ToyMinLP()
        p2 = _ToyMinLP()
        bat = BatchedLP([p1, p2])

        A, lb, ub = bat.linear_constraints_matrix()
        # 2 constraints total (1 per problem), 4 vars total
        self.assertEqual(lb.shape, (2,))
        self.assertEqual(ub.shape, (2,))
        self.assertEqual(A.shape, (2, 4))

        # The constraint matrix should look like:
        # [1 1 0 0]
        # [0 0 1 1]
        Ad = A.toarray() if hasattr(A, "toarray") else np.asarray(A)
        np.testing.assert_allclose(Ad, np.array([[1, 1, 0, 0],
                                                [0, 0, 1, 1]], dtype=float))
        np.testing.assert_allclose(lb, np.array([1.0, 1.0]))
        self.assertTrue(np.isinf(ub[0]) and np.isinf(ub[1]))

    def test_split_and_merge_assignments(self):
        p1 = _ToyMinLP()
        p2 = _ToyMinLP()
        bat = BatchedLP([p1, p2])

        x = np.array([1.0, 0.0, 0.0, 1.0])
        xs = bat.split_assignment(x)
        self.assertEqual(len(xs), 2)
        np.testing.assert_allclose(xs[0], np.array([1.0, 0.0]))
        np.testing.assert_allclose(xs[1], np.array([0.0, 1.0]))

        x2 = bat.merge_assignments(xs)
        np.testing.assert_allclose(x2, x)

    def test_batchedlp_inconsistent_sense_raises(self):
        p1 = _ToyMinLP()
        p2 = _ToyMaxLP()
        with self.assertRaises(ValueError):
            BatchedLP([p1, p2])


class TestSolutionAdapters(unittest.TestCase):
    def test_from_solver_unknown_raises(self):
        with self.assertRaises(KeyError):
            Solution.from_solver("nope", anything=123)

    def test_from_any_scipy_infers(self):
        class _FakeSciPyRes:
            def __init__(self):
                self.x = np.array([1.0, 0.0])

        r = _FakeSciPyRes()
        sol = Solution.from_any(r)
        self.assertEqual(sol.solver, "scipy")
        np.testing.assert_allclose(sol.x, np.array([1.0, 0.0]))

    def test_from_any_requires_solver_when_ambiguous(self):
        class _NoX:
            pass

        with self.assertRaises(ValueError):
            Solution.from_any(_NoX())


def _has_mip():
    try:
        import mip  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_has_mip(), "python-mip not installed")
class TestPythonMipAdapter(unittest.TestCase):
    def test_lp_solve_python_mip_adapter(self):
        pb = _ToyMinLP(c=(1.0, 2.0))
        sol = pb.solve(solver="cbc", relax_integrality=False)

        self.assertTrue(sol.solver.startswith("python-mip:"))
        x = np.round(np.asarray(sol.x, dtype=float))
        np.testing.assert_allclose(x, np.array([1.0, 0.0]))

        # Adapter should populate duals/slacks arrays with correct sizes
        self.assertIsNotNone(sol.duals)
        self.assertIsNotNone(sol.slacks)
        self.assertEqual(sol.duals.shape, (1,))
        self.assertEqual(sol.slacks.shape, (1,))


if __name__ == "__main__":
    unittest.main()

