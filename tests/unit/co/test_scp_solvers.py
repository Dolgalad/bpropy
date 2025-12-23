# tests/test_scp_solvers.py
import itertools
import shutil
import unittest
import numpy as np

from bpropy.co.scp import SCP


def _tiny_scp_for_optimality() -> SCP:
    # 3 items, 3 sets:
    # F0={0,1}, F1={1,2}, F2={0,2}
    # Make optimum unique: pick sets 0 and 1 (cost 2.0), set 2 is expensive.
    A = [0, 1, 2]
    F = [[0, 1], [1, 2], [0, 2]]
    w = np.array([1.0, 1.0, 10.0])
    return SCP(A, F, weights=w)


def _objective(pb: SCP, x) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(np.dot(pb.w, x))


def _bruteforce_opt(pb: SCP):
    """Return (x_best, obj_best) by enumerating all 0/1 vectors (nv must be small)."""
    n = pb.nv()
    best_obj = float("inf")
    best_x = None
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=int)
        if pb.check_solution(x):
            obj = _objective(pb, x)
            if obj < best_obj:
                best_obj = obj
                best_x = x
    return best_x, best_obj


def _has_mip() -> bool:
    try:
        import mip  # noqa: F401
        return True
    except Exception:
        return False


def _has_scipy() -> bool:
    try:
        import scipy  # noqa: F401
        return True
    except Exception:
        return False


class TestSCPSolvers(unittest.TestCase):
    def test_solve_greedy_dispatch(self):
        pb = _tiny_scp_for_optimality()

        res_direct = pb.greedy()
        res_solve = pb.solve(solver="greedy")

        # Both must be feasible
        self.assertTrue(pb.check_solution(np.asarray(res_direct.x, dtype=int)))
        self.assertTrue(pb.check_solution(np.asarray(res_solve.x, dtype=int)))

        # Same heuristic => same x (should be deterministic for argmax policy)
        np.testing.assert_array_equal(
            np.asarray(res_direct.x, dtype=float),
            np.asarray(res_solve.x, dtype=float),
        )

        # Solver label should be set by greedy()
        self.assertEqual(res_solve.solver, "greedy")

        # objective is not guaranteed to be populated; compute it
        self.assertAlmostEqual(_objective(pb, res_direct.x), _objective(pb, res_solve.x), places=9)

    @unittest.skipUnless(_has_scipy(), "SciPy not installed")
    def test_solve_scipy_relax_integrality(self):
        pb = _tiny_scp_for_optimality()
        res = pb.solve(solver="scipy", relax_integrality=True)

        self.assertEqual(res.solver, "scipy")

        x = np.asarray(res.x, dtype=float)
        self.assertEqual(x.shape, (pb.nv(),))

        # Bounds [0,1] with small tolerance
        self.assertTrue(np.all(x >= -1e-8))
        self.assertTrue(np.all(x <= 1.0 + 1e-8))

        # Must satisfy coverage constraints A @ x >= 1 (tolerance)
        A, lb, ub = pb.linear_constraints_matrix()
        Ax = A @ x
        self.assertTrue(np.all(Ax + 1e-8 >= lb))

    @unittest.skipUnless(_has_scipy(), "SciPy not installed")
    def test_solve_scipy_integral_matches_bruteforce_on_tiny_instance(self):
        pb = _tiny_scp_for_optimality()
        x_star, obj_star = _bruteforce_opt(pb)
        self.assertIsNotNone(x_star, "Tiny SCP should be feasible")

        # SciPy MILP with integrality enforced
        res = pb.solve(solver="scipy", relax_integrality=False)
        self.assertEqual(res.solver, "scipy")

        x = np.asarray(res.x, dtype=float)

        # Should be (near-)integral; allow tiny numerical tolerance
        self.assertTrue(np.all(np.abs(x - np.round(x)) <= 1e-8))
        x_int = np.asarray(np.round(x), dtype=int)

        self.assertTrue(pb.check_solution(x_int))
        self.assertAlmostEqual(_objective(pb, x_int), float(obj_star), places=7)
        np.testing.assert_array_equal(x_int, x_star)

    @unittest.skipUnless(_has_mip(), "python-mip not installed")
    def test_solve_cbc_integral_matches_bruteforce_on_tiny_instance(self):
        pb = _tiny_scp_for_optimality()
        x_star, obj_star = _bruteforce_opt(pb)
        self.assertIsNotNone(x_star, "Tiny SCP should be feasible")

        res = pb.solve(solver="cbc", relax_integrality=False)

        # Adapter sets solver like "python-mip:cbc" (or other backend name)
        self.assertIsNotNone(res.solver)
        self.assertTrue(res.solver.startswith("python-mip:"))

        x = np.asarray(res.x, dtype=float)
        # python-mip returns floats; convert to int by rounding
        x_int = np.asarray(np.round(x), dtype=int)

        self.assertTrue(pb.check_solution(x_int))
        self.assertAlmostEqual(_objective(pb, x_int), float(obj_star), places=7)
        np.testing.assert_array_equal(x_int, x_star)

    @unittest.skipUnless(_has_mip(), "python-mip not installed")
    def test_solve_cbc_relax_integrality(self):
        pb = _tiny_scp_for_optimality()
        res = pb.solve(solver="cbc", relax_integrality=True)

        self.assertIsNotNone(res.solver)
        self.assertTrue(res.solver.startswith("python-mip:"))

        x = np.asarray(res.x, dtype=float)
        self.assertEqual(x.shape, (pb.nv(),))
        self.assertTrue(np.all(x >= -1e-8))
        self.assertTrue(np.all(x <= 1.0 + 1e-8))

        A, lb, ub = pb.linear_constraints_matrix()
        Ax = A @ x
        self.assertTrue(np.all(Ax + 1e-8 >= lb))

    def test_solve_unknown_solver_raises(self):
        pb = _tiny_scp_for_optimality()
        with self.assertRaises(Exception):
            pb.solve(solver="definitely_not_a_solver")

    def test_solve_scg3_feasible(self):
        pb = _tiny_scp_for_optimality()
        res = pb.solve(solver="scg3", relax_integrality=False)

        self.assertEqual(res.solver, "scg3")
        x = np.asarray(res.x, dtype=float)
        x_int = np.asarray(np.round(x), dtype=int)

        self.assertTrue(pb.check_solution(x_int))


if __name__ == "__main__":
    unittest.main()

