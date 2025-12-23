import os
import unittest
import importlib.util
import numpy as np
import networkx as nx

from bpropy.co.mis import MIS
from bpropy.clique_cover import clique_cover


def _has_mip():
    try:
        import mip  # noqa: F401
        return True
    except Exception:
        return False


def _has_scipy():
    try:
        import scipy  # noqa: F401
        return True
    except Exception:
        return False


def _has_docplex():
    return importlib.util.find_spec("docplex") is not None


def _has_kamis():
    # Your kamis_solve requires KAMIS_ROOT env var and uses os.system to run a binary. :contentReference[oaicite:3]{index=3}
    root = os.getenv("KAMIS_ROOT")
    if not root:
        return False
    # most common expected binary name in your code: redumis in deploy/
    return os.path.exists(os.path.join(root, "deploy", "redumis"))


def _tiny_triangle_mis():
    # triangle MIS optimum size = 1
    g = nx.cycle_graph(3)
    pb = MIS(g)
    pb.C = clique_cover(pb.g)
    return pb


class TestMISSolvers(unittest.TestCase):
    @unittest.skipUnless(_has_mip(), "python-mip not installed")
    def test_solve_default_python_mip_feasible(self):
        pb = _tiny_triangle_mis()
        sol = pb.solve(verbose=False)  # default solver python-mip/cbc
        self.assertTrue(pb.check_solution(sol.x))
        self.assertIsNotNone(sol.solver)
        self.assertTrue(sol.solver.startswith("python-mip:"))

    @unittest.skipUnless(_has_mip(), "python-mip not installed")
    def test_solve_cbc_integral(self):
        pb = _tiny_triangle_mis()
        sol = pb.solve(solver="cbc", relax_integrality=False)
        self.assertTrue(pb.check_solution(sol.x))
        self.assertTrue(sol.solver.startswith("python-mip:"))

        # should be integral (binary) (tolerance)
        x = np.asarray(sol.x, dtype=float)
        self.assertTrue(np.all(np.abs(x - np.round(x)) <= 1e-8))

    @unittest.skipUnless(_has_mip(), "python-mip not installed")
    def test_solve_cbc_relaxed(self):
        pb = _tiny_triangle_mis()
        sol = pb.solve(solver="cbc", relax_integrality=True)
        self.assertTrue(pb.check_solution(sol.x))
        x = np.asarray(sol.x, dtype=float)
        self.assertTrue(np.all(x >= -1e-8))
        self.assertTrue(np.all(x <= 1.0 + 1e-8))

    @unittest.skipUnless(_has_scipy(), "SciPy not installed")
    def test_solve_scipy_relaxed(self):
        pb = _tiny_triangle_mis()
        sol = pb.solve(solver="scipy", relax_integrality=True)
        self.assertTrue(pb.check_solution(sol.x))
        self.assertEqual(sol.solver, "scipy")

    @unittest.skipUnless(_has_scipy(), "SciPy not installed")
    def test_solve_scipy_integral(self):
        pb = _tiny_triangle_mis()
        sol = pb.solve(solver="scipy", relax_integrality=False)
        self.assertTrue(pb.check_solution(sol.x))
        self.assertEqual(sol.solver, "scipy")
        x = np.asarray(sol.x, dtype=float)
        self.assertTrue(np.all(np.abs(x - np.round(x)) <= 1e-8))

    def test_solve_greedy(self):
        pb = _tiny_triangle_mis()
        sol = pb.solve(solver="greedy")
        self.assertTrue(pb.check_solution(sol.x))
        self.assertEqual(sol.solver, "greedy")

    def test_solve_unknown_solver_raises(self):
        pb = _tiny_triangle_mis()
        with self.assertRaises(Exception):
            pb.solve(solver="nope")

    @unittest.skipUnless(_has_docplex(), "docplex not installed")
    def test_solve_cplex_relaxed(self):
        pb = _tiny_triangle_mis()
        sol = pb.solve(solver="cplex", relax_integrality=True)
        self.assertTrue(pb.check_solution(sol.x))
        self.assertEqual(sol.solver, "cplex")

    @unittest.skipUnless(_has_kamis(), "KAMIS_ROOT missing or KaMIS binary not found")
    def test_solve_kamis_runs_and_returns_solution(self):
        pb = _tiny_triangle_mis()
        sol = pb.solve(solver="kamis", verbose=False)
        # KaMIS solution file is integer list; Solution.x is int array (0/1 selection)
        self.assertEqual(sol.solver, "kamis")
        # Basic sanity: correct length and feasible
        self.assertEqual(np.asarray(sol.x).size, pb.nv())
        self.assertTrue(pb.check_solution(sol.x))


if __name__ == "__main__":
    unittest.main()

