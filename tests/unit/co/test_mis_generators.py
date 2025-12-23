import unittest
import numpy as np

from bpropy.co.mis import MIS
from bpropy.co.generators.mis import random_mis
from bpropy.clique_cover import is_edge_cover, clique_cover


class TestMISGenerators(unittest.TestCase):
    def test_random_mis_returns_mis(self):
        pb = random_mis()
        self.assertIsInstance(pb, MIS)
        self.assertGreaterEqual(pb.nv(), 0)
        self.assertGreaterEqual(pb.ne(), 0)

    def test_random_mis_deterministic_with_seed_if_supported(self):
        # If your random_mis supports seed, this test is meaningful.
        # If not, you can delete it or modify generator to accept seed.
        try:
            pb1 = random_mis(50, seed=123)
            pb2 = random_mis(50, seed=123)
        except TypeError:
            self.skipTest("random_mis does not accept seed parameter")

        # Compare edge sets + weights
        e1 = {tuple(sorted(map(int, e))) for e in pb1.g.edges()}
        e2 = {tuple(sorted(map(int, e))) for e in pb2.g.edges()}
        self.assertEqual(e1, e2)
        np.testing.assert_allclose(pb1.w, pb2.w)

    def test_cliques_cover_edges_default_or_recomputed(self):
        pb = random_mis(40)
        # Either pb.C already covers edges, or we can set it via clique_cover.
        if not is_edge_cover(pb.C, pb.g):
            pb.C = clique_cover(pb.g)
        self.assertTrue(is_edge_cover(pb.C, pb.g))

    def test_greedy_on_random_is_feasible(self):
        pb = random_mis(60)
        # ensure clique cover is valid for LP check_solution (if generator only uses edges it should already be ok)
        if not is_edge_cover(pb.C, pb.g):
            pb.C = clique_cover(pb.g)
        sol = pb.greedy()
        self.assertTrue(pb.check_solution(sol.x))


if __name__ == "__main__":
    unittest.main()

