# bpropy/test/test_scp_generators.py
import unittest
import numpy as np

from bpropy.co.scp import SCP
from bpropy.co.generators.scp import (
    random_scp_items_k,
    random_scp_items_k_balanced,
)


def _item_degrees(pb: SCP) -> np.ndarray:
    """deg[a] = number of sets containing item a"""
    deg = np.zeros(pb.ne(), dtype=int)
    for j, f in enumerate(pb.F):
        for a in f:
            deg[int(a)] += 1
    return deg


def _set_sizes(pb: SCP) -> np.ndarray:
    return np.array([len(f) for f in pb.F], dtype=int)


class TestSCPGenerators(unittest.TestCase):
    def test_random_scp_items_k_item_degree_exact(self):
        n, m, k = 25, 60, 3
        pb = random_scp_items_k(n=n, m=m, k=k, seed=0)

        self.assertEqual(pb.nv(), n)
        self.assertEqual(pb.ne(), m)

        deg = _item_degrees(pb)
        self.assertTrue(np.all(deg == k), msg=f"min={deg.min()}, max={deg.max()} expected={k}")

    def test_random_scp_items_k_deterministic_with_seed(self):
        pb1 = random_scp_items_k(n=17, m=40, k=2, seed=123)
        pb2 = random_scp_items_k(n=17, m=40, k=2, seed=123)

        # Structure should match exactly
        self.assertEqual(pb1.F, pb2.F)
        np.testing.assert_allclose(pb1.w, pb2.w)

    def test_random_scp_items_k_allows_empty_sets(self):
        # Total memberships = m*k; if m*k < n, empties MUST exist.
        n, m, k = 50, 10, 1  # total memberships 10 < 50 sets
        pb = random_scp_items_k(n=n, m=m, k=k, seed=0)

        sizes = _set_sizes(pb)
        self.assertTrue(np.any(sizes == 0), msg="Expected some empty sets when m*k < n")

        # Still, every item appears exactly k times
        deg = _item_degrees(pb)
        self.assertTrue(np.all(deg == k))

    def test_random_scp_items_k_balanced_no_empty_sets_when_feasible(self):
        # Here m*k >= n, so it is feasible (in principle) to avoid empty sets
        n, m, k = 30, 40, 2  # total memberships 80 >= 30 sets
        pb = random_scp_items_k_balanced(n=n, m=m, k=k, seed=0)

        sizes = _set_sizes(pb)
        self.assertTrue(np.all(sizes > 0), msg=f"Found empty sets: sizes={sizes}")

        # Item degree should remain exactly k
        deg = _item_degrees(pb)
        self.assertTrue(np.all(deg == k), msg=f"min={deg.min()}, max={deg.max()} expected={k}")

    def test_random_scp_items_k_balanced_infeasible_regime_does_not_break_degree(self):
        # Infeasible to have all sets non-empty while keeping exact item degree:
        # m*k < n means there are not enough memberships to fill all sets.
        n, m, k = 50, 10, 1  # memberships 10 < 50 sets
        pb = random_scp_items_k_balanced(n=n, m=m, k=k, seed=0)

        # We do NOT require "no empty sets" here (impossible in general)
        # but we do want the generator to remain sane and keep k-regularity on items.
        deg = _item_degrees(pb)
        self.assertTrue(np.all(deg == k), msg=f"min={deg.min()}, max={deg.max()} expected={k}")

    def test_generator_accepts_custom_weights(self):
        n, m, k = 11, 20, 2
        weights = np.linspace(1.0, 2.0, n)
        pb = random_scp_items_k(n=n, m=m, k=k, seed=0, weights=weights)

        self.assertEqual(len(pb.w), n)
        np.testing.assert_allclose(pb.w, weights)


if __name__ == "__main__":
    unittest.main()

