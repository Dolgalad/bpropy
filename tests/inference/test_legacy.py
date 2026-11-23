"""Legacy binary bit order must be preserved when importing thesis parameters."""
import unittest

import torch

from bpropy.inference import FactorGraph, lbp
from bpropy.inference.legacy import from_legacy_factor_graph, legacy_to_row_major, row_major_to_legacy


class _MockLegacy:
    num_graphs = 1

    def iter_variables(self):
        return [4, 8, 11]

    def iter_factors(self):
        return [12, 13]

    def factor_variables(self, f):
        return {12: [4, 8], 13: [4, 8, 11]}[f]


class LegacyTests(unittest.TestCase):
    def test_legacy_import_remaps_ids_and_preserves_scope(self):
        graph = from_legacy_factor_graph(_MockLegacy())
        self.assertEqual(graph.cardinalities.tolist(), [2, 2, 2])
        self.assertEqual(graph.scope(0), (0, 1))
        self.assertEqual(graph.scope(1), (0, 1, 2))

    def test_legacy_bit_order_and_gradient(self):
        graph = FactorGraph.from_scopes([2, 2, 2], [(0, 1), (0, 1, 2)])
        original = torch.arange(12, dtype=torch.float64, requires_grad=True)
        converted = legacy_to_row_major(graph, original)
        torch.testing.assert_close(converted, original[[0, 2, 1, 3, 4, 8, 6, 10, 5, 9, 7, 11]])
        torch.testing.assert_close(row_major_to_legacy(graph, converted), original)
        loss = lbp(graph.compile(), converted, iterations=7).marginal(0)[1]
        loss.backward()
        self.assertTrue(torch.isfinite(original.grad).all())

    def test_nonbinary_legacy_table_raises(self):
        graph = FactorGraph.from_scopes([3], [(0,)])
        with self.assertRaisesRegex(ValueError, "binary"):
            legacy_to_row_major(graph, torch.ones(3))


if __name__ == "__main__":
    unittest.main()
