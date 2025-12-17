import unittest
import copy
import torch
from torch_geometric.data import HeteroData

from bpropy.co.mis import random_mis, MIS
from bpropy.adapters.vc_to_data import problem_to_data

FEATURE_MODES = ["EB", "CC", "CF", "EC", "CP"]


def _num_nodes_from_store(store, fallback_edges=None, dim=0):
    """
    Try to infer num_nodes for a store from a tensor we expect to exist
    (neighbors/factor_sizes/x). Fall back to edge_index if needed.
    """
    if hasattr(store, "neighbors") and torch.is_tensor(store.neighbors):
        return int(store.neighbors.numel())
    if hasattr(store, "factor_sizes") and torch.is_tensor(store.factor_sizes):
        return int(store.factor_sizes.numel())
    if hasattr(store, "x") and torch.is_tensor(store.x):
        return int(store.x.size(0))
    if fallback_edges is not None and torch.is_tensor(fallback_edges):
        return int(fallback_edges[dim].max().item() + 1) if fallback_edges.numel() > 0 else 0
    return None


def _check_bidir_edges(v2c, c2v, nv, nc):
    # Shapes match
    assert v2c.dim() == 2 and c2v.dim() == 2 and v2c.size(0) == 2 and c2v.size(0) == 2
    # Same number of edges
    assert v2c.size(1) == c2v.size(1)
    # Bounds
    assert v2c[0].max().item() < nv and v2c[1].max().item() < nc
    assert c2v[0].max().item() < nc and c2v[1].max().item() < nv
    # Optional: exact bidirectionality after sorting columns
    # Build sets of pairs for a robust equality check
    v2c_set = set(map(tuple, v2c.t().tolist()))
    c2v_set = set(map(lambda rc: (rc[1], rc[0]), c2v.t().tolist()))
    assert v2c_set == c2v_set


class TestMISHeteroData(unittest.TestCase):
    def test_hetero_data_creation_across_feature_modes(self):
        # Base problem (will be copied and mutated per mode)
        pb0 = random_mis(120)  # medium size so all modes have something to do
        for mode in FEATURE_MODES:
            with self.subTest(feature_mode=mode):
                pb = copy.deepcopy(pb0)  # feature extraction may mutate problem
                data = problem_to_data(pb, hetero=True, feature_mode=mode)

                # Type and tags
                self.assertIsInstance(data, HeteroData)
                self.assertTrue(hasattr(data, "problem_type_name"))
                self.assertTrue(hasattr(data, "problem_type_code"))
                self.assertTrue(hasattr(data, "feature_mode"))
                self.assertEqual(data.problem_type_name, "MIS")
                self.assertEqual(int(data.problem_type_code[0].item()), 0)  # 0 = MIS (as agreed)
                self.assertEqual(data.feature_mode, mode)

                # Node/edge stores present
                self.assertIn(("variable", "v2c", "constraint"), data.edge_types)
                self.assertIn(("constraint", "c2v", "variable"), data.edge_types)
                self.assertIn("variable", data.node_types)
                self.assertIn("constraint", data.node_types)

                v2c = data["variable", "v2c", "constraint"].edge_index
                c2v = data["constraint", "c2v", "variable"].edge_index

                # Infer counts from stores (preferred) or edge_index (fallback)
                nv = _num_nodes_from_store(data["variable"], fallback_edges=c2v, dim=1)
                nc = _num_nodes_from_store(data["constraint"], fallback_edges=v2c, dim=1)
                self.assertIsNotNone(nv)
                self.assertIsNotNone(nc)

                # The problem has been mutated by features; pb.nv() must match
                self.assertEqual(nv, pb.nv())
                # Constraints count equals mutated problem's constraints
                # MIS exposes cliques as constraints via pb.C
                self.assertEqual(nc, len(pb.C))

                # Edges are bidirectional & consistent
                _check_bidir_edges(v2c, c2v, nv, nc)

                # Optional: feature presence/shape checks when provided
                if hasattr(data["variable"], "x") and torch.is_tensor(data["variable"].x):
                    self.assertEqual(data["variable"].x.size(0), nv)
                if hasattr(data["constraint"], "x") and torch.is_tensor(data["constraint"].x):
                    self.assertEqual(data["constraint"].x.size(0), nc)
                if hasattr(data["variable", "v2c", "constraint"], "edge_attr") and \
                        torch.is_tensor(data["variable", "v2c", "constraint"].edge_attr):
                    self.assertEqual(
                        data["variable", "v2c", "constraint"].edge_attr.size(0),
                        v2c.size(1)
                    )

                # Sanity: factor-graph count must match constraints + variables
                # (for MIS, factors = variables + cliques)
                # Delay import to avoid heavy deps if not installed in minimal env
                from bpropy.mrf.mis_factor_graph import MISFactorGraph
                fg = MISFactorGraph(pb)
                self.assertEqual(fg.nf(), nv + nc)

                # Bounds and coherence
                self.assertLess(v2c.numel(), 10_000_000)  # safeguard vs explosion
                self.assertGreaterEqual(nv, 1)
                self.assertGreaterEqual(nc, 0)

    def test_cuda_move_keeps_tensors_on_device(self):
        # Smoke test: move hetero graph to CUDA and verify tensor attributes
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        pb = random_mis(64)
        data = problem_to_data(pb, hetero=True, feature_mode="EB")
        data = data.to("cuda")
        # Walk node/edge stores and assert any tensor fields are on CUDA
        for store in [data["variable"], data["constraint"]]:
            for key, val in store.items():
                if torch.is_tensor(val):
                    self.assertTrue(val.is_cuda, f"{key} not on CUDA")
        for key, store in data.edge_items():
            for attr, val in store.items():
                if torch.is_tensor(val):
                    self.assertTrue(val.is_cuda, f"{key}.{attr} not on CUDA")


if __name__ == "__main__":
    unittest.main()

