import unittest
import copy
import torch
import numpy as np

from torch_geometric.data import HeteroData

from bpropy.co.mis import MIS, random_mis, mis_equal
from bpropy.adapters.vc_to_data import problem_to_data, data_to_lp
from bpropy.mrf.mis_factor_graph import MISFactorGraph
from bpropy.mrf.sufficient_statistics import x2ss, ss2x


FEATURE_MODES = ["EB", "CC", "CF", "EC", "CP"]


def _canonicalize_cliques(C):
    """Sort nodes inside each clique and sort the list of cliques for order-agnostic compare."""
    return sorted([tuple(sorted(c)) for c in C])


class TestMISHeteroRoundtrip(unittest.TestCase):
    def test_roundtrip_per_feature_mode(self):
        # Start from a base MIS and let feature extraction mutate a copy per mode
        base = random_mis(120)

        for mode in FEATURE_MODES:
            with self.subTest(feature_mode=mode):
                pb = copy.deepcopy(base)

                # 1) Build HeteroData from problem (feature extraction may mutate pb)
                data = problem_to_data(pb, hetero=True, feature_mode=mode)

                # Basic type/tags
                self.assertIsInstance(data, HeteroData)
                self.assertEqual(data.problem_type_name, "MIS")
                self.assertEqual(int(data.problem_type_code[0].item()), 0)
                self.assertEqual(data.feature_mode, mode)

                # 2) Reconstruct MIS from HeteroData
                pb_rt = data_to_lp(data)
                self.assertIsInstance(pb_rt, MIS)

                # 3) Structural equality: nv, number of constraints (cliques), and clique sets
                self.assertEqual(pb_rt.nv(), pb.nv())
                self.assertEqual(len(pb_rt.C), len(pb.C))
                self.assertEqual(_canonicalize_cliques(pb_rt.C),
                                 _canonicalize_cliques(pb.C))

                # 4) Factor-graph shape consistency (factors = variables + cliques for MIS)
                fg = MISFactorGraph(pb_rt)
                self.assertEqual(fg.nf(), pb_rt.nv() + len(pb_rt.C))

                # 5) Sufficient statistics round-trip on a feasible solution
                x = pb_rt.greedy()  # any feasible independent set
                self.assertTrue(pb_rt.check_solution(x))
                ss = x2ss(x.astype(int), fg)
                x_back = ss2x(ss, fg)
                self.assertTrue(np.all(x_back == x))
                self.assertTrue(pb_rt.check_solution(x_back))

                # 6) Edge & node counts in hetero graph match nv / |C|
                v2c = data["variable", "v2c", "constraint"].edge_index
                c2v = data["constraint", "c2v", "variable"].edge_index
                nv = pb_rt.nv()
                nc = len(pb_rt.C)

                self.assertEqual(int(v2c[0].max().item()) + 1 if v2c.numel() else 0, nv)
                self.assertEqual(int(v2c[1].max().item()) + 1 if v2c.numel() else 0, nc)
                self.assertEqual(v2c.size(1), c2v.size(1))  # bidirectional count

                # Optional: if node features are present, row counts should match
                if "x" in data["variable"]:
                    self.assertEqual(data["variable"]["x"].size(0), nv)
                if "x" in data["constraint"]:
                    self.assertEqual(data["constraint"]["x"].size(0), nc)

    def test_cuda_move_smoke(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        pb = random_mis(64)
        data = problem_to_data(pb, hetero=True, feature_mode="EB").to("cuda")

        # Ensure tensor fields in node & edge stores are on CUDA
        for store in [data["variable"], data["constraint"]]:
            for k, v in store.items():
                if torch.is_tensor(v):
                    self.assertTrue(v.is_cuda, f"{k} not on CUDA")

        for _, store in data.edge_items():
            for k, v in store.items():
                if torch.is_tensor(v):
                    self.assertTrue(v.is_cuda, f"edge.{k} not on CUDA")


if __name__ == "__main__":
    unittest.main()

