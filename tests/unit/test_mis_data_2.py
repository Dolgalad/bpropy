import unittest
import torch
import numpy as np
from torch_geometric.data import Batch

from bpropy.co.mis import MIS, mis_equal, random_mis         # MIS API and equality helper
from bpropy.adapters.vc_to_data import problem_to_data, data_to_lp
from bpropy.mrf.mis_factor_graph import MISFactorGraph
from bpropy.mrf.sufficient_statistics import x2ss, ss2x

from tests.helpers.mis import simple_mis  # tiny fixture if you use it elsewhere


def _tensor_items(d):
    for k, v in d.items():
        if torch.is_tensor(v):
            yield k, v


class TestMISData(unittest.TestCase):
    def test_problem_to_data_and_back(self):
        pb = random_mis(100)
        data = problem_to_data(pb)  # LPProblem -> Data

        # Structural checks (same semantics as before)
        self.assertEqual(data.edge_index.max().item() + 1, pb.nv() + pb.ne())
        self.assertEqual(data.edge_index.shape[1], 4 * pb.ne())
        nV = int((data.bipartite == 0).sum().item())
        nF = int((data.bipartite == 1).sum().item())
        self.assertEqual(nV, pb.nv())
        self.assertEqual(nF, pb.ne())
        # edge features aligned (if you attach them in adapter)
        if hasattr(data, "e"):
            self.assertEqual(data.edge_index.shape[1], data.e.shape[0])

        # Round-trip via generic dispatcher
        pb1 = data_to_lp(data)
        self.assertIsInstance(pb1, MIS)
        self.assertTrue(mis_equal(pb, pb1))

    def test_batch_roundtrip(self):
        pb1, pb2 = random_mis(100), random_mis(150)
        d1 = problem_to_data(pb1)
        d2 = problem_to_data(pb2)

        # Make node feature widths consistent before batching (matches your older test)
        d1.x = torch.zeros(pb1.nv() + pb1.ne(), 3)
        d2.x = torch.zeros(pb2.nv() + pb2.ne(), 3)

        batch = Batch.from_data_list([d1, d2])
        self.assertEqual(batch.num_graphs, 2)
        self.assertEqual(int(batch.bipartite.sum().item()), pb1.ne() + pb2.ne())

        # Slice back per-graph and round-trip each
        back = [data_to_lp(batch.get_example(i)) for i in range(batch.num_graphs)]
        self.assertTrue(mis_equal(pb1, back[0]))
        self.assertTrue(mis_equal(pb2, back[1]))

    def test_solution_and_factor_graph_roundtrip(self):
        pb = random_mis(50)
        x = pb.greedy()  # any feasible solution
        data = problem_to_data(pb)
        pb1 = data_to_lp(data)
        self.assertTrue(pb1.check_solution(x))

        # Exercise multiple feature categories if your adapter supports a param
        for cat in ["EB", "CC", "CF", "EC", "CP"]:
            data_cat = problem_to_data(pb, feature_mode=cat)  # or problem_to_data(pb, feature_category=cat) if exposed
            pb_cat = data_to_lp(data_cat)
            self.assertTrue(pb_cat.check_solution(x))

            # Factor-graph & SS round-trip invariants
            fg = MISFactorGraph(pb_cat)
            x_ind = x2ss(x.astype(int), fg)
            y = ss2x(x_ind, fg)

            self.assertTrue(np.all(y == x))
            self.assertTrue(pb.check_solution(y) and pb_cat.check_solution(y))

            # clique ↔ factor alignment (MIS factors after variables correspond to cliques)
            self.assertEqual(len(pb.C), fg.nf() - pb.nv())
            fg_factors = [fg.factor_variables(F) for F in fg.iter_factors()]
            self.assertTrue(
                np.all([np.all(c1 == c2) for c1, c2 in zip(pb.C, fg_factors[pb.nv():])])
            )

            # each factor contributes exactly one active SS in x_ind; y respects clique ≤ 1
            off = 0
            for F in fg.iter_factors():
                nF = fg.factor_size(F)
                self.assertEqual(int(x_ind[off: off + 2 ** nF].sum()), 1)
                self.assertLessEqual(int(y[fg.factor_variables(F)].sum()), 1)
                off += 2 ** nF

    def test_device_move(self):
        if torch.cuda.is_available():
            pb = random_mis(50)
            data = problem_to_data(pb, with_lbp=True)  # if your adapter exposes these flags
            data = data.to("cuda")

            # Only assert tensor attributes moved (skip objects like lbp_edges/nmf_edges)
            for k, v in _tensor_items(data.to_dict()):
                self.assertTrue(v.is_cuda, f"{k} not on CUDA")

    def test_ordering_stable_through_adapters(self):
        from bpropy.adapters.vc_to_data import problem_to_data, data_to_lp
        pb = random_mis(50)
        data = problem_to_data(pb, hetero=True, feature_mode="EB")  # or MIS as needed
        pb_rt = data_to_lp(data)
        # Variable index identity
        self.assertEqual(pb_rt.nv(), pb.nv())
        # Constraint list outer order preserved (inner canonical already)
        # or for MIS
        self.assertTrue(all([(a==b).all() for a,b in zip(pb_rt.C, pb.C)]))



if __name__ == "__main__":
    unittest.main()

