import os
import json
import tempfile
import unittest
import numpy as np

from bpropy.co.scp import SCP
from bpropy.co.generators.scp import random_scp_items_k, random_scp_items_k_balanced

def _F_as_sets(F):
    return [set(map(int, f)) for f in F]

class TestSCPSerde(unittest.TestCase):
    def test_small_roundtrip_inline_json(self):
        # Simple 3×3 SCP
        A = [0,1,2]
        F = [[0,1], [1,2], [0,2]]
        w = np.array([1.0, 2.0, 1.5], dtype=float)
        pb = SCP(A, F, w)

        with tempfile.TemporaryDirectory() as td:
            stem = os.path.join(td, "scp_small")
            pb.save(stem)
            self.assertTrue(os.path.exists(stem + ".json"))

            # manifest sanity
            with open(stem + ".json", "r") as f:
                man = json.load(f)
            self.assertEqual(man["problem_type"], "SCP")

            pb2 = SCP.load(stem)
            self.assertEqual(pb2.nv(), pb.nv())
            self.assertEqual(pb2.ne(), pb.ne())
            self.assertTrue(np.allclose(pb2.w, pb.w))
            # F round-trip (order-agnostic within each set, and across sets)
            self.assertEqual(sorted(map(tuple, _F_as_sets(pb2.F))),
                             sorted(map(tuple, _F_as_sets(pb.F))))

    def test_large_roundtrip_npz_incidence(self):
        # K-regular on items; big enough to push NPZ CSR path
        pb = random_scp_items_k(n=10_000, m=2_000, k=3, seed=1)

        with tempfile.TemporaryDirectory() as td:
            stem = os.path.join(td, "scp_large")
            pb.save(stem, split_large=True, threshold_bytes=8_000)  # force arrays.npz
            self.assertTrue(os.path.exists(stem + ".json"))
            self.assertTrue(os.path.exists(stem + ".arrays.npz"))

            pb2 = SCP.load(stem)
            self.assertEqual(pb2.nv(), pb.nv())
            self.assertEqual(pb2.ne(), pb.ne())
            self.assertTrue(np.allclose(pb2.w, pb.w))
            # Degree on items stays exactly k
            deg = np.zeros(pb2.ne(), dtype=int)
            for j, f in enumerate(pb2.F):
                deg[np.array(f, dtype=int)] += 1
            self.assertTrue(np.all(deg == 3))
    def test_scp_save_load_ordering_json_and_csr(self):
        A = [0,1,2,3,4]
        F = [[2,2,1,0], [4,3,3], [1], [4,0,0], []]  # duplicates + unsorted, empty allowed
        w = np.ones(len(F))
        pb = SCP(A, F, w)
    
        with tempfile.TemporaryDirectory() as td:
            # JSON path (inline)
            stem_json = os.path.join(td, "scp_json")
            pb.save(stem_json, split_large=False)
            pb_json = SCP.load(stem_json)
    
            # CSR path (force npz)
            stem_csr = os.path.join(td, "scp_csr")
            pb.save(stem_csr, split_large=True, threshold_bytes=8_000)
            pb_csr = SCP.load(stem_csr)
    
        # Outer order preserved, inner canonical, across both paths
        F_can = [sorted(dict.fromkeys(f)) for f in F]
        self.assertEqual(pb_json.F, F_can)
        self.assertEqual(pb_csr.F, F_can)



if __name__ == "__main__":
    unittest.main()

