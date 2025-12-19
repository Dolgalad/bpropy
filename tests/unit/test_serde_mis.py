import os
import json
import shutil
import tempfile
import unittest
import numpy as np
from typing import Dict

from bpropy.co.mis import MIS

def _edges_set(pb: MIS):
    # You can replace by pb.edges() if you expose it
    # This helper expects pb has an adjacency list or a way to iterate edges
    # Here we assume pb has an attribute/list `E` of undirected edges.
    r = set(tuple(e) for e in pb.edges())
    return r
    E = getattr(pb, "E", None)
    if E is None:
        raise AttributeError("Provide MIS.edges() or MIS.E (list of (u,v)) for test_serde_mis")
    return set(tuple(sorted(e)) for e in E)

class TestMISSerde(unittest.TestCase):
    def test_small_roundtrip_inline_json(self):
        # Small MIS: inline everything in JSON
        nv = 8
        edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(6,7)]
        w = np.linspace(1.0, 2.5, nv, dtype=float)
        pb = MIS.from_edges(nv=nv, edges=edges, weights=w)
        # Add a couple of cliques to serialize (optional)
        pb.C = [[0,1,2], [4,5]]

        with tempfile.TemporaryDirectory() as td:
            stem = os.path.join(td, "mis_small")
            pb.save(stem)                 # writes mis_small.json
            self.assertTrue(os.path.exists(stem + ".json"))

            # Inspect manifest quickly
            with open(stem+".json", "r") as f:
                man = json.load(f)
            self.assertEqual(man["problem_type"], "MIS")

            # Load and compare
            pb2 = MIS.load(stem)
            self.assertEqual(pb2.nv(), pb.nv())
            self.assertTrue(np.allclose(pb2.w, pb.w))
            # Edge set equality (order-agnostic)
            self.assertEqual(_edges_set(pb2), _edges_set(pb))
            # Clique equality (if present)
            self.assertEqual(sorted(map(tuple, pb2.C)), sorted(map(tuple, pb.C)))

    def test_large_roundtrip_with_npz(self):
        # Big enough to trigger NPZ split for weights
        nv = 50_000
        # A sparse path graph for determinism
        edges = [(i, i+1) for i in range(nv-1)]
        w = np.ones(nv, dtype=float)
        pb = MIS.from_edges(nv=nv, edges=edges, weights=w)

        with tempfile.TemporaryDirectory() as td:
            stem = os.path.join(td, "mis_large")
            pb.save(stem, split_large=True, threshold_bytes=8_000)  # force NPZ for large arrays
            self.assertTrue(os.path.exists(stem + ".json"))
            self.assertTrue(os.path.exists(stem + ".arrays.npz"))

            pb2 = MIS.load(stem)
            self.assertEqual(pb2.nv(), pb.nv())
            self.assertTrue(np.allclose(pb2.w, pb.w))
            # spot-check a few edges
            for i in [0, 100, nv-2]:
                self.assertIn(tuple(sorted((i, i+1))), _edges_set(pb2))
    def test_mis_save_load_ordering(self):
        nv = 8
        edges = [(1,0),(2,1),(4,3)]  # scrambled input
        C = [[3,2,2,1], [7,6], [5,5,4,4,3]]  # duplicates + unsorted
        w = np.arange(nv) + 1.0
        pb = MIS.from_edges(nv, edges, weights=w)
        pb.C = C  # inject cliques with messy order
    
        # normalize happens in __init__ or just before save
        with tempfile.TemporaryDirectory() as td:
            stem = os.path.join(td, "mis_order")
            pb.save(stem)              # writes inner lists sorted & deduped
            pb2 = MIS.load(stem)
    
        # Outer list order preserved; inner lists canonical
        self.assertEqual(len(pb2.C), len(C))
        for i,(c_ref,c_rt) in enumerate(zip(
            [sorted(dict.fromkeys(map(int, c))) for c in C], pb2.C)):
            self.assertEqual(c_rt, c_ref)



if __name__ == "__main__":
    unittest.main()

