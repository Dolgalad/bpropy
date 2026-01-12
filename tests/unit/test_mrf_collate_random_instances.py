import unittest
from typing import List, Tuple, Dict, Any

import torch
from torch_geometric.data import Data

# --- Use your real collator ---
from bpropy.datasets.collator import mrf_collate

# --- Use your real generators (as requested) ---
# Prefer importing from your repo package paths if they exist, else fallback to local modules.
from bpropy.co.generators.mis import random_mis, random_mis_bounded_degree  # type: ignore
from bpropy.co.generators.scp import random_scp_items_k, random_scp_items_k_balanced  # type: ignore
from bpropy.mrf.factor_graph import FactorGraph  # type: ignore
from bpropy.mrf.factor_graph_approx import build_tree_cover, build_pairwise_factors  # type: ignore


def _vcg_to_pyg_data(vcg: FactorGraph) -> Data:
    """
    Convert a networkx-like FactorGraph into a minimal PyG Data for batching tests.
    We only need num_nodes and edge_index to be consistent; features are not needed.
    """
    n = vcg.number_of_nodes()
    edges = list(vcg.edges())
    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # minimal x so Batch.from_data_list is happy everywhere
    x = torch.zeros((n, 1), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, num_nodes=n)


def _make_instance_data_with_approx(
    pb,
    *,
    approx_kind: str = "tree",   # "tree" or "pairwise"
    seed: int = 0,
) -> Tuple[Any, Data, torch.Tensor, Dict[str, Any]]:
    """
    Build (pb, data, y, meta) where:
      - pb comes from your generators
      - vcg is FactorGraph(pb, add_unaries=False)
      - fg_true is FactorGraph(pb, add_unaries=True)
      - approx is built by your builder (tree cover or pairwise factors)
      - data has .factor_sizes and .fg_map as expected by mrf_collate
      - y is a random binary assignment of length n_vars (for batching test)
    """
    # "vcg" is the original variable-constraint bipartite graph (no unaries)
    vcg = FactorGraph(pb, add_unaries=False)

    # "fg_true" is the full factor graph (typically includes unaries + constraints)
    fg_true = FactorGraph(pb, add_unaries=True)

    if approx_kind == "tree":
        fg_apx, fg_map, fg_factor_sizes = build_tree_cover(
            pb=pb, vcg=vcg, fg_true=fg_true, approx={"seed": seed}
        )
    elif approx_kind == "pairwise":
        fg_apx, fg_map, fg_factor_sizes = build_pairwise_factors(
            pb=pb, vcg=vcg, fg_true=fg_true, approx={}
        )
    else:
        raise ValueError(f"Unknown approx_kind: {approx_kind}")

    # Build PyG Data corresponding to vcg (node indexing must match fg_map col0)
    data = _vcg_to_pyg_data(vcg)
    data.factor_sizes = torch.as_tensor(fg_factor_sizes, dtype=torch.long).clone()
    data.fg_map = torch.as_tensor(fg_map, dtype=torch.long).clone()

    # assignment length must match number of variables in the vcg
    n_vars = vcg.number_of_variables()
    y = torch.randint(0, 2, (n_vars,), dtype=torch.long)

    meta = {
        "num_nodes": int(data.num_nodes),
        "n_factors": int(data.factor_sizes.numel()),
        "fg_map": data.fg_map.clone(),
        "factor_sizes": data.factor_sizes.clone(),
        "y": y.clone(),
    }
    return pb, data, y, meta


def _expected_batched_fg_map(metas: List[Dict[str, Any]]) -> torch.Tensor:
    maps = []
    node_off = 0
    fac_off = 0
    for m in metas:
        fm = m["fg_map"].clone()
        fm[:, 0] += node_off
        fm[:, 1] += fac_off
        maps.append(fm)
        node_off += int(m["num_nodes"])
        fac_off += int(m["n_factors"])
    return torch.cat(maps, dim=0) if maps else torch.empty((0, 2), dtype=torch.long)


class TestMRFCollateWithRealGenerators(unittest.TestCase):
    def _run_batch_check(self, triples: List[Tuple[Any, Data, torch.Tensor]], metas: List[Dict[str, Any]]):
        pbs, data_batch, y_batch = mrf_collate(triples)

        # pbs should preserve order
        self.assertEqual(len(pbs), len(triples))

        # y concatenation
        expected_y = torch.cat([m["y"] for m in metas], dim=0)
        self.assertTrue(torch.equal(y_batch, expected_y))

        # factor_sizes concatenation
        expected_fs = torch.cat([m["factor_sizes"] for m in metas], dim=0)
        self.assertTrue(torch.equal(data_batch.factor_sizes, expected_fs))

        # fg_map offset correctness
        expected_map = _expected_batched_fg_map(metas)
        self.assertTrue(torch.equal(data_batch.fg_map, expected_map))

        # safety bounds (same spirit as asserts in collator)
        self.assertLess(int(data_batch.fg_map[:, 0].max()), int(data_batch.num_nodes))
        self.assertLess(int(data_batch.fg_map[:, 1].max()), int(data_batch.factor_sizes.numel()))

    def test_mis_tree_cover_batching(self):
        metas = []
        batch = []

        # Use your MIS generator
        for i in range(3):
            pb = random_mis(n=40 + 5 * i, p=0.12, cliques=False, seed=100 + i)
            pb, data, y, meta = _make_instance_data_with_approx(pb, approx_kind="tree", seed=0)
            metas.append(meta)
            batch.append((pb, data, y))

        self._run_batch_check(batch, metas)

    def test_scp_tree_cover_batching(self):
        metas = []
        batch = []

        # Use your SCP generator
        for i in range(3):
            # n sets, m items, each item appears in k sets
            pb = random_scp_items_k_balanced(n=40 + 5 * i, m=30 + 5 * i, k=6, seed=200 + i)
            pb, data, y, meta = _make_instance_data_with_approx(pb, approx_kind="tree", seed=0)
            metas.append(meta)
            batch.append((pb, data, y))

        self._run_batch_check(batch, metas)

    def test_mixed_mis_scp_tree_cover_batching(self):
        metas = []
        batch = []

        pb1 = random_mis_bounded_degree(nv=50, max_deg=5, target_avg_deg=3.0, seed=301)
        pb1, data1, y1, meta1 = _make_instance_data_with_approx(pb1, approx_kind="tree", seed=0)
        metas.append(meta1)
        batch.append((pb1, data1, y1))

        pb2 = random_scp_items_k(n=45, m=35, k=5, seed=302)
        pb2, data2, y2, meta2 = _make_instance_data_with_approx(pb2, approx_kind="tree", seed=0)
        metas.append(meta2)
        batch.append((pb2, data2, y2))

        pb3 = random_mis(n=55, p=0.10, cliques=False, seed=303)
        pb3, data3, y3, meta3 = _make_instance_data_with_approx(pb3, approx_kind="tree", seed=0)
        metas.append(meta3)
        batch.append((pb3, data3, y3))

        self._run_batch_check(batch, metas)

    def test_pairwise_builder_batching(self):
        metas = []
        batch = []

        for i in range(2):
            pb = random_mis(n=35 + 5 * i, p=0.18, cliques=False, seed=400 + i)
            pb, data, y, meta = _make_instance_data_with_approx(pb, approx_kind="pairwise", seed=0)
            metas.append(meta)
            batch.append((pb, data, y))

        self._run_batch_check(batch, metas)


if __name__ == "__main__":
    unittest.main()

