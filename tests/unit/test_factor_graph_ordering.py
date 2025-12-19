import unittest
import numpy as np

try:
    import scipy.sparse as sp
except Exception:
    sp = None

from bpropy.mrf.factor_graph import FactorGraph
#from bpropy.co.mis import mis_batch
from bpropy.co.mis import MIS
from bpropy.co.generators.mis import random_mis


def scopes_from_A(A) -> list[tuple[int, ...]]:
    """Return LOCAL scopes for each constraint row: tuple of column indices where A[i,j] != 0."""
    scopes: list[tuple[int, ...]] = []
    if sp is not None and sp.issparse(A):
        A = A.tocsr()
        for i in range(A.shape[0]):
            s, e = A.indptr[i], A.indptr[i + 1]
            cols = A.indices[s:e]
            scopes.append(tuple(int(c) for c in cols))
    else:
        A = np.asarray(A)
        for i in range(A.shape[0]):
            cols = np.flatnonzero(A[i] != 0)
            scopes.append(tuple(int(c) for c in cols))
    return scopes


def sorted_scopes(scopes: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Canonicalize each scope (sorted vars) and sort list by (size, lex)."""
    canon = [tuple(sorted(s)) for s in scopes]
    canon.sort(key=lambda s: (len(s), s))
    return canon


def factor_scopes_in_factor_id_order(fg: FactorGraph) -> list[tuple[int, ...]]:
    """
    Extract scopes from fg in factor-node id order.
    Assumes factor nodes are bipartite=1 and factor ids are assigned monotonically.
    """
    factors = [n for n, d in fg.nodes(data=True) if d.get("bipartite") == 1]
    factors.sort()
    out: list[tuple[int, ...]] = []
    for f in factors:
        vars_ = [v for v in fg.neighbors(f) if fg.nodes[v].get("bipartite") == 0]
        out.append(tuple(sorted(int(v) for v in vars_)))
    return out


class TestFactorGraphOrdering(unittest.TestCase):
    def setUp(self):
        # keep sizes moderate so tests stay fast but non-trivial
        self.pbs = [random_mis(60), random_mis(60), random_mis(60)]
        #self.pb_batch = mis_batch(self.pbs)
        self.pb_batch = MIS.batch(self.pbs)


    def test_single_instance_order_without_unaries(self):
        pb = self.pbs[0]
        A, lb, ub = pb.linear_constraints_matrix()
        expected = sorted_scopes(scopes_from_A(A))

        fg = FactorGraph(pb, add_unaries=False)
        got = factor_scopes_in_factor_id_order(fg)

        self.assertEqual(got, expected)

    def test_single_instance_order_with_unaries(self):
        pb = self.pbs[0]
        A, lb, ub = pb.linear_constraints_matrix()
        n = pb.nv()

        expected = sorted_scopes(scopes_from_A(A) + [(i,) for i in range(n)])

        fg = FactorGraph(pb, add_unaries=True)
        got = factor_scopes_in_factor_id_order(fg)

        self.assertEqual(got, expected)

    def test_batched_order_instance_then_size_lex(self):
        fg_batch = FactorGraph(self.pb_batch, add_unaries=True)
    
        expected_global: list[tuple[int, ...]] = []
        base = 0
        for pb in self.pbs:
            A, lb, ub = pb.linear_constraints_matrix()
            n = pb.nv()
    
            # constraint scopes (local)
            constraint_scopes = scopes_from_A(A)            # list[tuple[int,...]]
            constraint_scopes = [tuple(sorted(s)) for s in constraint_scopes]
    
            # add unaries (local)
            local = sorted_scopes(constraint_scopes + [(i,) for i in range(n)])
    
            # offset local variable ids into GLOBAL ids
            expected_global.extend(tuple(base + v for v in s) for s in local)
    
            # IMPORTANT: advance base by the total node block size of this instance,
            # because your append_instance uses base = number_of_nodes().
            n_constraints = A.shape[0]
            n_factors = n_constraints + n  # constraints + unaries
            base += (n + n_factors)
    
        got = factor_scopes_in_factor_id_order(fg_batch)
        self.assertEqual(got, expected_global)
    def test_get_subgraph_preserves_factor_order(self):
        fg_batch = FactorGraph(self.pb_batch, add_unaries=True)
        for i, pb in enumerate(self.pbs):
            fg_pb = FactorGraph(pb, add_unaries=True)
            fg_sg = fg_batch.get_subgraph(i)

            self.assertEqual(
                factor_scopes_in_factor_id_order(fg_sg),
                factor_scopes_in_factor_id_order(fg_pb),
                msg=f"Factor order mismatch in get_subgraph({i})",
            )

    def test_determinism_build_twice(self):
        fg1 = FactorGraph(self.pb_batch, add_unaries=True)
        fg2 = FactorGraph(self.pb_batch, add_unaries=True)

        self.assertEqual(
            factor_scopes_in_factor_id_order(fg1),
            factor_scopes_in_factor_id_order(fg2),
        )

    def test_constructor_vs_manual_append_instance_equivalence(self):
        """
        Ensures FactorGraph(pb) and FactorGraph(); append_instance(pb)
        produce identical factor order.
        """
        pb = self.pbs[0]

        fg_ctor = FactorGraph(pb, add_unaries=True)

        fg_manual = FactorGraph()
        fg_manual.append_instance(pb, add_unaries=True)

        self.assertEqual(
            factor_scopes_in_factor_id_order(fg_ctor),
            factor_scopes_in_factor_id_order(fg_manual),
        )


if __name__ == "__main__":
    unittest.main()

