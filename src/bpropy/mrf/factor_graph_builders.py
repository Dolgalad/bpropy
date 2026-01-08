import numpy as np
import scipy.sparse as sp
from bpropy.mrf.factor_graph import FactorGraph
from bpropy.utils.set_sort import lenlex_sort

def scopes_from_matrix(A, *, drop_zeros=True):
    """Return factor scopes (list of variable indices) from rows of A."""
    if sp.issparse(A):
        A = A.tocsr()
        scopes = [A.indices[A.indptr[i]:A.indptr[i+1]].tolist() for i in range(A.shape[0])]
    else:
        A = np.asarray(A)
        scopes = [np.flatnonzero(A[i]).tolist() for i in range(A.shape[0])]
    if drop_zeros:
        scopes = [s for s in scopes if len(s) > 0]
    return scopes

def build_factor_graph(
    nv: int,
    scopes,
    *,
    add_unary: bool = True,
    sort_scopes: bool = True,
):
    """
    Build a bipartite FactorGraph with:
      - variable nodes: 0..nv-1
      - factor nodes: nv..nv+nf-1
    """
    fg = FactorGraph()
    fg.add_nodes_from(range(nv), bipartite=0)

    if add_unary:
        # unary scopes: [ [0], [1], ... ]
        unary = [[i] for i in range(nv)]
    else:
        unary = []

    scopes = list(scopes)
    all_scopes = unary + scopes
    if sort_scopes:
        all_scopes = lenlex_sort(all_scopes)

    # Create factor nodes in one contiguous block (fast and predictable)
    start_f = nv
    fg.add_nodes_from(range(start_f, start_f + len(all_scopes)), bipartite=1)

    # Add edges
    for iF, scope in enumerate(all_scopes):
        F = start_f + iF
        for v in scope:
            fg.add_edge(int(v), F)

    return fg

