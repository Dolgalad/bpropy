# src/bpropy/mrf/factor_graph_approx.py
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Set, Tuple

import numpy as np
import networkx as nx
import torch

from bpropy.mrf.factor_graph import FactorGraph

Pair = Tuple[int, int]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _sorted_pair(i: int, j: int) -> Pair:
    return (i, j) if i <= j else (j, i)


def _all_pairs(scope: Sequence[int]) -> List[Pair]:
    scope = list(map(int, scope))
    out: List[Pair] = []
    for a in range(len(scope)):
        for b in range(a + 1, len(scope)):
            out.append(_sorted_pair(scope[a], scope[b]))
    return out


def _constraint_nodes_in_canonical_order(vcg) -> List[int]:
    # Constraints are bipartite==1. Sorted node id gives stable/canonical order.
    return sorted(int(n) for n, d in vcg.nodes(data=True) if int(d.get("bipartite", 0)) == 1)


def _n_vars_from_vcg(vcg) -> int:
    # Preferred: VCGraph.nv()
    if hasattr(vcg, "nv") and callable(getattr(vcg, "nv")):
        return int(vcg.nv())

    # Fallback: infer from bipartite attribute assuming variable ids are 0..nv-1
    var_nodes = [int(n) for n, d in vcg.nodes(data=True) if int(d.get("bipartite", 0)) == 0]
    if not var_nodes:
        raise ValueError("Could not infer number of variables from vcg (no bipartite==0 nodes).")
    return max(var_nodes) + 1


def _build_fg_from_pair_scopes(n_vars: int, pair_scopes: Sequence[Pair]) -> FactorGraph:
    """
    Build a single-instance FactorGraph with:
      - unary factors for each variable
      - pairwise factors for each (i,j) in pair_scopes

    Canonical order is delegated to FactorGraph.append_instance_from_scopes,
    which sorts by (size, lex) within the instance.
    """
    fg = FactorGraph()
    fg.append_instance_from_scopes(
        n_vars,
        scopes=[list(p) for p in pair_scopes],
        add_unaries=True,
    )
    return fg


#def _unary_factor_node_by_var(fg: FactorGraph) -> Dict[int, int]:
#    out: Dict[int, int] = {}
#    for F in fg.iter_factors():
#        scope = tuple(map(int, fg.factor_variables(F)))
#        if len(scope) == 1:
#            out[int(scope[0])] = int(F)
#    return out
#
#
#def _pair_factor_node_by_scope(fg: FactorGraph) -> Dict[Pair, int]:
#    out: Dict[Pair, int] = {}
#    for F in fg.iter_factors():
#        scope = tuple(map(int, fg.factor_variables(F)))
#        if len(scope) == 2:
#            out[_sorted_pair(scope[0], scope[1])] = int(F)
#    return out
def _unary_factor_node_by_var(fg: FactorGraph) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for iF, F in enumerate(fg.iter_factors()):
        scope = tuple(map(int, fg.factor_variables(F)))
        if len(scope) == 1:
            out[int(scope[0])] = int(iF)   # <-- iF, not F
    return out


def _pair_factor_node_by_scope(fg: FactorGraph) -> Dict[Pair, int]:
    out: Dict[Pair, int] = {}
    for iF, F in enumerate(fg.iter_factors()):
        scope = tuple(map(int, fg.factor_variables(F)))
        if len(scope) == 2:
            out[_sorted_pair(scope[0], scope[1])] = int(iF)  # <-- iF, not F
    return out


def _compute_fg_factor_sizes(fg: FactorGraph) -> torch.Tensor:
    """
    Returns a vector sized [fg.number_of_nodes()] aligned with FG node ids.
    factor_sizes[n] = factor arity if node is a factor, else 0 for variable nodes.
    """
    out = torch.ones(fg.nf(), dtype=torch.long)
    for iF,F in enumerate(fg.iter_factors()):
        out[iF] = int(fg.factor_size(F))
    return out

# ---------------------------------------------------------------------
# Mapping: VCGraph nodes -> approx FactorGraph *factor nodes*
# ---------------------------------------------------------------------

def _build_vc_to_apx_map(
    *,
    vcg,
    fg_apx: FactorGraph,
    keep_pairs: Set[Pair],
) -> torch.Tensor:
    """
    fg_map shape [n_map, 2]:
      - src: VCGraph node id (indexes data.x rows)
      - dst: approx FactorGraph node id of the TARGET FACTOR

    Convention:
      - VC variable node i  -> unary factor [i] in fg_apx
      - VC constraint node c -> for every kept pair (i,j) within its neighborhood scope,
                                map c -> pairwise factor (i,j) in fg_apx
    """
    n_vars = _n_vars_from_vcg(vcg)
    con_nodes = _constraint_nodes_in_canonical_order(vcg)

    apx_unary = _unary_factor_node_by_var(fg_apx)
    apx_pair = _pair_factor_node_by_scope(fg_apx)

    rows: List[Tuple[int, int]] = []

    # Variables -> unary factors (1-to-1)
    for i in range(n_vars):
        Fu = apx_unary.get(i, None)
        if Fu is None:
            raise ValueError(f"Approx FG has no unary factor for variable {i}.")
        rows.append((i, int(Fu)))

    # Constraints -> included pairwise factors (many-to-one)
    for c in con_nodes:
        scope = sorted(int(v) for v in vcg.neighbors(c))
        for p in _all_pairs(scope):
            if p not in keep_pairs:
                continue
            Fp = apx_pair.get(p, None)
            if Fp is None:
                # Shouldn't happen if fg_apx was built from keep_pairs, but be safe.
                continue
            rows.append((int(c), int(Fp)))
    for i,p in enumerate(keep_pairs):
        Fp = apx_pair.get(p, None)
        #print("\t", Fp, p)
        rows.append((int(p[0]), int(Fp)))
        rows.append((int(p[1]), int(Fp)))
    r = torch.as_tensor(rows, dtype=torch.long)
    return r


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------

# New return type: (fg_apx, fg_map, fg_factor_sizes)
BuilderOut = Tuple[FactorGraph, torch.Tensor, torch.Tensor]


def build_pairwise_factors(*, pb, vcg, fg_true: FactorGraph, approx: dict) -> BuilderOut:
    """
    Keep all unique pairwise edges (i,j) that appear in ANY constraint neighborhood scope.
    Always includes unary factors (needed for NMF).
    """
    n_vars = _n_vars_from_vcg(vcg)

    con_nodes = _constraint_nodes_in_canonical_order(vcg)
    pair_set: Set[Pair] = set()
    for c in con_nodes:
        scope = sorted(int(v) for v in vcg.neighbors(c))
        pair_set.update(_all_pairs(scope))

    pair_scopes = sorted(pair_set)  # canonical order
    fg_apx = _build_fg_from_pair_scopes(n_vars, pair_scopes)

    fg_map = _build_vc_to_apx_map(vcg=vcg, fg_apx=fg_apx, keep_pairs=set(pair_scopes))
    fg_factor_sizes = _compute_fg_factor_sizes(fg_apx)
    return fg_apx, fg_map, fg_factor_sizes


def build_tree_cover(*, pb, vcg, fg_true: FactorGraph, approx: dict) -> BuilderOut:
    """
    Keep a spanning tree over the induced variable graph:
      add edge (i,j) if i and j co-occur in at least one constraint scope,
      then extract a (seeded) minimum spanning tree (random weights).
    If disconnected, this yields a spanning forest.
    Always includes unary factors (needed for NMF).
    """
    n_vars = _n_vars_from_vcg(vcg)
    seed = int(approx.get("seed", 0))
    rng = np.random.RandomState(seed)

    # Induced variable graph from constraint scopes
    G = nx.Graph()
    G.add_nodes_from(range(n_vars))

    con_nodes = _constraint_nodes_in_canonical_order(vcg)
    all_pairs: Set[Pair] = set()
    for c in con_nodes:
        scope = sorted(int(v) for v in vcg.neighbors(c))
        all_pairs.update(_all_pairs(scope))

    for (i, j) in all_pairs:
        G.add_edge(i, j, weight=float(rng.rand()))

    tree = nx.minimum_spanning_tree(G, weight="weight")
    keep_pairs = {_sorted_pair(int(u), int(v)) for (u, v) in tree.edges()}

    pair_scopes = sorted(keep_pairs)  # canonical

    fg_apx = _build_fg_from_pair_scopes(n_vars, pair_scopes)

    fg_map = _build_vc_to_apx_map(vcg=vcg, fg_apx=fg_apx, keep_pairs=set(pair_scopes))
    fg_factor_sizes = _compute_fg_factor_sizes(fg_apx)
    return fg_apx, fg_map, fg_factor_sizes


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

APPROX_BUILDERS: Dict[str, Callable[..., BuilderOut]] = {
    "tree": build_tree_cover,
    "tree_cover": build_tree_cover,
    "pairwise": build_pairwise_factors,
    "pairwise_factors": build_pairwise_factors,
}

