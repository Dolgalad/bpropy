"""Opt-in adapters for experiments written against the old ``bpropy.mrf`` API.

The old implementation numbers states of a binary factor with the *first*
variable's bit as the least significant one. The new PyTorch-native API uses
row-major order, with the last variable varying fastest. Explicitly convert
all factor tables when migrating. These adapters do not import legacy bpropy.
"""
from __future__ import annotations

import torch

from .graph import FactorGraph


def from_legacy_factor_graph(legacy_graph) -> FactorGraph:
    """Convert a legacy ``bpropy.mrf.FactorGraph`` to the new topology.

    Accepts any duck-typed legacy graph exposing ``iter_variables``,
    ``iter_factors`` and ``factor_variables``. Supports disjoint legacy batches
    via ``get_subgraph(i)``; original numeric IDs are remapped to zero-based
    indices. All legacy variables are binary.
    """
    num_graphs = int(getattr(legacy_graph, "num_graphs", 1))
    if num_graphs > 1:
        return FactorGraph.batch([from_legacy_factor_graph(legacy_graph.get_subgraph(i))
                                  for i in range(num_graphs)])
    variables = list(legacy_graph.iter_variables())
    index = {v: i for i, v in enumerate(variables)}
    scopes = [[index[v] for v in legacy_graph.factor_variables(f)]
              for f in legacy_graph.iter_factors()]
    return FactorGraph.from_scopes([2] * len(variables), scopes)


def _reversed_binary_indices(k: int, device: torch.device) -> torch.Tensor:
    """Original little-endian table index for each row-major table entry."""
    rows = torch.arange(1 << k, dtype=torch.long, device=device)
    old = torch.zeros_like(rows)
    for i in range(k):
        old = old | (((rows >> (k - 1 - i)) & 1) << i)
    return old


def legacy_to_row_major(graph: FactorGraph, values: torch.Tensor) -> torch.Tensor:
    """Reorder binary legacy ``theta``/factor belief tables to row-major.

    Values remain differentiable. ``legacy_to_row_major`` is its own inverse
    for binary factors (bit reversal). It does not alter the contents of the
    original tensor or assume the factors have a particular interpretation.
    """
    if values.ndim != 1 or values.numel() != graph.num_configurations:
        raise ValueError("Values must contain one entry per factor configuration")
    if not bool((graph.cardinalities == 2).all().item()):
        raise ValueError("Legacy little-endian conversion supports binary variables only")
    ptr = graph.potential_ptr.cpu().tolist()
    sizes = graph.factor_sizes.cpu().tolist()
    chunks = []
    for f, k in enumerate(sizes):
        chunks.append(values[ptr[f]:ptr[f + 1]].index_select(
            0, _reversed_binary_indices(k, values.device)))
    return torch.cat(chunks) if chunks else values.new_empty((0,))


def row_major_to_legacy(graph: FactorGraph, values: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`legacy_to_row_major` for binary factor tables."""
    return legacy_to_row_major(graph, values)
