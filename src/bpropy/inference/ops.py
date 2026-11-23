"""Differentiable, segmented tensor reductions using native PyTorch only.

All operations support CPU and CUDA and remain part of the autograd graph.
They avoid a mandatory torch-scatter/PyG binary-extension dependency.
"""
from __future__ import annotations

import torch


def segment_sum(values: torch.Tensor, group: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Sum one-dimensional values for each group; empty groups return zero."""
    return values.new_zeros(n_groups).scatter_add(0, group, values)


def segment_logsumexp(values: torch.Tensor, group: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Stable grouped logsumexp. Empty or all-``-inf`` groups return ``-inf``.

    ``amax`` and ``scatter_add`` are differentiable PyTorch operations. Explicit
    max handling avoids the undefined ``-inf - (-inf)`` operation for impossible
    factor assignments, without bringing the tensors back to the CPU.
    """
    max_per_group = values.new_full((n_groups,), -torch.inf)
    max_per_group = max_per_group.scatter_reduce(0, group, values,
                                                  reduce="amax", include_self=True)
    safe_max = torch.where(torch.isfinite(max_per_group), max_per_group,
                           torch.zeros_like(max_per_group))
    shifted = (values - safe_max[group]).exp()
    norm = segment_sum(shifted, group, n_groups)
    return safe_max + norm.log()


def segment_log_softmax(values: torch.Tensor, group: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Normalize flat log weights over groups, keeping the log representation."""
    return values - segment_logsumexp(values, group, n_groups)[group]


def sum_excluding(values: torch.Tensor, group: torch.Tensor, n_groups: int) -> torch.Tensor:
    """For each entry, sum all *other* entries in the same group.

    Unlike ``total[group] - values``, this is well-defined for ``-inf``
    messages (hard constraints). Each excluded sum is ``-inf`` iff another
    entry in the group is ``-inf``. Only additions and gathers are required.
    """
    negative_infinity = torch.isneginf(values)
    finite = torch.where(negative_infinity, torch.zeros_like(values), values)
    sums = segment_sum(finite, group, n_groups)
    bad_counts = torch.zeros((n_groups,), device=values.device, dtype=torch.long)
    bad_counts = bad_counts.scatter_add(0, group, negative_infinity.to(torch.long))
    remaining_bad = bad_counts[group] - negative_infinity.to(torch.long)
    return torch.where(remaining_bad > 0, values.new_full(values.shape, -torch.inf),
                       sums[group] - finite)


def sum_including_negative_infinity(values: torch.Tensor, group: torch.Tensor,
                                    n_groups: int) -> torch.Tensor:
    """Grouped sums where ``-inf`` propagates within its group."""
    negative_infinity = torch.isneginf(values)
    safe = torch.where(negative_infinity, torch.zeros_like(values), values)
    sums = segment_sum(safe, group, n_groups)
    count = torch.zeros((n_groups,), device=values.device, dtype=torch.long)
    count = count.scatter_add(0, group, negative_infinity.to(torch.long))
    return torch.where(count > 0, sums.new_full(sums.shape, -torch.inf), sums)
