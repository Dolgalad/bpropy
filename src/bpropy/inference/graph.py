"""Compact, immutable topology for discrete factor graphs.

The topology stores *no* potentials, trainable parameters, or messages.  It is
suitable for re-use across many neural-network forward passes.  Factors may
have repeated scopes (they represent distinct factors), isolated variables,
and even empty scopes (constant factors).

A factor's flattened potential table uses C / PyTorch row-major ordering:
the *last* variable in its ordered scope varies fastest.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from typing import Iterable, Sequence

import torch


EdgeType = ("variable", "in", "factor")
ReverseEdgeType = ("factor", "has", "variable")


def _long_cpu(values: Sequence[int] | torch.Tensor) -> torch.Tensor:
    """Return a fresh, contiguous, 1-D CPU index tensor."""
    return torch.as_tensor(values, dtype=torch.long, device="cpu").reshape(-1).clone()


@dataclass(frozen=True, eq=False)
class FactorGraph:
    """A general discrete factor graph, with explicit incidence order.

    Args:
        cardinalities: ``[V]`` state counts (one or more per variable).
        factor_ptr: ``[F+1]`` CSR offsets for each factor's scope.
        factor_vars: ``[E]`` concatenated variable IDs, in factor order.
        variable_graph_ptr: ``[B+1]`` offsets for batched variable sets.
        factor_graph_ptr: ``[B+1]`` offsets for batched factor sets.

    Unlike a NetworkX graph, factors can have *identical* scopes and still be
    distinct. Factor potentials live in a separate floating-point tensor.
    """

    cardinalities: torch.Tensor
    factor_ptr: torch.Tensor
    factor_vars: torch.Tensor
    variable_graph_ptr: torch.Tensor
    factor_graph_ptr: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("cardinalities", "factor_ptr", "factor_vars",
                     "variable_graph_ptr", "factor_graph_ptr"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.dtype != torch.long:
                raise TypeError(f"{name} must be a one-dimensional torch.long tensor")
            if value.device != self.cardinalities.device:
                raise ValueError("All topology tensors must be on the same device")

        # These checks are topology-construction-time only. They are intentionally
        # not part of the forward pass of a compiled model.
        v = self.cardinalities.cpu().tolist()
        fp = self.factor_ptr.cpu().tolist()
        fv = self.factor_vars.cpu().tolist()
        vp = self.variable_graph_ptr.cpu().tolist()
        gp = self.factor_graph_ptr.cpu().tolist()
        if any(c < 1 for c in v):
            raise ValueError("Every variable cardinality must be >= 1")
        if not fp or fp[0] != 0 or fp[-1] != len(fv) or any(a > b for a, b in zip(fp, fp[1:])):
            raise ValueError("factor_ptr must start at 0, be nondecreasing and end at len(factor_vars)")
        if any(not (0 <= x < len(v)) for x in fv):
            raise ValueError("factor_vars contains an out-of-range variable index")
        if not vp or not gp or len(vp) != len(gp):
            raise ValueError("variable_graph_ptr and factor_graph_ptr must have matching lengths")
        if (vp[0], vp[-1]) != (0, len(v)) or (gp[0], gp[-1]) != (0, len(fp) - 1):
            raise ValueError("Batch pointers must span the variables and factors")
        if any(a > b for a, b in zip(vp, vp[1:])) or any(a > b for a, b in zip(gp, gp[1:])):
            raise ValueError("Batch pointers must be nondecreasing")
        for f in range(len(fp) - 1):
            scope = fv[fp[f]:fp[f + 1]]
            if len(scope) != len(set(scope)):
                raise ValueError(f"Factor {f} has a repeated variable in its scope")
        for b in range(len(vp) - 1):
            for f in range(gp[b], gp[b + 1]):
                if any(not (vp[b] <= x < vp[b + 1]) for x in fv[fp[f]:fp[f + 1]]):
                    raise ValueError(f"Factor {f} connects variables in another batch graph")

    @classmethod
    def from_scopes(
        cls,
        cardinalities: Sequence[int] | torch.Tensor,
        scopes: Iterable[Sequence[int]],
        *,
        device: torch.device | str | None = None,
    ) -> "FactorGraph":
        """Build from ordered variable scopes without depending on PyG.

        Example: ``FactorGraph.from_scopes([2, 3], [(0,), (0, 1)])``.
        """
        cards = _long_cpu(cardinalities)
        ptr, variables = [0], []
        for scope in scopes:
            variables.extend(int(i) for i in scope)
            ptr.append(len(variables))
        graph = cls(
            cardinalities=cards,
            factor_ptr=_long_cpu(ptr),
            factor_vars=_long_cpu(variables),
            variable_graph_ptr=_long_cpu([0, cards.numel()]),
            factor_graph_ptr=_long_cpu([0, len(ptr) - 1]),
        )
        return graph if device is None else graph.to(device)

    @classmethod
    def batch(cls, graphs: Sequence["FactorGraph"]) -> "FactorGraph":
        """Disjoint-union batch, including already-batched graphs.

        The result's indices refer to the *global* concatenated variable and
        factor arrays; the graph pointers preserve each independent instance.
        """
        if not graphs:
            raise ValueError("Cannot batch an empty sequence of graphs")
        device = graphs[0].device
        if any(g.device != device for g in graphs):
            raise ValueError("Move all graphs to the same device before batching")
        cards, fvars = [], []
        fptr, vptr, gptr = [0], [0], [0]
        vbase = fbase = ibase = 0
        for g in graphs:
            cards.append(g.cardinalities)
            fvars.append(g.factor_vars + vbase)
            fptr.extend((g.factor_ptr[1:] + ibase).cpu().tolist())
            vptr.extend((g.variable_graph_ptr[1:] + vbase).cpu().tolist())
            gptr.extend((g.factor_graph_ptr[1:] + fbase).cpu().tolist())
            vbase += g.num_variables
            fbase += g.num_factors
            ibase += g.num_incidences
        return cls(torch.cat(cards), _long_cpu(fptr).to(device),
                   torch.cat(fvars), _long_cpu(vptr).to(device),
                   _long_cpu(gptr).to(device))

    @property
    def device(self) -> torch.device:
        return self.cardinalities.device

    @property
    def num_variables(self) -> int:
        return int(self.cardinalities.numel())

    @property
    def num_factors(self) -> int:
        return int(self.factor_ptr.numel() - 1)

    @property
    def num_incidences(self) -> int:
        return int(self.factor_vars.numel())

    @property
    def num_graphs(self) -> int:
        return int(self.variable_graph_ptr.numel() - 1)

    @property
    def factor_sizes(self) -> torch.Tensor:
        return self.factor_ptr[1:] - self.factor_ptr[:-1]

    @property
    def potential_sizes(self) -> torch.Tensor:
        cards = self.cardinalities.cpu().tolist()
        ptr = self.factor_ptr.cpu().tolist()
        variables = self.factor_vars.cpu().tolist()
        sizes = [prod(cards[x] for x in variables[ptr[f]:ptr[f + 1]])
                 for f in range(self.num_factors)]
        return _long_cpu(sizes).to(self.device)

    @property
    def potential_ptr(self) -> torch.Tensor:
        sizes = self.potential_sizes
        return torch.cat((torch.zeros(1, dtype=torch.long, device=self.device), sizes.cumsum(0)))

    @property
    def num_configurations(self) -> int:
        return int(self.potential_sizes.sum().item())

    def scope(self, factor: int) -> tuple[int, ...]:
        """Ordered variables for one factor (CPU-access convenience)."""
        if not (0 <= factor < self.num_factors):
            raise IndexError(factor)
        a, b = self.factor_ptr[factor:factor + 2].cpu().tolist()
        return tuple(self.factor_vars[a:b].cpu().tolist())

    def to(self, device: torch.device | str) -> "FactorGraph":
        """Copy topology tensors to the requested device."""
        return replace(self, **{name: getattr(self, name).to(device) for name in (
            "cardinalities", "factor_ptr", "factor_vars", "variable_graph_ptr", "factor_graph_ptr")})

    def compile(self, *, max_factor_states: int = 1_000_000,
                max_entries: int = 8_000_000) -> "CompiledFactorGraph":
        """Precompute scatter/gather wiring once, independently of potentials."""
        from .plan import CompiledFactorGraph
        return CompiledFactorGraph.from_graph(
            self, max_factor_states=max_factor_states, max_entries=max_entries)

    def to_pyg(self, log_potentials: torch.Tensor | None = None):
        """Export an ordinary PyG ``HeteroData`` with reciprocal edges.

        ``log_potentials`` is stored as a *flat* factor attribute so regular
        ``torch_geometric.loader.DataLoader`` can concatenate it across graphs.
        Per-factor offsets are reconstructed from variable cardinalities and
        factor scopes, not batched as fragile absolute pointer attributes.
        """
        try:
            from torch_geometric.data import HeteroData
        except ImportError as exc:
            raise ImportError("Install bpropy[pyg] to use PyG conversion") from exc
        data = HeteroData()
        data["variable"].num_nodes = self.num_variables
        data["variable"].cardinalities = self.cardinalities
        data["factor"].num_nodes = self.num_factors
        data["factor"].table_size = self.potential_sizes
        if self.num_graphs > 1:
            # A directly exported disjoint-union graph retains its batch
            # identity. Do not pass an already-batched graph through another
            # PyG DataLoader; instead batch the original unbatched graphs.
            data["variable"].ptr = self.variable_graph_ptr
            data["factor"].ptr = self.factor_graph_ptr
            batch_ids = torch.arange(self.num_graphs, device=self.device)
            data["variable"].batch = torch.repeat_interleave(
                batch_ids, self.variable_graph_ptr[1:] - self.variable_graph_ptr[:-1])
            data["factor"].batch = torch.repeat_interleave(
                batch_ids, self.factor_graph_ptr[1:] - self.factor_graph_ptr[:-1])
        inc_factor = torch.repeat_interleave(
            torch.arange(self.num_factors, device=self.device), self.factor_sizes)
        slot = torch.arange(self.num_incidences, device=self.device) - torch.repeat_interleave(
            self.factor_ptr[:-1], self.factor_sizes)
        data[EdgeType].edge_index = torch.stack((self.factor_vars, inc_factor))
        data[EdgeType].slot = slot
        data[ReverseEdgeType].edge_index = torch.stack((inc_factor, self.factor_vars))
        data[ReverseEdgeType].slot = slot
        if log_potentials is not None:
            if log_potentials.ndim != 1 or log_potentials.numel() != self.num_configurations:
                raise ValueError("log_potentials must be the flattened factor tables")
            data["factor"].log_potentials = log_potentials
        return data

    @classmethod
    def from_pyg(cls, data) -> "FactorGraph":
        """Import ``HeteroData`` or ``Batch``, respecting factor slots.

        Only the variable->factor incidence edge type is required. The optional
        reverse edge type is ignored. Standard PyG batching correctly offsets
        the bipartite indices.
        """
        if EdgeType not in data.edge_types:
            raise ValueError(f"PyG data requires edge type {EdgeType!r}")
        store = data[EdgeType]
        if "slot" not in store:
            raise ValueError("PyG incidence edges need a 'slot' index")
        card = data["variable"].cardinalities
        n_factors = int(data["factor"].num_nodes)
        edge_index = store.edge_index
        slots = store.slot
        edges = [dict() for _ in range(n_factors)]
        for var, factor, slot in zip(edge_index[0].cpu().tolist(),
                                     edge_index[1].cpu().tolist(), slots.cpu().tolist()):
            if slot in edges[factor]:
                raise ValueError("Duplicate incidence slot for a factor")
            edges[factor][slot] = var
        scopes = []
        for f, incidence in enumerate(edges):
            if set(incidence) != set(range(len(incidence))):
                raise ValueError(f"Factor {f} has missing or nonconsecutive slots")
            scopes.append([incidence[k] for k in range(len(incidence))])
        g = cls.from_scopes(card, scopes, device=card.device)
        variable_ptr = getattr(data["variable"], "ptr", None)
        factor_ptr = getattr(data["factor"], "ptr", None)
        if variable_ptr is not None or factor_ptr is not None:
            if variable_ptr is None or factor_ptr is None:
                raise ValueError("PyG batch must have pointers for both node types")
            g = replace(g, variable_graph_ptr=variable_ptr.to(card.device),
                        factor_graph_ptr=factor_ptr.to(card.device))
        return g

    @staticmethod
    def potentials_from_pyg(data) -> torch.Tensor:
        """Retrieve the flat, differentiable log-potentials from PyG data."""
        if "log_potentials" not in data["factor"]:
            raise ValueError("The PyG factor store has no log_potentials")
        return data["factor"].log_potentials
