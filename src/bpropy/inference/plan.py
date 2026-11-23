"""Compile factor graph topology into reusable GPU scatter/gather indices.

The plan has no learnable values. Building is a one-off CPU operation; moving a
compiled plan to CUDA creates tensors that inference uses without further graph
traversal or Python loops over factors.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

import torch

from .graph import FactorGraph


@dataclass(frozen=True, eq=False)
class CompiledFactorGraph:
    """Tensor-only computation plan for sum-product BP and naive mean field.

    Shapes and semantics:

    * ``state_ptr [V+1]``: offsets of the variable-state beliefs.
    * ``state_var [S]``: variable index for every variable-state slot.
    * ``config_ptr [F+1]``: offsets of each factor's configuration table.
    * ``config_factor [T]``: factor index for every configuration.
    * ``message_ptr [E+1]``: offsets of variable-to-factor messages, one
      normalized categorical distribution per *incidence*.
    * ``message_incidence [M]``: incidence index per message-state slot.
    * ``message_var_state [M]``: variable-state slot corresponding to a
      message-state slot.
    * ``entry_config [A]`` and ``entry_message [A]``: each factor
      configuration's participating incidence-state slots, in configuration
      order. ``A = sum_f (degree(f) * configurations(f))``.

    These indices permit all factors of different arities/cardinalities and all
    graphs in a batch to be processed simultaneously on the GPU.
    """

    graph: FactorGraph
    state_ptr: torch.Tensor
    state_var: torch.Tensor
    config_ptr: torch.Tensor
    config_factor: torch.Tensor
    message_ptr: torch.Tensor
    message_incidence: torch.Tensor
    message_var_state: torch.Tensor
    entry_config: torch.Tensor
    entry_message: torch.Tensor

    @classmethod
    def from_graph(cls, graph: FactorGraph, *, max_factor_states: int = 1_000_000,
                   max_entries: int = 8_000_000) -> "CompiledFactorGraph":
        if max_factor_states < 1 or max_entries < 1:
            raise ValueError("Compilation limits must be positive")
        cards = graph.cardinalities.cpu().tolist()
        factor_ptr = graph.factor_ptr.cpu().tolist()
        factors_vars = graph.factor_vars.cpu().tolist()
        state_ptr = [0]
        for c in cards:
            state_ptr.append(state_ptr[-1] + c)
        msg_ptr = [0]
        for var in factors_vars:
            msg_ptr.append(msg_ptr[-1] + cards[var])
        config_ptr = [0]
        entry_cfgs, entry_msgs = [], []
        n_entries = 0
        for f in range(graph.num_factors):
            scope = factors_vars[factor_ptr[f]:factor_ptr[f + 1]]
            factor_cards = [cards[v] for v in scope]
            n = prod(factor_cards)
            if n > max_factor_states:
                raise ValueError(
                    f"Factor {f} has {n} states, exceeding max_factor_states={max_factor_states}. "
                    "Dense factor tables grow exponentially with the scope size.")
            n_entries += len(scope) * n
            if n_entries > max_entries:
                raise ValueError(f"Compiled scatter plan exceeds max_entries={max_entries}")
            if scope:
                config = torch.arange(n, dtype=torch.long).unsqueeze(1)
                # Last scope variable varies fastest, like standard torch.reshape.
                strides = [prod(factor_cards[i + 1:]) for i in range(len(scope))]
                index = torch.tensor(strides, dtype=torch.long)
                states = (config // index) % torch.tensor(factor_cards, dtype=torch.long)
                msg_starts = torch.tensor(msg_ptr[factor_ptr[f]:factor_ptr[f + 1]], dtype=torch.long)
                msg_idx = (states + msg_starts).reshape(-1)
                cfg_idx = torch.arange(config_ptr[-1], config_ptr[-1] + n).repeat_interleave(len(scope))
                entry_cfgs.append(cfg_idx)
                entry_msgs.append(msg_idx)
            config_ptr.append(config_ptr[-1] + n)

        cptr = torch.tensor(config_ptr, dtype=torch.long)
        sptr = torch.tensor(state_ptr, dtype=torch.long)
        mptr = torch.tensor(msg_ptr, dtype=torch.long)
        var_for_incidence = graph.factor_vars.cpu()
        msg_incidence = torch.repeat_interleave(torch.arange(graph.num_incidences),
                                                mptr[1:] - mptr[:-1])
        msg_var_state = torch.cat([
            torch.arange(state_ptr[var], state_ptr[var + 1]) for var in factors_vars
        ]) if factors_vars else torch.empty((0,), dtype=torch.long)
        plan = cls(
            graph=graph.to("cpu"),
            state_ptr=sptr,
            state_var=torch.repeat_interleave(torch.arange(graph.num_variables), sptr[1:] - sptr[:-1]),
            config_ptr=cptr,
            config_factor=torch.repeat_interleave(torch.arange(graph.num_factors), cptr[1:] - cptr[:-1]),
            message_ptr=mptr,
            message_incidence=msg_incidence,
            message_var_state=msg_var_state,
            entry_config=torch.cat(entry_cfgs) if entry_cfgs else torch.empty(0, dtype=torch.long),
            entry_message=torch.cat(entry_msgs) if entry_msgs else torch.empty(0, dtype=torch.long),
        )
        return plan if graph.device.type == "cpu" else plan.to(graph.device)

    @property
    def device(self) -> torch.device:
        return self.state_ptr.device

    @property
    def num_configurations(self) -> int:
        return self.config_factor.numel()

    @property
    def num_states(self) -> int:
        return self.state_var.numel()

    @property
    def num_message_states(self) -> int:
        return self.message_incidence.numel()

    def to(self, device: torch.device | str) -> "CompiledFactorGraph":
        """Move all index buffers, including topology, to CPU or GPU."""
        return replace(self, graph=self.graph.to(device), **{
            field: getattr(self, field).to(device) for field in (
                "state_ptr", "state_var", "config_ptr", "config_factor",
                "message_ptr", "message_incidence", "message_var_state",
                "entry_config", "entry_message")})

    def validate_potentials(self, log_potentials: torch.Tensor) -> None:
        """Check argument invariants before inference (not inside the GPU loop)."""
        if not isinstance(log_potentials, torch.Tensor):
            raise TypeError("log_potentials must be a torch.Tensor")
        if log_potentials.ndim != 1 or log_potentials.numel() != self.num_configurations:
            raise ValueError(f"Expected a 1-D log-potential vector of length {self.num_configurations}")
        if not log_potentials.dtype.is_floating_point:
            raise TypeError("log_potentials must have floating-point dtype")
        if log_potentials.device != self.device:
            raise ValueError(f"plan is on {self.device}; log_potentials are on {log_potentials.device}")
        # NaN / +Inf are disallowed by convention; -Inf encodes hard constraints.
