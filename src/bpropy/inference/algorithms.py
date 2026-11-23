"""Differentiable, GPU-vectorized discrete factor-graph inference.

Each iteration uses a small number of PyTorch gathers/scatter reductions.  No
Python loop iterates through factors or edges inside the inference kernels.
Compile a topology once and reuse it with new learned log potentials.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log, log1p

import torch
from torch import nn

from .graph import FactorGraph
from .ops import (segment_log_softmax, segment_logsumexp, segment_sum,
                  sum_excluding, sum_including_negative_infinity)
from .plan import CompiledFactorGraph


@dataclass(eq=False)
class InferenceResult:
    """Inference result in ragged/flat tensor format.

    ``variable_beliefs`` holds normalized probabilities for each variable's
    states, in ``plan.state_ptr`` order. ``factor_beliefs`` contains marginal
    distributions over factor configurations if available (only BP).
    ``residual`` is a scalar tensor; ``converged`` is ``None`` when early
    stopping wasn't requested.
    """

    plan: CompiledFactorGraph
    variable_beliefs: torch.Tensor
    factor_beliefs: torch.Tensor | None
    iterations: int
    residual: torch.Tensor
    converged: bool | None
    algorithm: str

    def marginal(self, variable: int) -> torch.Tensor:
        """Return a variable's normalized state distribution."""
        if not (0 <= variable < self.plan.graph.num_variables):
            raise IndexError(variable)
        # Indexing by scalar tensor bounds is a convenience outside the kernel.
        a, b = self.plan.state_ptr[variable:variable + 2].cpu().tolist()
        return self.variable_beliefs[a:b]

    def factor_marginal(self, factor: int) -> torch.Tensor:
        """Return a factor's normalized configuration distribution (BP only)."""
        if self.factor_beliefs is None:
            raise ValueError("Factor marginals are not provided by this inference method")
        if not (0 <= factor < self.plan.graph.num_factors):
            raise IndexError(factor)
        a, b = self.plan.config_ptr[factor:factor + 2].cpu().tolist()
        return self.factor_beliefs[a:b]


def _validate_parameters(plan: CompiledFactorGraph, theta: torch.Tensor,
                         iterations: int, damping: float, tol: float | None) -> None:
    plan.validate_potentials(theta)
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not (0.0 <= damping < 1.0):
        raise ValueError("damping must be in the interval [0, 1)")
    if tol is not None and tol < 0.0:
        raise ValueError("tol must be nonnegative or None")


def _initial_log_messages(plan: CompiledFactorGraph, theta: torch.Tensor,
                          initial_messages: torch.Tensor | None) -> torch.Tensor:
    if initial_messages is not None:
        if initial_messages.shape != (plan.num_message_states,):
            raise ValueError("initial_messages has incompatible shape")
        if initial_messages.device != theta.device or initial_messages.dtype != theta.dtype:
            raise ValueError("initial_messages must use the potentials' device and dtype")
        return segment_log_softmax(initial_messages, plan.message_incidence,
                                   plan.graph.num_incidences)
    sizes = plan.message_ptr[1:] - plan.message_ptr[:-1]
    return -sizes[plan.message_incidence].to(theta.dtype).log()


def _factor_to_variable(theta: torch.Tensor, log_v2f: torch.Tensor,
                        plan: CompiledFactorGraph) -> torch.Tensor:
    """Parallel factor-to-variable marginalization over every configuration."""
    entry_input = log_v2f[plan.entry_message]
    other_inputs = sum_excluding(entry_input, plan.entry_config, plan.num_configurations)
    logits = theta[plan.entry_config] + other_inputs
    return segment_logsumexp(logits, plan.entry_message, plan.num_message_states)


def _variable_to_factor(log_f2v: torch.Tensor, plan: CompiledFactorGraph) -> torch.Tensor:
    """Parallel variable-to-factor cavity messages, per-state normalized."""
    all_but_target = sum_excluding(log_f2v, plan.message_var_state, plan.num_states)
    return segment_log_softmax(all_but_target, plan.message_incidence,
                               plan.graph.num_incidences)


def _bp_beliefs(theta: torch.Tensor, log_v2f: torch.Tensor,
                log_f2v: torch.Tensor, plan: CompiledFactorGraph,
                *, return_factor_beliefs: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    variable_logits = sum_including_negative_infinity(
        log_f2v, plan.message_var_state, plan.num_states)
    variable_log_beliefs = segment_log_softmax(variable_logits, plan.state_var,
                                               plan.graph.num_variables)
    factor_probs = None
    if return_factor_beliefs:
        config_inputs = sum_including_negative_infinity(
            log_v2f[plan.entry_message], plan.entry_config, plan.num_configurations)
        factor_logits = theta + config_inputs
        factor_probs = segment_log_softmax(factor_logits, plan.config_factor,
                                            plan.graph.num_factors).exp()
    return variable_log_beliefs.exp(), factor_probs


def lbp(
    plan: CompiledFactorGraph,
    log_potentials: torch.Tensor,
    *,
    iterations: int = 30,
    damping: float = 0.0,
    tol: float | None = None,
    initial_messages: torch.Tensor | None = None,
    return_factor_beliefs: bool = True,
) -> InferenceResult:
    """Synchronous sum-product loopy belief propagation in log-space.

    Args:
        plan: Precompiled general discrete factor graph, on the same device.
        log_potentials: Flattened factor tables ``[T]``, differentiable.
        iterations: Fixed iteration count unless ``tol`` is provided.
        damping: In [0,1); probability-space convex mixing of old and new
            normalized variable-to-factor messages. 0 means no damping.
        tol: Optional maximum probability-message change for early stopping.
            ``None`` (the default) avoids GPU-to-CPU synchronization and keeps
            the iteration count static for training and ``torch.compile``.
        initial_messages: Optional flat initial *log* messages ``[M]``.
        return_factor_beliefs: Whether to compute normalized factor beliefs.

    Returns:
        :class:`InferenceResult` with variable marginals and optionally factor
        marginals. Gradients propagate through every unrolled iteration.

    ``-inf`` log potentials express impossible configurations. A model with no
    valid global assignment cannot have meaningful normalized marginals.
    """
    _validate_parameters(plan, log_potentials, iterations, damping, tol)
    q = _initial_log_messages(plan, log_potentials, initial_messages)
    residual = log_potentials.new_zeros(())
    converged: bool | None = None if tol is None else False
    for i in range(iterations):
        r = _factor_to_variable(log_potentials, q, plan)
        fresh = _variable_to_factor(r, plan)
        if damping:
            q_next = torch.logaddexp(fresh + log1p(-damping), q + log(damping))
        else:
            q_next = fresh
        residual = ((q_next.exp() - q.exp()).abs().amax()
                    if plan.num_message_states else log_potentials.new_zeros(()))
        q = q_next
        if tol is not None and bool((residual <= tol).item()):
            converged = True
            break
    # Recompute from final q, avoiding a one-iteration belief/message mismatch.
    r = _factor_to_variable(log_potentials, q, plan)
    variable, factor = _bp_beliefs(log_potentials, q, r, plan,
                                   return_factor_beliefs=return_factor_beliefs)
    return InferenceResult(plan, variable, factor, i + 1, residual, converged, "lbp")


def _nmf_update(theta: torch.Tensor, q: torch.Tensor, plan: CompiledFactorGraph,
                eps: float) -> torch.Tensor:
    # The list of all states participating in each configuration is compiled.
    # Summing logs gives the product over each factor configuration; subtract
    # the target variable's contribution to obtain the mean-field cavity.
    logq = q.clamp_min(eps).log()
    entry_var_state = plan.message_var_state[plan.entry_message]
    entry_logq = logq[entry_var_state]
    config_total = segment_sum(entry_logq, plan.entry_config, plan.num_configurations)
    cavity_weight = (config_total[plan.entry_config] - entry_logq).exp()
    expected_theta = theta[plan.entry_config] * cavity_weight
    logits = segment_sum(expected_theta, entry_var_state, plan.num_states)
    log_probs = segment_log_softmax(logits, plan.state_var, plan.graph.num_variables)
    return log_probs.exp()


def mean_field(
    plan: CompiledFactorGraph,
    log_potentials: torch.Tensor,
    *,
    iterations: int = 30,
    damping: float = 0.0,
    tol: float | None = None,
    eps: float = 1e-12,
    initial_beliefs: torch.Tensor | None = None,
    validate_finite: bool = True,
) -> InferenceResult:
    """Parallel synchronous naive-mean-field fixed-point inference.

    Uses the same factor topology and flattened log potentials as ``lbp``.
    Naive mean-field is an approximation ``q(x)=prod_i q_i(x_i)``. All finite
    log potentials are supported; ``-inf`` potentials require special handling
    of feasibility/entropy and are rejected in the current implementation.
    The returned factor_beliefs is ``None`` (the variational factor products
    are not actual factor marginals of the original model).

    ``eps`` prevents ``log(0)`` inside products; use a small positive value.
    Set ``validate_finite=False`` only when potentials are known to be finite;
    it avoids a GPU-to-CPU synchronization and enables fullgraph torch.compile.
    """
    _validate_parameters(plan, log_potentials, iterations, damping, tol)
    if not (0.0 < eps < 1.0):
        raise ValueError("eps must be between zero and one")
    if validate_finite and not bool(torch.isfinite(log_potentials).all().item()):
        raise ValueError("Mean-field currently requires finite log potentials; soften hard constraints")
    cards = plan.graph.cardinalities[plan.state_var].to(log_potentials.dtype)
    if initial_beliefs is None:
        q = 1 / cards
    else:
        if initial_beliefs.shape != (plan.num_states,):
            raise ValueError("initial_beliefs must have shape [sum(cardinalities)]")
        if initial_beliefs.device != log_potentials.device or initial_beliefs.dtype != log_potentials.dtype:
            raise ValueError("initial_beliefs must use potentials' device and dtype")
        if not bool(((initial_beliefs >= 0) & torch.isfinite(initial_beliefs)).all().item()):
            raise ValueError("initial_beliefs must be finite nonnegative probabilities")
        q = segment_log_softmax(initial_beliefs.clamp_min(eps).log(),
                                plan.state_var, plan.graph.num_variables).exp()
    residual = log_potentials.new_zeros(())
    converged: bool | None = None if tol is None else False
    for i in range(iterations):
        fresh = _nmf_update(log_potentials, q, plan, eps)
        q_next = (1 - damping) * fresh + damping * q
        residual = ((q_next - q).abs().amax() if plan.num_states
                    else log_potentials.new_zeros(()))
        q = q_next
        if tol is not None and bool((residual <= tol).item()):
            converged = True
            break
    return InferenceResult(plan, q, None, i + 1, residual, converged, "mean_field")


class LBP(nn.Module):
    """Stateless ``torch.nn.Module`` adapter for the ``lbp`` routine."""

    def __init__(self, *, iterations: int = 30, damping: float = 0.0,
                 return_factor_beliefs: bool = True):
        super().__init__()
        self.iterations = iterations
        self.damping = damping
        self.return_factor_beliefs = return_factor_beliefs

    def forward(self, plan: CompiledFactorGraph,
                log_potentials: torch.Tensor) -> InferenceResult:
        return lbp(plan, log_potentials, iterations=self.iterations,
                   damping=self.damping, return_factor_beliefs=self.return_factor_beliefs)


class NaiveMeanField(nn.Module):
    """Stateless ``torch.nn.Module`` adapter for the ``mean_field`` routine."""

    def __init__(self, *, iterations: int = 30, damping: float = 0.0,
                 validate_finite: bool = True):
        super().__init__()
        self.iterations = iterations
        self.damping = damping
        self.validate_finite = validate_finite

    def forward(self, plan: CompiledFactorGraph,
                log_potentials: torch.Tensor) -> InferenceResult:
        return mean_field(plan, log_potentials, iterations=self.iterations,
                          damping=self.damping, validate_finite=self.validate_finite)
