from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch

from bpropy.flow_matching import (
    flow_rollout,
    sample_evidence_aware_flow_source,
)

from bpropy.mrf.evidence_factory import EvidenceMaskFactory
from bpropy.mrf.lbp_edges import LBPEdges
from bpropy.mrf.lbp import loopy_belief_propagation
from bpropy.mrf.nmf_edges import NMFEdges
from bpropy.mrf.naive_mean_field import naive_mean_field


Evidence = Dict[int, int]


@dataclass
class _FlowMRFBackendBase:
    """
    Shared base for FlowLBPBackend and FlowNMFBackend.

    Protocol:
      - reset(pb): initialize factor graph, evidence factory, and inference edges
      - observe(): run conditional flow rollout, extract theta, clamp evidence,
                   then run inference
      - update(evidence): add decimation evidence and invalidate cached beliefs

    The flow-specific pieces are delegated to bpropy.flow_matching:
      - sample_evidence_aware_flow_source(...)
      - flow_rollout(..., return_final_theta=True)
    """

    fg_builder: Callable[[Any], Any]
    theta_builder: Callable[[Any, Any, torch.device, torch.dtype], torch.Tensor]

    model: torch.nn.Module
    adapter: Any
    data: Any
    cfg: Dict[str, Any]

    run_every: int = 1
    neg_inf: Optional[float] = None

    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    rollout_steps: Optional[int] = None

    # Internal state.
    _pb: Any = None
    _fg: Any = None
    _base_theta: Optional[torch.Tensor] = None
    _theta: Optional[torch.Tensor] = None
    _evidence_factory: Any = None
    _mu: Optional[torch.Tensor] = None
    _step: int = 0
    _evidence: Optional[Evidence] = None

    def _reset_common(self, pb: Any) -> None:
        self._pb = pb
        self._fg = self.fg_builder(pb)

        base_theta = self.theta_builder(
            pb,
            self._fg,
            self.device,
            self.dtype,
        )

        if torch.is_tensor(base_theta):
            base_theta = base_theta.detach().clone()

        self._base_theta = base_theta
        self._theta = None

        self._evidence_factory = EvidenceMaskFactory(
            self._fg,
            theta=self._base_theta,
            device=self._base_theta.device,
            neg_inf=self.neg_inf,
        )

        self._mu = None
        self._step = 0
        self._evidence = {}

    def _num_variables(self) -> int:
        if self._pb is not None and hasattr(self._pb, "nv"):
            return int(self._pb.nv())

        if hasattr(self.data, "node_types") and (
            "variable" in getattr(self.data, "node_types", [])
        ):
            return int(self.data["variable"].num_nodes)

        if hasattr(self.data, "bipartite"):
            return int((self.data.bipartite == 0).sum().item())

        raise RuntimeError(
            "Could not infer number of variables. Expected pb.nv(), "
            "hetero data['variable'].num_nodes, or homogeneous data.bipartite."
        )

    def _make_p0(self) -> torch.Tensor:
        flow_cfg = self.cfg.get("flow", {})

        return sample_evidence_aware_flow_source(
            num_variables=self._num_variables(),
            evidence=self._evidence,
            device=self.device,
            dtype=self.dtype,
            source_prob=float(flow_cfg.get("source_prob", 0.5)),
            source=flow_cfg.get("source", "uniform"),
        )

    def _run_flow_theta(self) -> torch.Tensor:
        """
        Run a full conditional flow rollout and return final theta.

        Requires bpropy.flow_matching.flow_rollout to support:
            return_final_theta=True

        and to return an object with a `.theta` field.
        """
        p0 = self._make_p0()

        result = flow_rollout(
            model=self.model,
            p_init=p0,
            adapter=self.adapter,
            data=self.data,
            pb=[self._pb],
            device=self.device,
            cfg=self.cfg,
            return_final_theta=True,
            n_steps=self.rollout_steps,
        )

        theta = result.theta
        if theta is None:
            raise RuntimeError(
                "flow_rollout(..., return_final_theta=True) returned theta=None. "
                "Add adapter.theta_fn(model, flow_data, cfg), or make "
                "flow_matching.default_theta_fn able to call model(..., inference='theta')."
            )

        if not torch.is_tensor(theta):
            raise TypeError(
                f"Expected flow rollout theta to be a torch.Tensor, got {type(theta)!r}."
            )

        return theta.detach().clone()

    def update(self, evidence: Evidence) -> None:
        if evidence:
            assert self._evidence is not None
            assert self._evidence_factory is not None

            evidence = {int(k): int(v) for k, v in evidence.items()}

            self._evidence.update(evidence)
            self._evidence_factory.add_evidence(evidence, strict=True)

            # Current cached beliefs were computed under older evidence.
            self._mu = None

        self._step += 1

    def _should_run(self) -> bool:
        return (
            self._mu is None
            or self.run_every <= 1
            or self._step % self.run_every == 0
        )


@dataclass
class FlowLBPBackend(_FlowMRFBackendBase):
    """
    Flow-conditioned LBP backend.

    observe():
      1. Builds p0 respecting current evidence.
      2. Runs full flow rollout p0 -> p1.
      3. Extracts final model theta.
      4. Applies EvidenceMaskFactory to theta.
      5. Runs loopy belief propagation.
      6. Returns mu_v reshaped as (nV, 2).
    """

    max_iter: int = 1000
    tol: float = 1e-6
    damping: float = 0.5
    warm_start: bool = True

    _edges: Any = None
    _v2f: Optional[torch.Tensor] = None

    def reset(self, pb: Any) -> None:
        self._reset_common(pb)

        assert self._fg is not None
        assert self._base_theta is not None

        self._edges = LBPEdges(self._fg)

        n_msg = int(self._edges.n_f2v_msg.sum().item())
        self._v2f = torch.zeros(
            n_msg,
            device=self._base_theta.device,
            dtype=self._base_theta.dtype,
        )

    def observe(self) -> Optional[torch.Tensor]:
        assert self._evidence_factory is not None
        assert self._edges is not None
        assert self._v2f is not None

        if self._should_run():
            self._theta = self._run_flow_theta()

            # Match existing LBPBackend behavior: evidence mask is applied
            # immediately before inference.
            th = self._evidence_factory.apply_inplace(self._theta)

            if not self.warm_start:
                self._v2f.zero_()

            res = loopy_belief_propagation(
                th,
                self._v2f,
                self._edges,
                max_iter=self.max_iter,
                tol=self.tol,
                damping=self.damping,
            )

            self._mu = torch.nan_to_num(
                res.mu_v.reshape(-1, 2),
                nan=0.0,
            )

        return self._mu


@dataclass
class FlowNMFBackend(_FlowMRFBackendBase):
    """
    Flow-conditioned naive mean-field backend.

    observe():
      1. Builds p0 respecting current evidence.
      2. Runs full flow rollout p0 -> p1.
      3. Extracts final model theta.
      4. Applies EvidenceMaskFactory to theta.
      5. Runs naive mean field.
      6. Returns mu reshaped as (nV, 2).
    """

    max_iter: int = 1000
    tol: float = 1e-6
    damping: float = 0.0

    _edges: Any = None

    def reset(self, pb: Any) -> None:
        self._reset_common(pb)

        assert self._fg is not None
        assert self._base_theta is not None

        self._edges = NMFEdges(self._fg).to(self._base_theta.device)

    def observe(self) -> Optional[torch.Tensor]:
        assert self._evidence_factory is not None
        assert self._edges is not None

        if self._should_run():
            self._theta = self._run_flow_theta()

            # Match existing NMFBackend behavior: use non-inplace evidence apply.
            th = self._evidence_factory.apply(self._theta)

            sol = naive_mean_field(
                th,
                self._edges,
                max_iter=self.max_iter,
                tol=self.tol,
                damping=self.damping,
            )

            self._mu = torch.nan_to_num(
                sol.mu.reshape(-1, 2),
                nan=0.0,
            )

        return self._mu

@dataclass
class FlowScoreBackend:
    """
    Flow-conditioned score backend for variable-logit / variable-probability models.

    This backend is for models that do NOT produce factor theta.

    Protocol:
      - reset(pb): initialize per-instance state
      - observe(): build evidence-aware p0, run full flow rollout, return scores
      - update(evidence): store evidence and invalidate cached scores

    observe() returns final rollout probabilities p1 as shape (nV,).
    """

    model: torch.nn.Module
    adapter: Any
    data: Any
    cfg: Dict[str, Any]

    run_every: int = 1
    rollout_steps: Optional[int] = None

    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    _pb: Any = None
    _scores: Optional[torch.Tensor] = None
    _step: int = 0
    _evidence: Optional[Evidence] = None

    def reset(self, pb: Any) -> None:
        self._pb = pb
        self._scores = None
        self._step = 0
        self._evidence = {}

    def _num_variables(self) -> int:
        if self._pb is not None and hasattr(self._pb, "nv"):
            return int(self._pb.nv())

        if hasattr(self.data, "node_types") and (
            "variable" in getattr(self.data, "node_types", [])
        ):
            return int(self.data["variable"].num_nodes)

        if hasattr(self.data, "bipartite"):
            return int((self.data.bipartite == 0).sum().item())

        raise RuntimeError(
            "Could not infer number of variables. Expected pb.nv(), "
            "hetero data['variable'].num_nodes, or homogeneous data.bipartite."
        )

    def _make_p0(self) -> torch.Tensor:
        flow_cfg = self.cfg.get("flow", {})

        return sample_evidence_aware_flow_source(
            num_variables=self._num_variables(),
            evidence=self._evidence,
            device=self.device,
            dtype=self.dtype,
            source_prob=float(flow_cfg.get("source_prob", 0.5)),
            source=flow_cfg.get("source", "uniform"),
        )

    def _should_run(self) -> bool:
        return (
            self._scores is None
            or self.run_every <= 1
            or self._step % self.run_every == 0
        )

    def observe(self) -> Optional[torch.Tensor]:
        if self._should_run():
            p0 = self._make_p0()

            p1 = flow_decoding(
                model=self.model,
                p_init=p0,
                adapter=self.adapter,
                data=self.data,
                pb=[self._pb],
                device=self.device,
                cfg=self.cfg,
                update_mode="endpoint_velocity",
                n_steps=self.rollout_steps,
            )

            p1 = p1.to(device=self.device, dtype=self.dtype).clamp(0.0, 1.0)

            # Hard-clamp returned probabilities too. The source is already
            # evidence-aware, but this makes the backend output consistent
            # with current decimation evidence.
            if self._evidence:
                for i, value in self._evidence.items():
                    p1[int(i)] = float(value)

            self._scores = p1

        return self._scores

    def update(self, evidence: Evidence) -> None:
        if evidence:
            assert self._evidence is not None
            evidence = {int(k): int(v) for k, v in evidence.items()}
            self._evidence.update(evidence)

            # Current cached scores were computed under older evidence.
            self._scores = None

        self._step += 1
