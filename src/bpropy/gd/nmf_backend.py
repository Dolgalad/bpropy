from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch

Evidence = Dict[int, int]


@dataclass
class NMFBackend:
    """
    Mean-field backend that owns:
      - fg + theta (built from pb using fg_builder/theta_builder)
      - EvidenceMaskFactory to clamp theta
      - NMFEdges
      - MF schedule + hyperparams (run_every, max_iter, tol, damping)
    """
    fg_builder: Callable[[Any], Any]
    theta_builder: Callable[[Any, Any, torch.device, torch.dtype], torch.Tensor]

    run_every: int = 1
    max_iter: int = 1000
    tol: float = 1e-6
    damping: float = 0.0
    neg_inf: Optional[float] = None

    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    # internal
    _theta: Optional[torch.Tensor] = None
    _evidence_factory: Any = None
    _edges: Any = None
    _mu: Optional[torch.Tensor] = None
    _step: int = 0

    def reset(self, pb: Any) -> None:
        from bpropy.mrf.nmf_edges import NMFEdges
        from bpropy.mrf.naive_mean_field import naive_mean_field
        from bpropy.mrf.evidence_factory import EvidenceMaskFactory

        self._nmf = naive_mean_field

        fg = self.fg_builder(pb)
        self._theta = self.theta_builder(pb, fg, self.device, self.dtype)

        # NMFEdges has .to(device) in your implementation
        self._edges = NMFEdges(fg).to(self._theta.device)

        self._evidence_factory = EvidenceMaskFactory(
            fg, theta=self._theta, device=self._theta.device, neg_inf=self.neg_inf
        )

        self._mu = None
        self._step = 0

    def observe(self) -> Optional[torch.Tensor]:
        """
        Returns mu as (nV, 2) tensor.
        Only runs mean-field on schedule.
        """
        assert self._theta is not None

        if (self._mu is None) or (self.run_every <= 1) or (self._step % self.run_every == 0):
            th = self._evidence_factory.apply(self._theta)

            sol = self._nmf(
                th,
                self._edges,
                max_iter=self.max_iter,
                tol=self.tol,
                damping=self.damping,
            )

            # naive_mean_field returns flat (2*nV,) in your code; reshape to (nV,2)
            mu = sol.mu.reshape(-1, 2)
            mu = torch.nan_to_num(mu, nan=0.0)

            self._mu = mu
        return self._mu

    def update(self, evidence: Evidence) -> None:
        if evidence:
            self._evidence_factory.add_evidence(evidence, strict=True)
        self._step += 1

