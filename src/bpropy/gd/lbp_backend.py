from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple, List

import numpy as np
import torch

from bpropy.mrf.lbp_edges import LBPEdges
from bpropy.mrf.lbp import loopy_belief_propagation
from bpropy.mrf.evidence_factory import EvidenceMaskFactory


Evidence = Dict[int, int]

@dataclass
class LBPBackend:
    fg_builder: Callable[[Any], Any]
    theta_builder: Callable[[Any, Any, torch.device, torch.dtype], torch.Tensor]

    run_every: int = 1
    max_iter: int = 1000
    tol: float = 1e-6
    damping: float = 0.5
    neg_inf: Optional[float] = None
    warm_start: bool = True
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    _theta: Optional[torch.Tensor] = None
    _evidence_factory: Any = None
    _edges: Any = None
    _v2f: Optional[torch.Tensor] = None
    _mu: Optional[torch.Tensor] = None
    _step: int = 0

    def reset(self, pb: Any) -> None:

        self._loopy_bp = loopy_belief_propagation

        fg = self.fg_builder(pb)
        theta = self.theta_builder(pb, fg, self.device, self.dtype)

        if torch.is_tensor(theta):
            theta = theta.detach().clone()
        self._theta = theta

        self._edges = LBPEdges(fg)
        self._evidence_factory = EvidenceMaskFactory(
            fg, theta=self._theta, device=self._theta.device, neg_inf=self.neg_inf
        )

        n_msg = int(self._edges.n_f2v_msg.sum().item())
        self._v2f = torch.zeros(n_msg, device=self._theta.device, dtype=self._theta.dtype)

        self._mu = None
        self._step = 0

    def observe(self) -> Optional[torch.Tensor]:
        assert self._theta is not None

        if (self._mu is None) or (self.run_every <= 1) or (self._step % self.run_every == 0):
            #th = self._evidence_factory.apply(self._theta)
            th = self._evidence_factory.apply_inplace(self._theta)

            if not self.warm_start:
                self._v2f.zero_()
            res = self._loopy_bp(
                th, self._v2f, self._edges,
                max_iter=self.max_iter, tol=self.tol, damping=self.damping
            )

            if torch.isnan(res.mu_v).any():
                self._mu = torch.nan_to_num(res.mu_v.reshape(-1, 2), nan=0.0)
            else:
                self._mu = res.mu_v.reshape(-1, 2)
        return self._mu

    def update(self, evidence: Evidence) -> None:
        if evidence:
            self._evidence_factory.add_evidence(evidence, strict=True)
        self._step += 1

