from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from bpropy.gd.inference_backend import InferenceBackend


@dataclass
class ScoreBackend(InferenceBackend):
    logits: Optional[torch.Tensor] = None
    evidence: Optional[Dict[int, int]] = None  # var -> {0,1}

    def reset(self, pb: Any) -> None:
        # pb unused for score-only backend, but kept for interface
        self.evidence = {}

    def set_logits(self, logits: torch.Tensor) -> None:
        self.logits = logits

    def update(self, evidence: Dict[int, int]) -> None:
        if self.evidence is None:
            self.evidence = {}
        self.evidence.update({int(k): int(v) for k, v in evidence.items()})

    def observe(self) -> torch.Tensor:
        if self.logits is None:
            raise RuntimeError("ScoreBackend.observe called before set_logits().")

        # logits -> P(x=1)
        p1 = torch.sigmoid(self.logits)

        mu_v = torch.stack([1.0 - p1, p1], dim=-1)  # [n,2]

        # clamp evidence
        if self.evidence:
            for i, v in self.evidence.items():
                mu_v[i, 0] = 1.0 if v == 0 else 0.0
                mu_v[i, 1] = 1.0 if v == 1 else 0.0

        return mu_v

