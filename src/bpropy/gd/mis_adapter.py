from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple, List

import numpy as np
import torch

import numpy as np

class MISAdapter:
    """
    Decisions based ONLY on pb.g (original MIS graph).
    Evidence semantics: x[v]=1 implies x[u]=0 for all u in N(v).
    """
    def __init__(self):
        self.pb = None
        self.active = None   # boolean mask or set
        self.x = None

    def reset(self, pb):
        self.pb = pb
        n = int(pb.nv()) if hasattr(pb, "nv") else int(pb.n)
        self.active = np.ones(n, dtype=bool)
        self.x = -np.ones(n, dtype=int)   # -1 = undecided, 0/1 decided

    def is_done(self) -> bool:
        return not self.active.any()

    def step(self, backend_output):
        """
        backend_output expected: mu shaped (n,2) or (2n,) convertible.
        If you want a no-inference MIS heuristic later, you can ignore mu here.
        """
        mu = backend_output
        if mu is None:
            raise ValueError("MISAdapter currently expects mu from an inference backend.")

        if hasattr(mu, "reshape"):
            mu = mu.reshape(-1, 2)
        scores = np.asarray(mu[:, 1].detach().cpu() if hasattr(mu, "detach") else mu[:, 1])
        scores = np.nan_to_num(scores, nan=0.0)

        # pick best active node
        active_idx = np.flatnonzero(self.active)
        u = int(active_idx[np.argmax(scores[active_idx])])

        evidence = {u: 1}
        for v in self.pb.g.neighbors(u):
            v = int(v)
            if self.active[v]:
                evidence[v] = 0

        # commit evidence in adapter state
        for v, val in evidence.items():
            self.x[v] = val
            self.active[v] = False

        return evidence, {"chosen": u, "score": float(scores[u])}

    def solution(self):
        # undecided -> 0 (or leave as -1 if you prefer)
        x = self.x.copy()
        x[x < 0] = 0
        return x

