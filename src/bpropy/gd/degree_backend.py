from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import torch

@dataclass
class DegreeBackend:
    """
    No-inference backend for MIS.
    Returns a (n,2) 'mu-like' tensor whose second column is a score.

    We use score = -deg_active(v) so that MISAdapter's argmax chooses min-degree.
    """
    recompute_each_step: bool = True  # degrees change as nodes are removed

    pb: Any = None
    active: Optional[np.ndarray] = None
    _step: int = 0

    def reset(self, pb) -> None:
        self.pb = pb
        n = int(pb.nv()) if hasattr(pb, "nv") else int(pb.n)
        self.active = np.ones(n, dtype=bool)
        self._step = 0

    def observe(self) -> torch.Tensor:
        assert self.pb is not None and self.active is not None
        n = self.active.size

        # compute degree restricted to currently active nodes
        deg = np.zeros(n, dtype=np.int32)
        active_set = set(np.flatnonzero(self.active))
        for v in active_set:
            # neighbors among active nodes only
            deg[v] = sum((int(u) in active_set) for u in self.pb.g.neighbors(int(v)))

        # mu-like: second column is score
        mu = torch.zeros((n, 2), dtype=torch.float32)
        mu[:, 1] = torch.from_numpy(-deg.astype(np.float32))  # argmax -> min-degree
        return mu

    def update(self, evidence: dict[int, int]) -> None:
        # Keep the backend's active set in sync with the adapter via evidence:
        # any variable fixed (0 or 1) is no longer active.
        if self.active is None:
            return
        for v in evidence.keys():
            self.active[int(v)] = False
        self._step += 1

