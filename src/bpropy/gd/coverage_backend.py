from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Dict
import numpy as np
import torch


@dataclass
class CoverageBackend:
    """
    No-inference backend for SCP (classic greedy set cover).
    Returns a (n_sets,2) 'mu-like' tensor whose second column is a greedy score.

      score(j) = newly_covered(j) / cost(j)

    Adapter should pick argmax(score).

    Notes:
      - We track uncovered elements internally.
      - evidence is expected to include chosen set decisions {j: 1} (and optionally {j:0}).
    """
    pb: Any = None
    uncovered: Optional[np.ndarray] = None  # length pb.ne()
    active_sets: Optional[np.ndarray] = None  # length pb.nv()
    _step: int = 0

    def reset(self, pb) -> None:
        self.pb = pb
        m = int(pb.ne())
        n = int(pb.nv())
        self.uncovered = np.ones(m, dtype=bool)
        self.active_sets = np.ones(n, dtype=bool)
        self._step = 0

    def observe(self) -> torch.Tensor:
        assert self.pb is not None and self.uncovered is not None and self.active_sets is not None
        n = self.active_sets.size

        # compute gain for each set: number of uncovered elements it would newly cover
        gains = np.zeros(n, dtype=np.float32)
        for j in np.flatnonzero(self.active_sets):
            elems = self.pb.F[int(j)]
            if len(elems) == 0:
                continue
            gains[int(j)] = float(np.sum(self.uncovered[np.asarray(elems, dtype=int)]))

        # divide by cost (weights)
        w = np.asarray(self.pb.w, dtype=np.float32).reshape(-1)
        denom = np.maximum(w, 1e-12)
        scores = gains / denom

        mu = torch.zeros((n, 2), dtype=torch.float32)
        mu[:, 1] = torch.from_numpy(scores)
        return mu

    def update(self, evidence: Dict[int, int]) -> None:
        """
        Sync backend state with adapter via evidence:
          - any fixed variable (0 or 1) becomes inactive
          - if evidence sets x[j]=1, we mark elements in F[j] as covered
        """
        if self.pb is None or self.uncovered is None or self.active_sets is None:
            return

        for j, val in evidence.items():
            j = int(j)
            self.active_sets[j] = False
            if int(val) == 1:
                for a in self.pb.F[j]:
                    self.uncovered[int(a)] = False

        self._step += 1

