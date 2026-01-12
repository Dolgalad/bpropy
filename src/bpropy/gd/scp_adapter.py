from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _get_costs(pb, n_sets: int) -> np.ndarray:
    """Try to pull set costs/weights from pb; default to 1."""
    for name in ("w", "cost", "costs", "weights", "c"):
        if hasattr(pb, name):
            c = np.asarray(getattr(pb, name), dtype=float).reshape(-1)
            if c.size == n_sets:
                return c
    return np.ones(n_sets, dtype=float)

def _extract_incidence(pb) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    SCP incidence extraction for bpropy.co.scp.SCP:

      - pb.F: list of sets, each is a list of covered element indices
      - pb.ne(): number of elements (constraints)
      - pb.nv(): number of sets (variables)

    Returns:
      S[j] = np.array(elements covered by set j)
      C[i] = np.array(sets covering element i)
    """
    # Prefer SCP's native representation
    if hasattr(pb, "F") and callable(getattr(pb, "ne", None)) and callable(getattr(pb, "nv", None)):
        n_sets = int(pb.nv())
        n_elems = int(pb.ne())

        S = [np.asarray(pb.F[j], dtype=int) for j in range(n_sets)]

        tmp = [[] for _ in range(n_elems)]
        for j, elems in enumerate(S):
            for i in elems:
                ii = int(i)
                if 0 <= ii < n_elems:
                    tmp[ii].append(int(j))
                else:
                    raise ValueError(f"SCP incidence: element id {ii} out of range [0,{n_elems})")
        C = [np.asarray(t, dtype=int) for t in tmp]
        return S, C

    # Fallback to generic LPProblem view: use the constraint matrix Ax >= 1
    if hasattr(pb, "linear_constraints_matrix"):
        A, lb, ub = pb.linear_constraints_matrix()
        # A is shape (m, n) = (ne, nv)
        # Build S from columns. This works for dense or sparse.
        if hasattr(A, "tocsc"):  # scipy sparse
            Ac = A.tocsc()
            m, n = Ac.shape
            S = []
            for j in range(n):
                start, end = Ac.indptr[j], Ac.indptr[j + 1]
                S.append(Ac.indices[start:end].astype(int))
        else:
            A = np.asarray(A)
            if A.ndim != 2:
                raise ValueError(f"Expected 2D constraint matrix, got shape {A.shape}")
            m, n = A.shape
            S = [np.flatnonzero(A[:, j]).astype(int) for j in range(n)]

        tmp = [[] for _ in range(m)]
        for j, elems in enumerate(S):
            for i in elems:
                tmp[int(i)].append(int(j))
        C = [np.asarray(t, dtype=int) for t in tmp]
        return S, C

    raise AttributeError("SCPAdapter: cannot extract incidence (expected SCP with F/ne/nv or LPProblem with linear_constraints_matrix).")

class SCPAdapter:
    """
    Adapter for SCP decimation guided by an inference backend.

    State:
      - active_sets[j] indicates if set-variable j is still undecided
      - uncovered[i] indicates if element i is not yet covered
      - x[j] ∈ {-1,0,1} set decision

    Step rule:
      - choose an active set j that maximizes (mu[j,1] / cost[j]) by default
      - set x[j]=1, mark its covered elements as covered
      - stop when all elements are covered; remaining sets become 0 in solution()
    """

    def __init__(self, normalize_by_cost: bool = True):
        self.normalize_by_cost = normalize_by_cost

        self.pb: Any = None
        self.S: List[np.ndarray] = []
        self.C: List[np.ndarray] = []
        self.cost: Optional[np.ndarray] = None

        self.active_sets: Optional[np.ndarray] = None
        self.uncovered: Optional[np.ndarray] = None
        self.x: Optional[np.ndarray] = None

    def reset(self, pb: Any) -> None:
        self.pb = pb
        self.S, self.C = _extract_incidence(pb)
        n_sets = len(self.S)
        n_elems = len(self.C)
        self.cost = _get_costs(pb, n_sets)

        self.active_sets = np.ones(n_sets, dtype=bool)
        self.uncovered = np.ones(n_elems, dtype=bool)
        self.x = -np.ones(n_sets, dtype=int)

    def is_done(self) -> bool:
        assert self.uncovered is not None
        return not self.uncovered.any()

    def step(self, backend_output: Any) -> Tuple[Dict[int, int], Dict[str, Any]]:
        """
        backend_output expected: mu shaped (n_sets,2) or (2*n_sets,) convertible.
        Uses mu[:,1] as "probability/score of selecting set".
        """
        assert self.active_sets is not None and self.uncovered is not None and self.x is not None
        assert self.cost is not None

        mu = backend_output
        if mu is None:
            raise ValueError("SCPAdapter expects mu from an inference backend.")

        # Convert to (n_sets,2) numpy
        if hasattr(mu, "detach"):
            mu_np = mu.detach().cpu().numpy()
        else:
            mu_np = np.asarray(mu)
        mu_np = mu_np.reshape(-1, 2)

        scores = np.nan_to_num(mu_np[:, 1].astype(float), nan=0.0)

        # Mask inactive sets
        active_idx = np.flatnonzero(self.active_sets)
        if active_idx.size == 0:
            # No sets left; fail-safe: stop (solution() will fill 0s)
            return {}, {"chosen": None, "reason": "no_active_sets"}

        if self.normalize_by_cost:
            denom = np.maximum(self.cost, 1e-12)
            eff = scores / denom
        else:
            eff = scores

        # entropy
        entropy = -eff * np.log(eff+1e-12) - (1.0 - eff) * np.log(1.0 - eff+1e-12)

        #jj = int(active_idx[np.argmax(eff[active_idx])])
        #jj_val = 1
        jj = int(active_idx[np.argmin(entropy[active_idx])])
        jj_val = np.argmax(mu_np[jj])


        # Commit: choose set j
        #evidence = {j: 1}
        #self.x[j] = 1
        #self.active_sets[j] = False
        evidence = {jj: jj_val}
        self.x[jj] = int(jj_val)
        self.active_sets[jj] = False

        # Update coverage
        newly = []
        if jj_val == 1:
            for i in self.S[jj]:
                i = int(i)
                if self.uncovered[i]:
                    self.uncovered[i] = False
                    newly.append(i)
        else:
            # check for items covered by S[j] covered by only 1 active set
            active_idx = np.flatnonzero(self.active_sets)
            for a in self.S[jj]:
                a = int(a)
                if self.uncovered[a]:
                    sets_covering_a = []
                    for s_idx in active_idx:
                        if a in self.S[s_idx]:
                            sets_covering_a.append(s_idx)
                    if len(sets_covering_a)==1:
                        s_idx = int(sets_covering_a[0])
                        self.uncovered[a] = False
                        newly.append(a)
                        evidence[s_idx] = 1
                        self.x[s_idx] = 1
                        self.active_sets[s_idx] = False
                        for ii in self.S[s_idx]:
                            ii = int(ii)
                            if self.uncovered[ii]:
                                self.uncovered[ii] = False
                                newly.append(ii)

        info = {
            "chosen": jj,
            "mu1": float(scores[jj]),
            "cost": float(self.cost[jj]),
            "newly_covered": newly,
            "n_uncovered": int(self.uncovered.sum()),
        }
        return evidence, info

    def solution(self) -> np.ndarray:
        """Undecided -> 0, and also set any remaining active sets to 0."""
        assert self.x is not None
        x = self.x.copy()
        x[x < 0] = 0
        return x



