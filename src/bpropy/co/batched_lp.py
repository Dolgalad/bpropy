# bpropy/co/batched_lp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal

import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

from bpropy.co.base_lp import LPProblem


@dataclass
class BatchedLP(LPProblem):
    """
    Generic batched LP/ILP via variable concatenation and block-diagonal constraints.

    This is the default batching strategy for any LPProblem.
    """

    problems: List[LPProblem]

    def __post_init__(self):
        if len(self.problems) == 0:
            raise ValueError("BatchedLP requires at least one problem.")
        # Require a consistent objective sense across the batch
        sense = self.problems[0].objective_sense
        if any(p.objective_sense != sense for p in self.problems):
            raise ValueError("All problems in a BatchedLP must share the same objective_sense.")

        self._var_sizes = np.array([p.nv() for p in self.problems], dtype=int)
        self._offsets = np.concatenate([[0], np.cumsum(self._var_sizes)[:-1]])

    @property
    def objective_sense(self) -> Literal["min", "max"]:
        return self.problems[0].objective_sense

    def nv(self) -> int:
        return int(self._var_sizes.sum())

    @property
    def num_subproblems(self) -> int:
        return len(self.problems)

    @property
    def var_sizes(self) -> np.ndarray:
        return self._var_sizes.copy()

    @property
    def var_offsets(self) -> np.ndarray:
        """Starting index of each subproblem block in the concatenated x."""
        return self._offsets.copy()

    def objective_coeffs(self) -> np.ndarray:
        return np.concatenate([np.asarray(p.objective_coeffs(), dtype=float).reshape(-1) for p in self.problems])

    def variable_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        lbs = []
        ubs = []
        for p in self.problems:
            vb = p.variable_bounds()
            if vb is None:
                lbs.append(np.full(p.nv(), -np.inf))
                ubs.append(np.full(p.nv(), np.inf))
            else:
                lbx, ubx = vb
                lbs.append(np.asarray(lbx, dtype=float).reshape(-1))
                ubs.append(np.asarray(ubx, dtype=float).reshape(-1))
        return np.concatenate(lbs), np.concatenate(ubs)

    def linear_constraints_matrix(self):
        As = []
        lbs = []
        ubs = []
        for p in self.problems:
            A, lb, ub = p.linear_constraints_matrix()
            As.append(A)
            lbs.append(np.asarray(lb, dtype=float).reshape(-1))
            ubs.append(np.asarray(ub, dtype=float).reshape(-1))

        lb = np.concatenate(lbs) if lbs else np.zeros(0, dtype=float)
        ub = np.concatenate(ubs) if ubs else np.zeros(0, dtype=float)

        # Build block diagonal A
        if sp is not None:
            As2 = []
            for A in As:
                if sp.issparse(A):
                    As2.append(A.tocsr())
                else:
                    As2.append(sp.csr_matrix(np.asarray(A, dtype=float)))
            Ablk = sp.block_diag(As2, format="csr")
            return Ablk, lb, ub

        # Dense fallback
        blocks = [np.asarray(A, dtype=float) for A in As]
        if len(blocks) == 0:
            return np.zeros((0, self.nv()), dtype=float), lb, ub

        m_total = sum(B.shape[0] for B in blocks)
        Ablk = np.zeros((m_total, self.nv()), dtype=float)
        row = 0
        for i, B in enumerate(blocks):
            m, n = B.shape
            col = int(self._offsets[i])
            Ablk[row:row + m, col:col + n] = B
            row += m
        return Ablk, lb, ub

    def split_assignment(self, x: np.ndarray) -> List[np.ndarray]:
        x = np.asarray(x).reshape(-1)
        if x.size != self.nv():
            raise ValueError("x size mismatch.")
        out = []
        for off, n in zip(self._offsets, self._var_sizes):
            out.append(x[int(off):int(off + n)])
        return out

    def merge_assignments(self, xs: List[np.ndarray]) -> np.ndarray:
        if len(xs) != len(self.problems):
            raise ValueError("Wrong number of blocks.")
        return np.concatenate([np.asarray(v).reshape(-1) for v in xs])

