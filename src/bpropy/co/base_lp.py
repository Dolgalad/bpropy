# bpropy/co/base_lp.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Iterable, Sequence, Optional
from scipy.optimize import LinearConstraint

try:
    import scipy.sparse as sp
except Exception:
    sp = None

@dataclass
class LPSolveConfig:
    solver: str = "cbc"            # "cbc" | "scipy" | "cplex"
    relax_integrality: bool = False
    disp: bool = False

class LPProblem(ABC):
    """LP-like problems that can expose a variable–constraint (VC) graph."""
    # --- required minimal API ---
    @abstractmethod
    def nv(self) -> int: ...
    @abstractmethod
    def constraints(self) -> Sequence[LinearConstraint]: ...
    @abstractmethod
    def weights(self) -> np.ndarray: ...
    # --- optional batching meta (defaults = single graph) ---
    def num_graphs(self) -> int: return 1
    def graph_sizes(self) -> np.ndarray: return np.array([self.nv()])
    # --- generic helpers (shared across MIS/SCP) ---
    def to_vc_graph(self):
        from bpropy.graphs.vc_graph import VCGraph
        return VCGraph(self)
    def to_data(self, *, with_lbp=False, with_nmf=False, y=None, hetero=False):
        from bpropy.adapters.vc_to_data import problem_to_data
        return problem_to_data(self, with_lbp=with_lbp, with_nmf=with_nmf, y=y, hetero=hetero)
    @classmethod
    def from_data(cls, data):
        from bpropy.adapters.vc_to_data import data_to_lp
        return data_to_lp(data)  # returns an LPProblem subclass instance
    @classmethod
    def batch(cls, problems: Iterable["LPProblem"]) -> "LPProblem":
        from bpropy.adapters.vc_to_data import batch_lp
        return batch_lp(problems)

    #def check_solution(self, x, tol=1e-8):
    #    print("x.shape, ", x.shape)
    #    for c in self.constraints():
    #        out = c.A @ x
    #        if c.lb > out + tol or c.ub < out - tol:
    #            print(c.lb , out, c.ub)
    #            print(c.lb > out, c.ub < out)
    #            print(c.A)
    #            print(x, x.shape, flush=True)
    #            print("AAAAAAAAAAAA")

    #            return False
    #    return True

    def linear_constraints_matrix(self) -> Tuple[object, np.ndarray, np.ndarray]:
        """
        Return (A, lb, ub) describing Ax in [lb, ub].
        A can be numpy.ndarray (ne, nv) or a scipy.sparse matrix.
        lb, ub are 1-D float arrays of length ne.
        Use +/-np.inf for unbounded sides.
        """
        raise NotImplementedError

    def variable_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Optional: return (lbx, ubx), each 1-D length nv, or None to skip.
        Use +/-np.inf for unbounded sides.
        Default: no explicit per-variable bounds check.
        """
        return None

    # --- Generic feasibility check implemented once here:

    def check_solution(
        self,
        x: np.ndarray,
        *,
        tol: float = 1e-9,
        return_slack: bool = False,
        check_var_bounds: bool = True,
    ):
        """
        Generic feasibility: lb <= A @ x <= ub (and optionally lbx <= x <= ubx).
        Works for binary or fractional x.

        Parameters
        ----------
        x : array-like, shape (nv,)
        tol : float
            Numerical tolerance for inequalities.
        return_slack : bool
            If True, return (is_feasible, (lb_slack, ub_slack, var_lb_slack, var_ub_slack))
        check_var_bounds : bool
            If True and variable_bounds() is provided, also enforce lbx <= x <= ubx.

        Returns
        -------
        bool or (bool, tuple-of-slacks)
        """
        if self.nv()==0:
            return True
        x = np.asarray(x, dtype=float).reshape(-1)

        A, lb, ub = self.linear_constraints_matrix()
        # Support dense or sparse A
        if sp is not None and sp.issparse(A):
            Ax = A @ x
            Ax = np.asarray(Ax, dtype=float).reshape(-1)
        else:
            A = np.asarray(A, dtype=float)
            Ax = A @ x

        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)

        if Ax.shape[0] != lb.shape[0] or Ax.shape[0] != ub.shape[0]:
            raise ValueError(f"Ax has length {Ax.shape[0]}, lb {lb.shape[0]}, ub {ub.shape[0]}")

        # Treat +/-inf as open bounds; compare with tolerance elsewhere
        lb_mask = np.isfinite(lb)
        ub_mask = np.isfinite(ub)

        # Slacks (positive means satisfied)
        lb_slack = np.full_like(Ax, np.inf)
        ub_slack = np.full_like(Ax, np.inf)
        lb_slack[lb_mask] = Ax[lb_mask] - lb[lb_mask]
        ub_slack[ub_mask] = ub[ub_mask] - Ax[ub_mask]

        feas = True
        if np.any(lb_slack[lb_mask] < -tol):  # Ax < lb - tol
            feas = False
        if np.any(ub_slack[ub_mask] < -tol):  # Ax > ub + tol
            feas = False

        var_lb_slack = var_ub_slack = None
        if check_var_bounds:
            vb = self.variable_bounds()
            if vb is not None:
                lbx, ubx = vb
                lbx = np.asarray(lbx, dtype=float).reshape(-1)
                ubx = np.asarray(ubx, dtype=float).reshape(-1)
                if lbx.size != x.size or ubx.size != x.size:
                    raise ValueError("variable_bounds() size mismatch with x")
                v_lb_mask = np.isfinite(lbx)
                v_ub_mask = np.isfinite(ubx)
                var_lb_slack = np.full_like(x, np.inf)
                var_ub_slack = np.full_like(x, np.inf)
                var_lb_slack[v_lb_mask] = x[v_lb_mask] - lbx[v_lb_mask]
                var_ub_slack[v_ub_mask] = ubx[v_ub_mask] - x[v_ub_mask]
                if np.any(var_lb_slack[v_lb_mask] < -tol) or np.any(var_ub_slack[v_ub_mask] < -tol):
                    feas = False

        if return_slack:
            return feas, (lb_slack, ub_slack, var_lb_slack, var_ub_slack)
        return feas
