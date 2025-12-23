# bpropy/co/base_lp.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Literal, List, ClassVar

import numpy as np
from scipy.optimize import LinearConstraint

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

from bpropy.co.solution import Solution

@dataclass
class LPSolveConfig:
    solver: str = "scipy"              # "scipy" | "cbc" | "cplex" | ...
    relax_integrality: bool = False
    verbose: bool = False


class LPProblem(ABC):
    """
    Base class for ILP/LP-like problems.

    Subclasses provide an ILP view via:
      - objective_coeffs() and objective_sense()
      - linear_constraints_matrix(): (A, lb, ub) encoding lb <= A x <= ub
      - variable_bounds(): (lbx, ubx) for box bounds (often [0,1]^n)

    This base class provides:
      - a generic feasibility check
      - a generic SciPy MILP solve fallback (optional)
      - a default batching strategy (block-diagonal) via BatchedLP
    """
    _VCGRAPH_REGISTRY: ClassVar[dict[str, type["LPProblem"]]] = {}


    # ---------- required ILP view ----------
    @abstractmethod
    def nv(self) -> int:
        """Number of decision variables."""

    @abstractmethod
    def linear_constraints_matrix(self) -> Tuple[object, np.ndarray, np.ndarray]:
        """
        Return (A, lb, ub) describing lb <= A @ x <= ub.

        A can be a numpy.ndarray or a scipy.sparse matrix.
        lb, ub are float arrays of length m (number of constraints).
        Use +/-np.inf for open bounds.
        """

    @abstractmethod
    def objective_coeffs(self) -> np.ndarray:
        """Return objective coefficients c (length nv)."""

    @property
    @abstractmethod
    def objective_sense(self) -> Literal["min", "max"]:
        """Objective sense: 'min' or 'max'."""

    # ---------- optional convenience / metadata ----------
    def variable_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Optional: return (lbx, ubx) each length nv.
        Use +/-np.inf for open bounds.
        Default: no explicit per-variable bounds check.
        """
        return None

    @property
    def num_subproblems(self) -> int:
        """Batch metadata: number of subproblems (default 1)."""
        return 1

    @property
    def var_sizes(self) -> np.ndarray:
        """Batch metadata: variable sizes per subproblem (default [nv])."""
        return np.array([self.nv()], dtype=int)

    # Backwards-compat helpers (keep if your adapters expect them)
    def weights(self) -> np.ndarray:
        """
        Backwards compat for code expecting `weights()`.

        By convention, we return coefficients in a *maximize* view.
        Prefer objective_coeffs/objective_sense in new code.
        """
        c = np.asarray(self.objective_coeffs(), dtype=float)
        return c if self.objective_sense == "max" else -c

    # ---------- helpers ----------
    def constraints(self) -> Sequence[LinearConstraint]:
        """Convenience wrapper returning a list with a single LinearConstraint(A, lb, ub)."""
        A, lb, ub = self.linear_constraints_matrix()
        return [LinearConstraint(A, lb=lb, ub=ub)]

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
        """
        n = self.nv()
        if n == 0:
            return True

        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != n:
            raise ValueError(f"x has size {x.size}, expected {n}")

        A, lb, ub = self.linear_constraints_matrix()

        if sp is not None and sp.issparse(A):
            Ax = (A @ x).astype(float)
            Ax = np.asarray(Ax).reshape(-1)
        else:
            A = np.asarray(A, dtype=float)
            Ax = A @ x

        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)

        if Ax.size != lb.size or Ax.size != ub.size:
            raise ValueError("Constraint dimension mismatch in (A,lb,ub).")

        lb_mask = np.isfinite(lb)
        ub_mask = np.isfinite(ub)

        lb_slack = np.full_like(Ax, np.inf)
        ub_slack = np.full_like(Ax, np.inf)
        lb_slack[lb_mask] = Ax[lb_mask] - lb[lb_mask]
        ub_slack[ub_mask] = ub[ub_mask] - Ax[ub_mask]

        feas = True
        if np.any(lb_slack[lb_mask] < -tol):
            feas = False
        if np.any(ub_slack[ub_mask] < -tol):
            feas = False

        var_lb_slack = var_ub_slack = None
        if check_var_bounds:
            vb = self.variable_bounds()
            if vb is not None:
                lbx, ubx = vb
                lbx = np.asarray(lbx, dtype=float).reshape(-1)
                ubx = np.asarray(ubx, dtype=float).reshape(-1)
                if lbx.size != n or ubx.size != n:
                    raise ValueError("variable_bounds size mismatch with nv.")
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

    # ---------- generic solve fallback ----------
    def solve(
        self,
        *,
        config: Optional[LPSolveConfig] = None,
        solver: Optional[str] = None,
        relax_integrality: Optional[bool] = None,
        verbose: Optional[bool] = None,
        # override hook for MIS(cliques=...)
        override_linear_constraints: Optional[Tuple[object, np.ndarray, np.ndarray]] = None,
    ) -> Solution:
        """
        Generic LP/MILP solve for problems that provide the LP view.

        Specialized problem classes (MIS/SCP/...) can implement special solvers,
        and delegate here for generic solvers.

        Parameters
        ----------
        solver:
            "scipy" | "python-mip" | "cbc" | "cplex" | ...
        relax_integrality:
            If True, solve the LP relaxation.
        verbose:
            Verbosity (passed to backend if supported).
        override_linear_constraints:
            Optional (A, lb, ub) to use instead of self.linear_constraints_matrix().
            Useful for MIS with custom cliques.

        Returns
        -------
        Solution
        """
        if config is None:
            config = LPSolveConfig()
        if solver is not None:
            config.solver = solver
        if relax_integrality is not None:
            config.relax_integrality = bool(relax_integrality)
        if verbose is not None:
            config.verbose = bool(verbose)

        slv = (config.solver or "scipy").lower()

        # --- SciPy backend ---
        if slv == "scipy":
            if override_linear_constraints is None:
                r = self.solve_scipy_milp(
                    relax_integrality=config.relax_integrality,
                    verbose=config.verbose,
                )
                return Solution.from_solver("scipy", r=r)
            else:
                # Use SciPy milp directly with override constraints.
                from scipy.optimize import milp, Bounds, LinearConstraint
                n = self.nv()
                c = np.asarray(self.objective_coeffs(), dtype=float).reshape(-1)
                c_min = c if self.objective_sense == "min" else -c

                integrality = np.zeros(n) if config.relax_integrality else np.ones(n)

                vb = self.variable_bounds()
                if vb is None:
                    lbx = np.full(n, -np.inf)
                    ubx = np.full(n, np.inf)
                else:
                    lbx, ubx = vb

                A, lb, ub = override_linear_constraints
                r = milp(
                    c_min,
                    integrality=integrality,
                    constraints=[LinearConstraint(A, lb=lb, ub=ub)],
                    bounds=Bounds(lbx, ubx),
                    options={"disp": config.verbose},
                )
                return Solution.from_solver("scipy", r=r)

        # --- python-mip family backend ---
        # allow "cbc" as alias; store solver_name accordingly
        if slv in {"python-mip", "cbc", "cplex", "gurobi", "grb"}:
            import mip

            # map alias -> mip solver_name if you like
            solver_name = None if slv in {"python-mip", "cbc"} else slv.upper()
            if solver_name is None:
                md = mip.Model("lp")
            else:
                md = mip.Model("lp", solver_name=solver_name)


            n = self.nv()
            c = np.asarray(self.objective_coeffs(), dtype=float).reshape(-1)

            vb = self.variable_bounds()
            if vb is None:
                lbx = np.full(n, -np.inf)
                ubx = np.full(n, np.inf)
            else:
                lbx, ubx = vb
                lbx = np.asarray(lbx, dtype=float).reshape(-1)
                ubx = np.asarray(ubx, dtype=float).reshape(-1)

            # choose var type: continuous if relaxed, binary otherwise
            var_type = mip.CONTINUOUS if config.relax_integrality else mip.BINARY
            x = [md.add_var(lb=float(lbx[i]), ub=float(ubx[i]), var_type=var_type) for i in range(n)]

            # objective
            expr = mip.xsum(float(c[i]) * x[i] for i in range(n))
            md.objective = mip.minimize(expr) if self.objective_sense == "min" else mip.maximize(expr)

            # constraints: lb <= A x <= ub
            if override_linear_constraints is None:
                A, lb, ub = self.linear_constraints_matrix()
            else:
                A, lb, ub = override_linear_constraints

            # build constraints row-wise (works for dense or sparse A)
            # (for sparse, you can use A.getrow(i) with scipy, but keep it simple here)
            A = A.toarray() if hasattr(A, "toarray") else np.asarray(A, dtype=float)
            lb = np.asarray(lb, dtype=float).reshape(-1)
            ub = np.asarray(ub, dtype=float).reshape(-1)

            for i in range(A.shape[0]):
                row = A[i]
                row_expr = mip.xsum(float(row[j]) * x[j] for j in range(n) if row[j] != 0.0)
                if np.isfinite(lb[i]):
                    md += row_expr >= float(lb[i])
                if np.isfinite(ub[i]):
                    md += row_expr <= float(ub[i])

            md.verbose = 1 if config.verbose else 0
            md.optimize(relax=config.relax_integrality)

            return Solution.from_solver("python-mip", x=x, md=md)

        raise ValueError(f"Unknown generic solver={config.solver!r}")

    def solve_scipy_milp(
        self,
        *,
        relax_integrality: bool = False,
        verbose: bool = False,
    ):
        """
        Generic SciPy MILP solver for problems that provide (A,lb,ub) and bounds.
        Returns the raw SciPy result object.

        Subclasses typically wrap this into a problem-specific Solution class.
        """
        from scipy.optimize import milp, Bounds

        n = self.nv()
        c = np.asarray(self.objective_coeffs(), dtype=float).reshape(-1)
        if c.size != n:
            raise ValueError("objective_coeffs size mismatch with nv.")

        # SciPy milp is a minimizer.
        c_min = c if self.objective_sense == "min" else -c

        integrality = np.zeros(n) if relax_integrality else np.ones(n)

        vb = self.variable_bounds()
        if vb is None:
            # fallback: free vars
            lbx = np.full(n, -np.inf)
            ubx = np.full(n, np.inf)
        else:
            lbx, ubx = vb

        return milp(
            c_min,
            integrality=integrality,
            constraints=self.constraints(),
            bounds=Bounds(lbx, ubx),
            options={"disp": verbose},
        )

    # ---------- default batching ----------
    @classmethod
    def batch(cls, problems: Iterable["LPProblem"]) -> "LPProblem":
        """
        Default batching: returns a BatchedLP (block-diagonal constraints).
        Individual problem classes may override for more efficient native batching.
        """
        from bpropy.co.batched_lp import BatchedLP
        return BatchedLP(list(problems))

    # ---------- existing helpers you already used ----------
    def to_vc_graph(self):
        from bpropy.graphs.vc_graph import VCGraph
        return VCGraph(self)

    def to_data(self, *, with_lbp=False, with_nmf=False, y=None, hetero=False):
        from bpropy.adapters.vc_to_data import problem_to_data
        return problem_to_data(self, with_lbp=with_lbp, with_nmf=with_nmf, y=y, hetero=hetero)

    @classmethod
    def from_data(cls, data):
        from bpropy.adapters.vc_to_data import data_to_lp
        return data_to_lp(data)

    @classmethod
    def register_vcgraph_type(cls, name: str, problem_cls: type["LPProblem"]) -> None:
        cls._VCGRAPH_REGISTRY[name] = problem_cls

    @classmethod
    def from_vcgraph(cls, vcg):
        ptype = vcg.graph.get("problem_type", None)
        if ptype is None:
            raise ValueError("VCGraph missing graph['problem_type']; cannot infer LPProblem subclass.")

        pb_cls = cls._VCGRAPH_REGISTRY.get(ptype, None)
        if pb_cls is None:
            raise ValueError(f"Unknown problem_type={ptype!r}. Register it with LPProblem.register_vcgraph_type().")

        # convention: the subclass implements from_vcgraph
        if not hasattr(pb_cls, "from_vcgraph"):
            raise ValueError(f"{pb_cls.__name__} does not implement from_vcgraph(vcg).")

        return pb_cls.from_vcgraph(vcg)

