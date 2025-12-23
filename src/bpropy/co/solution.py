from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional, Dict, ClassVar
import numpy as np


Adapter = Callable[..., "Solution"]


@dataclass(slots=True)
class Solution:
    x: np.ndarray
    duals: Optional[np.ndarray] = None
    slacks: Optional[np.ndarray] = None
    varbasis_onehot: Optional[np.ndarray] = None
    reduced_costs: Optional[np.ndarray] = None
    solve_time: Optional[float] = None

    status: Optional[str] = None
    objective: Optional[float] = None
    solver: Optional[str] = None
    raw: Any = None

    # ---- adapter registry ----
    _ADAPTERS: ClassVar[Dict[str, Adapter]] = {}

    @classmethod
    def register_adapter(cls, name: str) -> Callable[[Adapter], Adapter]:
        """Decorator to register a solver adapter by name."""
        key = name.lower()

        def deco(fn: Adapter) -> Adapter:
            cls._ADAPTERS[key] = fn
            return fn

        return deco

    @classmethod
    def from_solver(cls, solver: str, /, **kwargs) -> "Solution":
        """Explicit construction: Solution.from_solver('cplex', sol=..., md=...)."""
        key = solver.lower()
        if key not in cls._ADAPTERS:
            raise KeyError(
                f"Unknown solver adapter '{solver}'. "
                f"Known: {sorted(cls._ADAPTERS.keys())}"
            )
        return cls._ADAPTERS[key](**kwargs)

    @classmethod
    def from_any(cls, result: Any, md: Any = None, solver: Optional[str] = None, **kwargs) -> "Solution":
        """
        Best-effort construction.
        - If solver is given, dispatch to that adapter.
        - Otherwise, try to infer from 'result' shape/attributes.
        """
        if solver is not None:
            return cls.from_solver(solver, result=result, md=md, **kwargs)

        # 1) SciPy milp-like: has attribute 'x'
        if hasattr(result, "x"):
            return cls.from_solver("scipy", r=result)

        # 2) python-mip model-like: md has .vars/.constrs (often you pass md)
        if md is not None and hasattr(md, "vars") and hasattr(md, "constrs"):
            # some code paths pass x explicitly; accept both
            return cls.from_solver("python-mip", x=getattr(result, "x", None), md=md)

        # 3) CPLEX docplex solution-like: typically sol + md needed, so inference is hard.
        raise ValueError(
            "Could not infer solver type. Pass solver='cplex'/'scipy'/'python-mip'."
        )


# ----------------- adapters -----------------

@Solution.register_adapter("scipy")
def _from_scipy_milp(r) -> Solution:
    return Solution(x=np.asarray(r.x, dtype=float), raw=r, solver="scipy")


@Solution.register_adapter("cplex")
def _from_cplex(sol, md) -> Solution:
    n = md.number_of_variables
    varlist = [md.get_var_by_name(f"x{i}") for i in range(n)]

    varbasis = np.array([bstt.value for bstt in md.var_basis_statuses(varlist)], dtype=int)
    varbasis_onehot = np.zeros((varbasis.size, 4), dtype=float)
    varbasis_onehot[np.arange(varbasis.size), varbasis] = 1.0

    x = np.array([sol.get_value(var) for var in varlist], dtype=float)
    reduced_costs = np.array(sol.get_reduced_costs(varlist), dtype=float)

    constraints = list(md.iter_constraints())
    slacks = np.array(md.slack_values(constraints), dtype=float)
    duals = np.array(md.dual_values(constraints), dtype=float)

    return Solution(
        x=x,
        duals=duals,
        slacks=slacks,
        varbasis_onehot=varbasis_onehot,
        reduced_costs=reduced_costs,
        solver="cplex",
        raw=sol,
    )


@Solution.register_adapter("python-mip")
def _from_python_mip(x, md) -> Solution:
    n = len(md.vars)
    m = len(md.constrs)

    varbasis = np.zeros(n, dtype=int)
    xv = np.zeros(n, dtype=float)
    rc = np.zeros(n, dtype=float)
    duals = np.zeros(m, dtype=float)
    slacks = np.zeros(m, dtype=float)

    for v in md.vars:
        if v.rc == 0:
            varbasis[v.idx] = 1
        else:
            if v.x == v.lb:
                varbasis[v.idx] = 0
            elif v.x == v.ub:
                varbasis[v.idx] = 2
            else:
                varbasis[v.idx] = 3
        xv[v.idx] = v.x
        rc[v.idx] = v.rc

    for c in md.constrs:
        duals[c.idx] = c.pi
        slacks[c.idx] = c.slack

    varbasis_onehot = np.zeros((varbasis.size, 4), dtype=float)
    varbasis_onehot[np.arange(varbasis.size), varbasis] = 1.0

    return Solution(
        x=xv,
        duals=duals,
        slacks=slacks,
        varbasis_onehot=varbasis_onehot,
        reduced_costs=rc,
        raw=md,
        solver="python-mip:"+md.solver_name.lower(),
    )

