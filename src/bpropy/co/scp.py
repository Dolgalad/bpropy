# bpropy/co/scp.py
from __future__ import annotations

from typing import Dict, Any, List, Optional, Literal
import numpy as np
import mip
from mip import xsum, BINARY
from scipy.optimize import milp, LinearConstraint, Bounds

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

from bpropy.co.base_lp import LPProblem
from bpropy.io.serde import SerializableLPMixin
from bpropy.co.scp_solution import SCPSolution


def number_containing_sets(a, F):
    return sum([a in f for f in F])


def sort_covering(A, F):
    return sorted(A, key=lambda a: number_containing_sets(a, F))


class SCP(LPProblem, SerializableLPMixin):
    """
    Set Covering Problem (SCP):
      minimize   w^T x
      s.t.       A x >= 1   (each element covered at least once)
                 0 <= x <= 1, x binary (unless relaxed)
    """

    def __init__(self, A, F, weights=None):
        self.A = sort_covering(A, F)
        self.F = [sorted(dict.fromkeys(map(int, f))) for f in F]
        self.w = np.ones(len(self.F), dtype=float) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
        if self.w.size != len(self.F):
            raise ValueError("weights length must match number of sets |F|")

    @property
    def objective_sense(self) -> Literal["min", "max"]:
        return "min"

    def objective_coeffs(self) -> np.ndarray:
        return self.w

    def ne(self) -> int:
        return len(self.A)

    def nv(self) -> int:
        return len(self.F)

    def variable_bounds(self):
        n = self.nv()
        return np.zeros(n), np.ones(n)

    def adj_vec(self, f):
        return np.array([a in f for a in self.A], dtype=bool)

    def adj_matrix(self):
        # shape (ne, nv)
        return np.stack([self.adj_vec(f) for f in self.F], axis=1).astype(int)

    def linear_constraints_matrix(self):
        A = self.adj_matrix().astype(float)  # (ne, nv)
        lb = np.ones(self.ne(), dtype=float)
        ub = np.full(self.ne(), np.inf, dtype=float)
        return A, lb, ub

    def constraints(self):
        # Prefer a single LinearConstraint with full matrix.
        A, lb, ub = self.linear_constraints_matrix()
        return [LinearConstraint(A, lb=lb, ub=ub)]

    def solve(self, disp: bool = False, solver: str = "cbc", relax_integrality: bool = False):
        if solver == "scipy":
            n = self.nv()
            integrality = np.zeros(n) if relax_integrality else np.ones(n)
            return SCPSolution.from_scipy_milp(
                milp(
                    self.w,  # SciPy minimizes => direct
                    integrality=integrality,
                    constraints=self.constraints(),
                    bounds=Bounds(np.zeros(n), np.ones(n)),
                    options={"disp": disp},
                )
            )

        # default: CBC via python-mip
        md = mip.Model("scp")
        x = [md.add_var(lb=0.0, ub=1.0, var_type=BINARY) for _ in range(self.nv())]
        md.objective = mip.minimize(xsum(self.w[i] * x[i] for i in range(self.nv())))
        for a in self.A:
            md += mip.xsum(x[i] for i, f in enumerate(self.F) if a in f) >= 1
        md.verbose = 0
        md.optimize(relax=relax_integrality)
        return SCPSolution.from_python_mip(x, md)

    # ---- serialization (keep your existing logic, but align names) ----
    def _to_serializable(self, use_npz: bool, arrays: dict, threshold_bytes: int = 200_000):
        payload = {"nv": self.nv(), "weights": {"source": "inline", "key": "w"}, "scp": {}}

        w = np.asarray(self.w, dtype=float).reshape(-1)
        if use_npz and w.nbytes >= threshold_bytes:
            arrays["w"] = w
            payload["weights"] = {"source": "npz", "key": "w"}
        else:
            payload.setdefault("_arrays", {})["w"] = w.tolist()

        ne, nv = self.ne(), self.nv()
        nnz = sum(len(f) for f in self.F)
        estimated_bytes = nnz * 8

        if use_npz and estimated_bytes >= threshold_bytes:
            # store CSR rows=items, cols=sets
            data, indices, indptr = [], [], [0]
            for a_idx in range(ne):
                col_idx = [j for j, f in enumerate(self.F) if a_idx in f]
                col_idx.sort()
                data.extend([1.0] * len(col_idx))
                indices.extend(col_idx)
                indptr.append(len(indices))
            arrays.update({
                "A_data": np.asarray(data, dtype=float),
                "A_indices": np.asarray(indices, dtype=np.int32),
                "A_indptr": np.asarray(indptr, dtype=np.int32),
                "A_shape": np.asarray([ne, nv], dtype=np.int64),
            })
            payload["scp"]["A"] = {"source": "csr", "keys": ["A_data", "A_indices", "A_indptr", "A_shape"]}
        else:
            payload["scp"]["A"] = {"source": "list-of-sets"}
            payload["scp"]["F"] = [sorted(dict.fromkeys(map(int, f))) for f in self.F]

        return payload

    @classmethod
    def _from_serializable(cls, man: Dict[str, Any], arrays: Dict[str, np.ndarray]):
        wsrc = man["weights"]
        if wsrc["source"] == "npz":
            w = arrays[wsrc["key"]]
        else:
            w = np.array(man.get("_arrays", {}).get(wsrc["key"]), dtype=float)

        scp = man["scp"]
        if scp["A"]["source"] == "csr":
            keys = scp["A"]["keys"]
            data = arrays[keys[0]]
            indices = arrays[keys[1]]
            indptr = arrays[keys[2]]
            shape = tuple(arrays[keys[3]].tolist())
            if sp is None:
                raise RuntimeError("scipy.sparse required to load CSR SCP instances")
            A_csr = sp.csr_matrix((data, indices, indptr), shape=shape)
            # Convert back to list-of-sets F representation:
            ne, nv = shape
            F = []
            for j in range(nv):
                rows = A_csr[:, j].nonzero()[0].tolist()
                F.append(rows)
            A_univ = list(range(ne))
            return cls(A_univ, F, weights=w)

        # list-of-sets
        F = scp.get("F",[])
        flat = [a for f in F for a in f]  # ignores empties naturally

        if flat:
            ne = max(flat) + 1
        else:
            # if all sets are empty, we cannot infer ne from F
            ne = int(man.get("ne", 0))
        
        A_univ = list(range(ne))
        #A_univ = list(range(max(max(f) for f in F) + 1)) if len(F) else []
        return cls(A_univ, F, weights=w)

LPProblem.register_vcgraph_type("SCP", SCP)

