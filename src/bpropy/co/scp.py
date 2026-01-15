# bpropy/co/scp.py
from __future__ import annotations
import os
import tempfile
from typing import Dict, Any, List, Optional, Literal
import numpy as np
import mip
from mip import xsum, BINARY
from scipy.optimize import milp, LinearConstraint, Bounds

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

from bpropy.co.base_lp import LPProblem, LPSolveConfig
from bpropy.io.serde import SerializableLPMixin
from bpropy.co.solution import Solution
from bpropy.co.scg3 import scp_to_orlib_sc, run_scg3, parse_scg3_solution_file


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
    def solve(
        self,
        verbose : bool = False,
        solver: str = "cbc",
        relax_integrality: bool = False,
    ) -> Solution:

        slv = (solver or "cbc").lower()

        # specialized
        if slv == "greedy":
            return self.greedy()

        if slv == "scg3":
            with tempfile.TemporaryDirectory() as d:
                scp_path = os.path.join(d, "instance.sc")
                solution_path = os.path.join(d, "solution")
                scp_to_orlib_sc(self, scp_path, rows_are_zero_based=True)
                run_scg3(scp_path, solution_path, verbose=verbose)
                x = parse_scg3_solution_file(solution_path, n_cols=self.nv(), one_based_cols=True)
                return Solution(solver="scg3", x=x)

        # generic
        return super().solve(
            config=LPSolveConfig(solver=slv, relax_integrality=relax_integrality, verbose=verbose)
        )

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

    def greedy(
        self,
        *,
        policy: Literal["argmax", "sample"] = "argmax",
        temperature: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        use_costs: bool = True,
        scores: Optional[np.ndarray] = None,
    ) -> SCPSolution:
        """
        Greedy Set Cover heuristic.
    
        Parameters
        ----------
        policy : {"argmax","sample"}
            "argmax": pick best score each step.
            "sample": sample via softmax(scores/temperature).
        temperature : float
            Softmax temperature for sampling.
        rng : np.random.Generator, optional
            RNG for sampling.
        use_costs : bool
            If True, score = newly_covered / w[j]. If False, score = newly_covered.
    
        Returns
        -------
        sol : SCPSolution
        """
        x = scp_greedy(self, policy=policy, temperature=temperature, rng=rng, use_costs=use_costs, scores=scores)
        return Solution(x=x, solver="greedy")

LPProblem.register_vcgraph_type("SCP", SCP)

def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    z = (x - np.max(x)) / temperature
    e = np.exp(z)
    s = e.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full_like(e, 1.0 / e.size, dtype=float)
    return e / s


def scp_greedy(
    pb,
    *,
    policy: Literal["argmax", "sample"] = "argmax",
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    use_costs: bool = True,
    scores: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Classic greedy heuristic for Set Cover:

      pick j maximizing gain(j) / w[j]   (if use_costs=True)
      pick j maximizing gain(j)          (if use_costs=False)

    where gain(j) is the number of currently uncovered elements in pb.F[j].

    Returns x ∈ {0,1}^nv.
    """
    n = pb.nv()
    m = pb.ne()

    if policy == "sample" and rng is None:
        rng = np.random.default_rng()

        # Optional override scores (static per set)
    if scores is not None:
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if scores.size != n:
            raise ValueError(f"scores must have shape ({n},), got ({scores.size},)")

    # coverage state
    uncovered = np.ones(m, dtype=bool)
    active_sets = np.ones(n, dtype=bool)

    x = np.zeros(n, dtype=float)

    # Fail-fast if there are elements no set can cover
    # (optional but nice)
    covered_by_any = np.zeros(m, dtype=bool)
    for j in range(n):
        for a in pb.F[j]:
            covered_by_any[int(a)] = True
    if not covered_by_any.all():
        # infeasible SCP instance
        return x

    while uncovered.any():
        active_idx = np.flatnonzero(active_sets)
        if active_idx.size == 0:
            # No sets left but still uncovered elements => infeasible under given representation
            break

        # Compute gain for each active set
        gains = np.zeros(active_idx.size, dtype=float)
        for k, j in enumerate(active_idx):
            elems = np.asarray(pb.F[int(j)], dtype=int)
            if elems.size:
                gains[k] = float(np.sum(uncovered[elems]))
        # If all gains are 0, we're stuck
        if np.max(gains) <= 0:
            break

        if scores is not None:
            step_scores = scores[active_idx]
        else:
            if use_costs:
                costs = np.asarray(pb.w, dtype=float).reshape(-1)
                denom = np.maximum(costs[active_idx], 1e-12)
                step_scores = gains / denom
            else:
                step_scores = gains


        if policy == "argmax":
            j = int(active_idx[np.argmax(step_scores)])
        elif policy == "sample":
            # sample proportional to exp(scores / temperature)
            p = _softmax(step_scores, temperature=temperature)
            j = int(rng.choice(active_idx, p=p))
        else:
            raise ValueError(f"Unknown policy={policy!r} (expected 'argmax' or 'sample')")
        #print(f"selected j={j} val={np.max(step_scores)}")
        #print("\tnewly covered = ", sum(uncovered[int(a)] for a in pb.F[j]))

        # select set j
        x[j] = 1.0
        active_sets[j] = False

        # mark covered
        for a in pb.F[j]:
            uncovered[int(a)] = False
        # remove all sets that don't cover anything new
        gains = np.zeros(active_idx.size, dtype=float)
        for k, j in enumerate(active_idx):
            elems = np.asarray(pb.F[int(j)], dtype=int)
            if elems.size:
                gains[k] = float(np.sum(uncovered[elems]))
        # If all gains are 0, we're stuck
        if np.max(gains) <= 0:
            break
        for aidx, g in zip(active_idx, gains):
            if g==0 and active_sets[aidx]:
                active_sets[aidx] = False
                x[aidx] = 0.0

    return x

