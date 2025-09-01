import numpy as np
import mip
from mip import xsum, minimize, BINARY
from scipy.optimize import milp, LinearConstraint, Bounds
from typing import Iterable, Optional, Literal, Dict, Any, List
try:
    import scipy.sparse as sp
except Exception:
    sp = None

from bpropy.co.scp_solution import SCPSolution
from bpropy.co.base_lp import LPProblem

from bpropy.io.serde import SerializableLPMixin

def number_containing_sets(a, F):
    return sum([a in f for f in F])

def sort_covering(A, F):
    return sorted(A, key=lambda a: number_containing_sets(a, F))

class SCP(LPProblem, SerializableLPMixin):
    def __init__(self, A, F, weights=None):
        super(LPProblem).__init__()
        super(SerializableLPMixin).__init__()
        self.A = sort_covering(A, F)
        self.F = [list(f) for f in F]                       # normalize
        self.F = [sorted(dict.fromkeys(map(int, f))) for f in self.F]

        self.w = np.ones(len(self.F)) if weights is None else np.asarray(weights, dtype=float)

    def ne(self): return len(self.A)
    def nv(self): return len(self.F)
    def number_of_variables(self): return self.nv()
    def number_of_constraints(self): return len(self.A)     # O(1)
    def linear_constraints_matrix(self):
        A = self.adj_matrix().astype(float)         # shape (ne, nv), 0/1
        lb = np.ones(self.ne(), dtype=float)        # each item covered at least once
        ub = np.full(self.ne(), np.inf, dtype=float)
        return A, lb, ub
    
    def variable_bounds(self):
        # 0 <= x <= 1 (LP-relax); if you want only integrality, bounds are still [0,1]
        nv = self.nv()
        return np.zeros(nv), np.ones(nv)

    def constraints(self):
        c = []
        for a in self.A:
            row = np.array([a in f for f in self.F], dtype=float)
            c.append(LinearConstraint(row, lb=1, ub=np.inf))
        return c

    #def constraints(self) -> LinearConstraint:
    #    A = self.adj_matrix().astype(float)                # shape (ne, nv), float64
    #    lb = np.ones(self.ne(), dtype=float)               # each item covered at least once
    #    ub = np.full(self.ne(), np.inf, dtype=float)       # no explicit upper bound
    #    return [LinearConstraint(A, lb=lb, ub=ub)]

    # Optional: unify LPProblem "maximize" view
    def weights(self):                                      # generic API for adapters
        return -self.w                                      # minimize -> maximize

    def solve(self, disp=False, solver="cbc", relax_integrality=False):
        if solver == "scipy":
            integrality = np.zeros(self.nv()) if relax_integrality else np.ones(self.nv())
            return SCPSolution.from_scipy_milp(milp(
                self.w,                                     # correct sign for SciPy minimize
                integrality=integrality,
                constraints=self.constraints(),
                bounds=Bounds(np.zeros(self.nv()), np.ones(self.nv())),
                options={"disp": disp}
            ))
        # CBC
        md = mip.Model("scp")
        x = [md.add_var(lb=0.0, ub=1.0, var_type=BINARY) for _ in range(self.nv())]
        md.objective = mip.minimize(xsum(self.w[i] * x[i] for i in range(self.nv())))  # weighted
        for a in self.A:
            md += mip.xsum(x[i] for i, f in enumerate(self.F) if a in f) >= 1
        md.verbose = 0
        status = md.optimize(relax=relax_integrality)
        return SCPSolution.from_python_mip(x, md)

    def adj_vec(self, f):
        return np.array([a in f for a in self.A], dtype=bool)

    def adj_matrix(self):
        return np.stack([self.adj_vec(f) for f in self.F], axis=1).astype(int)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def greedy(self, scores=None):
        #print("###########################################")
        covered = np.zeros(self.ne(), dtype=bool)
        x = np.zeros(self.nv(), dtype=bool)
        while not np.all(covered):
            cand = [i for i in range(self.nv()) if x[i] == 0]
            if scores is None:
                def ratio(i):
                    gain = (np.logical_not(covered) & self.adj_vec(self.F[i])).sum()
                    return np.inf if gain == 0 else (self.w[i] / gain)
            else:
                def ratio(i):
                    gain = (np.logical_not(covered) & self.adj_vec(self.F[i])).sum()
                    #return np.inf if gain == 0 else (-scores[i] / gain)
                    return np.inf if gain == 0 else (1 - self.sigmoid(scores[i]))
                    return np.inf if gain == 0 else ((1 - self.sigmoid(scores[i])) / gain)


            #print("remaining candidates : ", len(cand))
            #print("candidates: ", cand)
            #print("scores    : ", [scores[c] for c in cand])

            idx = min(cand, key=ratio)
            #print("new : ", idx)
            #print("\t", self.adj_vec(self.F[idx]).sum())
            #print("\t", ratio(idx))

            x[idx] = 1
            covered |= self.adj_vec(self.F[idx])
        return x

    #def check_solution(self, x, verbose: bool = False):
    #    if self.nv() == 0: return True
    #    r = np.stack([self.adj_vec(self.F[i]) for i in range(self.nv()) if x[i] == 1]).sum(axis=0)
    #    if verbose:
    #        print(r)
    #    return np.all(r >= 1)

    def _to_serializable(self, use_npz: bool, arrays: dict, threshold_bytes: int = 200_000):
        payload = {"nv": self.nv(), "weights": {"source": "inline", "key": "w"}, "scp": {}}
    
        # weights: decide by the same threshold
        w = np.asarray(self.w, dtype=float).reshape(-1)
        if use_npz and w.nbytes >= threshold_bytes:
            arrays["w"] = w
            payload["weights"] = {"source": "npz", "key": "w"}
        else:
            payload.setdefault("_arrays", {})["w"] = w.tolist()
    
        # incidence: estimate by nnz (each 1.0 as float64 ~ 8 bytes)
        ne, nv = self.ne(), self.nv()
        nnz = sum(len(f) for f in self.F)                 # ~= n*k for your generator
        estimated_bytes = nnz * 8
    
        if use_npz and estimated_bytes >= threshold_bytes:
            # store CSR rows=items, cols=sets
            data, indices, indptr = [], [], [0]
            for a in range(ne):
                # collect sets j s.t. a in F[j]
                col_idx = [j for j, f in enumerate(self.F) if a in f]
                col_idx.sort()
                data.extend([1.0] * len(col_idx))
                indices.extend(col_idx)
                indptr.append(len(indices))
            arrays.update({
                "A_data":   np.asarray(data, dtype=float),
                "A_indices":np.asarray(indices, dtype=np.int32),
                "A_indptr": np.asarray(indptr, dtype=np.int32),
                "A_shape":  np.asarray([ne, nv], dtype=np.int64),
            })
            payload["scp"]["A"] = {"source":"csr","keys":["A_data","A_indices","A_indptr","A_shape"]}
        else:
            payload["scp"]["A"] = {"source":"list-of-sets"}
            #payload["scp"]["F"] = [list(map(int, set(f))) for f in self.F]
            payload["scp"]["F"] = [sorted(dict.fromkeys(map(int, f))) for f in self.F]
    
        return payload

    @classmethod
    def _from_serializable(cls, man: Dict[str, Any], arrays: Dict[str, np.ndarray]):
        nv = int(man["nv"])
        wsrc = man["weights"]
        if wsrc["source"] == "npz":
            w = arrays[wsrc["key"]]
        else:
            w = np.array(man.get("_arrays", {}).get(wsrc["key"]), dtype=float)

        scp = man["scp"]["A"]
        if scp["source"] == "list-of-sets":
            #F = [list(map(int, f)) for f in man["scp"]["F"]]
            #A = sorted(set(a for f in F for a in f))
            #return cls(A=A, F=F, weights=w)
            F = [sorted(dict.fromkeys(map(int, f))) for f in man["scp"]["F"]]
            A = list(range(len({a for f in F for a in f})))  # or better: pass in ne explicitly if stored
            return cls(A=A, F=F, weights=w)

        elif scp["source"] == "csr":
            A_data = arrays["A_data"]; A_indices = arrays["A_indices"]; A_indptr = arrays["A_indptr"]; A_shape = arrays["A_shape"]
            ne, nv_ = int(A_shape[0]), int(A_shape[1])
            assert nv_ == nv
            # reconstruct F from CSR (cols=sets): items per column j
            # (Our CSR was rows=items over columns=sets; we need column lists)
            # Build CSC quickly to iterate columns
            if sp is None:
                # Fallback: dense (ok for moderate)
                A_dense = sp.csr_matrix((A_data, A_indices, A_indptr), shape=(ne, nv_)).toarray()
                F = [np.where(A_dense[:, j] != 0)[0].astype(int).tolist() for j in range(nv_)]
            else:
                #A_csr = sp.csr_matrix((A_data, A_indices, A_indptr), shape=(ne, nv_))
                #A_csc = A_csr.tocsc()
                #F = [A_csc.indices[A_csc.indptr[j]:A_csc.indptr[j+1]].astype(int).tolist() for j in range(nv_)]
                A_csr = sp.csr_matrix((A_data, A_indices, A_indptr), shape=(ne, nv))
                A_csc = A_csr.tocsc()
                F = [A_csc.indices[A_csc.indptr[j]:A_csc.indptr[j+1]].astype(int).tolist() for j in range(nv)]
                # F[j] is sorted; keep outer order j=0..nv-1 as is.

            A_items = list(range(ne))
            return cls(A=A_items, F=F, weights=w)
        else:
            raise ValueError("Unknown SCP A source")


def random_scp_items_k(n: int,
                       m: int,
                       k: int,
                       *,
                       weights: Optional[Iterable[float]] = None,
                       seed: Optional[int] = None) -> SCP:
    """
    Make an SCP with n items (A = range(n)) and m sets (F),
    where each item appears in exactly k distinct sets.
    Set sizes are unconstrained (random).
    """
    if not (0 <= k <= m):
        raise ValueError("k must satisfy 0 <= k <= m")
    rng = np.random.default_rng(seed)

    F = [[] for _ in range(m)]
    # For each item, pick k distinct sets and add the item there.
    for a in range(n):
        cols = rng.choice(m, size=k, replace=False)
        for j in cols:
            F[j].append(a)

    # Optionally random or unit weights
    if weights is None:
        w = np.ones(m, dtype=float)
    else:
        w = np.asarray(list(weights), dtype=float)
        if w.size != m:
            raise ValueError("weights must have length m")

    return SCP(list(range(n)), [sorted(set(s)) for s in F], weights=w)

def random_scp_items_k_balanced(n: int,
                                m: int,
                                k: int,
                                *,
                                weights: Optional[Iterable[float]] = None,
                                seed: Optional[int] = None,
                                max_repair_iters: int = 3) -> SCP:
    """
    As above, but repairs empties and nudges towards balanced set sizes
    without violating 'each item appears in exactly k sets'.
    """
    scp = random_scp_items_k(n, m, k, weights=weights, seed=seed)
    F = [set(f) for f in scp.F]  # work on sets for O(1) membership

    # 1) Ensure no empty sets: move a random item into empty sets,
    #    removing it from one of its current sets with size > 1.
    empties = [j for j in range(m) if len(F[j]) == 0]
    if empties:
        rng = np.random.default_rng(seed)
        # Precompute item -> sets map
        item_sets = {a: set() for a in range(n)}
        for j in range(m):
            for a in F[j]:
                item_sets[a].add(j)
        for j in empties:
            # pick an item that currently belongs to >=1 sets
            # and can be moved from a set of size > 1
            moved = False
            cand_items = rng.permutation(n)
            for a in cand_items:
                donors = [t for t in item_sets[a] if len(F[t]) > 1]
                if donors:
                    t = rng.choice(donors)
                    F[t].remove(a)
                    F[j].add(a)
                    item_sets[a].remove(t)
                    item_sets[a].add(j)
                    moved = True
                    break
            if not moved:
                # fallback: pick any item a, take from a random donor
                a = int(rng.integers(n))
                t = next(iter(item_sets[a])) if item_sets[a] else int(rng.integers(m))
                if a in F[t]:
                    F[t].remove(a)
                F[j].add(a)

    # 2) Light balancing pass (optional): move items from largest sets to smallest,
    #    preserving the 'k per item' constraint (we only move between sets of the same item).
    for _ in range(max_repair_iters):
        sizes = np.array([len(s) for s in F])
        j_big = int(np.argmax(sizes))
        j_small = int(np.argmin(sizes))
        if sizes[j_big] - sizes[j_small] <= 1:
            break
        # try to move an item that is in j_big but NOT in j_small
        moved = False
        for a in list(F[j_big]):
            if a not in F[j_small]:
                F[j_big].remove(a)
                F[j_small].add(a)
                moved = True
                break
        if not moved:
            break

    return SCP(list(range(n)), [sorted(s) for s in F], weights=scp.w)

