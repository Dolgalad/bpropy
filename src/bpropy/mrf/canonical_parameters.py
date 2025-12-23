from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import torch

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None


def mis_canonical_parameters(fg, M=10., device=None, dtype=None):
    theta = torch.zeros(fg.number_sufficient_statistics(), device=device, requires_grad=False)
    c = 0
    for F in fg.iter_factors():
        nF = fg.factor_size(F)
        if nF == 1:
            theta[c+1] = 1.
            theta[c] = 0.
            c += 2
        else:
            for v in range(2**nF):
                vals = np.unpackbits(np.array([v]).astype(np.uint8), count=nF, bitorder="little")
                if vals.sum()>1:
                    theta[c] = M
                else:
                    theta[c] = 0.
                c += 1
    return theta

def scp_canonical_parameters(fg, M=10.):
    theta = torch.zeros(fg.number_sufficient_statistics(), device=device, requires_grad=False)
    c = 0
    for F in fg.iter_factors():
        nF = fg.factor_size(F)
        if nF == 1:
            theta[c+1] = 0.
            theta[c] = 1.
            c += 2
        else:
            for v in range(2**nF):
                vals = np.unpackbits(np.array([v]).astype(np.uint8), count=nF, bitorder="little")
                if vals.sum()==0:
                    theta[c] = M
                else:
                    theta[c] = 0.
                c += 1
    return theta


def canonical_parameters(
    pb,
    *,
    M: float = 10.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    add_unaries: bool = True,
    unary_bias: float = 0.0,
    include_empty_constraints: bool = False,
    tol: float = 0.0,
) -> torch.Tensor:
    """
    Build theta in the same overcomplete SS order as FactorGraph(pb, add_unaries=...):

    - Unary factors (optional): [x=0, x=1] get a bias if unary_bias != 0.
        * If pb.objective_sense == "max": prefer x=1 -> theta(x=1)=+unary_bias
        * If pb.objective_sense == "min": prefer x=0 -> theta(x=0)=+unary_bias
      (Set unary_bias=0.0 if you truly want only {0, -M}.)

    - Constraint factors: for each row i, for each assignment x_scope:
        theta = 0   if lb_i <= a_i·x <= ub_i
        theta = -M  otherwise

    Row order is sorted by (|scope|, scope_lex) to match FactorGraph ordering. 
    Assignment order is little-endian to match SS indexing. :contentReference[oaicite:6]{index=6}
    """
    if device is None:
        device = torch.device("cpu")

    A, lb, ub = pb.linear_constraints_matrix()
    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)

    n = int(pb.nv()) if hasattr(pb, "nv") else int(A.shape[1])

    # ---- extract per-row scopes and coefficients ----
    rows: List[Tuple[Tuple[int, ...], int]] = []

    if sp is not None and sp.issparse(A):
        A_csr = A.tocsr()
        for i in range(A_csr.shape[0]):
            start, end = A_csr.indptr[i], A_csr.indptr[i + 1]
            cols = A_csr.indices[start:end]
            if cols.size == 0 and not include_empty_constraints:
                continue
            scope = tuple(sorted(int(j) for j in cols))
            rows.append((scope, i))
    else:
        A_arr = np.asarray(A, dtype=float)
        for i in range(A_arr.shape[0]):
            cols = np.flatnonzero(A_arr[i] != 0)
            if cols.size == 0 and not include_empty_constraints:
                continue
            scope = tuple(sorted(int(j) for j in cols))
            rows.append((scope, i))

    # ---- sort rows to match FactorGraph factor ordering (after unaries) ----
    rows.sort(key=lambda t: (len(t[0]), list(t[0])))

    # ---- compute total SS length ----
    ss_len = 0
    if add_unaries:
        ss_len += 2 * n
    for scope, _ in rows:
        ss_len += 1 << len(scope)

    theta = torch.zeros(ss_len, device=device, dtype=dtype)

    c = 0

    # ---- unary factors (optional) ----
    if add_unaries:
        prefer_one = (pb.objective_sense == "max")  # defined on LPProblem :contentReference[oaicite:7]{index=7}
        if unary_bias != 0.0:
            for v in range(n):
                if prefer_one:
                    theta[c + 1] = float(unary_bias)  # x=1 preferred
                else:
                    theta[c + 0] = float(unary_bias)  # x=0 preferred
                c += 2
        else:
            c += 2 * n

    # Helper to evaluate row dot product on a partial assignment
    def row_dot(scope: Tuple[int, ...], row_idx: int, x_bits: int) -> float:
        if sp is not None and sp.issparse(A):
            # fetch row coefficients for just the scope columns
            # (CSR row slice then index into columns)
            A_csr = A  # original may not be csr; ensure csr once
            if not sp.isspmatrix_csr(A_csr):
                A_csr = A_csr.tocsr()
            start, end = A_csr.indptr[row_idx], A_csr.indptr[row_idx + 1]
            cols = A_csr.indices[start:end]
            data = A_csr.data[start:end]
            coef_map = {int(j): float(a) for j, a in zip(cols, data)}
            s = 0.0
            for k, j in enumerate(scope):
                if (x_bits >> k) & 1:
                    s += coef_map.get(j, 0.0)
            return s
        else:
            A_arr = np.asarray(A, dtype=float)
            s = 0.0
            for k, j in enumerate(scope):
                if (x_bits >> k) & 1:
                    s += float(A_arr[row_idx, j])
            return s

    # ---- constraint factors ----
    for scope, i in rows:
        a = len(scope)
        if a == 0:
            # empty-scope constraint: Ax is constant 0
            Ax = 0.0
            violated = (np.isfinite(lb[i]) and Ax < lb[i] - tol) or (np.isfinite(ub[i]) and Ax > ub[i] + tol)
            theta[c + 0] = -float(M) if violated else 0.0
            c += 1
            continue

        for bits in range(1 << a):
            Ax = row_dot(scope, i, bits)
            violated = (np.isfinite(lb[i]) and Ax < lb[i] - tol) or (np.isfinite(ub[i]) and Ax > ub[i] + tol)
            theta[c] = -float(M) if violated else 0.0
            c += 1

    return theta

