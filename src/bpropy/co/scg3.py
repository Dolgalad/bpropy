from __future__ import annotations
from typing import Iterable, Sequence, List, Tuple, Optional
import numpy as np
import re

def write_orlib_sc(
    path: str,
    m: int,
    n: int,
    costs: Sequence[int] | np.ndarray,
    col_covers: Sequence[Iterable[int]],
    *,
    rows_are_zero_based: bool = True,
    costs_per_line: int = 10,
) -> None:
    """
    Write an OR-Library style Set Covering instance (.sc).

    File format (as used by scg3):
      line 1: m (#rows)
      line 2: n (#cols)
      line 3: nnz (total incidences)
      line 4: 0
      line 5: 0
      then n costs
      then for each column j=1..n:
        list of covered row indices (1-based), terminated by 0
    """
    if len(costs) != n:
        raise ValueError(f"costs length {len(costs)} != n {n}")
    if len(col_covers) != n:
        raise ValueError(f"col_covers length {len(col_covers)} != n {n}")

    # Normalize covers to 1-based sorted unique lists
    covers_1based: List[List[int]] = []
    nnz = 0
    for j, cov in enumerate(col_covers):
        cov_list = list(cov)
        if rows_are_zero_based:
            cov_list = [r + 1 for r in cov_list]
        # Validate bounds
        for r in cov_list:
            if not (1 <= r <= m):
                raise ValueError(f"Column {j}: row index {r} out of bounds 1..{m}")
        # Sort/unique for reproducibility
        cov_list = sorted(set(cov_list))
        nnz += len(cov_list)
        covers_1based.append(cov_list)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{m}\n")
        f.write(f"{n}\n")
        f.write(f"{nnz}\n")
        f.write("0\n")
        f.write("0\n")

        # Costs (spread across lines)
        c = list(map(int, costs))
        for i in range(0, n, costs_per_line):
            chunk = c[i : i + costs_per_line]
            f.write(" ".join(map(str, chunk)) + "\n")

        # Column coverage lists terminated by 0
        for cov in covers_1based:
            if cov:
                f.write(" ".join(map(str, cov)) + " 0\n")
            else:
                # empty column (rare, but legal)
                f.write("0\n")

def scp_to_orlib_sc(pb, path, rows_are_zero_based=True):
    A, lb, ub = pb.linear_constraints_matrix()  # A is (m,n)
    m, n = A.shape

    # costs = pb objective coefficients (SCP uses weights w)
    costs = np.asarray(pb.objective_coeffs(), dtype=float).reshape(-1)
    if costs.size != n:
        raise ValueError(f"objective_coeffs has size {costs.size}, expected {n}")

    # Build column covers (list of row indices per column)
    if hasattr(A, "tocsc"):  # scipy sparse
        A_csc = A.tocsc()
        col_covers = [A_csc[:, j].indices for j in range(n)]  # 0-based rows
    else:
        A_dense = np.asarray(A)
        col_covers = [np.flatnonzero(A_dense[:, j]).tolist() for j in range(n)]  # 0-based rows

    write_orlib_sc(
        path,
        m,
        n,
        costs=costs,
        col_covers=col_covers,
        rows_are_zero_based=rows_are_zero_based,
    )


def parse_scg3_solution_file(
    path: str,
    n_cols: int,
    *,
    one_based_cols: bool = True,
) -> np.ndarray:
    """
    Parse a scg3 output solution file into a 0/1 numpy vector x of length n_cols.

    It looks for a block starting near:
      - 'SOLUZIONE' or 'SOLUTION' or
      - 'Number of columns = ...'
    then reads lines with at least 2 integers:
      (k, j, ...)  -> j is the chosen column id
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Heuristic: locate a likely solution block
    # Prefer "Number of columns" since it appears in your file.
    anchors = [
        r"Number\s+of\s+columns\s*=\s*\d+",
        r"SOLUZIONE",
        r"SOLUTION",
    ]
    start = None
    for a in anchors:
        m = re.search(a, text, flags=re.IGNORECASE)
        if m:
            start = m.start()
            break
    if start is None:
        raise ValueError("Could not find solution block (missing 'Number of columns' / 'SOLUZIONE').")

    tail = text[start:]

    chosen: List[int] = []
    for line in tail.splitlines():
        # Stop if we reach the big stats box or another section
        if line.strip().startswith("###") or "TEST DI RIDUZIONE" in line or "RIDUZIONE" in line:
            # Not always a stop, but usually after the chosen columns
            # We'll stop only if we've already collected something.
            if chosen:
                break

        # Extract integers on the line
        ints = re.findall(r"-?\d+", line)
        if len(ints) >= 2:
            # In scg3 format: "k  j  0" -> second integer is column id
            j = int(ints[1])
            # Ignore obviously non-sensical indices
            if 0 < j <= (n_cols if one_based_cols else n_cols - 1):
                chosen.append(j)

    if not chosen:
        raise ValueError("Found solution block anchor but did not parse any chosen columns.")

    # Build x
    x = np.zeros(n_cols, dtype=np.int8)
    if one_based_cols:
        for j in chosen:
            x[j - 1] = 1
    else:
        for j in chosen:
            x[j] = 1
    return x

