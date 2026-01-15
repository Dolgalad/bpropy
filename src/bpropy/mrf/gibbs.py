from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

from typing import Callable, Iterable, List, Optional, Tuple, Union
import numpy as np


TempSchedule = Union[
    float,
    Callable[[int], float],
    List[Tuple[int, float]],  # [(sweep0, T0), (sweep1, T1), ...] piecewise-constant
]


def make_temperature_schedule(
    *,
    kind: str = "constant",
    T0: float = 1.0,
    T1: float = 1.0,
    total_sweeps: int = 1,
    decay: float = 0.99,
    floor: float = 1e-3,
) -> Callable[[int], float]:
    """
    Returns temp(sweep) -> T.

    kind:
      - "constant": always T0
      - "linear": linearly from T0 to T1 over total_sweeps
      - "exp": T = max(floor, T0 * decay**sweep)
    """
    if kind == "constant":
        return lambda s: float(T0)

    if kind == "linear":
        total = max(int(total_sweeps), 1)
        def temp(s: int) -> float:
            a = min(max(s, 0), total - 1) / (total - 1 if total > 1 else 1)
            return float((1 - a) * T0 + a * T1)
        return temp

    if kind == "exp":
        def temp(s: int) -> float:
            return float(max(floor, T0 * (decay ** max(s, 0))))
        return temp

    raise ValueError(f"Unknown schedule kind: {kind}")


def piecewise_temperature(points: List[Tuple[int, float]]) -> Callable[[int], float]:
    """
    points = [(0, 5.0), (100, 2.0), (500, 1.0)]
    means:
      sweeps [0..99]   -> 5.0
      sweeps [100..499]-> 2.0
      sweeps [500..]   -> 1.0
    """
    pts = sorted((int(s), float(T)) for s, T in points)

    def temp(sweep: int) -> float:
        s = int(sweep)
        T = pts[0][1]
        for sw, Tv in pts:
            if s >= sw:
                T = Tv
            else:
                break
        return float(T)

    return temp



# ---------------------------
# Utilities
# ---------------------------

def _logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    return m + float(np.log(np.sum(np.exp(a - m))))


def _sample_from_logp(rng: np.random.Generator, logp: np.ndarray) -> int:
    """Sample categorical with probs proportional to exp(logp)."""
    z = logp - np.max(logp)  # stability
    p = np.exp(z)
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0.0:
        # fallback uniform
        return int(rng.integers(0, len(logp)))
    p /= s
    return int(rng.choice(len(logp), p=p))


def _assignment_index_little_endian(bits: np.ndarray) -> int:
    """
    Matches FactorGraph.factor_sufficient_statistics:
        idx = sum(((b & 1) << k) for k, b in enumerate(xF))
    where xF is in the factor's variable order (sorted variables). :contentReference[oaicite:1]{index=1}
    """
    # bits is shape (k,), dtype int, values in {0,1}
    idx = 0
    for k, b in enumerate(bits):
        idx |= (int(b) & 1) << k
    return idx


# ---------------------------
# Factor slice bookkeeping
# ---------------------------

@dataclass(frozen=True)
class FactorSlice:
    """Represents a factor node and the slice of theta corresponding to it."""
    F: int                      # factor node id in the FactorGraph
    vars: Tuple[int, ...]        # sorted variable node ids adjacent to F
    start: int                   # start offset in flattened theta
    size: int                    # number of entries (2**arity)


def _build_factor_slices(fg) -> Tuple[List[FactorSlice], int]:
    """
    Build factor slices in the canonical factor iteration order fg.iter_factors(). :contentReference[oaicite:2]{index=2}
    """
    slices: List[FactorSlice] = []
    off = 0
    for F in fg.iter_factors():
        vlist = tuple(fg.factor_variables(F))  # sorted per your implementation :contentReference[oaicite:3]{index=3}
        arity = len(vlist)
        sz = 1 << arity  # 2**arity
        slices.append(FactorSlice(F=F, vars=vlist, start=off, size=sz))
        off += sz
    return slices, off


def _theta_as_flat(theta: Union[np.ndarray, "object"]) -> np.ndarray:
    """
    Accept numpy array or torch tensor-like (has .detach().cpu().numpy()).
    """
    if isinstance(theta, np.ndarray):
        t = theta
    else:
        # torch.Tensor / similar
        t = theta.detach().cpu().numpy()
    return np.asarray(t, dtype=np.float64).ravel()


# ---------------------------
# Coloring (blocks)
# ---------------------------

def _build_primal_adjacency(fg) -> List[set]:
    """
    Primal adjacency: variables share an edge if they appear together in any factor. :contentReference[oaicite:4]{index=4}
    """
    n = fg.nv()  # number_of_variables convenience :contentReference[oaicite:5]{index=5}
    adj = [set() for _ in range(n)]
    for F in fg.iter_factors():
        scope = fg.factor_variables(F)
        for i in range(len(scope)):
            vi = scope[i]
            for j in range(i + 1, len(scope)):
                vj = scope[j]
                adj[vi].add(vj)
                adj[vj].add(vi)
    return adj


def _greedy_coloring(adj: List[set]) -> List[List[int]]:
    """
    Greedy coloring -> color classes are valid blocks for chromatic Gibbs.
    """
    n = len(adj)
    order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    color = [-1] * n
    blocks: List[List[int]] = []

    for v in order:
        forbidden = {color[u] for u in adj[v] if color[u] != -1}
        c = 0
        while c in forbidden:
            c += 1
        color[v] = c
        while len(blocks) <= c:
            blocks.append([])
        blocks[c].append(v)

    for b in blocks:
        b.sort()
    return blocks


# ---------------------------
# Block Gibbs Sampler
# ---------------------------

def block_gibbs_sample(
    fg,
    theta: Union[np.ndarray, "object"],
    max_samples: int,
    *,
    burn_in_sweeps: int = 0,
    thinning: int = 1,
    seed: Optional[int] = None,
    init_x: Optional[np.ndarray] = None,
    blocks: Optional[List[List[int]]] = None,
    pb: Optional[Any] = None
) -> np.ndarray:
    """
    Block (chromatic) Gibbs sampler for your FactorGraph with binary variables.

    Parameters
    ----------
    fg : FactorGraph
        Your bpropy FactorGraph. Must support iter_factors(), factor_variables(F),
        containing_factors(i), nv(). :contentReference[oaicite:6]{index=6}
    theta : array-like
        Flattened log-potentials in overcomplete representation:
        concatenation over factors (in fg.iter_factors() order) of length 2**|F|.
        (If theta is shaped, it will be flattened.)
    max_samples : int
        Number of samples to *return* (after burn-in and thinning).
    burn_in_sweeps : int
        Number of full sweeps over all blocks to discard.
    thinning : int
        Keep one sample every `thinning` sweeps after burn-in.
    seed : int
        RNG seed.
    init_x : np.ndarray
        Optional initial state, shape (nv,), entries in {0,1}.
    blocks : list[list[int]]
        Optional precomputed blocks. If None, will compute greedy coloring blocks.

    Returns
    -------
    samples : np.ndarray
        Shape (max_samples, nv) with entries in {0,1}.
    """
    rng = np.random.default_rng(seed)
    n = fg.nv()  # :contentReference[oaicite:7]{index=7}

    temp_fn = make_temperature_schedule(kind="exp", T0=5.0, decay=0.9995, floor=1.0)

    # Prepare theta slices per factor
    theta_flat = _theta_as_flat(theta)
    fac_slices, total_needed = _build_factor_slices(fg)
    if theta_flat.size != total_needed:
        raise ValueError(
            f"theta has {theta_flat.size} entries, but factor graph needs {total_needed} "
            f"(sum of 2**arity over factors in iter_factors() order)."
        )

    # Incident factor slices per variable (use fg.containing_factors(i)) :contentReference[oaicite:8]{index=8}
    # Map factor node id -> FactorSlice for quick lookup
    slice_by_F = {fs.F: fs for fs in fac_slices}
    incident: List[List[FactorSlice]] = []
    for i in range(n):
        Fs = fg.containing_factors(i)  # sorted list of factor node ids :contentReference[oaicite:9]{index=9}
        incident.append([slice_by_F[F] for F in Fs])

    # Blocks
    if blocks is None:
        adj = _build_primal_adjacency(fg)
        blocks = _greedy_coloring(adj)

    # Init state
    if init_x is None:
        x = rng.integers(0, 2, size=(n,), dtype=np.int64)
    else:
        x = np.asarray(init_x, dtype=np.int64).copy()
        if x.shape != (n,):
            raise ValueError(f"init_x must have shape ({n},), got {x.shape}")
        if not np.all((x == 0) | (x == 1)):
            raise ValueError("init_x must be binary (0/1).")

    def factor_logpot_for_current_state(fs: FactorSlice) -> float:
        """Compute log-pot for factor fs under current x."""
        bits = x[list(fs.vars)]
        idx = _assignment_index_little_endian(bits)
        return float(theta_flat[fs.start + idx])

    def local_log_score_var(i: int, val: int) -> float:
        """Sum of incident factor log-potentials when x[i]=val (others fixed)."""
        old = int(x[i])
        x[i] = val
        s = 0.0
        for fs in incident[i]:
            s += factor_logpot_for_current_state(fs)
        x[i] = old
        return float(s)

    # Sampling loop
    samples = np.empty((max_samples, n), dtype=np.int64)
    kept = 0

    total_sweeps = burn_in_sweeps + max_samples * thinning
    for sweep in range(total_sweeps):
        for block in blocks:
            # Within a color class, variables share no factor => independent conditional updates.
            # We still update sequentially (safe and simple).
            for i in block:
                #logp0 = local_log_score_var(i, 0)
                #logp1 = local_log_score_var(i, 1)
                #x[i] = _sample_from_logp(rng, np.array([logp0, logp1], dtype=np.float64))
                T = temp_fn(sweep)        # > 0
                invT = 1.0 / max(T, 1e-12)
                logp = invT * np.array([local_log_score_var(i, 0), local_log_score_var(i, 1)], dtype=np.float64)
                x[i] = _sample_from_logp(rng, logp)
        if pb is not None:
            print("sweep ", sweep, " ", np.dot(pb.w, x), " ", pb.check_solution(x), " ", T)

        if sweep >= burn_in_sweeps and ((sweep - burn_in_sweeps) % thinning == 0):
            samples[kept] = x
            kept += 1
            if kept == max_samples:
                break

    return samples

