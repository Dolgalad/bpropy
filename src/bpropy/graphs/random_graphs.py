import numpy as np
from typing import Optional

def random_bounded_degree(
    nv: int,
    max_deg: int,
    *,
    target_avg_deg: Optional[float] = None,
    p: Optional[float] = None,
    ensure_min_deg: int = 0,
    seed: Optional[int] = None,
    max_trials_factor: int = 20,
):
    """
    Generate a random simple graph G with nv nodes (0..nv-1) and degree(v) <= max_deg,
    then return (nv, edges, weights) ready for MIS.from_edges(...).

    Strategy
    --------
    - Choose a target number of edges E_target from either target_avg_deg or p.
    - Sample random node pairs and add an edge iff both endpoints remain <= max_deg.
    - Stop when E_target edges are added or we run out of trials.

    Parameters
    ----------
    nv : int
    max_deg : int
        Hard cap per node: degree(v) <= max_deg.
    target_avg_deg : float | None
        Desired average degree; implies E_target ≈ nv * target_avg_deg / 2.
    p : float | None
        ER-like density; implies E_target ≈ p * nv * (nv-1) / 2.
        (Use either target_avg_deg or p, not both.)
    ensure_min_deg : int
        If > 0, do a light repair pass to lift nodes below this minimum degree
        while still respecting the max_deg cap (best-effort).
    seed : int | None
    max_trials_factor : int
        Safety multiplier for the max number of edge-sampling attempts.

    Returns
    -------
    nv : int
    edges : list[tuple[int,int]]
    weights : np.ndarray (float, shape [nv])  # default all-ones
    """
    if nv <= 0:
        return 0, [], np.zeros(0, dtype=float)
    if max_deg < 0:
        raise ValueError("max_deg must be >= 0")
    if target_avg_deg is not None and p is not None:
        raise ValueError("Provide only one of target_avg_deg or p.")

    rng = np.random.default_rng(seed)

    # Upper bound on the number of edges given the degree cap
    max_edges_cap = (nv * min(max_deg, nv - 1)) // 2

    if target_avg_deg is not None:
        desired_edges = int(round(nv * target_avg_deg / 2))
    elif p is not None:
        desired_edges = int(round(p * nv * (nv - 1) / 2))
    else:
        # default: half of the cap, a reasonable sparse-ish graph
        desired_edges = max_edges_cap // 2

    E_target = max(0, min(desired_edges, max_edges_cap))

    # Fast path: no edges possible
    if E_target == 0:
        return nv, [], np.ones(nv, dtype=float)

    deg = np.zeros(nv, dtype=np.int32)
    edges = set()

    # We sample pairs uniformly at random; expected O(E_target) acceptance
    max_trials = max_trials_factor * max(1, E_target)
    trials = 0
    while len(edges) < E_target and trials < max_trials:
        trials += 1
        u = int(rng.integers(nv))
        v = int(rng.integers(nv - 1))
        if v >= u:
            v += 1  # map to [0..u-1]∪[u+1..nv-1]
        if u > v:
            u, v = v, u

        if (u, v) in edges:
            continue
        if deg[u] >= max_deg or deg[v] >= max_deg:
            continue

        # accept
        edges.add((u, v))
        deg[u] += 1
        deg[v] += 1

    # Optional: lift very low-degree nodes (best-effort), still respecting max_deg.
    if ensure_min_deg > 0:
        # Build adjacency for quick checks
        nbrs = [set() for _ in range(nv)]
        for (u, v) in edges:
            nbrs[u].add(v); nbrs[v].add(u)

        low = [i for i in range(nv) if deg[i] < ensure_min_deg]
        # Try to connect low-degree nodes with random partners that keep caps and avoid duplicates
        for u in low:
            attempts = 0
            while deg[u] < ensure_min_deg and attempts < nv * 2:
                attempts += 1
                v = int(rng.integers(nv - 1))
                if v >= u: v += 1
                if deg[u] >= max_deg or deg[v] >= max_deg: 
                    continue
                a, b = (u, v) if u < v else (v, u)
                if a in nbrs[b] or a == b:
                    continue
                # add edge
                edges.add((a, b))
                nbrs[a].add(b); nbrs[b].add(a)
                deg[a] += 1; deg[b] += 1

    # All-ones weights by default (you can pass your own later if needed)
    w = np.ones(nv, dtype=float)
    return nv, sorted(edges), w
