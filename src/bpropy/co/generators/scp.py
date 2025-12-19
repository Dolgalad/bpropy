import numpy as np
from typing import Dict, Any, List, Optional, Literal, Iterable


from bpropy.co.scp import SCP

## Generators
def random_scp_items_k(n: int,
                       m: int,
                       k: int,
                       *,
                       weights: Optional[Iterable[float]] = None,
                       seed: Optional[int] = None) -> SCP:
    """
    Make an SCP with m items (A = range(m)) and n sets (F),
    where each item appears in exactly k distinct sets.
    Set sizes are unconstrained (random).
    """
    if not (0 <= k <= n):
        raise ValueError("k must satisfy 0 <= k <= n")
    rng = np.random.default_rng(seed)

    F = [[] for _ in range(n)]
    # For each item, pick k distinct sets and add the item there.
    for a in range(m):
        cols = rng.choice(n, size=k, replace=False)
        for j in cols:
            F[j].append(a)

    # Optionally random or unit weights
    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(list(weights), dtype=float)
        if w.size != n:
            raise ValueError("weights must have length n")

    return SCP(list(range(m)), [sorted(set(s)) for s in F], weights=w)

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
    empties = [j for j in range(n) if len(F[j]) == 0]
    if empties:
        rng = np.random.default_rng(seed)
        # Precompute item -> sets map
        item_sets = {a: set() for a in range(m)}
        for j in range(n):
            for a in F[j]:
                item_sets[a].add(j)
        for j in empties:
            # pick an item that currently belongs to >=1 sets
            # and can be moved from a set of size > 1
            moved = False
            cand_items = rng.permutation(m)
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
                a = int(rng.integers(m))
                t = next(iter(item_sets[a])) if item_sets[a] else int(rng.integers(n))
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

    return SCP(list(range(m)), [sorted(s) for s in F], weights=scp.w)
