from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _as_mu2_numpy(mu: Any) -> np.ndarray:
    """Convert backend output to a finite numpy array of shape (n, 2)."""
    if hasattr(mu, "detach"):
        mu_np = mu.detach().cpu().numpy()
    else:
        mu_np = np.asarray(mu)

    mu_np = np.asarray(mu_np, dtype=float).reshape(-1, 2)
    return np.nan_to_num(mu_np, nan=0.0, posinf=1.0, neginf=0.0)


def _binary_entropy_from_mu1(mu1: np.ndarray) -> np.ndarray:
    """Binary entropy H(x) from p(x=1)."""
    p = np.clip(np.asarray(mu1, dtype=float), 1e-12, 1.0 - 1e-12)
    return -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)


def _mean_or_none(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return None
    return float(np.mean(x))


def _candidate_diagnostics(
    *,
    chosen: int,
    value: int,
    candidates: np.ndarray,
    mu_np: np.ndarray,
    selection_score: np.ndarray,
) -> Dict[str, Any]:
    """
    Shared logging diagnostics for GD decisions.

    candidates:
        Variable indices that were eligible before the step.
    mu_np:
        Shape (n, 2), raw variable marginals.
    selection_score:
        Shape (n,), the score actually used by the adapter to rank candidates.
        For MIS this is usually mu1.
        For SCP this may be mu1 / cost.
    """
    candidates = np.asarray(candidates, dtype=int)
    other = candidates[candidates != int(chosen)]

    mu1 = np.nan_to_num(mu_np[:, 1].astype(float), nan=0.0)
    entropy = _binary_entropy_from_mu1(mu1)

    chosen_mu1 = float(mu1[int(chosen)])
    chosen_entropy = float(entropy[int(chosen)])
    chosen_selection_score = float(selection_score[int(chosen)])

    avg_mu1_candidates = _mean_or_none(mu1[candidates])
    avg_mu1_other_candidates = _mean_or_none(mu1[other])

    avg_entropy_candidates = _mean_or_none(entropy[candidates])
    avg_entropy_other_candidates = _mean_or_none(entropy[other])

    avg_selection_score_candidates = _mean_or_none(selection_score[candidates])
    avg_selection_score_other_candidates = _mean_or_none(selection_score[other])

    return {
        "chosen": int(chosen),
        "value": int(value),

        "chosen_mu1": chosen_mu1,
        "avg_mu1_candidates": avg_mu1_candidates,
        "avg_mu1_other_candidates": avg_mu1_other_candidates,
        "delta_mu1_chosen_vs_avg_others": (
            None if avg_mu1_other_candidates is None
            else float(chosen_mu1 - avg_mu1_other_candidates)
        ),

        "chosen_entropy": chosen_entropy,
        "avg_entropy_candidates": avg_entropy_candidates,
        "avg_entropy_other_candidates": avg_entropy_other_candidates,
        "delta_entropy_chosen_vs_avg_others": (
            None if avg_entropy_other_candidates is None
            else float(chosen_entropy - avg_entropy_other_candidates)
        ),

        "selection_score": chosen_selection_score,
        "avg_selection_score_candidates": avg_selection_score_candidates,
        "avg_selection_score_other_candidates": avg_selection_score_other_candidates,
        "delta_selection_score_chosen_vs_avg_others": (
            None if avg_selection_score_other_candidates is None
            else float(chosen_selection_score - avg_selection_score_other_candidates)
        ),

        "n_candidates": int(candidates.size),
    }

def _get_costs(pb, n_sets: int) -> np.ndarray:
    """Try to pull set costs/weights from pb; default to 1."""
    for name in ("w", "cost", "costs", "weights", "c"):
        if hasattr(pb, name):
            c = np.asarray(getattr(pb, name), dtype=float).reshape(-1)
            if c.size == n_sets:
                return c
    return np.ones(n_sets, dtype=float)

def _extract_incidence(pb) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    SCP incidence extraction for bpropy.co.scp.SCP:

      - pb.F: list of sets, each is a list of covered element indices
      - pb.ne(): number of elements (constraints)
      - pb.nv(): number of sets (variables)

    Returns:
      S[j] = np.array(elements covered by set j)
      C[i] = np.array(sets covering element i)
    """
    # Prefer SCP's native representation
    if hasattr(pb, "F") and callable(getattr(pb, "ne", None)) and callable(getattr(pb, "nv", None)):
        n_sets = int(pb.nv())
        n_elems = int(pb.ne())

        S = [np.asarray(pb.F[j], dtype=int) for j in range(n_sets)]

        tmp = [[] for _ in range(n_elems)]
        for j, elems in enumerate(S):
            for i in elems:
                ii = int(i)
                if 0 <= ii < n_elems:
                    tmp[ii].append(int(j))
                else:
                    raise ValueError(f"SCP incidence: element id {ii} out of range [0,{n_elems})")
        C = [np.asarray(t, dtype=int) for t in tmp]
        return S, C

    # Fallback to generic LPProblem view: use the constraint matrix Ax >= 1
    if hasattr(pb, "linear_constraints_matrix"):
        A, lb, ub = pb.linear_constraints_matrix()
        # A is shape (m, n) = (ne, nv)
        # Build S from columns. This works for dense or sparse.
        if hasattr(A, "tocsc"):  # scipy sparse
            Ac = A.tocsc()
            m, n = Ac.shape
            S = []
            for j in range(n):
                start, end = Ac.indptr[j], Ac.indptr[j + 1]
                S.append(Ac.indices[start:end].astype(int))
        else:
            A = np.asarray(A)
            if A.ndim != 2:
                raise ValueError(f"Expected 2D constraint matrix, got shape {A.shape}")
            m, n = A.shape
            S = [np.flatnonzero(A[:, j]).astype(int) for j in range(n)]

        tmp = [[] for _ in range(m)]
        for j, elems in enumerate(S):
            for i in elems:
                tmp[int(i)].append(int(j))
        C = [np.asarray(t, dtype=int) for t in tmp]
        return S, C

    raise AttributeError("SCPAdapter: cannot extract incidence (expected SCP with F/ne/nv or LPProblem with linear_constraints_matrix).")

class SCPAdapter:
    """
    Adapter for SCP decimation guided by an inference backend.

    State:
      - active_sets[j] indicates if set-variable j is still undecided
      - uncovered[i] indicates if element i is not yet covered
      - x[j] ∈ {-1,0,1} set decision

    Step rule:
      - choose an active set j that maximizes (mu[j,1] / cost[j]) by default
      - set x[j]=1, mark its covered elements as covered
      - stop when all elements are covered; remaining sets become 0 in solution()
    """

    def __init__(self, 
                 normalize_by_cost: bool = True,
                 pick_mode: str = "min_entropy", # "min_entropy" | "softmax_eff" | "softmax_neg_entropy" | "max_eff"
                 value_mode: str = "argmax"    , # "argmax" | "sample" | "add"
                 j_temperature: float = 1.0,
                 seed: Optional[int] = None,
                 fix_prop: Optional[float] = None,
    ):
        self.normalize_by_cost = normalize_by_cost
        self.pick_mode = pick_mode
        self.value_mode = value_mode
        self.j_temperature = j_temperature
        self.rng = np.random.default_rng(seed)
        if fix_prop is not None:
            self.fix_prop = fix_prop
        else:
            self.fix_prop = 1.0

        self.pb: Any = None
        self.S: List[np.ndarray] = []
        self.C: List[np.ndarray] = []
        self.cost: Optional[np.ndarray] = None

        self.active_sets: Optional[np.ndarray] = None
        self.uncovered: Optional[np.ndarray] = None
        self.x: Optional[np.ndarray] = None

        ## Perf improvements
        self.useful_sets: Optional[np.ndarray] = None
        self.remaining_cover_count: Optional[np.ndarray] = None
        self.enable_pruning: bool = True
        self.active_coverer_count: Optional[np.ndarray] = None

    def reset(self, pb: Any) -> None:
        self.pb = pb
        self.S, self.C = _extract_incidence(pb)
        n_sets = len(self.S)
        n_elems = len(self.C)
        self.cost = _get_costs(pb, n_sets)

        self.active_sets = np.ones(n_sets, dtype=bool)
        self.uncovered = np.ones(n_elems, dtype=bool)
        self.x = -np.ones(n_sets, dtype=int)

        ## Perf improvements
        self.remaining_cover_count = np.fromiter(
                    (len(self.S[j]) for j in range(n_sets)),
                        dtype=np.int32,
                            count=n_sets,
                            )
        self.useful_sets = self.remaining_cover_count > 0

        self.active_coverer_count = np.fromiter(
                    (len(self.C[i]) for i in range(n_elems)),
                        dtype=np.int32,
                            count=n_elems,
                            )

    def is_done(self) -> bool:
        assert self.uncovered is not None
        return (not self.uncovered.any()) or (self.active_sets.sum() <= self.pb.nv() * (1.0 - self.fix_prop))
    def _deactivate_set(self, j: int, value: int, evidence: Dict[int, int]) -> None:
        if not self.active_sets[j]:
            return

        self.active_sets[j] = False
        self.useful_sets[j] = False
        self.x[j] = value
        evidence[j] = value

        for i in self.S[j]:
            i = int(i)
            if self.uncovered[i]:
                self.active_coverer_count[i] -= 1
    def _cover_element(self, i: int, evidence: Dict[int, int], newly: list[int]) -> None:
        if not self.uncovered[i]:
            return

        self.uncovered[i] = False
        newly.append(i)

        for s_idx in self.C[i]:
            s_idx = int(s_idx)
            if self.active_sets[s_idx]:
                self.remaining_cover_count[s_idx] -= 1
                if self.remaining_cover_count[s_idx] <= 0:
                    self.useful_sets[s_idx] = False
                    self.active_sets[s_idx] = False
                    if self.x[s_idx] < 0:
                        self.x[s_idx] = 0
                    evidence[s_idx] = 0

    def step(self, backend_output: Any, return_log: bool = False) -> Tuple[Dict[int, int], Dict[str, Any]]:
        """
        backend_output expected: mu shaped (n_sets,2) or (2*n_sets,) convertible.
        Uses mu[:,1] as "probability/score of selecting set".
        """
        assert self.active_sets is not None and self.uncovered is not None and self.x is not None
        assert self.cost is not None

        mu = backend_output
        if mu is None:
            raise ValueError("SCPAdapter expects mu from an inference backend.")

        # Convert to (n_sets,2) numpy
        if hasattr(mu, "detach"):
            mu_np = mu.detach().cpu().numpy()
        else:
            mu_np = np.asarray(mu)
        mu_np = mu_np.reshape(-1, 2)

        scores = np.nan_to_num(mu_np[:, 1].astype(float), nan=0.0)

        # Mask inactive sets
        #active_idx = np.flatnonzero(self.active_sets)
        #feasible = np.array([self.uncovered[self.S[j]].any() for j in active_idx], dtype=bool)
        #cand = active_idx[feasible]

        ## Perf improvements
        cand = np.flatnonzero(self.active_sets & self.useful_sets)
        if cand.size == 0:
            info = {"chosen": None, "reason": "no_feasible_sets", "n_candidates": 0}
            return ({}, info) if return_log else ({}, None)
        
        if self.normalize_by_cost:
            denom = np.maximum(self.cost, 1e-12)
            eff = scores / denom
        else:
            eff = scores

        ## entropy
        #p = np.clip(eff, 1e-12, 1.0 - 1e-12)
        #entropy = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)

        # Selection entropy is based on the score actually used for selection.
        # For SCP this may be cost-normalized.
        p_eff = np.clip(eff, 1e-12, 1.0 - 1e-12)
        selection_entropy = -p_eff * np.log(p_eff) - (1.0 - p_eff) * np.log(1.0 - p_eff)

        # pick variable index
        #active_idx = np.flatnonzero(self.active_sets)

        jj_score = 0.0

        #if self.pick_mode == "min_entropy":
        #    #jj = int(active_idx[np.argmin(entropy[active_idx])])
        #    jj = int(cand[np.argmin(entropy[cand])])
        #    jj_score = entropy[cand][np.argmin(entropy[cand])]

        #elif self.pick_mode == "max_eff":
        #    #jj = int(active_idx[np.argmax(eff[active_idx])])
        #    jj = int(cand[np.argmax(eff[cand])])
        #    jj_score = eff[cand][np.argmax(eff[cand])]

        #elif self.pick_mode == "softmax_eff":
        #    #logits = np.log(np.clip(eff[active_idx], 1e-12, None))
        #    logits = np.log(np.clip(eff[cand], 1e-12, None))

        #    local = _sample_softmax(self.rng, logits, temperature=self.j_temperature)
        #    #jj = int(active_idx[local])
        #    jj = int(cand[local])

        #
        #elif self.pick_mode == "softmax_neg_entropy":
        #    #logits = -entropy[active_idx]
        #    logits = -entropy[cand]

        #    local = _sample_softmax(self.rng, logits, temperature=self.j_temperature)
        #    #jj = int(active_idx[local])
        #    jj = int(cand[local])

        if self.pick_mode == "min_entropy":
            jj = int(cand[np.argmin(selection_entropy[cand])])
        
        elif self.pick_mode == "max_eff":
            jj = int(cand[np.argmax(eff[cand])])
        
        elif self.pick_mode == "softmax_eff":
            logits = np.log(np.clip(eff[cand], 1e-12, None))
            local = _sample_softmax(self.rng, logits, temperature=self.j_temperature)
            jj = int(cand[local])
        
        elif self.pick_mode == "softmax_neg_entropy":
            logits = -selection_entropy[cand]
            local = _sample_softmax(self.rng, logits, temperature=self.j_temperature)
            jj = int(cand[local])
        else:
            raise ValueError(f"Unknown pick_mode={self.pick_mode}")
        
        # --- pick jj_val ---
        if self.value_mode == "argmax":
            jj_val = int(np.argmax(mu_np[jj]))
        elif self.value_mode == "sample":
            p = mu_np[jj].astype(float)
            p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            p = np.clip(p, 0.0, None)
            p = p / p.sum() if p.sum() > 0 else np.array([0.5, 0.5])
            jj_val = int(self.rng.choice(2, p=p))
        elif self.value_mode == "add":
            jj_val = 1
        else:
            raise ValueError(f"Unknown value_mode={self.value_mode}")

        info = None
        
        if return_log:
            info = _candidate_diagnostics(
                chosen=jj,
                value=jj_val,
                candidates=cand,
                mu_np=mu_np,
                selection_score=eff,
            )
        
            info.update({
                "score": float(eff[jj]),             # generic legacy field
                "mu1": float(scores[jj]),            # raw marginal p(x_j=1)
                "cost": float(self.cost[jj]),
                "selection_entropy": float(selection_entropy[jj]),
            })
        # Commit: choose set j
        evidence = {}
        self._deactivate_set(jj, int(jj_val), evidence)

        # Update coverage
        newly = []
        if jj_val == 1:
            #for i in self.S[jj]:
            #    i = int(i)
            #    if self.uncovered[i]:
            #        self.uncovered[i] = False
            #        newly.append(i)
            ## Perf improvements
            for i in self.S[jj]:
                self._cover_element(int(i), evidence, newly)
        else:
            # check for items covered by S[j] covered by only 1 active set
            #active_idx = np.flatnonzero(self.active_sets)
            #for a in self.S[jj]:
            #    a = int(a)
            #    if self.uncovered[a]:
            #        sets_covering_a = []
            #        for s_idx in active_idx:
            #            if a in self.S[s_idx]:
            #                sets_covering_a.append(s_idx)
            #        if len(sets_covering_a)==1:
            #            s_idx = int(sets_covering_a[0])
            #            self.uncovered[a] = False
            #            newly.append(a)
            #            evidence[s_idx] = 1
            #            self.x[s_idx] = 1
            #            self.active_sets[s_idx] = False
            #            for ii in self.S[s_idx]:
            #                ii = int(ii)
            #                if self.uncovered[ii]:
            #                    self.uncovered[ii] = False
            #                    newly.append(ii)

            ## Perf improvements
            forced = []
            for a in self.S[jj]:
                a = int(a)
                if self.uncovered[a] and self.active_coverer_count[a] == 1:
                    coverers = self.C[a]
                    active_coverers = coverers[self.active_sets[coverers]]
                    if active_coverers.size == 1:
                        forced.append(int(active_coverers[0]))

            # deduplicate
            if forced:
                forced = list(dict.fromkeys(forced))

            for s_idx in forced:
                if self.active_sets[s_idx]:
                    self._deactivate_set(s_idx, 1, evidence)
                    for ii in self.S[s_idx]:
                        self._cover_element(int(ii), evidence, newly)
        # --- NEW: deactivate sets that cannot cover anything new anymore ---
        # Any active set that has zero uncovered elements left is useless.
        if newly:
            touched_sets = set()
            for i in newly:
                for s_idx in self.C[int(i)]:   # sets that contain element i
                    if self.active_sets[int(s_idx)]:
                        touched_sets.add(int(s_idx))

            for s_idx in touched_sets:
                # check if this set still covers at least one uncovered element
                elems = self.S[s_idx]
                if elems.size == 0 or (not self.uncovered[elems].any()):
                    self.active_sets[s_idx] = False
                    if self.x[s_idx] < 0:
                        self.x[s_idx] = 0
                    evidence[s_idx] = 0
        if return_log:
            assert info is not None

            cand_entropy_vals = selection_entropy[cand]
            cand_mu_vals = scores[cand]
            cand_eff_vals = eff[cand]
        
            entropy_sorted = np.sort(cand_entropy_vals)
            mu_sorted_desc = np.sort(cand_mu_vals)[::-1]
            eff_sorted_desc = np.sort(cand_eff_vals)[::-1]
        
            info.update({
                "newly_covered": newly,
                "n_uncovered": int(self.uncovered.sum()),
                "n_active_after": int(self.active_sets.sum()),
                "n_fixed_by_step": int(len(evidence)),
        
                # Backward-compatible aggregate names.
                "candidate_entropy_mean": float(cand_entropy_vals.mean()),
                "candidate_entropy_std": float(cand_entropy_vals.std()),
                "candidate_entropy_min": float(cand_entropy_vals.min()),
                "candidate_entropy_max": float(cand_entropy_vals.max()),
        
                "candidate_mu_mean": float(cand_mu_vals.mean()),
                "candidate_mu_std": float(cand_mu_vals.std()),
                "candidate_mu_min": float(cand_mu_vals.min()),
                "candidate_mu_max": float(cand_mu_vals.max()),
        
                "chosen_eff": float(eff[jj]),
        
                # Legacy gap-to-mean names.
                "entropy_gap_to_mean": float(selection_entropy[jj] - cand_entropy_vals.mean()),
                "mu1_gap_to_mean": float(scores[jj] - cand_mu_vals.mean()),
                "eff_gap_to_mean": float(eff[jj] - cand_eff_vals.mean()),
        
                "entropy_rank_asc": int(1 + np.sum(cand_entropy_vals < selection_entropy[jj])),
                "mu1_rank_desc": int(1 + np.sum(cand_mu_vals > scores[jj])),
                "eff_rank_desc": int(1 + np.sum(cand_eff_vals > eff[jj])),
        
                "candidate_entropy_second_min": (
                    float(entropy_sorted[1]) if cand.size > 1 else None
                ),
                "candidate_mu_second_max": (
                    float(mu_sorted_desc[1]) if cand.size > 1 else None
                ),
                "candidate_eff_second_max": (
                    float(eff_sorted_desc[1]) if cand.size > 1 else None
                ),
        
                "entropy_gap_best_second": (
                    float(entropy_sorted[1] - entropy_sorted[0]) if cand.size > 1 else None
                ),
                "mu_gap_best_second": (
                    float(mu_sorted_desc[0] - mu_sorted_desc[1]) if cand.size > 1 else None
                ),
                "eff_gap_best_second": (
                    float(eff_sorted_desc[0] - eff_sorted_desc[1]) if cand.size > 1 else None
                ),
            })
        
        return (evidence, info) if return_log else (evidence, None)
    def _prune_nonexclusive_coverers(self, x: np.ndarray, S: List[np.ndarray], n_elems: int) -> np.ndarray:
        """
        Remove redundant selected sets: any selected set that does NOT exclusively cover
        at least one element in the *current* selected cover.
    
        This is done safely via iterative pruning:
          - maintain coverage counts per element under current x==1
          - remove a set j if all elements it covers have count >= 2 (i.e., it covers nothing uniquely)
          - repeat until no more sets can be removed
    
        Returns a new x (does not modify input).
        """
        x = np.asarray(x, dtype=int).copy()
        if x.ndim != 1:
            x = x.reshape(-1)
    
        selected = np.flatnonzero(x == 1)
        if selected.size == 0 or n_elems <= 0:
            return x
    
        # coverage counts
        cover_count = np.zeros(int(n_elems), dtype=int)
        for j in selected:
            elems = S[int(j)]
            if elems.size:
                cover_count[elems] += 1
    
        in_sol = np.zeros_like(x, dtype=bool)
        in_sol[selected] = True
    
        changed = True
        while changed:
            changed = False
            current = np.flatnonzero(in_sol)
            for j in current:
                elems = S[int(j)]
                if elems.size == 0:
                    in_sol[int(j)] = False
                    x[int(j)] = 0
                    changed = True
                    continue
    
                if np.all(cover_count[elems] >= 2):
                    in_sol[int(j)] = False
                    x[int(j)] = 0
                    cover_count[elems] -= 1
                    changed = True
   
        return x

    def solution(self) -> np.ndarray:
        """Undecided -> 0, and also set any remaining active sets to 0."""
        assert self.x is not None
        x = self.x.copy()
        x[x < 0] = 0

        if self.fix_prop < 1.0:
            return x, self.active_sets

        if self.S and self.C:
            x = self._prune_nonexclusive_coverers(x, self.S, n_elems=len(self.C))

        return x


def _sample_softmax(rng: np.random.Generator, logits: np.ndarray, temperature: float = 1.0) -> int:
    z = logits / max(temperature, 1e-12)
    z = z - np.max(z)              # stability
    p = np.exp(z)
    p = p / np.sum(p)
    return int(rng.choice(len(p), p=p))

