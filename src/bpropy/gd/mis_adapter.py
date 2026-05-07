from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple, List

import numpy as np
import torch

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


class MISAdapter:
    """
    Decisions based ONLY on pb.g (original MIS graph).
    Evidence semantics: x[v]=1 implies x[u]=0 for all u in N(v).

    For fix-and-solve:
      - x[v] = -1 means undecided
      - active[v] = True means free / undecided
      - solution() returns (x, active) when fix_prop < 1.0
    """

    def __init__(
        self,
        pick_mode: str = "max_mu",
        value_mode: str = "add",
        seed: Optional[int] = None,
        fix_prop: Optional[float] = None,
        **kwargs,
    ):
        self.pick_mode = pick_mode
        self.value_mode = value_mode
        self.rng = np.random.default_rng(seed)

        if fix_prop is not None:
            self.fix_prop = float(fix_prop)
        else:
            self.fix_prop = 1.0

        self.pb = None
        self.active = None
        self.x = None
        self.n = None

    def reset(self, pb):
        self.pb = pb
        self.n = int(pb.nv()) if hasattr(pb, "nv") else int(pb.n)
        self.active = np.ones(self.n, dtype=bool)
        self.x = -np.ones(self.n, dtype=int)   # -1 = undecided, 0/1 decided

    def is_done(self) -> bool:
        assert self.active is not None
        assert self.n is not None

        # Stop when no vertices remain, or when the requested fixed proportion
        # has been reached. This matches the SCP fix-and-solve behavior.
        return (not self.active.any()) or (
            self.active.sum() <= self.n * (1.0 - self.fix_prop)
        )

    def step(self, backend_output, return_log: bool = False):
        """
        backend_output expected: mu shaped (n,2) or (2n,) convertible.
        """
        assert self.active is not None and self.x is not None

        mu = backend_output
        if mu is None:
            raise ValueError("MISAdapter currently expects mu from an inference backend.")

        mu_np = _as_mu2_numpy(mu)
        scores = mu_np[:, 1].astype(float)

        active_idx = np.flatnonzero(self.active)
        if active_idx.size == 0:
            info = {
                "chosen": None,
                "value": None,
                "reason": "no_active_vertices",
                "n_candidates": 0,
            }
            return ({}, info) if return_log else ({}, None)

        entropy = _binary_entropy_from_mu1(scores)

        if self.pick_mode in ("max_mu", "max_score"):
            u = int(active_idx[np.argmax(scores[active_idx])])
        elif self.pick_mode == "min_entropy":
            u = int(active_idx[np.argmin(entropy[active_idx])])
        elif self.pick_mode == "softmax_mu":
            logits = np.log(np.clip(scores[active_idx], 1e-12, None))
            local = _sample_softmax(self.rng, logits)
            u = int(active_idx[local])
        else:
            raise ValueError(f"Unknown MIS pick_mode={self.pick_mode}")

        # For MIS, the natural decimation action is to add the chosen vertex
        # to the independent set and force its neighbors to 0.
        if self.value_mode in ("add", "argmax"):
            u_val = 1
        elif self.value_mode == "sample":
            p = mu_np[u].astype(float)
            p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            p = np.clip(p, 0.0, None)
            p = p / p.sum() if p.sum() > 0 else np.array([0.5, 0.5])
            u_val = int(self.rng.choice(2, p=p))
        else:
            raise ValueError(f"Unknown MIS value_mode={self.value_mode}")

        evidence = {u: u_val}

        if u_val == 1:
            for v in self.pb.g.neighbors(u):
                v = int(v)
                if self.active[v]:
                    evidence[v] = 0

        for v, val in evidence.items():
            self.x[int(v)] = int(val)
            self.active[int(v)] = False

        info = None
        if return_log:
            info = _candidate_diagnostics(
                chosen=u,
                value=u_val,
                candidates=active_idx,
                mu_np=mu_np,
                selection_score=scores,
            )

            candidate_scores = scores[active_idx]
            candidate_entropy = entropy[active_idx]
            entropy_sorted = np.sort(candidate_entropy)
            mu_sorted_desc = np.sort(candidate_scores)[::-1]

            info.update({
                "score": float(scores[u]),
                "mu1": float(scores[u]),
                "n_active_after": int(self.active.sum()),
                "n_fixed_by_step": int(len(evidence)),

                "candidate_entropy_mean": float(candidate_entropy.mean()),
                "candidate_entropy_std": float(candidate_entropy.std()),
                "candidate_entropy_min": float(candidate_entropy.min()),
                "candidate_entropy_max": float(candidate_entropy.max()),

                "candidate_mu_mean": float(candidate_scores.mean()),
                "candidate_mu_std": float(candidate_scores.std()),
                "candidate_mu_min": float(candidate_scores.min()),
                "candidate_mu_max": float(candidate_scores.max()),

                "entropy_gap_to_mean": float(info["chosen_entropy"] - candidate_entropy.mean()),
                "mu1_gap_to_mean": float(scores[u] - candidate_scores.mean()),

                "entropy_rank_asc": int(1 + np.sum(candidate_entropy < info["chosen_entropy"])),
                "mu1_rank_desc": int(1 + np.sum(candidate_scores > scores[u])),

                "candidate_entropy_second_min": (
                    float(entropy_sorted[1]) if active_idx.size > 1 else None
                ),
                "candidate_mu_second_max": (
                    float(mu_sorted_desc[1]) if active_idx.size > 1 else None
                ),

                "entropy_gap_best_second": (
                    float(entropy_sorted[1] - entropy_sorted[0]) if active_idx.size > 1 else None
                ),
                "mu_gap_best_second": (
                    float(mu_sorted_desc[0] - mu_sorted_desc[1]) if active_idx.size > 1 else None
                ),
            })

        return (evidence, info) if return_log else (evidence, None)

    def solution(self):
        assert self.x is not None
        assert self.active is not None

        x = self.x.copy()

        #if self.fix_prop < 1.0:
        #    # Important: keep undecided variables free for the reduced solver.
        #    # Do NOT convert them to 0 before returning the partial solution.
        #    return x, self.active.copy()
        if self.fix_prop < 1.0:
            # Match SCPAdapter semantics: sol contains numeric values everywhere,
            # while free tells the reduced solver which variables remain unfixed.
            x[x < 0] = 0
            return x, self.active.copy()

        # Full greedy decode: undecided -> 0.
        x[x < 0] = 0
        return x

def _sample_softmax(rng: np.random.Generator, logits: np.ndarray, temperature: float = 1.0) -> int:
    z = logits / max(temperature, 1e-12)
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    return int(rng.choice(len(p), p=p))
