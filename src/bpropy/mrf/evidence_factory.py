from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, List

import torch


Evidence = Mapping[int, int]  # var -> {0,1}


def _compute_offsets_from_scopes(scopes: Sequence[Sequence[int]]) -> List[int]:
    offsets = [0]
    total = 0
    for scope in scopes:
        total += 1 << len(scope)
        offsets.append(total)
    return offsets


def _build_var_to_ss_indices(
    n_vars: int,
    factor_scopes: Sequence[Sequence[int]],
    factor_offsets: Sequence[int],
    *,
    little_endian: bool,
    device: torch.device,
) -> Tuple[List[torch.LongTensor], List[torch.LongTensor]]:
    idx0: List[List[int]] = [[] for _ in range(n_vars)]
    idx1: List[List[int]] = [[] for _ in range(n_vars)]

    for f_id, scope in enumerate(factor_scopes):
        k = len(scope)
        start = factor_offsets[f_id]
        block = 1 << k

        for local_pos, v in enumerate(scope):
            if v < 0 or v >= n_vars:
                raise ValueError(f"Variable index out of range: v={v} for n_vars={n_vars}")

            bitpos = local_pos if little_endian else (k - 1 - local_pos)

            for a in range(block):
                bit = (a >> bitpos) & 1
                if bit == 0:
                    idx0[v].append(start + a)
                else:
                    idx1[v].append(start + a)

    idx0_t = [torch.tensor(lst, dtype=torch.long, device=device) for lst in idx0]
    idx1_t = [torch.tensor(lst, dtype=torch.long, device=device) for lst in idx1]
    return idx0_t, idx1_t


@dataclass
class EvidenceMaskFactory:
    """
    Evidence clamping helper for theta/SS vectors built from a FactorGraph.

    Assumptions (must match your theta/SS construction):
      - theta is the concatenation of per-factor truth-table parameters
      - factor blocks are ordered the same as fg.iter_factors()
      - within each factor, the assignment index a in [0,2^k) encodes bits for variables
      - little_endian controls whether bit j corresponds to scope[j] (True) or scope[k-1-j] (False)

    The factory maintains a cumulative boolean mask over theta indices that remain allowed.
    """
    fg: object  # FactorGraph duck-type
    n_vars: int
    factor_scopes: Sequence[Sequence[int]]
    factor_offsets: Sequence[int]
    idx0: Sequence[torch.LongTensor]
    idx1: Sequence[torch.LongTensor]

    device: torch.device
    mask: torch.BoolTensor
    evidence: Dict[int, int]
    neg_inf: float
    little_endian: bool

    def __init__(
        self,
        fg: object,
        *,
        theta: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        little_endian: bool = True,
        neg_inf: Optional[float] = None,
        validate_theta_length: bool = True,
    ) -> None:
        self.fg = fg
        self.little_endian = little_endian

        # ---- Extract graph structure (order matters!) ----
        # These methods match what you used in your script.
        n_vars = int(fg.number_of_variables()) if hasattr(fg, "number_of_variables") else int(fg.number_of_nodes())
        self.n_vars = n_vars

        factors = list(fg.iter_factors())
        scopes = [list(fg.factor_variables(F)) for F in factors]
        self.factor_scopes = scopes
        self.factor_offsets = _compute_offsets_from_scopes(scopes)

        total_size = self.factor_offsets[-1]

        # ---- Device selection ----
        if device is None:
            device = theta.device if theta is not None else torch.device("cpu")
        self.device = device

        # ---- neg_inf selection ----
        if neg_inf is None:
            if theta is not None and theta.dtype.is_floating_point:
                self.neg_inf = float(torch.finfo(theta.dtype).min / 2)
            else:
                self.neg_inf = -1e9
        else:
            self.neg_inf = float(neg_inf)

        # ---- Optional sanity checks ----
        if theta is not None and validate_theta_length and theta.numel() != total_size:
            raise ValueError(
                f"theta length mismatch: theta.numel()={theta.numel()} but expected {total_size} "
                f"(sum_f 2^|scope_f| over fg.iter_factors() order)"
            )

        # ---- Precompute index maps and initialize state ----
        self.idx0, self.idx1 = _build_var_to_ss_indices(
            n_vars,
            scopes,
            self.factor_offsets,
            little_endian=little_endian,
            device=device,
        )

        self.mask = torch.ones(total_size, dtype=torch.bool, device=device)
        self.evidence = {}

    @property
    def size(self) -> int:
        return int(self.mask.numel())

    def reset(self) -> None:
        self.mask.fill_(True)
        self.evidence.clear()

    def clone(self) -> "EvidenceMaskFactory":
        new = EvidenceMaskFactory(
            self.fg,
            theta=None,  # not needed for clone
            device=self.device,
            little_endian=self.little_endian,
            neg_inf=self.neg_inf,
            validate_theta_length=False,
        )
        # share precomputed maps/scopes/offsets
        new.n_vars = self.n_vars
        new.factor_scopes = self.factor_scopes
        new.factor_offsets = self.factor_offsets
        new.idx0 = self.idx0
        new.idx1 = self.idx1
        # copy state
        new.mask = self.mask.clone()
        new.evidence = dict(self.evidence)
        return new

    def add_evidence(self, ev: Evidence, *, strict: bool = True) -> None:
        """
        Add evidence cumulatively (monotone). Contradictions raise unless strict=False.
        """
        for v, x in ev.items():
            x = int(x)
            if x not in (0, 1):
                raise ValueError(f"Evidence value must be 0 or 1, got {x} for v={v}")
            if v < 0 or v >= self.n_vars:
                raise ValueError(f"Variable index out of range: v={v} for n_vars={self.n_vars}")

            if v in self.evidence and self.evidence[v] != x:
                if strict:
                    raise ValueError(f"Contradictory evidence for v={v}: {self.evidence[v]} vs {x}")
                else:
                    continue

            if self.evidence.get(v, None) == x:
                continue

            self.evidence[v] = x

            # Clamp: forbid assignments where variable takes the opposite value
            bad = self.idx1[v] if x == 0 else self.idx0[v]
            if bad.device != self.device:
                bad = bad.to(self.device)
            self.mask[bad] = False

    def apply(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Return clamped copy of theta where masked-out entries are set to neg_inf.
        """
        if theta.numel() != self.size:
            raise ValueError(f"theta length mismatch: {theta.numel()} vs mask size {self.size}")
        if theta.device != self.device:
            raise ValueError(f"theta.device {theta.device} != factory.device {self.device}")

        neg = torch.full_like(theta, self.neg_inf)
        return torch.where(self.mask, theta, neg)

    def apply_inplace(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Clamp theta in-place (modifies theta).
        """
        if theta.numel() != self.size:
            raise ValueError(f"theta length mismatch: {theta.numel()} vs mask size {self.size}")
        if theta.device != self.device:
            raise ValueError(f"theta.device {theta.device} != factory.device {self.device}")

        theta[~self.mask] = self.neg_inf
        return theta

    def forbidden_fraction(self) -> float:
        return float((~self.mask).float().mean().item())

