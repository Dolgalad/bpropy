import math
import random
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


def _is_tensor(x) -> bool:
    return isinstance(x, torch.Tensor)


def _pretty(x):
    if _is_tensor(x):
        return f"Tensor(shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device})"
    return f"{type(x)}"


def _get_factor_sizes_for_ss(data) -> Optional[torch.Tensor]:
    """
    Try to infer the list of factor arities that correspond to the flattened sufficient-statistics vector.
    This is library-specific, so we attempt a few common conventions.

    Returns a 1D tensor of factor sizes, or None if we can't infer.
    """
    # Case A: data already has factor_sizes_ss or similar
    for name in ["factor_sizes_ss", "ss_factor_sizes", "factor_sizes_factors"]:
        if hasattr(data, name):
            fs = getattr(data, name)
            if _is_tensor(fs) and fs.ndim == 1:
                return fs

    # Case B: homo bipartite graphs: factor nodes are bipartite==1 (constraints/factors)
    if hasattr(data, "bipartite") and hasattr(data, "factor_sizes"):
        bip = data.bipartite
        fs_all = data.factor_sizes
        if _is_tensor(bip) and _is_tensor(fs_all) and bip.ndim == 1 and fs_all.ndim == 1:
            # typical convention: factor nodes have bipartite==1
            fs = fs_all[bip == 1]
            if fs.numel() > 0:
                return fs

    return None


def _block_offsets_from_factor_sizes(fs: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given factor sizes [k1,k2,...], blocks have lengths 2^k.
    Returns (starts, lengths) as 1D int64 tensors.
    """
    lens = torch.tensor([1 << int(k) for k in fs], dtype=torch.long)
    starts = torch.zeros_like(lens)
    if lens.numel() > 0:
        starts[1:] = torch.cumsum(lens[:-1], dim=0)
    return starts, lens


@torch.no_grad()
def audit_batch(
    data: Any,
    pred: Optional[Any] = None,
    cfg: Optional[Any] = None,
    *,
    max_blocks_to_check: int = 10,
    seed: int = 0,
    strict: bool = True,
) -> None:
    """
    Run a series of audits on a single batch.

    - Checks feature tensors (x/e), bipartite mask, factor sizes
    - Checks that y_ss is a valid one-hot concatenation over factor blocks
    - Checks expected y_ss length matches sum(2^k)
    - Checks model outputs contract vs loss (best effort, based on pred structure)

    If strict=True, raises AssertionError on hard failures.
    """
    rnd = random.Random(seed)

    print("\n" + "=" * 88)
    print("[AUDIT] Batch overview")
    print("=" * 88)

    # ---- 1) Feature tensors
    print("\n[AUDIT-1] Feature tensors")
    if hasattr(data, "x"):
        print("data.x:", _pretty(data.x))
        if strict:
            assert _is_tensor(data.x), f"data.x must be Tensor, got {_pretty(data.x)}"
            assert data.x.ndim == 2, f"data.x expected 2D [num_nodes, feat_dim], got {tuple(data.x.shape)}"
    else:
        print("data.x: <missing>")

    if hasattr(data, "e"):
        print("data.e:", _pretty(data.e))
        if hasattr(data, "edge_index") and _is_tensor(data.edge_index) and _is_tensor(data.e):
            E = int(data.edge_index.shape[1])
            print(f"  edge_index: shape={tuple(data.edge_index.shape)}  (#edges={E})")
            print(f"  data.e: shape={tuple(data.e.shape)}")
            if data.e.ndim == 1:
                if strict:
                    assert data.e.numel() == E, (
                        f"1D data.e should have length E (#edges). "
                        f"Got len(e)={data.e.numel()} vs E={E}. "
                        f"This usually means e is built on a different edge list than edge_index."
                    )
            elif data.e.ndim == 2:
                if strict:
                    assert data.e.shape[0] == E, (
                        f"2D data.e should be [E,F]. Got e.shape[0]={data.e.shape[0]} vs E={E}."
                    )
    else:
        print("data.e: <missing>")

    # ---- 2) Bipartite + factor_sizes sanity
    print("\n[AUDIT-2] Bipartite mask and factor sizes")
    if hasattr(data, "bipartite"):
        bip = data.bipartite
        print("data.bipartite:", _pretty(bip))
        if strict:
            assert _is_tensor(bip) and bip.ndim == 1, "data.bipartite should be 1D Tensor"
        if _is_tensor(bip):
            nv = int((bip == 0).sum().item())
            nf = int((bip == 1).sum().item())
            print(f"  #vars(bip==0)={nv}  #factors/constraints(bip==1)={nf}  total={bip.numel()}")
    else:
        bip = None
        print("data.bipartite: <missing>")

    if hasattr(data, "factor_sizes"):
        fs_all = data.factor_sizes
        print("data.factor_sizes:", _pretty(fs_all))
        if _is_tensor(fs_all):
            uniq = torch.unique(fs_all.cpu())
            print("  factor_sizes unique:", uniq.tolist())
            if strict:
                assert fs_all.ndim == 1, "data.factor_sizes should be 1D"
                assert torch.all(fs_all >= 1), "factor_sizes must be >=1"
            if _is_tensor(bip) and fs_all.numel() == bip.numel():
                # Often variable nodes have unary size == 1
                vars_fs = fs_all[bip == 0]
                if vars_fs.numel() > 0:
                    frac_unary = float((vars_fs == 1).float().mean().item())
                    print(f"  vars unary fraction (factor_sizes==1): {frac_unary:.3f}")
    else:
        fs_all = None
        print("data.factor_sizes: <missing>")

    # ---- 3) y_ss (sufficient statistics) structure checks
    print("\n[AUDIT-3] y_ss structure (one-hot blocks)")
    y_ss = getattr(data, "y_ss", None)
    if y_ss is None and hasattr(data, "y"):
        # Some pipelines store it as y; best-effort
        y_ss = getattr(data, "y")

    if y_ss is None:
        print("data.y_ss / data.y: <missing>  (skip SS checks)")
    else:
        print("target vector:", _pretty(y_ss))
        if strict:
            assert _is_tensor(y_ss) and y_ss.ndim == 1, "target SS vector must be 1D Tensor"

        fs_ss = _get_factor_sizes_for_ss(data)
        if fs_ss is None:
            print("  Could not infer factor sizes for SS blocks. "
                  "If you have them, store as data.factor_sizes_ss.")
        else:
            fs_ss_cpu = fs_ss.detach().cpu().long().tolist()
            starts, lens = _block_offsets_from_factor_sizes(fs_ss_cpu)
            total_expected = int(lens.sum().item())
            total_actual = int(y_ss.numel())

            print(f"  inferred #SS blocks={len(fs_ss_cpu)}  expected_len=sum(2^k)={total_expected}  actual_len={total_actual}")
            if strict:
                assert total_expected == total_actual, (
                    f"SS length mismatch: expected {total_expected} from factor sizes, got {total_actual}. "
                    "This strongly suggests SS/FG ordering mismatch or wrong FG used for SS."
                )

            # Check one-hot on a random subset of blocks
            idxs = list(range(len(fs_ss_cpu)))
            rnd.shuffle(idxs)
            idxs = idxs[: min(max_blocks_to_check, len(idxs))]

            bad = 0
            for bi in idxs:
                s = int(starts[bi].item())
                L = int(lens[bi].item())
                block = y_ss[s : s + L]
                sm = float(block.sum().item())
                mx = float(block.max().item())
                mn = float(block.min().item())
                # if it’s one-hot, sum==1 and max==1 and values are 0/1
                is_binary = bool(torch.all((block == 0) | (block == 1)).item())
                ok = (abs(sm - 1.0) < 1e-6) and (abs(mx - 1.0) < 1e-6) and (mn >= 0.0) and is_binary
                if not ok:
                    bad += 1
                    print(f"  [BAD BLOCK] i={bi} k={fs_ss_cpu[bi]} L={L} sum={sm:.6f} min={mn:.3f} max={mx:.3f} binary={is_binary}")
            print(f"  checked {len(idxs)} blocks: bad={bad}")
            if strict:
                assert bad == 0, (
                    "Some SS blocks are not one-hot / not binary. "
                    "This indicates a bug in x2ss/target generation or a mismatch between factor sizes and SS layout."
                )

            # Optional: check approx assumption (unary+pairwise)
            uniq_ss = sorted(set(fs_ss_cpu))
            print("  SS factor-size set:", uniq_ss)
            if cfg is not None:
                # If you *expect* only {1,2}, enforce it:
                if strict and all(isinstance(k, int) for k in uniq_ss):
                    assert set(uniq_ss).issubset({1, 2}), (
                        f"Expected only unary/pairwise factors for approx runs, got factor sizes {uniq_ss}. "
                        "This can cause irreducible loss floors if the model only represents [1,2]."
                    )

    # ---- 4) Output/Loss contract checks (best effort)
    print("\n[AUDIT-4] pred / loss contract (best effort)")
    if pred is None:
        print("pred: <none> (skip output checks)")
    else:
        if isinstance(pred, dict):
            print("pred keys:", list(pred.keys()))
            for k, v in pred.items():
                if _is_tensor(v):
                    print(f"  pred[{k}] -> {tuple(v.shape)} {v.dtype}")
        else:
            print("pred:", _pretty(pred))

        # If cfg is available, check loss selection
        loss_name = None
        if cfg is not None:
            # try common patterns
            loss_name = getattr(cfg, "loss", None)
            if isinstance(loss_name, dict):
                loss_name = loss_name.get("name") or loss_name.get("type")
        print("loss_name (from cfg, best effort):", loss_name)

        # Heuristic assertions:
        # - If loss is fce_logits -> must have factor logits-like tensor
        if loss_name in ("fce_logits",):
            ok = isinstance(pred, dict) and any("logit" in k or "factor" in k for k in pred.keys())
            if strict:
                assert ok, (
                    "Using fce_logits but pred does not appear to contain factor logits. "
                    "Ensure forward returns factor logits for each factor configuration."
                )

    print("\n[AUDIT] OK (no hard failures)" if not strict else "\n[AUDIT] OK (passed strict checks)")
    print("=" * 88 + "\n")

