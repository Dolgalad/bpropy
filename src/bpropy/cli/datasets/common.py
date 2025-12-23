# src/bpropy/cli/common.py
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


# -----------------------------
# Timing
# -----------------------------
def time_f(func, *args, **kwargs):
    t0 = time.time()
    r = func(*args, **kwargs)
    return time.time() - t0, r


# -----------------------------
# Range parsing / sampling
# -----------------------------
IntRange = Tuple[int, int]
FloatRange = Tuple[float, float]


def parse_int_range(s: str) -> IntRange:
    """Parse 'k' or 'a:b' (inclusive) -> (a, b)."""
    if ":" in s:
        a, b = s.split(":")
        lo, hi = int(a), int(b)
    else:
        lo = hi = int(s)
    if lo > hi:
        raise ValueError(f"Invalid int range '{s}' (lo>hi).")
    return lo, hi


def parse_float_range(s: Optional[str]) -> Optional[FloatRange]:
    """Parse None or 'x' or 'a:b' (inclusive-ish) -> (a, b)."""
    if s is None:
        return None
    if ":" in s:
        a, b = s.split(":")
        lo, hi = float(a), float(b)
    else:
        lo = hi = float(s)
    if lo > hi:
        raise ValueError(f"Invalid float range '{s}' (lo>hi).")
    return lo, hi


def sample_int(rng: np.random.Generator, bounds: IntRange) -> int:
    lo, hi = bounds
    return int(rng.integers(lo, hi + 1))


def sample_float(rng: np.random.Generator, bounds: FloatRange) -> float:
    lo, hi = bounds
    if lo == hi:
        return float(lo)
    # Make the upper bound effectively inclusive.
    x = rng.uniform(lo, np.nextafter(hi, np.float64(np.inf)))
    return float(x)


# -----------------------------
# Splits
# -----------------------------
SplitCounts = Tuple[int, int, int]  # (train, val, test)
SplitName = str  # "train" | "val" | "test"


def normalize_splits(splits: Optional[Sequence[int]], n_total: int) -> Optional[SplitCounts]:
    """
    Convert split percentages (or weights) into exact counts that sum to n_total.
    Example: [80,10,10] and n_total=57 -> (46, 6, 5) or similar.
    """
    if splits is None:
        return None
    if len(splits) != 3 or sum(splits) <= 0:
        raise ValueError("--splits expects three non-negative ints with positive sum (e.g. 80 10 10).")
    p = np.array(splits, dtype=float)
    p /= p.sum()
    counts = np.floor(p * n_total + 1e-9).astype(int)
    # distribute remainder
    for i in range(n_total - int(counts.sum())):
        counts[i % 3] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def split_from_index(idx_global: int, counts: SplitCounts) -> SplitName:
    """
    Stable split membership depending only on idx_global and global counts.
    """
    n_tr, n_va, _n_te = counts
    if idx_global < n_tr:
        return "train"
    if idx_global < n_tr + n_va:
        return "val"
    return "test"


def ensure_split_dirs(base: Union[str, Path], split_counts: Optional[SplitCounts]) -> None:
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    if split_counts is not None:
        for sub in ("train", "val", "test"):
            (base / sub).mkdir(parents=True, exist_ok=True)


# -----------------------------
# Sharding / indexing
# -----------------------------
@dataclasses.dataclass(frozen=True)
class ShardPlan:
    shard_id: int
    num_shards: int
    start: int
    stop: int  # exclusive

    @property
    def size(self) -> int:
        return self.stop - self.start


def block_partition(n_total: int, shard_id: int, num_shards: int) -> ShardPlan:
    """
    Partition [0, n_total) into num_shards contiguous blocks (balanced by at most 1).
    """
    if num_shards <= 0:
        raise ValueError("num_shards must be >= 1.")
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id must be in [0, {num_shards-1}].")
    q, r = divmod(n_total, num_shards)
    start = shard_id * q + min(shard_id, r)
    stop = start + q + (1 if shard_id < r else 0)
    return ShardPlan(shard_id=shard_id, num_shards=num_shards, start=start, stop=stop)


def iter_indices(
    n_total: int,
    *,
    # Preferred:
    shard_id: Optional[int] = None,
    num_shards: Optional[int] = None,
    # Backward compat:
    index_offset: int = 0,
    n_local: Optional[int] = None,
) -> Tuple[Iterable[int], int, Optional[ShardPlan]]:
    """
    Returns (indices, n_total_global, shardplan_or_none).

    Preferred mode: pass shard_id and num_shards; indices are a block partition of [0, n_total).
    Backward mode: pass index_offset and n_local; indices are [index_offset, index_offset+n_local).
    """
    if shard_id is not None or num_shards is not None:
        if shard_id is None or num_shards is None:
            raise ValueError("Provide both --shard-id and --num-shards.")
        plan = block_partition(n_total, shard_id, num_shards)
        return range(plan.start, plan.stop), n_total, plan

    # backward compatibility
    if n_local is None:
        if index_offset < 0 or index_offset > n_total:
            raise ValueError("--index-offset must be in [0, total].")
        n_local = n_total - index_offset
    if index_offset < 0:
        raise ValueError("--index-offset must be >= 0.")
    return range(index_offset, index_offset + n_local), n_total, None


# -----------------------------
# Output helpers / resume
# -----------------------------
def default_instance_stem(output_dir: Union[str, Path], idx_global: int, split: Optional[str], prefix: str = "") -> Path:
    out = Path(output_dir)
    if split:
        out = out / split
    return out / f"{prefix}{idx_global}"


def exists_any_artifact(stem: Union[str, Path]) -> bool:
    """
    Your pb.save(...) likely writes <stem>.json (+ optional <stem>.npz).
    We'll consider existence of either as 'already generated'.
    """
    stem = Path(stem)
    return stem.with_suffix(".json").exists() or stem.with_suffix(".npz").exists() or stem.exists()


def write_metadata_once(output_dir: Union[str, Path], metadata: Mapping[str, Any], *, filename: str = "metadata.json") -> None:
    """
    Create output_dir/metadata.json if it does not exist yet.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    if path.exists():
        return
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(metadata), f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


# -----------------------------
# Stable split hashing (optional)
# -----------------------------
def stable_bucket(seed: int, idx_global: int, num_buckets: int = 10_000) -> int:
    """
    Deterministic bucket in [0, num_buckets).
    Useful if you want splits independent of N_total counts.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    h.update(b":")
    h.update(str(idx_global).encode("utf-8"))
    return int.from_bytes(h.digest(), "little") % num_buckets

