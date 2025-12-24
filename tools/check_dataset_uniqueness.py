#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from collections import defaultdict

def file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: check_dataset_uniqueness.py DATASET_DIR", file=sys.stderr)
        return 2

    root = Path(sys.argv[1])
    if not root.exists():
        print(f"Directory not found: {root}", file=sys.stderr)
        return 2

    # Consider only instance-defining files
    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix in {".json", ".npz"}
        and not p.name.endswith("_sol.npy")
    )

    if not files:
        print("No dataset files found.")
        return 1

    seen = defaultdict(list)

    for p in files:
        h = file_hash(p)
        seen[h].append(p)

    duplicates = {h: ps for h, ps in seen.items() if len(ps) > 1}

    print(f"Checked {len(files)} files")
    print(f"Unique instances: {len(seen)}")

    if not duplicates:
        print("✅ No duplicates found")
        return 0

    print(f"❌ Found {len(duplicates)} duplicate groups:\n")
    for h, ps in duplicates.items():
        print(f"Hash {h}:")
        for p in ps:
            print(f"  - {p}")
        print()

    return 1

if __name__ == "__main__":
    raise SystemExit(main())

