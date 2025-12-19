# bpropy/io/serde.py
from __future__ import annotations
import json, gzip, os
from dataclasses import asdict
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union, Iterable

import numpy as np
try:
    import scipy.sparse as sp
except Exception:
    sp = None

# ---------- helpers ----------
def _open_text(path: Union[str, Path], mode="rt"):
    p = str(path)
    return gzip.open(p, mode) if p.endswith(".gz") else open(p, mode)

def _dump_json(obj: Dict[str, Any], path: Union[str, Path]):
    with _open_text(path, "wt") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def _load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with _open_text(path, "rt") as f:
        return json.load(f)

def _npz_save(path_npz: Union[str, Path], **arrays):
    np.savez_compressed(path_npz, **arrays)

def _npz_load(path_npz: Union[str, Path]) -> Dict[str, np.ndarray]:
    return dict(np.load(path_npz, allow_pickle=False))

# ---------- LPProblem base mixin ----------
class SerializableLPMixin:
    SCHEMA = "bpropy-lp"
    VERSION = 1

    # --- required on subclasses ---
    def _to_serializable(self, use_npz: bool, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def _from_serializable(cls, man: Dict[str, Any], arrays: Dict[str, np.ndarray]):
        raise NotImplementedError

    # --- public API ---
    def save(self, stem: Union[str, Path], *, split_large: bool = True, threshold_bytes: int = 200_000):
        """
        Save to stem.json[.gz] (+ optional stem.arrays.npz).
        If split_large True and arrays exceed threshold, store in NPZ.
        """
        stem = Path(stem)
        arrays: Dict[str, np.ndarray] = {}
        manifest = {
            "schema": self.SCHEMA,
            "version": self.VERSION,
            "problem_type": type(self).__name__.upper(),   # "MIS" or "SCP"
            "metadata": {}
        }
        use_npz = split_large
        payload = self._to_serializable(use_npz=use_npz, arrays=arrays, threshold_bytes=threshold_bytes)
        manifest.update(payload)

        # Decide whether to split
        total_bytes = sum(a.nbytes for a in arrays.values())
        if not split_large or total_bytes == 0 or total_bytes < threshold_bytes:
            # inline arrays as lists
            for k, a in arrays.items():
                manifest.setdefault("_arrays", {})[k] = a.tolist()
            _dump_json(manifest, stem.with_suffix(".json"))
        else:
            # write manifest + npz
            manifest["arrays_file"] = stem.with_suffix(".arrays.npz").name
            _dump_json(manifest, stem.with_suffix(".json"))
            _npz_save(stem.with_suffix(".arrays.npz"), **arrays)

    @classmethod
    def load(cls, stem: Union[str, Path]):
        stem = Path(stem)
        man = _load_json(stem if stem.suffix in (".json",".gz") else stem.with_suffix(".json"))

        # schema/version check
        assert man.get("schema") == cls.SCHEMA, f"Unexpected schema: {man.get('schema')}"
        # Forward-compat: allow newer version if subclass handles it; we keep ==1 here
        assert int(man.get("version", 0)) == 1, "Unsupported version"

        # Gather arrays
        arrays: Dict[str, np.ndarray] = {}
        if "arrays_file" in man:
            arrays.update(_npz_load(stem.with_name(man["arrays_file"])))
        if "_arrays" in man:
            for k, lst in man["_arrays"].items():
                arrays[k] = np.array(lst)

        # Dispatch to the right class
        pt = man.get("problem_type", "").upper()
        if pt == "MIS":
            from bpropy.co.mis import MIS
            return MIS._from_serializable(man, arrays)
        elif pt == "SCP":
            from bpropy.co.scp import SCP
            return SCP._from_serializable(man, arrays)
        else:
            raise ValueError(f"Unknown problem_type {pt}")

    def save_assignments(
        self,
        stem: str,
        assignments: Dict[str, np.ndarray],
        *,
        threshold_bytes: int = 8000,
        arrays_key_prefix: str = "assign_",
    ) -> None:
        """
        Save a collection of variable assignments (name -> vector) alongside the problem.
        Stores vectors inline (JSON) or into the NPZ sidecar depending on threshold_bytes.
        """
        man_path = stem + ".json"
        with open(man_path, "r") as f:
            manifest = json.load(f)

        # Load existing arrays if sidecar present, so we can append
        arrays = {}
        sidecar_path = None
        if "arrays_file" in manifest:
            sidecar_path = os.path.join(os.path.dirname(stem), manifest["arrays_file"])
            arrays.update(dict(np.load(sidecar_path, allow_pickle=False)))

        manifest.setdefault("assignments", {})
        manifest.setdefault("_arrays", {})  # used only for inline storage

        for name, x in assignments.items():
            if x is None:
                continue
            x = np.asarray(x, dtype=float).reshape(-1)
            key = f"{arrays_key_prefix}{name}"

            if x.nbytes >= threshold_bytes:
                arrays[key] = x
                manifest["assignments"][name] = {"source": "npz", "key": key}
            else:
                manifest["_arrays"][key] = x.tolist()
                manifest["assignments"][name] = {"source": "inline", "key": key}

        # If we have any arrays (pre-existing or newly added), ensure arrays_file is set and write the sidecar
        if arrays:
            if "arrays_file" not in manifest:
                manifest["arrays_file"] = os.path.basename(stem + ".arrays.npz")
            sidecar_path = os.path.join(os.path.dirname(stem), manifest["arrays_file"])
            np.savez_compressed(sidecar_path, **arrays)

        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    @classmethod
    def load_assignments(cls, stem: str) -> Dict[str, np.ndarray]:
        """
        Load all named assignments saved by save_assignments().
        Returns a dict: name -> np.ndarray
        """
        man_path = stem + ".json"
        with open(man_path, "r") as f:
            manifest = json.load(f)

        arrays = {}
        if "arrays_file" in manifest:
            sidecar_path = os.path.join(os.path.dirname(stem), manifest["arrays_file"])
            arrays.update(dict(np.load(sidecar_path, allow_pickle=False)))

        out = {}
        for name, ref in manifest.get("assignments", {}).items():
            if ref["source"] == "npz":
                out[name] = np.asarray(arrays[ref["key"]], dtype=float)
            else:
                out[name] = np.asarray(manifest["_arrays"][ref["key"]], dtype=float)
        return out
