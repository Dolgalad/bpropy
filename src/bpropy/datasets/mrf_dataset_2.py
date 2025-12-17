# bpropy/data/mrf_dataset.py
import inspect
import os, json, hashlib, time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, HeteroData

from bpropy.io.serde import SerializableLPMixin  # base mixin your MIS/SCP inherit
from bpropy.co.mis import MIS
from bpropy.co.scp import SCP
from bpropy.adapters.vc_to_data import problem_to_data  # your helper

ProblemT = Union[MIS, SCP]
PreprocFn = Callable[[ProblemT], Data]

@dataclass
class MRFDatasetConfig:
    feature_mode: Optional[str] = None      # e.g., "EB","CC","CF","EC","CP" (for MIS)
    preproc_version: str = "v1"             # bump to invalidate caches
    yield_mode: str = "both"                # "data" | "problem" | "both"
    # if True, skip cache and recompute; if False, read/write cache normally
    refresh: bool = False
    return_assignment: bool = False
    assignment_solver: Optional[str] = None

def _list_stems(split_dir: str) -> List[str]:
    stems = []
    for fname in os.listdir(split_dir):
        if fname.endswith(".json"):
            stems.append(os.path.join(split_dir, fname[:-5]))  # strip .json
    stems.sort()
    return stems

def _read_manifest(stem: str) -> dict:
    with open(stem + ".json", "r") as f:
        return json.load(f)

def _infer_type(man: dict) -> str:
    t = man.get("problem_type")
    if t in ("MIS", "SCP"):
        return t
    # fallback: folder name or fields present
    raise ValueError(f"Cannot infer problem_type for {man}")

def _load_problem_by_type(stem: str, prob_type: str) -> ProblemT:
    if prob_type == "MIS":
        return MIS.load(stem)
    elif prob_type == "SCP":
        return SCP.load(stem)
    else:
        raise ValueError(prob_type)

def _load_assignment_from_stem(pb, stem: str, solver: Optional[str] = None):
    """
    Returns either the whole dict of assignments (solver -> assignment),
    or a single solver's assignment if `solver` is provided.
    """
    # Prefer classmethod on your mixin’d problem class:
    if hasattr(pb.__class__, "load_assignments"):
        assignments = pb.__class__.load_assignments(stem)
    else:
        # Optional fallback to a base LP class if you still keep it around:
        try:
            from bpropy.co.base_lp import LPProblem
            if hasattr(LPProblem, "load_assignments"):
                assignments = LPProblem.load_assignments(stem)
            else:
                return None
        except Exception:
            return None

    if assignments is None:
        return None
    if solver is None:
        return assignments
    if solver in assignments:
        return assignments[solver]
    # Nice error with available choices:
    raise KeyError(f"Solver '{solver}' not found. Available: {list(assignments.keys())}")

def _cache_paths(processed_dir: str, split: str, stem_base: str) -> Tuple[str, str]:
    out_dir = os.path.join(processed_dir, split)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, stem_base + ".pt"), os.path.join(out_dir, stem_base + ".meta.json")

def _hash_for_cache(man: dict, pb: ProblemT, cfg: MRFDatasetConfig, stem: str) -> str:
    # Lightweight cache key: size+mtime of manifest + type + cfg knobs
    st = os.stat(stem + ".json")
    h = hashlib.sha256()
    h.update(str(st.st_mtime_ns).encode())
    h.update(str(st.st_size).encode())
    h.update((_infer_type(man)).encode())
    h.update((cfg.feature_mode or "none").encode())
    h.update(cfg.preproc_version.encode())
    # You can add more signals (e.g., versions) if needed
    return h.hexdigest()

class MRFDataset(Dataset):
    """
    Loads problems from root/split (split in {'train','val','test'}),
    optionally preprocesses to PyG.Data via a user-supplied callable,
    and caches results. Supports yielding Data, Problem, or both.
    """
    def __init__(
        self,
        root: str,
        split: str,
        *,
        preprocess: Optional[PreprocFn] = None,
        processed_dir: Optional[str] = None,
        cfg: Optional[MRFDatasetConfig] = None,
    ):
        assert split in {"train", "val", "test"}
        self.root = root
        self.split = split
        self.split_dir = os.path.join(root, split)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(self.split_dir)

        self.stems = _list_stems(self.split_dir)
        self.cfg = cfg or MRFDatasetConfig()
        self.preprocess = preprocess or (lambda pb: problem_to_data(pb, feature_mode=self.cfg.feature_mode))
        self.processed_dir = processed_dir or os.path.join(root, "processed")

    def __len__(self) -> int:
        return len(self.stems)

    def _load_from_cache(self, stem: str, cache_key: str) -> Optional[Union[Data,HeteroData]]:
        stem_base = os.path.basename(stem)
        pt_path, meta_path = _cache_paths(self.processed_dir, self.split, stem_base)
        if not (os.path.exists(pt_path) and os.path.exists(meta_path)):
            return None
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("cache_key") != cache_key:
                return None
            try:
                # Explicitly disable "weights-only" to load PyG Data
                data = torch.load(pt_path, map_location="cpu", weights_only=False)
            except TypeError:
                # Older torch without the kwarg
                data = torch.load(pt_path, map_location="cpu")
            return data
        except Exception as e:
            # Optional: surface the reason to help debugging
            print(f"[MRFDataset] cache read failed for {pt_path}: {e}")
            return None
    #def _load_from_cache(self, stem: str, cache_key: str) -> Optional[Data]:
    #    import json, os
    #    import torch
    #    from torch_geometric.data import Data
    #
    #    stem_base = os.path.basename(stem)
    #    pt_path, meta_path = _cache_paths(self.processed_dir, self.split, stem_base)
    #    if not (os.path.exists(pt_path) and os.path.exists(meta_path)):
    #        return None
    #    try:
    #        # 1) verify cache key
    #        with open(meta_path, "r") as f:
    #            meta = json.load(f)
    #        if meta.get("cache_key") != cache_key:
    #            return None
    #
    #        # 2) load cached graph (PyG Data is not "weights-only")
    #        try:
    #            data = torch.load(pt_path, map_location="cpu", weights_only=False)
    #        except TypeError:
    #            # older torch without the kwarg
    #            data = torch.load(pt_path, map_location="cpu")
    #
    #        # 3) (optional) attach assignment into the Data as `.gt`
    #        #    - only if caller wants assignments
    #        #    - only when a single solver is selected (so it's a 1-D array, not a dict)
    #        if (
    #            data is not None
    #            and isinstance(data, Data)
    #            and getattr(self.cfg, "return_assignment", False)
    #            and getattr(self.cfg, "assignment_solver", None) is not None
    #            and not hasattr(data, "gt")  # don't overwrite if already present
    #        ):
    #            try:
    #                # Prefer the mixin classmethod to avoid needing a Problem instance:
    #                if hasattr(SerializableLPMixin, "load_assignments"):
    #                    assignments = SerializableLPMixin.load_assignments(stem)
    #                else:
    #                    assignments = None
    #
    #                if assignments is not None:
    #                    solver = self.cfg.assignment_solver
    #                    if solver in assignments:
    #                        gt = assignments[solver]  # expected 1-D numpy array, len == num_nodes
    #                        import torch as _torch
    #                        gt_t = _torch.as_tensor(gt)  # accepts numpy directly
    #                        # Optional sanity check against node count if present:
    #                        if hasattr(data, "num_nodes") and data.num_nodes is not None:
    #                            assert gt_t.numel() == int(data.num_nodes), \
    #                                f"Assignment length {gt_t.numel()} != num_nodes {data.num_nodes}"
    #                        # Attach as a field to be batched by PyG:
    #                        data.gt = gt_t
    #                    # else: silently skip if solver not present (the separate assignment return path still handles errors)
    #            except Exception as e:
    #                # Non-fatal: keep cached data usable even if assignment attach fails
    #                print(f"[MRFDataset] attach assignment to Data failed for {stem_base}: {e}")
    #
    #        return data
    #    except Exception as e:
    #        print(f"[MRFDataset] cache read failed for {stem_base}: {e}")
    #        return None


    def _save_to_cache(self, stem: str, cache_key: str, data: Data, prob_type: str):
        stem_base = os.path.basename(stem)
        pt_path, meta_path = _cache_paths(self.processed_dir, self.split, stem_base)
        torch.save(data, pt_path)
        meta = {
            "cache_key": cache_key,
            "problem_type": prob_type,
            "feature_mode": self.cfg.feature_mode,
            "preproc_version": self.cfg.preproc_version,
            "created": time.time(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def _get_item(self, index: int) -> Tuple[Optional[ProblemT], Optional[Data], Optional[object]]:
        stem = self.stems[index]
        man = _read_manifest(stem)                      # existing:contentReference[oaicite:2]{index=2}
        prob_type = _infer_type(man)                    # existing:contentReference[oaicite:3]{index=3}
        cache_key = _hash_for_cache(man, None, self.cfg, stem)
    
        data = None
        if not self.cfg.refresh:
            data = self._load_from_cache(stem, cache_key)  # existing:contentReference[oaicite:4]{index=4}
    
        pb = None
        assignment = None

        if data is None or self.cfg.yield_mode in {"problem", "both"}:
            pb = _load_problem_by_type(stem, prob_type)    # existing:contentReference[oaicite:5]{index=5}
    
        if data is None:
            # inspect the preprocess function, if it takes two arguments pass the labels
            sig = inspect.signature(self.preprocess)
            params = sig.parameters
            required = [p for p in params.values() if p.default is inspect._empty and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            if len(required)==1:
                data = self.preprocess(pb)                     # existing:contentReference[oaicite:6]{index=6}
            elif len(required)==2:
                assignment = _load_assignment_from_stem(pb, stem, solver=self.cfg.assignment_solver)
                data = self.preprocess(pb, assignment)                     # existing:contentReference[oaicite:6]{index=6}
            else:
                raise Exception(f"Preprocess function should admit 1 or 2 arguments. Found {len(required)}")

            self._save_to_cache(stem, cache_key, data, prob_type)
    
    
        if self.cfg.return_assignment:
            if pb is None:
                pb = _load_problem_by_type(stem, prob_type)
            if assignment is None:
                assignment = _load_assignment_from_stem(pb, stem, solver=self.cfg.assignment_solver)
        return pb, data, assignment

    def __getitem__(self, index: int):
        pb, data, assignment = self._get_item(index)
    
        if not self.cfg.return_assignment:
            if self.cfg.yield_mode == "data":
                return data
            elif self.cfg.yield_mode == "problem":
                return pb
            elif self.cfg.yield_mode == "both":
                return pb, data
            else:
                raise ValueError(self.cfg.yield_mode)
        else:
            # Return assignment as an extra element (keeps BC: just appended)
            if self.cfg.yield_mode == "data":
                return data, assignment
            elif self.cfg.yield_mode == "problem":
                return pb, assignment
            elif self.cfg.yield_mode == "both":
                return pb, data, assignment
            else:
                raise ValueError(self.cfg.yield_mode)

