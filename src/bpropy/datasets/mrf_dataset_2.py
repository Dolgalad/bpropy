# bpropy/data/mrf_dataset.py
import inspect
import os, json, hashlib, time
from dataclasses import dataclass, field
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

def time_f(func, *args, **kwargs):
    t0 = time.time()
    r = func(*args, **kwargs)
    return time.time() - t0, r


@dataclass
class MRFDatasetConfig:
    feature_mode: Optional[str] = None      # e.g., "EB","CC","CF","EC","CP" (for MIS)
    preproc_version: str = "v1"             # bump to invalidate caches
    yield_mode: str = "both"                # "data" | "problem" | "both"
    # if True, skip cache and recompute; if False, read/write cache normally
    refresh: bool = False
    return_assignment: bool = False
    assignment_solver: Optional[str] = None
    # Extra preprocessing knobs that must be part of the cache key (must be JSON-serializable)
    preprocess_kwargs: dict = field(default_factory=dict)


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
    # Adjust if you have explicit tag in manifest
    if "problem" in man:
        print(man)
        return man["problem"].lower()
    # heuristic fallback
    if "C" in man and "n" in man and "m" in man:
        return "scp"
    if "n" in man and ("E" in man or "edges" in man):
        return "mis"
    # last resort
    return "unknown"


def _load_problem_by_type(stem: str, prob_type: str) -> ProblemT:
    # Uses your SerializableLPMixin interface
    if prob_type == "mis":
        return MIS.load(stem + ".json")
    if prob_type == "scp":
        return SCP.load(stem + ".json")
    # try generic
    return SerializableLPMixin.load(stem + ".json")  # type: ignore

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


#def _cache_paths(processed_dir: str, split: str, stem: str, cache_key: str) -> Tuple[str, str]:
#    stem_base = os.path.basename(stem)
#    out_dir = os.path.join(processed_dir, split)
#    os.makedirs(out_dir, exist_ok=True)
#    return os.path.join(out_dir, stem_base + ".pt"), os.path.join(out_dir, stem_base + ".meta.json")
def _cache_paths(
    processed_dir: str,
    split: str,
    stem: str,
    cache_key: str,
) -> Tuple[str, str]:
    stem_base = os.path.basename(stem)
    out_dir = os.path.join(processed_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    fname = f"{stem_base}__{cache_key}"
    return (
        os.path.join(out_dir, fname + ".pt"),
        os.path.join(out_dir, fname + ".meta.json"),
    )



def _hash_for_cache(
    man: dict,
    pb: ProblemT,
    cfg: MRFDatasetConfig,
    stem: str,
    preprocess_kwargs: Optional[dict] = None,
) -> str:
    # Lightweight cache key: size+mtime of manifest + type + cfg knobs
    st = os.stat(stem + ".json")
    h = hashlib.sha256()
    h.update(str(st.st_mtime_ns).encode())
    h.update(str(st.st_size).encode())
    h.update((_infer_type(man)).encode())
    h.update((cfg.feature_mode or "none").encode())
    h.update(cfg.preproc_version.encode())

    # Include preprocessing knobs (e.g., hetero/with_lbp/with_nmf/with_ss) so different modes do not share cache files.
    pk = preprocess_kwargs if preprocess_kwargs is not None else getattr(cfg, "preprocess_kwargs", {})
    if pk:
        h.update(json.dumps(pk, sort_keys=True, separators=(",", ":")).encode())

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
        preprocess: Optional[Callable[..., Data]] = None,
        cfg: Optional[MRFDatasetConfig] = None,
        processed_dir: str = "processed",
    ):
        self.root = root
        self.split = split
        self.split_dir = os.path.join(root, split)
        self.processed_dir = os.path.join(root, processed_dir)
        self.cfg = cfg or MRFDatasetConfig()
        self.preprocess = preprocess or (lambda pb: pb)  # type: ignore

        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Dataset split directory not found: {self.split_dir}")

        self.stems = _list_stems(self.split_dir)

    def __len__(self) -> int:
        return len(self.stems)

    #def _load_from_cache(self, stem: str, cache_key: str) -> Optional[Data]:
    #    data_path, meta_path = _cache_paths(self.processed_dir, self.split, stem, cache_key)
    #    if not os.path.exists(data_path) or not os.path.exists(meta_path):
    #        return None
    #    try:
    #        # meta is used for debug only
    #        _ = json.load(open(meta_path, "r"))
    #        data = torch.load(data_path)
    #        return data
    #    except Exception:
    #        return None
    def _load_from_cache(self, stem: str) -> Optional[Data]:
        man = _read_manifest(stem)
        cache_key = _hash_for_cache(man, None, self.cfg, stem, preprocess_kwargs=self.cfg.preprocess_kwargs)

        data_path, meta_path = _cache_paths(
            self.processed_dir, self.split, stem, cache_key
        )

        if not (os.path.exists(data_path) and os.path.exists(meta_path)):
            return None

        try:
            meta = json.load(open(meta_path, "r"))
            # cache_key match is now implicit in filename, but keep check for safety
            if meta.get("cache_key") != cache_key:
                return None
            return torch.load(data_path)
        except Exception:
            return None


    #def _save_to_cache(self, stem: str, cache_key: str, data: Data, prob_type: str) -> None:
    #    data_path, meta_path = _cache_paths(self.processed_dir, self.split, stem, cache_key)
    #    torch.save(data, data_path)
    #    meta = {
    #        "cache_key": cache_key,
    #        "prob_type": prob_type,
    #        "feature_mode": self.cfg.feature_mode,
    #        "preproc_version": self.cfg.preproc_version,
    #        "preprocess_kwargs": self.cfg.preprocess_kwargs,
    #        "created": time.time(),
    #    }
    #    with open(meta_path, "w") as f:
    #        json.dump(meta, f, indent=2)
    def _save_to_cache(self, stem: str, data: Data, prob_type: str, preprocess_meta: dict = {}) -> None:
        man = _read_manifest(stem)
        cache_key = _hash_for_cache(man, None, self.cfg, stem, preprocess_kwargs=self.cfg.preprocess_kwargs)
    
        data_path, meta_path = _cache_paths(self.processed_dir, self.split, stem, cache_key)
    
        torch.save(data, data_path)
        meta = {
            "cache_key": cache_key,
            "prob_type": prob_type,
            "feature_mode": self.cfg.feature_mode,
            "preproc_version": self.cfg.preproc_version,
            "preprocess_kwargs": self.cfg.preprocess_kwargs,
            "created": time.time(),
            **preprocess_meta,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


    def _get_item(self, index: int) -> Tuple[Optional[ProblemT], Optional[Data], Optional[object]]:
        stem = self.stems[index]
        man = _read_manifest(stem)
        prob_type = _infer_type(man)

        #cache_key = _hash_for_cache(man, None, self.cfg, stem, preprocess_kwargs=self.cfg.preprocess_kwargs)

        data = None
        if not self.cfg.refresh:
            #data = self._load_from_cache(stem, cache_key)
            data = self._load_from_cache(stem)
            data.preprocess_time_s = float(man.get("preprocess_time_s", 0.0))
            data.preprocess_cached = True


        pb = None
        assignment = None

        if data is None or self.cfg.yield_mode in {"problem", "both"}:
            pb = _load_problem_by_type(stem, prob_type)

        if data is None:
            # inspect the preprocess function, if it takes two arguments pass the labels
            sig = inspect.signature(self.preprocess)
            params = sig.parameters
            required = [
                p for p in params.values()
                if p.default is inspect._empty
                and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]

            if len(required) == 1:
                t_preprocess, data = time_f(self.preprocess, pb)
            elif len(required) == 2:
                t_preprocess = time.time()
                assignment = _load_assignment_from_stem(pb, stem, solver=self.cfg.assignment_solver)
                data = self.preprocess(pb, assignment)
                t_preprocess = time.time() - t_preprocess
            else:
                raise Exception(f"Preprocess function should admit 1 or 2 arguments. Found {len(required)}")

            #self._save_to_cache(stem, cache_key, data, prob_type)
            meta = {"preprocess_time_s": t_preprocess}
            self._save_to_cache(stem, data, prob_type, meta)
            data.preprocess_time_s = float(t_preprocess)
            data.preprocess_cached = False


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
                raise ValueError(f"Unknown yield_mode: {self.cfg.yield_mode}")

        # return assignment too
        if self.cfg.yield_mode == "data":
            return data, assignment
        elif self.cfg.yield_mode == "problem":
            return pb, assignment
        elif self.cfg.yield_mode == "both":
            return pb, data, assignment
        else:
            raise ValueError(f"Unknown yield_mode: {self.cfg.yield_mode}")

