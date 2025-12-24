# bpropy/adapters/vc_to_data.py
from __future__ import annotations
import numpy as np
import torch
from typing import Iterable, Literal, Optional, Tuple, Union
from torch_geometric.data import Data, HeteroData

# Problem classes & converters
from bpropy.co.mis import MIS, mis_batch                       # batching/mis specifics :contentReference[oaicite:0]{index=0}
from bpropy.co.scp import SCP                                  # scp specifics (constraints, weights, greedy) :contentReference[oaicite:1]{index=1}
from bpropy.co.mis_solution import MISSolution                  # (kept for type hints / symmetry) :contentReference[oaicite:2]{index=2}
from bpropy.co.scp_solution import SCPSolution                  # (kept for type hints / symmetry) :contentReference[oaicite:3]{index=3}

# VC graph + utilities
from bpropy.graphs.vc_graph import VCGraph, variable_constraint_graph  # shared VC builder (nodes tagged bipartite=0/1) :contentReference[oaicite:4]{index=4}
from bpropy.utils.graph_utils import get_edge_index             # used in both mis/scp adapters today :contentReference[oaicite:5]{index=5}

# Factor-graph + wiring
#from bpropy.mrf.mis_factor_graph import MISFactorGraph
#from bpropy.mrf.scp_factor_graph import SCPFactorGraph
from bpropy.mrf.factor_graph import FactorGraph
from bpropy.mrf.lbp_edges import LBPEdges
from bpropy.mrf.nmf_edges import NMFEdges                       # has .batch, __getitem__, to(device) you rely on :contentReference[oaicite:6]{index=6}
from bpropy.mrf.sufficient_statistics import x2ss

# Feature builders (mirror your current usage)
from bpropy.datasets.mis_features import get_features as mis_features
from bpropy.datasets.scp_features import get_features as scp_features
from bpropy.datasets.types import FeatureExtractionResult

# ---- problem registry / tags -------------------------------------------------

NAME2CODE = {"MIS": 0, "SCP": 1}
CODE2NAME = {v: k for k, v in NAME2CODE.items()}

def _infer_problem_name(pb) -> str:
    if isinstance(pb, MIS): return "MIS"
    if isinstance(pb, SCP): return "SCP"
    raise TypeError(f"Unknown problem class: {type(pb)}")

# ---- common VC->PyG (homogeneous) -------------------------------------------

def _vc_to_homogeneous(vcg: VCGraph) -> Tuple[Data, torch.Tensor, torch.Tensor]:
    # Same structure you build in mis/scp adapters: edge_index, bipartite, factor_sizes, neighbors
    edge_index = get_edge_index(vcg)                                           # :contentReference[oaicite:7]{index=7}
    bipartite = torch.zeros(vcg.number_of_nodes(), dtype=torch.long)
    bipartite[[n for n, d in vcg.nodes(data=True) if d["bipartite"]]] = 1

    factor_sizes = torch.ones(vcg.number_of_nodes(), dtype=torch.long)
    neighbors    = torch.zeros(vcg.number_of_nodes(), dtype=torch.long)
    for n, d in vcg.nodes(data=True):
        if d["bipartite"] == 1:
            factor_sizes[n] = len(list(vcg.neighbors(n)))
        neighbors[n] = len(list(vcg.neighbors(n)))

    data = Data(edge_index=edge_index,
                bipartite=bipartite,
                factor_sizes=factor_sizes,
                neighbors=neighbors)
    return data, factor_sizes, neighbors

# ---- common features (normalize exactly as you do now) -----------------------
def _make_features(pb, name: str, feature_mode=None) -> FeatureExtractionResult:
    if name == "MIS":
        if feature_mode is None:
            feature_mode = "EB"
        C, nfeat, efeat, cfeat = mis_features(pb, mode=feature_mode)
    elif name == "SCP":
        C = pb.constraints()  # SCP exposes real constraints, not cliques
        nfeat, efeat, cfeat = scp_features(pb)
    else:
        raise ValueError(f"Unknown problem type {name}")

    # zero-center per feature block
    nfeat = nfeat - np.mean(nfeat, axis=0, keepdims=True)
    cfeat = cfeat - np.mean(cfeat, axis=0, keepdims=True)
    efeat = efeat - np.mean(efeat, axis=0, keepdims=True)

    return FeatureExtractionResult(
        node_features=nfeat,
        constraint_features=cfeat,
        edge_features=efeat,
        constraints=C,
        feature_mode=feature_mode,
    )

# ---- public: LPProblem -> (Data|HeteroData) ---------------------------------
def problem_to_data(pb,
               *,
               with_lbp: bool = False,
               with_nmf: bool = False,
               with_ss: bool = False,
               y: Optional[torch.Tensor] = None,
               hetero: bool = False,
               feature_mode: str = None) -> Union[Data, HeteroData]:
    """
    Build a VCGraph from `pb` and convert to a PyG (Data|HeteroData),
    attaching optional LBP/NMF wiring and problem type tags.

    Mirrors the logic in mis_to_data/scp_to_data, but unified.  """
    name = _infer_problem_name(pb)

    # 1) Extract features (may mutate `pb` by adding constraints)
    fx = _make_features(pb, name, feature_mode=feature_mode)  # returns FeatureExtractionResult
    # Normalize exactly as you already do
    nfeat = fx.node_features - fx.node_features.mean(axis=0, keepdims=True)
    cfeat = fx.constraint_features - fx.constraint_features.mean(axis=0, keepdims=True)
    efeat = fx.edge_features - fx.edge_features.mean(axis=0, keepdims=True)

    # 2) Rebuild VCGraph from the mutated problem
    vcg = VCGraph(pb)  # builds from pb.constraints(...) which now reflect mutations. :contentReference[oaicite:2]{index=2}

    if not hetero:
        data, factor_sizes, neighbors = _vc_to_homogeneous(vcg)                # aligns with both adapters :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}
        # features
        nn = nfeat.shape[1] - cfeat.shape[1]
        x = np.concatenate((nfeat, np.pad(cfeat, ((0,0), (0,nn)))), axis=0)
        data.x = torch.from_numpy(x).to(torch.float32)
        data.e = torch.from_numpy(efeat).to(torch.float32)

        # problem type tags (survive batching)
        data.problem_type_name = name
        data.problem_type_code = torch.tensor([NAME2CODE[name]], dtype=torch.long)
        data.feature_mode = feature_mode

        # optional wiring via the right factor-graph class
        if with_lbp or with_nmf:
            fg = FactorGraph(pb)
            #if name == "MIS":
            #    fg = MISFactorGraph(pb)
            #else:
            #    fg = SCPFactorGraph(pb)
            if with_lbp:
                data.lbp_edges = LBPEdges(fg)                                  # exactly how you attach today :contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14}
            if with_nmf:
                data.nmf_edges = NMFEdges(fg)                                  # NMFEdges has batching/slicing utilities already :contentReference[oaicite:15]{index=15}
            if y is not None:
                data.y = y
                if with_ss:
                    data.y_ss = torch.from_numpy(x2ss(y.astype(int), fg)).to(torch.float32)
        else:
            if with_ss and (y is not None):
                fg = FactorGraph(pb)
                data.y_ss = torch.from_numpy(x2ss(y.astype(int), fg)).to(torch.float32)

        return data

    # HeteroData branch – mirrors your MIS hetero converter
    # (If you later add SCP hetero features, branch similarly here.)
    from bpropy.adapters.mis_data import get_vc_edge_index as _get_vc_edge_index # reuse helper you already have :contentReference[oaicite:16]{index=16}
    v2c_edge_index = _get_vc_edge_index(vcg, direction="v2c")
    c2v_edge_index = _get_vc_edge_index(vcg, direction="c2v")

    data = HeteroData()
    data["variable", "v2c", "constraint"].edge_index = v2c_edge_index
    data["constraint", "c2v", "variable"].edge_index = c2v_edge_index

    # minimal node attrs (you already do this for MIS hetero) :contentReference[oaicite:17]{index=17}
    nv, nc = vcg.nv(), vcg.nc()
    v_neighbors = torch.zeros(nv, dtype=torch.long)
    c_neighbors = torch.zeros(nc, dtype=torch.long)
    factor_sizes = torch.ones(nc, dtype=torch.long)
    for n, d in vcg.nodes(data=True):
        if d["bipartite"] == 1:  # constraint
            factor_sizes[n - nv] = len(list(vcg.neighbors(n)))
            c_neighbors[n - nv] = len(list(vcg.neighbors(n)))
        else:
            v_neighbors[n] = len(list(vcg.neighbors(n)))
    data["constraint"].factor_sizes = factor_sizes
    data["variable"].neighbors = v_neighbors
    data["constraint"].neighbors = c_neighbors
    data["variable"].x=torch.from_numpy(nfeat).to(dtype=torch.float32)
    data["constraint"].x=torch.from_numpy(cfeat).to(dtype=torch.float32)

    # tag
    data.problem_type_name = name
    data.problem_type_code = torch.tensor([NAME2CODE[name]], dtype=torch.long)
    data.feature_mode = feature_mode

    if with_lbp or with_nmf:
        #fg = MISFactorGraph(pb) if name == "MIS" else SCPFactorGraph(pb)
        fg = FactorGraph(pb)
        if with_lbp: data.lbp_edges = LBPEdges(fg)
        if with_nmf: data.nmf_edges = NMFEdges(fg)
        if y is not None:
            data.y = y
            data.y_ss = torch.from_numpy(x2ss(y.astype(int), fg)).to(torch.float32)
    else:
        if with_ss and (y is not None):
            fg = FactorGraph(pb)
            data.y_ss = torch.from_numpy(x2ss(y.astype(int), fg)).to(torch.float32)
    return data

# ---- public: Data/HeteroData -> problem instance ----------------------------

def data_to_lp(data: Union[Data, HeteroData], *, batch: bool = False):
    """Recover a concrete problem instance from a (batched) Data/HeteroData.
    Uses problem_type tag and your existing converters. """
    # Prefer explicit name; else fall back to numeric code
    name = getattr(data, "problem_type_name", None)
    if name is None and hasattr(data, "problem_type_code"):
        name = CODE2NAME[int(data.problem_type_code[0])]

    if name == "MIS":
        from bpropy.adapters.mis_data import data_to_mis, hetero_data_to_mis   # reuse your tested converters :contentReference[oaicite:18]{index=18}
        return hetero_data_to_mis(data, batch=batch) if isinstance(data, HeteroData) else data_to_mis(data, batch=batch)
    elif name == "SCP":
        from bpropy.adapters.scp_data import data_to_scp                       # reuse your tested converter :contentReference[oaicite:19]{index=19}
        if hasattr(data, "num_graphs") and batch:
            # build list then (optionally) design a batcher if you want a single SCP
            return [data_to_scp(data.get_example(i), batch=False) for i in range(data.num_graphs)]
        return data_to_scp(data, batch=False)
    else:
        raise ValueError("Unknown or missing problem_type tag on Data/HeteroData")

# ---- optional: batch a list of problems into one large problem --------------

def batch_lp(problems: Iterable[Union[MIS, SCP]]):
    """Return a single batched problem when all are of the same type.
    (MIS batching uses your existing mis_batch.) """
    problems = list(problems)
    if not problems:
        raise ValueError("Empty problem list")

    first = problems[0]
    if all(isinstance(p, MIS) for p in problems):
        return MIS.batch(problems)                                            # calls mis_batch under the hood :contentReference[oaicite:20]{index=20}
    if all(isinstance(p, SCP) for p in problems):
        # (Optional) implement SCP.batch similar to MIS if/when you need it.
        raise NotImplementedError("SCP batching not implemented yet")
    raise TypeError("Mixed problem types: batch separately per type")

#def problem_to_data(pb: LPProblem, **kwargs) -> Data | HeteroData:
#    """
#    Convenience: builds VCGraph(pb) then calls vcgraph_to_data(vcg, **kwargs).
#    """
#    return vc_to_data(VCGraph(pb), **kwargs)
