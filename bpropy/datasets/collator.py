import torch
from typing import List, Tuple, Union
from bpropy.co.scp import SCP
from bpropy.co.mis import MIS
from bpropy.mrf.lbp_edges import LBPEdges
from bpropy.mrf.nmf_edges import NMFEdges
from bpropy.adapters.vc_to_data import problem_to_data  # your helper


ProblemT = Union[SCP, MIS]
from torch_geometric.data import Data, HeteroData, Batch

# Treat both PyG graph classes as "graph-like"
GraphTypes = (Data, HeteroData)


def _maybe_batch_custom_edges(data_batch, originals):
    """
    After Batch.from_data_list, PyG leaves unknown attrs (like lbp_edges/nmf_edges)
    as lists. Use the class-provided .batch(...) to merge them back.
    """
    # LBP
    if hasattr(data_batch, "lbp_edges") and isinstance(data_batch.lbp_edges, (list, tuple)):
        lbs = [getattr(d, "lbp_edges", None) for d in originals]
        lbs = [e for e in lbs if isinstance(e, LBPEdges)]
        if lbs:
            data_batch.lbp_edges = LBPEdges.batch(lbs)

    # NMF
    if hasattr(data_batch, "nmf_edges") and isinstance(data_batch.nmf_edges, (list, tuple)):
        nms = [getattr(d, "nmf_edges", None) for d in originals]
        nms = [e for e in nms if isinstance(e, NMFEdges)]
        if nms:
            data_batch.nmf_edges = NMFEdges.batch(nms)

def mrf_collate(batch: List):
    first = batch[0]

    # (problem, graph, assignment)
    if isinstance(first, tuple) and len(first) == 3 and isinstance(first[1], GraphTypes):
        pbs, graphs, assigns = zip(*batch)
        data_batch = Batch.from_data_list(list(graphs))
        _maybe_batch_custom_edges(data_batch, graphs)
        # keep y as you prefer (flat cat per-variable); or return the list if you need idxs
        y = torch.cat([torch.as_tensor(a) for a in assigns], dim=0)
        return list(pbs), data_batch, y

    # (graph, assignment)
    if isinstance(first, tuple) and len(first) == 2 and isinstance(first[0], GraphTypes):
        graphs, assigns = zip(*batch)
        data_batch = Batch.from_data_list(list(graphs))
        _maybe_batch_custom_edges(data_batch, graphs)
        y = torch.cat([torch.as_tensor(a) for a in assigns], dim=0)
        return data_batch, y

    # graph only
    if isinstance(first, GraphTypes):
        data_batch = Batch.from_data_list(batch)
        _maybe_batch_custom_edges(data_batch, batch)
        return data_batch

    # (problem, assignment) — no graphs in either slot
    if isinstance(first, tuple) and len(first) == 2:
        a, b = zip(*batch)
        # neither item is a (Hetero)Data graph → (pb, y)
        from torch_geometric.data import Data, HeteroData
        if not isinstance(a[0], (Data, HeteroData)) and not isinstance(b[0], (Data, HeteroData)):
            return list(a), list(b)

    # (problem, graph)
    if isinstance(first, tuple) and len(first) == 2 and isinstance(first[1], GraphTypes):
        pbs, graphs = zip(*batch)
        data_batch = Batch.from_data_list(list(graphs))
        _maybe_batch_custom_edges(data_batch, graphs)
        return list(pbs), data_batch

    # problems only or anything else → pass through
    return list(batch)

