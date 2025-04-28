import torch
from torch_geometric.loader.dataloader import Collater

from bpropy.mrf.lbp_edges import LBPEdges
from bpropy.mrf.nmf_edges import NMFEdges

class MRFDataCollater(Collater):
    def __call__(self, batch):
        bdata = super().__call__(batch)
        # batch the LBPEdges
        if hasattr(batch[0], "lbp_edges"):
            bdata.lbp_edges = LBPEdges.batch([d.lbp_edges for d in batch])
        if hasattr(batch[0], "nmf_edges"):
            bdata.nmf_edges = NMFEdges.batch([d.nmf_edges for d in batch])

        return bdata

class MRFDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=None, exclude_keys=None, **kwargs):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        kwargs.pop("collate_fn", None)
        super().__init__(
                dataset,
                batch_size,
                shuffle,
                collate_fn=MRFDataCollater(dataset, follow_batch, exclude_keys),
                **kwargs
                )

