import torch
from torch.nn import Module, ModuleList

class EFCanonicalParam(Module):
    def __init__(self, in_channels, mlp, factor_sizes=[1]):
        super().__init__()
        self.factor_sizes = factor_sizes
        self.mlps = ModuleList([mlp(in_channels, 2**s) for s in factor_sizes])
    def forward(self, x, data):
        y = torch.concatenate([
            torch.flatten(self.mlps[i](x[data.factor_sizes==s])) for i,s in enumerate(self.factor_sizes)
            ])
        # reorder predictions by item in the batch
        if hasattr(data, "num_graphs") and hasattr(data, "lbp_edges"):
            g_ind = data.lbp_edges.graph_indicator()
            f_ind = data.factor_sizes.repeat_interleave(torch.pow(2, data.factor_sizes))
            r =  torch.concatenate([y[g_ind == i] for i in range(data.num_graphs)])
            return r[torch.argsort(torch.argsort(f_ind, stable=True), stable=True)]
        if hasattr(data, "num_graphs") and hasattr(data, "nmf_edges"):
            g_ind = data.nmf_edges.graph_indicator()
            f_ind = data.factor_sizes.repeat_interleave(torch.pow(2, data.factor_sizes))
            r =  torch.concatenate([y[g_ind == i] for i in range(data.num_graphs)])
            return r[torch.argsort(torch.argsort(f_ind, stable=True), stable=True)]

        return y

