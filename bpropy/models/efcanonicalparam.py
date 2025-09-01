import torch
from torch.nn import Module, ModuleList

class EFCanonicalParam(Module):
    def __init__(self, in_channels, mlp, factor_sizes=[1]):
        super().__init__()
        self.factor_sizes = factor_sizes
        self.mlps = ModuleList([mlp(in_channels, 2**s) for s in factor_sizes])
    # loop free fix : needs check
    def forward(self, x, data):
        """
        x: [num_factors, in_channels] — rows aligned with data.factor_sizes (factor order)
        data.factor_sizes: LongTensor [num_factors] with each factor's arity
        Returns:
            theta: 1-D tensor of length sum(2**arity) in factor order
        """
        fs = data.factor_sizes.to(dtype=torch.long)                              # [F]
        # SS block sizes per factor (2**arity) and their start offsets in θ
        ss_len = (1 << fs)                                                       # [F]
        offsets = torch.cumsum(ss_len, dim=0) - ss_len                           # [F]
        SS_total = int(ss_len.sum().item())
        theta = x.new_empty(SS_total)

        # we’ll fill θ bucket-by-bucket (no per-factor loop)
        # build a lookup: arity -> index into self.mlps
        arity2mlp = {int(s): i for i, s in enumerate(self.factor_sizes)}

        # unique arities present in this batch (small set)
        uniq = torch.unique(fs, sorted=True)

        for s in uniq.tolist():  # loop over arities only (not over factors)
            if s not in arity2mlp:
                continue  # no MLP defined for this arity
            mlp = self.mlps[arity2mlp[s]]

            idx = torch.nonzero(fs == s, as_tuple=True)[0]                       # [K]
            if idx.numel() == 0:
                continue

            # Run MLP once for the whole bucket, get [K, 2**s]
            out = mlp(x.index_select(0, idx))                                    # [K, 2**s]

            # Compute scatter positions in θ:
            # For each factor k, positions = [offset_k + 0, ..., offset_k + (2**s - 1)]
            base = offsets.index_select(0, idx).unsqueeze(1)                     # [K, 1]
            span = torch.arange(1 << s, device=x.device).view(1, -1)             # [1, 2**s]
            pos  = (base + span).reshape(-1)                                     # [K*2**s]

            theta.index_copy_(0, pos, out.reshape(-1))

        return theta
    #def forward(self, x, data):
    #    y = torch.concatenate([
    #        torch.flatten(self.mlps[i](x[data.factor_sizes==s])) for i,s in enumerate(self.factor_sizes)
    #        ])
    #    # temporary fix
    #    #out = []
    #    ## map arity -> which MLP to use
    #    #which = {s:i for i,s in enumerate(self.factor_sizes)}
    #    #for i, s in enumerate(data.factor_sizes.tolist()):
    #    #    o_i = self.mlps[which[int(s)]](x[i:i+1])  # shape: [1, 2**s]
    #    #    out.append(o_i.view(-1))                  # flatten this factor's SS
    #    #y = torch.cat(out, dim=0)
    #    #return y
    #    # reorder predictions by item in the batch
    #    if hasattr(data, "num_graphs") and hasattr(data, "lbp_edges"):
    #        g_ind = data.lbp_edges.graph_indicator()
    #        f_ind = data.factor_sizes.repeat_interleave(torch.pow(2, data.factor_sizes))
    #        r =  torch.concatenate([y[g_ind == i] for i in range(data.num_graphs)])
    #        return r[torch.argsort(torch.argsort(f_ind, stable=True), stable=True)]
    #    if hasattr(data, "num_graphs") and hasattr(data, "nmf_edges"):
    #        g_ind = data.nmf_edges.graph_indicator()
    #        f_ind = data.factor_sizes.repeat_interleave(torch.pow(2, data.factor_sizes))
    #        r =  torch.concatenate([y[g_ind == i] for i in range(data.num_graphs)])
    #        return r[torch.argsort(torch.argsort(f_ind, stable=True), stable=True)]

    #    return y

