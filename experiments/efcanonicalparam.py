import torch
from torch.nn import Module, ModuleList

class EFCanonicalParam(Module):
    def __init__(self, in_channels, mlp, factor_sizes=[1]):
        super().__init__()
        self.factor_sizes = factor_sizes
        self.mlps = ModuleList([mlp(in_channels, 2**s) for s in factor_sizes])
    def forward(self, x, factor_sizes):
        return torch.concatenate([
            torch.flatten(self.mlps[i](x[factor_sizes==s])) for i,s in enumerate(self.factor_sizes)
            ])

