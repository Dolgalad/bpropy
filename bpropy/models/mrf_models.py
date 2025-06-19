import torch
import torch_scatter
import torch.nn.functional as F
from torch.nn import ReLU, ELU, Dropout, BatchNorm1d
from torch_geometric.nn import GCNConv, Sequential, LayerNorm
from torch_geometric.nn.models import GCN
from torchvision.ops import MLP

from bpropy.mrf.lbp import loopy_belief_propagation
from bpropy.mrf.naive_mean_field import naive_mean_field
from bpropy.models.efcanonicalparam import EFCanonicalParam

class GCN_MRF_v4(torch.nn.Module):
    def __init__(self, in_dim, n_layers=4, hidden_dim=64, dropout_p=0.25, normalize=False, bp_max_iter=10, bp_tol=1e-6, bp_damping=0.5, factor_sizes=[1,2], jk="lstm", norm="LayerNorm", act="elu", inference="loopy_belief_propagation", mapping_aggr="mean", normalize_theta=False):
        super().__init__()
        self.inference = inference 
        self.bp_max_iter = bp_max_iter
        self.bp_tol = bp_tol
        self.bp_damping = bp_damping
        self.conv = GCN(in_dim, hidden_dim, n_layers, out_channels=hidden_dim, dropout=dropout_p, act=act, norm=norm, jk=jk)
        self.mapping_aggr=mapping_aggr
        self.normalize_theta = normalize_theta

        self.efc = EFCanonicalParam(hidden_dim, lambda in_channels, out_channels: MLP(in_channels, hidden_channels=[64, 64, out_channels], activation_layer=ELU, norm_layer=torch.nn.LayerNorm), factor_sizes=factor_sizes)
    def forward(self, data, fg_data=None, fg_map=None):
        x, edge_index = data.x, data.edge_index

        x = self.conv(x, edge_index)

        if not fg_data is None:
            # mapping
            if self.mapping_aggr=="mean":
                x = torch_scatter.scatter_mean(x[fg_map[:,0]], fg_map[:,1],dim=0)
            if self.mapping_aggr=="max":
                x,_ = torch_scatter.scatter_max(x[fg_map[:,0]], fg_map[:,1],dim=0)

            edge_data = fg_data
        else:
            edge_data = data


        x = self.efc(x, edge_data)
        if self.normalize_theta:
            x = (x - x.min()) / (x.max() - x.min())
        if self.inference == "loopy_belief_propagation":
            return loopy_belief_propagation(x, torch.zeros_like(x), edge_data.lbp_edges, max_iter=self.bp_max_iter, tol=self.bp_tol, damping=self.bp_damping).mu, x
        if self.inference == "naive_mean_field":
            return naive_mean_field(x, edge_data.nmf_edges, max_iter=self.bp_max_iter, tol=self.bp_tol).mu, x
