import torch
import torch_scatter
import torch.nn.functional as F
from torch.nn import ReLU, ELU, Dropout, BatchNorm1d
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn import GCNConv, Sequential, LayerNorm, to_hetero, HeteroDictLinear
from torch_geometric.nn.models import GCN, GAT
from torch_geometric.data import Data
from torchvision.ops import MLP

from bpropy.mrf.lbp import loopy_belief_propagation
from bpropy.mrf.naive_mean_field import naive_mean_field
from bpropy.models.efcanonicalparam import EFCanonicalParam

class GCN_MRF_v4(torch.nn.Module):
    def __init__(self, in_dim, n_layers=4, hidden_dim=64, dropout_p=0.25, normalize=False, bp_max_iter=10, bp_tol=1e-6, bp_damping=0.5, factor_sizes=[1,2], jk="lstm", norm="LayerNorm", act="elu", inference="loopy_belief_propagation", mapping_aggr="mean", normalize_theta=False, gnn_f="GCN"):
        super().__init__()
        self.inference = inference 
        self.bp_max_iter = bp_max_iter
        self.bp_tol = bp_tol
        self.bp_damping = bp_damping
        self.conv = eval(gnn_f)(in_dim, hidden_dim, n_layers, out_channels=hidden_dim, dropout=dropout_p, act=act, norm=norm, jk=jk)
        self.mapping_aggr=mapping_aggr
        self.normalize_theta = normalize_theta

        self.efc = EFCanonicalParam(hidden_dim, lambda in_channels, out_channels: MLP(in_channels, hidden_channels=[64, 64, out_channels], activation_layer=ELU, norm_layer=torch.nn.LayerNorm), factor_sizes=factor_sizes)
        if mapping_aggr=="gat":
            self.mapping = GATConv(hidden_dim, hidden_dim)
    def forward(self, data, fg_data=None, fg_map=None):
        x, edge_index = data.x, data.edge_index

        x = self.conv(x, edge_index)

        if not fg_data is None:
            # mapping
            if self.mapping_aggr=="mean":
                x = torch_scatter.scatter_mean(x[fg_map[:,0]], fg_map[:,1],dim=0)
            if self.mapping_aggr=="max":
                x,_ = torch_scatter.scatter_max(x[fg_map[:,0]], fg_map[:,1],dim=0)
            if self.mapping_aggr=="gat":
                M = fg_map#[:,[1,0]]
                if M[:,1].max() >= x.shape[0]:
                    x = torch.nn.functional.pad(x, (0,0,0,M.max() - x.shape[0]+1), "constant", 0)
                x = self.mapping(x, M.T)

            edge_data = fg_data
        else:
            edge_data = data


        x = self.efc(x, edge_data)
        if self.normalize_theta:
            x = (x - x.min()) / (x.max() - x.min())
        if self.inference == "loopy_belief_propagation":
            return loopy_belief_propagation(x, torch.zeros_like(x), edge_data.lbp_edges, max_iter=self.bp_max_iter, tol=self.bp_tol, damping=self.bp_damping), x
        if self.inference == "naive_mean_field":
            return naive_mean_field(x, edge_data.nmf_edges, max_iter=self.bp_max_iter, tol=self.bp_tol), x

class Hetero_GCN_MRF_v4(torch.nn.Module):
    def __init__(self, in_variable_dim=10, in_constraint_dim=4, n_layers=4, hidden_dim=64, dropout_p=0.25, normalize=False, bp_max_iter=10, bp_tol=1e-6, bp_damping=0.5, factor_sizes=[1,2], jk="lstm", norm="LayerNorm", act="elu", inference="loopy_belief_propagation", mapping_aggr="mean", normalize_theta=False, gnn_f="GAT"):
        super().__init__()
        self.inference = inference 
        self.bp_max_iter = bp_max_iter
        self.bp_tol = bp_tol
        self.bp_damping = bp_damping
        self.conv = eval(gnn_f)(hidden_dim, hidden_dim, n_layers, out_channels=hidden_dim, dropout=dropout_p, act=act, norm=norm, jk=jk, add_self_loops=False)
        # debug
        self.enc = HeteroDictLinear({'variable': in_variable_dim, 'constraint': in_constraint_dim}, hidden_dim)

        self.conv = to_hetero(self.conv, (['variable', 'constraint'], [('variable', 'v2c', 'constraint'), ('constraint', 'c2v', 'variable')]), aggr="sum")

        self.mapping_aggr=mapping_aggr
        self.normalize_theta = normalize_theta

        self.efc = EFCanonicalParam(hidden_dim, lambda in_channels, out_channels: MLP(in_channels, hidden_channels=[64, 64, out_channels], activation_layer=ELU, norm_layer=torch.nn.LayerNorm), factor_sizes=factor_sizes)
        if mapping_aggr=="gat":
            self.mapping = GATConv(hidden_dim, hidden_dim)
    def forward(self, data, fg_data=None, fg_map=None):
        #x, edge_index = data.x, data.edge_index

        x = self.enc(data.x_dict)

        x = self.conv(x, data.edge_index_dict)

        if not fg_data is None:
            # mapping
            if self.mapping_aggr=="mean":
                x = torch_scatter.scatter_mean(x[fg_map[:,0]], fg_map[:,1],dim=0)
            if self.mapping_aggr=="max":
                x,_ = torch_scatter.scatter_max(x[fg_map[:,0]], fg_map[:,1],dim=0)
            if self.mapping_aggr=="gat":
                M = fg_map#[:,[1,0]]
                if M[:,1].max() >= x.shape[0]:
                    x = torch.nn.functional.pad(x, (0,0,0,M.max() - x.shape[0]+1), "constant", 0)
                x = self.mapping(x, M.T)

            edge_data = fg_data
        else:
            edge_data = data

        xp = torch.cat((x["variable"], x["constraint"]))

        tmp_edge_data = Data()
        if hasattr(edge_data, "lbp_edges"):
            tmp_edge_data.lbp_edges = edge_data.lbp_edges
        if hasattr(edge_data, "nmf_edges"):
            tmp_edge_data.nmf_edges = edge_data.nmf_edges
        tmp_edge_data.factor_sizes = torch.concatenate([torch.ones(x["variable"].shape[0], dtype=torch.int, device=xp.device), edge_data["constraint"].factor_sizes])

        xp = self.efc(xp, tmp_edge_data)
        assert xp.numel() == int(edge_data.lbp_edges.n_ss.sum().item())
        if self.normalize_theta:
            xp = (xp - xp.min()) / (xp.max() - xp.min())
        if self.inference == "loopy_belief_propagation":
            #v2f = xp.new_zeros(int(edge_data.lbp_edges.n_f2v_msg.sum().item()))
            v2f = torch.zeros_like(xp)
            return loopy_belief_propagation(xp, v2f, edge_data.lbp_edges, max_iter=self.bp_max_iter, tol=self.bp_tol, damping=self.bp_damping), xp
        if self.inference == "naive_mean_field":
            return naive_mean_field(xp, edge_data.nmf_edges, max_iter=self.bp_max_iter, tol=self.bp_tol), xp
