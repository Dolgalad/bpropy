import torch
import torch.nn.functional as F
from torch.nn import ReLU, ELU, Dropout, BatchNorm1d
from torch_geometric.nn import GCNConv, Sequential, LayerNorm
from torch_geometric.nn.models import GCN
from torchvision.ops import MLP

from bpropy.lbp import _loopy_belief_propagation

class GCN_MRF_v1(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64, dropout_p=0.25, bp_max_iter=10, bp_tol=1e-6, bp_damping=0.5):
        super().__init__()
        self.bp_max_iter = bp_max_iter
        self.bp_tol = bp_tol
        self.bp_damping = bp_damping

        self.dropout_p = dropout_p
        self.conv1 = GCNConv(9, hidden_dim, normalize=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv3 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv4 = GCNConv(hidden_dim, hidden_dim, normalize=False)

        # canonical parameter predictors
        self.unary_predictor = torch.nn.Linear(hidden_dim, 2)
        self.binary_predictor = torch.nn.Linear(hidden_dim, 4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = x + self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = x + self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)


        x = self.conv4(x, edge_index)

        theta = torch.concatenate((
            torch.flatten(torch.nn.functional.normalize(self.unary_predictor(x[data.bipartite==0]), p=1, dim=1)),
            torch.flatten(torch.nn.functional.normalize(self.binary_predictor(x[data.bipartite==1]), p=1, dim=1))))

        #theta = torch.nn.functional.normalize(theta, p=1, dim=0)
        v2f = torch.zeros_like(theta, device=x.device)
        mu = _loopy_belief_propagation(data, theta, v2f, data.f2v_edges, data.r_edges, data.q_edges, data.m_edges, max_iter=self.bp_max_iter, tol=self.bp_tol, damping=self.bp_damping)

        return mu

class GCN_MRF_v2(torch.nn.Module):
    def __init__(self, in_dim, n_layers=4, hidden_dim=64, dropout_p=0.25, normalize=False, bp_max_iter=10, bp_tol=1e-6, bp_damping=0.5):
        super().__init__()
        self.bp_max_iter = bp_max_iter
        self.bp_tol = bp_tol
        self.bp_damping = bp_damping

        self.dropout_p = dropout_p
        self.conv_in = GCNConv(9, hidden_dim, normalize=normalize, aggr="max")
        self.act_in = ELU(inplace=True)
        self.dropout_in = Dropout(dropout_p)
        self.bn_in = BatchNorm1d(hidden_dim)
        self.layers = Sequential("x,edge_index", [
            (Sequential("x,edge_index", [
                (GCNConv(hidden_dim, hidden_dim, normalize=normalize), "x,edge_index -> x"),
                Dropout(dropout_p),
                BatchNorm1d(hidden_dim),
                ELU(inplace=True),
                ]), "x,edge_index -> x") for _ in range(n_layers)
            ])
        self.conv_out = GCNConv(hidden_dim, hidden_dim, normalize=normalize)
        self.dropout_out = Dropout(dropout_p)

        # canonical parameter predictors
        self.unary_predictor = torch.nn.Linear(hidden_dim, 2)
        self.binary_predictor = torch.nn.Linear(hidden_dim, 4)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv_in(x, edge_index)
        x = self.dropout_in(x)

        x = self.bn_in(x)

        x = self.act_in(x)


        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.layers(x, edge_index)


        x = self.conv_out(x, edge_index)
        x = self.dropout_out(x)


        theta = torch.concatenate((
            torch.flatten(torch.nn.functional.normalize(self.unary_predictor(x[data.bipartite==0]), p=1, dim=1)),
            torch.flatten(torch.nn.functional.normalize(self.binary_predictor(x[data.bipartite==1]), p=1, dim=1))))
        #theta = torch.nn.functional.normalize(theta, p=1, dim=0)
        v2f = torch.zeros_like(theta, device=x.device)
        mu = _loopy_belief_propagation(data, theta, v2f, data.f2v_edges, data.r_edges, data.q_edges, data.m_edges, max_iter=self.bp_max_iter, tol=self.bp_tol, damping=self.bp_damping)
        return mu

class GCN_MRF_v3(torch.nn.Module):
    def __init__(self, in_dim, n_layers=4, hidden_dim=64, dropout_p=0.25, normalize=False, bp_max_iter=10, bp_tol=1e-6, bp_damping=0.5):
        super().__init__()
        self.bp_max_iter = bp_max_iter
        self.bp_tol = bp_tol
        self.bp_damping = bp_damping
        self.conv = GCN(in_dim, hidden_dim, n_layers, out_channels=hidden_dim, dropout=dropout_p, act="elu", norm="LayerNorm", jk="lstm")

        # canonical parameter predictors
        self.unary_predictor = torch.nn.Linear(hidden_dim, 2)
        self.binary_predictor = torch.nn.Linear(hidden_dim, 4)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv(x, edge_index)

        theta = torch.concatenate((
            torch.flatten(torch.nn.functional.normalize(self.unary_predictor(x[data.bipartite==0]), p=1, dim=1)),
            torch.flatten(torch.nn.functional.normalize(self.binary_predictor(x[data.bipartite==1]), p=1, dim=1))))

        v2f = torch.zeros_like(theta, device=x.device)
        mu = _loopy_belief_propagation(data, theta, v2f, data.f2v_edges, data.r_edges, data.q_edges, data.m_edges, max_iter=self.bp_max_iter, tol=self.bp_tol, damping=self.bp_damping)

        return mu

from efcanonicalparam import EFCanonicalParam
class GCN_MRF_v4_tmp(torch.nn.Module):
    def __init__(self, in_dim, n_layers=4, hidden_dim=64, dropout_p=0.25, normalize=False, bp_max_iter=10, bp_tol=1e-6, bp_damping=0.5):
        super().__init__()
        self.bp_max_iter = bp_max_iter
        self.bp_tol = bp_tol
        self.bp_damping = bp_damping
        self.conv = GCN(in_dim, hidden_dim, n_layers, out_channels=hidden_dim, dropout=dropout_p, act="elu", norm="LayerNorm", jk="lstm")

        # canonical parameter predictors
        self.unary_predictor = torch.nn.Linear(hidden_dim, 2)
        self.binary_predictor = torch.nn.Linear(hidden_dim, 4)

        self.efc = EFCanonicalParam(hidden_dim, lambda in_channels, out_channels: MLP(in_channels, hidden_channels=[64, 64, out_channels], activation_layer=ELU, norm_layer=torch.nn.LayerNorm), factor_sizes=[1,2])
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv(x, edge_index)

        #theta = torch.concatenate((
        #    torch.flatten(torch.nn.functional.normalize(self.unary_predictor(x[data.bipartite==0]), p=1, dim=1)),
        #    torch.flatten(torch.nn.functional.normalize(self.binary_predictor(x[data.bipartite==1]), p=1, dim=1))))
        return self.efc(x, data.bipartite + 1)
        #print(theta.shape, theta_tmp.shape)

        #v2f = torch.zeros_like(theta, device=x.device)
        #mu = _loopy_belief_propagation(data, theta, v2f, data.f2v_edges, data.r_edges, data.q_edges, data.m_edges, max_iter=self.bp_max_iter, tol=self.bp_tol, damping=self.bp_damping)

        return mu

class GCN_MRF_v4(torch.nn.Module):
    def __init__(self, in_dim, n_layers=4, hidden_dim=64, dropout_p=0.25, normalize=False, bp_max_iter=10, bp_tol=1e-6, bp_damping=0.5):
        super().__init__()
        self.bp_max_iter = bp_max_iter
        self.bp_tol = bp_tol
        self.bp_damping = bp_damping
        self.conv = GCN(in_dim, hidden_dim, n_layers, out_channels=hidden_dim, dropout=dropout_p, act="elu", norm="LayerNorm", jk="lstm")

        # canonical parameter predictors
        self.unary_predictor = torch.nn.Linear(hidden_dim, 2)
        self.binary_predictor = torch.nn.Linear(hidden_dim, 4)

        self.efc = EFCanonicalParam(hidden_dim, lambda in_channels, out_channels: MLP(in_channels, hidden_channels=[64, 64, out_channels], activation_layer=ELU, norm_layer=torch.nn.LayerNorm), factor_sizes=[1,2])
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv(x, edge_index)

        theta = self.efc(x, data.bipartite + 1)

        v2f = torch.zeros_like(theta, device=x.device)
        mu = _loopy_belief_propagation(data, theta, v2f, data.f2v_edges, data.r_edges, data.q_edges, data.m_edges, max_iter=self.bp_max_iter, tol=self.bp_tol, damping=self.bp_damping)

        return mu

