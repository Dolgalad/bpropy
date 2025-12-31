# mrf_models_var_logits.py
import torch
from torch.nn import ELU
from torchvision.ops import MLP
from torch_geometric.nn import HeteroDictLinear, to_hetero, SAGEConv, GraphConv
from torch_geometric.nn.models import GCN, GAT
from torch_geometric.data import Data
import torch_scatter

from bpropy.models.efcanonicalparam import EFCanonicalParam

class GCNVarLogits(torch.nn.Module):
    """
    Homogeneous graph → one logit per *variable* node.
    We assume one of:
      - data.var_node_mask: BoolTensor [num_nodes]   (True = variable node), or
      - data.var_index / data.var_idx: LongTensor[K] (indices of variable nodes), or
      - data.num_vars: int and variables are [0..num_vars-1]; else fallback = all nodes.
    """
    def __init__(self, in_dim, n_layers=4, hidden_dim=64, dropout_p=0.25,
                 jk="lstm", norm="LayerNorm", act="elu", gnn_f="GCN",
                 readout_hidden=(64, 64)):
        super().__init__()
        Conv = dict(GCN=GCN, GAT=GAT)[gnn_f]
        self.conv = Conv(in_dim, hidden_dim, n_layers,
                         out_channels=hidden_dim, dropout=dropout_p,
                         act=act, norm=norm, jk=jk)
        self.readout = MLP(hidden_dim, hidden_channels=list(readout_hidden)+[1],
                           activation_layer=ELU, norm_layer=torch.nn.LayerNorm)

    def _var_idx(self, data):
        if hasattr(data, "var_node_mask"):
            return data.var_node_mask.nonzero(as_tuple=False).view(-1)
        if hasattr(data, "var_index"):
            return data.var_index.long()
        if hasattr(data, "var_idx"):
            return data.var_idx.long()
        if hasattr(data, "num_vars"):
            return torch.arange(int(data.num_vars), device=data.x.device)
        # fallback: all nodes
        return torch.arange(data.num_nodes, device=data.x.device)

    def forward(self, data):
        x = self.conv(data.x, data.edge_index)   # [N, hidden_dim]
        vidx = self._var_idx(data)               # [K]
        logits = self.readout(x[vidx]).squeeze(-1)  # [K]
        return logits


class HeteroGCNVarLogits(torch.nn.Module):
    """
    Hetero graph with types: 'variable' and 'constraint';
    one logit per node in the 'variable' store.
    """
    def __init__(self, in_variable_dim=10, in_constraint_dim=4, n_layers=4, hidden_dim=64, dropout_p=0.25,
                 jk="lstm", norm="LayerNorm", act="elu", gnn_f="GCN",
                 readout_hidden=(64, 64)):
        super().__init__()
        Conv = dict(GCN=GCN, GAT=GAT)[gnn_f]

        self.enc = HeteroDictLinear({'variable': in_variable_dim, 'constraint': in_constraint_dim}, hidden_dim)
        base = Conv(hidden_dim, hidden_dim, n_layers,
                    out_channels=hidden_dim, dropout=dropout_p,
                    act=act, norm=norm, jk=jk, add_self_loops=False)
        # Your hetero meta-graph matches the one used in your file:
        # (['variable','constraint'], [('variable','v2c','constraint'), ('constraint','c2v','variable')])
        self.conv = to_hetero(base,
                              (['variable','constraint'],
                               [('variable','v2c','constraint'),
                                ('constraint','c2v','variable')]),
                              aggr="mean")
        self.readout = MLP(hidden_dim, hidden_channels=list(readout_hidden)+[1],
                           activation_layer=ELU, norm_layer=torch.nn.LayerNorm)

    def forward(self, data):
        # x_dict: {'variable': [Nv, hid], 'constraint':[Nc, hid]}
        x_dict = self.enc(data.x_dict)
        x_dict = self.conv(x_dict, data.edge_index_dict)
        logits = self.readout(x_dict["variable"]).squeeze(-1)  # [Nv]
        return logits

class _BaseBipartiteGNN(torch.nn.Module):
    def __init__(self, hidden_dim: int, n_layers: int = 4, dropout_p: float = 0.25, conv="SAGE"):
        super().__init__()
        Conv = {"SAGE": SAGEConv, "GraphConv": GraphConv}[conv]
        self.convs = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        self.drop = torch.nn.Dropout(dropout_p)

        for _ in range(n_layers):
            # (-1, -1) enables bipartite (src, dst) dims inference
            self.convs.append(Conv((-1, -1), hidden_dim))
            self.acts.append(ELU())

    def forward(self, x, edge_index):
        # x may be Tensor or tuple(Tensor, Tensor) in hetero mode
        for conv, act in zip(self.convs, self.acts):
            x = conv(x, edge_index)
            x = act(x)
            x = self.drop(x)
        return x


class HeteroGCNVarLogits_v2(torch.nn.Module):
    def __init__(
        self,
        in_variable_dim=10,
        in_constraint_dim=4,
        n_layers=4,
        hidden_dim=64,
        dropout_p=0.25,
        readout_hidden=(64, 64),
        conv="GraphConv",          # "SAGE" or "GraphConv"
        aggr="mean",          # "mean" is typically stabler than "sum"
    ):
        super().__init__()
        self.enc = HeteroDictLinear(
            {"variable": in_variable_dim, "constraint": in_constraint_dim},
            hidden_dim,
        )

        base = _BaseBipartiteGNN(hidden_dim, n_layers=n_layers, dropout_p=dropout_p, conv=conv)

        self.conv = to_hetero(
            base,
            (["variable", "constraint"],
             [("variable", "v2c", "constraint"),
              ("constraint", "c2v", "variable")]),
            aggr=aggr,
        )

        self.readout = MLP(
            hidden_dim,
            hidden_channels=list(readout_hidden) + [1],
            activation_layer=ELU,
            norm_layer=torch.nn.LayerNorm,
        )

    def forward(self, data):
        x_dict = self.enc(data.x_dict)
        x_dict = self.conv(x_dict, data.edge_index_dict)
        return self.readout(x_dict["variable"]).squeeze(-1)

class HeteroGCNFactorLogits(torch.nn.Module):
    def __init__(self, in_variable_dim=10, in_constraint_dim=4, n_layers=4, hidden_dim=64, dropout_p=0.25, normalize=False, factor_sizes=[1,2], jk="lstm", norm="LayerNorm", act="elu", mapping_aggr="mean", gnn_f="GCN", add_self_loops=False, normalize_theta=False):
        super().__init__()
        self.conv = eval(gnn_f)(hidden_dim, hidden_dim, n_layers, out_channels=hidden_dim, dropout=dropout_p, act=act, norm=norm, jk=jk, add_self_loops=add_self_loops)
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


        # 2) Keys for a stable, vectorized lex-sort: by (graph_id, type) with variables first
        b_var = data["variable"].batch          # [n_var]
        b_con = data["constraint"].batch        # [n_con]
        g_ids = torch.cat([b_var, b_con], dim=0)                      # graph id per row in `xp`
        t_key = torch.cat([torch.zeros_like(b_var), torch.ones_like(b_con)], dim=0)  # 0=var, 1=constraint
        
        # Stable lexicographic sort: first by type (vars first), then by graph id
        idx = torch.arange(g_ids.numel(), device=g_ids.device)
        # secondary key (type), then primary (graph id) — both stable
        idx = idx[torch.argsort(t_key, stable=True)]
        idx = idx[torch.argsort(g_ids[idx], stable=True)]
        
        # 3) Apply the same permutation to `xp` and to factor sizes
        xp = xp.index_select(0, idx)
        
        fs_all = torch.cat([
            torch.ones(b_var.numel(), dtype=torch.long, device=xp.device),
            edge_data["constraint"].factor_sizes.to(torch.long)
        ], dim=0)
        fs_all = fs_all.index_select(0, idx)
        
        # 4) Pack into a temporary Data for EFCanonicalParam
        tmp_edge_data = Data()
        if hasattr(edge_data, "lbp_edges"):
            tmp_edge_data.lbp_edges = edge_data.lbp_edges
        if hasattr(edge_data, "nmf_edges"):
            tmp_edge_data.nmf_edges = edge_data.nmf_edges
        tmp_edge_data.factor_sizes = fs_all

        xp = self.efc(xp, tmp_edge_data)
        if hasattr(edge_data, "lbp_edges"):
            assert xp.numel() == int(edge_data.lbp_edges.n_ss.sum().item())
        if hasattr(edge_data, "nmf_edges"):
            assert xp.numel() == int(edge_data.nmf_edges.n_ss.sum().item())

        if self.normalize_theta:
            #xp = (xp - xp.min()) / (xp.max() - xp.min())
            if hasattr(edge_data, "lbp_edges"):
                m_idx = torch.arange(edge_data.lbp_edges.m_stride.size(0), device=xp.device).repeat_interleave(edge_data.lbp_edges.m_stride)
                m_stride = edge_data.lbp_edges.m_stride
            if hasattr(edge_data, "nmf_edges"):
                fs = tmp_edge_data.factor_sizes.to(torch.long)        # [n_factors]
                m_stride = (1 << fs)                                  # [n_factors], 2**arity
                m_idx = torch.arange(fs.numel(), device=xp.device).repeat_interleave(m_stride)
            # per-factor logsumexp of θ
            z_theta = torch_scatter.scatter_logsumexp(xp, m_idx)
            xp = xp - z_theta.repeat_interleave(m_stride)
            return xp

