# mrf_models_var_logits.py
import torch
import torch.nn.functional as F
from torch.nn import ELU
from torchvision.ops import MLP
from torch_geometric.nn import HeteroDictLinear, to_hetero, SAGEConv, GraphConv
from torch_geometric.nn.models import GCN, GAT, GraphSAGE
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
        Conv = dict(GCN=GCN, GAT=GAT, SAGE=GraphSAGE)[gnn_f]
        self.conv = Conv(in_dim, hidden_dim, n_layers,
                         out_channels=hidden_dim, dropout=dropout_p,
                         act=act, norm=norm, jk=jk)
        if norm is None:
            norm_layer = None
        elif norm == "LayerNorm":
            norm_layer = torch.nn.LayerNorm
        elif norm == "BatchNorm":
            norm_layer = torch.nn.BatchNorm1d
        else:
            norm_layer = None
        self.readout = MLP(hidden_dim, hidden_channels=list(readout_hidden)+[1],
                           activation_layer=ELU, norm_layer=norm_layer)

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


class ResidualSAGEVarLogits(torch.nn.Module):
    """
    Homogeneous graph -> one logit per variable node, with:
    logit_v = base(x_v) + delta(h_v)

    - base(x): MLP on raw node features (strong per-node baseline)
    - h: stacked SAGEConv with residual connections + LayerNorm
    - delta(h): small MLP correction head

    Matches GCNVarLogits' variable indexing conventions.
    """

    def __init__(
        self,
        in_dim: int,
        n_layers: int = 4,
        hidden_dim: int = 128,
        dropout_p: float = 0.25,
        sage_aggr: str = "mean",          # "mean", "max", "lstm", ...
        use_layernorm: bool = True,
        base_hidden=(128, 128),
        delta_hidden=(128, 64),
        init_base_scale: float = 1.0,     # can reduce if base dominates, e.g. 0.5
        init_delta_scale: float = 1.0,    # can reduce if delta too noisy, e.g. 0.5
        ):
        super().__init__()
        assert n_layers >= 1

        # Baseline: logits from raw features only
        self.base_mlp = MLP(
            in_dim,
            hidden_channels=list(base_hidden) + [1],
            activation_layer=ELU,
            norm_layer=torch.nn.LayerNorm,
            )

        # GNN trunk: SAGEConv stack
        self.input_proj = (
            torch.nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else torch.nn.Identity()
            )
        self.convs = torch.nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim, aggr=sage_aggr) for _ in range(n_layers)]
            )
        self.norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hidden_dim) for _ in range(n_layers)]
            ) if use_layernorm else None
        self.dropout_p = dropout_p

        # Delta head: correction logits from contextual embeddings
        self.delta_mlp = MLP(
            hidden_dim,
            hidden_channels=list(delta_hidden) + [1],
            activation_layer=ELU,
            norm_layer=torch.nn.LayerNorm,
            )

        # Optional learnable scales for stability / easy ablation
        self.base_scale = torch.nn.Parameter(torch.tensor(float(init_base_scale)))
        self.delta_scale = torch.nn.Parameter(torch.tensor(float(init_delta_scale)))

    def _var_idx(self, data):
        if hasattr(data, "var_node_mask"):
            return data.var_node_mask.nonzero(as_tuple=False).view(-1)
        if hasattr(data, "var_index"):
            return data.var_index.long()
        if hasattr(data, "var_idx"):
            return data.var_idx.long()
        if hasattr(data, "num_vars"):
            return torch.arange(int(data.num_vars), device=data.x.device)
        return torch.arange(data.num_nodes, device=data.x.device)

    def forward(self, data):
        x_raw = data.x
        edge_index = data.edge_index

        vidx = self._var_idx(data)

        # Base logits from raw features (no message passing)
        base_logits = self.base_mlp(x_raw[vidx]).squeeze(-1)

        # GNN trunk (residual SAGE)
        h = self.input_proj(x_raw)
        for i, conv in enumerate(self.convs):
            h_in = h
            h = conv(h, edge_index)
            if self.norms is not None:
                h = self.norms[i](h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout_p, training=self.training)

            # Residual connection (prevents over-smoothing / preserves identity)
            h = h + h_in

        # Delta correction from contextual embeddings
        delta_logits = self.delta_mlp(h[vidx]).squeeze(-1)

        # Residual prediction
        logits = self.base_scale * base_logits + self.delta_scale * delta_logits
        return logits
