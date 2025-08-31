# mrf_models_var_logits.py
import torch
from torch.nn import ELU
from torchvision.ops import MLP
from torch_geometric.nn import HeteroDictLinear, to_hetero
from torch_geometric.nn.models import GCN, GAT

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
                 jk="lstm", norm="LayerNorm", act="elu", gnn_f="GAT",
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
                              aggr="sum")
        self.readout = MLP(hidden_dim, hidden_channels=list(readout_hidden)+[1],
                           activation_layer=ELU, norm_layer=torch.nn.LayerNorm)

    def forward(self, data):
        # x_dict: {'variable': [Nv, hid], 'constraint':[Nc, hid]}
        x_dict = self.enc(data.x_dict)
        x_dict = self.conv(x_dict, data.edge_index_dict)
        logits = self.readout(x_dict["variable"]).squeeze(-1)  # [Nv]
        return logits

