import torch
import torch_scatter
import torch.nn.functional as F
from torch.nn import ReLU, ELU, Dropout, BatchNorm1d, GELU
from torch_geometric.nn.conv import GATConv
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GCNConv, Sequential, LayerNorm, to_hetero, HeteroDictLinear
from torch_geometric.nn.models import GCN, GAT
from torch_geometric.data import Data
from torchvision.ops import MLP

from bpropy.mrf.lbp import loopy_belief_propagation
from bpropy.mrf.naive_mean_field import naive_mean_field
from bpropy.models.efcanonicalparam import EFCanonicalParam

def concat_group_by(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    index_count = torch.bincount(index)
    fill_count = index_count.max() - index_count
    fill_zeros = torch.zeros_like(x[0], device=x.device).repeat(fill_count.sum(), *([1]*(len(x.shape)-1)))
    fill_index = torch.arange(0, fill_count.shape[0], device=x.device).repeat_interleave(fill_count)
    index_ = torch.cat([index, fill_index], dim=0)
    x_ = torch.cat([x, fill_zeros], dim=0)
    x_ = x_[torch.argsort(index_, stable=True)].view(index_count.shape[0], index_count.max(), *x.shape[1:])
    return x_
def segment_arange(lengths: torch.Tensor) -> torch.Tensor:
    total = int(lengths.sum().item())
    if total == 0:
        return torch.empty((0,), device=lengths.device, dtype=torch.long)
    end = torch.cumsum(lengths, 0)                 # [B]
    start = end - lengths                          # [B]
    base = torch.repeat_interleave(start, lengths) # [total]
    return torch.arange(total, device=lengths.device) - base

def count_pairwise_per_graph(edges) -> torch.Tensor:
    # factor_batch: [sum(n_factors)] -> graph id for each factor
    B = int(edges.n_factors.numel())
    factor_batch = torch.repeat_interleave(
        torch.arange(B, device=edges.device, dtype=torch.long),
        edges.n_factors.to(torch.long),
    )
    is_pair = (edges.m_stride.to(torch.long) == 4).to(torch.long)  # size=2
    n_pair = torch.zeros(B, device=edges.device, dtype=torch.long)
    n_pair.scatter_add_(0, factor_batch, is_pair)
    return n_pair

def interleave_theta(unary_theta: torch.Tensor, pair_theta: torch.Tensor, edges) -> torch.Tensor:
    # Flatten the predicted blocks
    unary_flat = unary_theta.reshape(-1)   # [2*sum(n_vars)]
    pair_flat  = pair_theta.reshape(-1)    # [4*sum(n_pair)]

    n_vars = edges.n_vars.to(torch.long)          # [B]
    n_pair = count_pairwise_per_graph(edges)      # [B]

    u_len = 2 * n_vars                             # [B]
    p_len = 4 * n_pair                             # [B]
    out_len = u_len + p_len                        # [B]
    out_ptr = torch.cumsum(out_len, 0) - out_len   # [B]

    # Build destination indices for unary part (first within each graph)
    u_local = segment_arange(u_len)                                  # [sum u_len]
    u_base  = torch.repeat_interleave(out_ptr, u_len)                # [sum u_len]
    out_u_idx = u_base + u_local                                     # [sum u_len]

    # Build destination indices for pairwise part (after unary within each graph)
    p_local = segment_arange(p_len)                                  # [sum p_len]
    p_base  = torch.repeat_interleave(out_ptr + u_len, p_len)        # [sum p_len]
    out_p_idx = p_base + p_local                                     # [sum p_len]

    # Scatter into final theta vector
    theta_flat = unary_flat.new_empty(int(out_len.sum().item()))
    theta_flat[out_u_idx] = unary_flat
    theta_flat[out_p_idx] = pair_flat
    return theta_flat
def debug_tensor(x, name, filename=None):
    if filename:
        with open(filename, "a") as f:
            print(name, "min ", x.min(dim=0), file=f)
            print(name, "max ", x.max(dim=0), file=f)
            print(name, "mean", x.mean(dim=0), file=f)
            print(name, "std ", x.std(dim=0), file=f)
    else:
        print(name, "min ", x.min(dim=0))
        print(name, "max ", x.max(dim=0))
        print(name, "mean", x.mean(dim=0))
        print(name, "std ", x.std(dim=0))

class GCN_MRF_v5(torch.nn.Module):
    def __init__(self, in_dim, n_layers=4, hidden_dim=64, dropout_p=0.25, normalize=False, bp_max_iter=10, bp_tol=1e-6, bp_damping=0.5, factor_sizes=[1,2], jk="lstm", norm="LayerNorm", act="elu", inference="loopy_belief_propagation", mapping_aggr="mean", normalize_theta=False, gnn_f="GCN"):
        super().__init__()
        self.inference = inference 
        self.bp_max_iter = bp_max_iter
        self.bp_tol = bp_tol
        self.bp_damping = bp_damping
        self.conv = eval(gnn_f)(in_dim, hidden_dim, n_layers, out_channels=hidden_dim, dropout=dropout_p, act=act, norm=norm, jk=jk)
        self.mapping_aggr=mapping_aggr
        self.normalize_theta = normalize_theta

        self.efc = EFCanonicalParam(hidden_dim, lambda in_channels, out_channels: MLP(in_channels, hidden_channels=[64, 64, out_channels], activation_layer=GELU, norm_layer=torch.nn.LayerNorm), factor_sizes=factor_sizes)
        self.mlp_unary = MLP(hidden_dim, hidden_channels=[64, 64, 2], activation_layer=GELU, norm_layer=torch.nn.LayerNorm)
        self.mlp_pairwise = MLP(3*hidden_dim, hidden_channels=[64, 64, 4], activation_layer=GELU, norm_layer=torch.nn.LayerNorm)

        if mapping_aggr=="gat":
            self.mapping = GATConv(hidden_dim, hidden_dim)
    def forward(self, data, fg_data=None, fg_map=None, inference: str | None = None):
        #print("in GCN_MRF_v4.forward")
        inference = self.inference if inference is None else inference

        x, edge_index = data.x, data.edge_index
        #print("\tmean(x) = ", torch.mean(x, dim=0))
        #print("\tstd(x)  = ", torch.std(x, dim=0))

        from torch_geometric.transforms import AddSelfLoops
        transform = AddSelfLoops(attr='edge_weight', fill_value=1.0)
        data = transform(data)

        x = self.conv(x, edge_index)
        #debug_tensor(x, "x", filename="debug.log")

        if hasattr(data, "fg_map"):
            m_stride = data.lbp_edges.m_stride.to(torch.long)
            is_unary = (m_stride == 2)
            is_pair = (m_stride == 4)

            unary_id_of_factor = torch.cumsum(is_unary.to(torch.long), 0) - 1
            pair_id_of_factor = torch.cumsum(is_pair.to(torch.long), 0) - 1

            n_unary = int(is_unary.sum().item())
            n_pair = int(is_pair.sum().item())

            node_idx  = data.fg_map[:, 0]
            factor_idx = data.fg_map[:, 1].to(torch.long)
            
            src_is_var = (data.bipartite[node_idx] == 0)
            src_is_con = ~src_is_var
            
            factor_is_unary = is_unary[factor_idx]
            factor_is_pair  = is_pair[factor_idx]

            mask_var_unary = src_is_var & factor_is_unary
            mask_con_unary = src_is_con & factor_is_unary
            mask_var_pair  = src_is_var & factor_is_pair
            mask_con_pair  = src_is_con & factor_is_pair

            unary_id_e = unary_id_of_factor[factor_idx]   # [E]
            pair_id_e  = pair_id_of_factor[factor_idx]    # [E]

            # variable endpoint per unary factor (usually exactly 1)
            var_x_unary = torch_scatter.scatter_max(
                x[node_idx[mask_var_unary]],
                unary_id_e[mask_var_unary],
                dim=0,
                dim_size=n_unary
            )[0]                                           # [n_unary, H]
            
            ## constraint endpoint aggregated per unary factor (maybe 0/1/many depending on your FG design)
            #con_x_unary = torch_scatter.scatter_max(
            #    x[node_idx[mask_con_unary]],
            #    unary_id_e[mask_con_unary],
            #    dim=0,
            #    dim_size=n_unary
            #)[0]                                           # [n_unary, H]
            
            #x_unary_f = torch.cat([var_x_unary, con_x_unary], dim=1)  # [n_unary, 2H]
            x_unary_f = var_x_unary

            # gather var endpoints for pair factors
            var_nodes = x[node_idx[mask_var_pair]]             # [E_var_pair, H]
            pid       = pair_id_e[mask_var_pair]               # [E_var_pair]
            
            var_x_pair = concat_group_by(var_nodes, pid).flatten(start_dim=1)  # [n_pair, 2H]
            con_x_pair = torch_scatter.scatter_max(
                x[node_idx[mask_con_pair]],
                pair_id_e[mask_con_pair],
                dim=0,
                dim_size=n_pair
            )[0]                                                # [n_pair, H]

            x_pair_f = torch.cat([var_x_pair, con_x_pair], dim=1)

            x = interleave_theta(self.mlp_unary(x_unary_f), self.mlp_pairwise(x_pair_f), data.lbp_edges)








        if inference in (None, "none", "theta", "no_inference"):
            return x
        if self.normalize_theta:
            x = (x - x.min()) / (x.max() - x.min())
        if inference == "loopy_belief_propagation":
            lbp_res = loopy_belief_propagation(x, torch.zeros_like(x), data.lbp_edges, max_iter=self.bp_max_iter, tol=self.bp_tol, damping=self.bp_damping)
            #debug_tensor(lbp_res.mu_v.reshape(NV,2), "mu_v (reshaped)", filename="debug.log")
            return lbp_res, x
        if inference == "naive_mean_field":
            return naive_mean_field(x, data.nmf_edges, max_iter=self.bp_max_iter, tol=self.bp_tol, damping=self.bp_damping), x
