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
        debug_tensor(x, "x", filename="debug.log")

        if hasattr(data, "fg_map"):
            NV = (data.bipartite==0).sum().item()

            ## build input tensor of unary variable embeddings, same as the original
            x_unary = x[:NV, :]
            debug_tensor(x_unary, "x_unary", filename="debug.log")

            #################################
            # pairwise factor count
            n_pair = int((data.factor_sizes == 2).sum().item())  # or nf - NV if guaranteed
            # but easiest if your factor ordering is unary first then pairwise:
            n = int(data.factor_sizes.numel() - NV)
            
            # --- VAR endpoints for pairwise factors ---
            var_fg_map = data.fg_map[(data.fg_map[:,0] < NV) & (data.fg_map[:,1] >= NV)]
            pair_idx = var_fg_map[:,1] - NV                      # <-- reindex to [0..n_pair-1]
            var_x_pairwise = concat_group_by(x[var_fg_map[:,0]], pair_idx).flatten(start_dim=1)
            
            # --- CONSTRAINT embedding aggregated per pairwise factor ---
            constr_fg_map = data.fg_map[(data.fg_map[:,0] >= NV) & (data.fg_map[:,1] >= NV)]
            pair_idx_c = constr_fg_map[:,1] - NV                 # <-- reindex too
            
            constr_x_pairwise, _ = torch_scatter.scatter_max(
                x[constr_fg_map[:,0]],
                pair_idx_c,
                dim=0,
                dim_size=n_pair                                  # <-- important: force size
            )
            
            x_pairwise = torch.cat((var_x_pairwise, constr_x_pairwise), dim=1)  # [n_pair, 3H]
            debug_tensor(x_pairwise, "x_pairwise", filename="debug.log")
            pairwise_theta = self.mlp_pairwise(x_pairwise)                  # [n_pair, 4]
            debug_tensor(pairwise_theta, "pairwise_theta", filename="debug.log")
            pairwise_theta = pairwise_theta.flatten()

            # unary params (2 per variable)
            unary_theta = self.mlp_unary(x_unary)          # shape: [2*NV]
            debug_tensor(unary_theta, "unary_theta", filename="debug.log")
            p1_unary = torch.softmax(unary_theta, dim=1)[:,1]
            debug_tensor(p1_unary, "p1_unary", filename="debug.log")

            unary_theta = unary_theta.flatten()
           
            # pairwise params already computed (4 per pairwise factor)
            #pairwise_theta = self.mlp_pairwise(x_pairwise).flatten() # shape: [4*n_pair]
            
            x = torch.cat([unary_theta, pairwise_theta], dim=0)
            
            #print("theta_flat", theta_flat.shape, "y_ss", data.y_ss.shape)
            #print("diff", theta_flat.shape[0] - data.y_ss.shape[0])


            # mapping
            #if self.mapping_aggr=="mean":
            #    x = torch_scatter.scatter_mean(x[data.fg_map[:,0]], data.fg_map[:,1],dim=0)
            #if self.mapping_aggr=="max":
            #    x,_ = torch_scatter.scatter_max(x[data.fg_map[:,0]], data.fg_map[:,1],dim=0)
            #if self.mapping_aggr=="gat":
            #    M = data.fg_map#[:,[1,0]]
            #    if M[:,1].max() >= x.shape[0]:
            #        x = torch.nn.functional.pad(x, (0,0,0,M.max() - x.shape[0]+1), "constant", 0)
            #    x = self.mapping(x, M.T)

            #print(_x.shape[0] - data.y_ss.shape[0])
            #print(x.shape)

        #x = self.efc(x, data)
        #print("after efc ", x.shape)


        if inference in (None, "none", "theta", "no_inference"):
            return x
        if self.normalize_theta:
            x = (x - x.min()) / (x.max() - x.min())
        if inference == "loopy_belief_propagation":
            lbp_res = loopy_belief_propagation(x, torch.zeros_like(x), data.lbp_edges, max_iter=self.bp_max_iter, tol=self.bp_tol, damping=self.bp_damping)
            debug_tensor(lbp_res.mu_v.reshape(NV,2), "mu_v (reshaped)", filename="debug.log")
            return lbp_res, x
        if inference == "naive_mean_field":
            return naive_mean_field(x, data.nmf_edges, max_iter=self.bp_max_iter, tol=self.bp_tol, damping=self.bp_damping), x
