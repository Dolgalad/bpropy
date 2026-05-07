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

############################## FIX FACTOR > 2
class GCN_MRF_v51(torch.nn.Module):
    """
    Generalized version of GCN_MRF_v5 that supports factors of size > 2.

    Key differences vs v5:
      - Ignores constraint embeddings completely.
      - Factor embedding = concat of variable embeddings in the factor scope (lex ordered).
      - Predicts theta for ALL factor arities using one MLP head per arity.
      - Packs theta in canonical factor order by emitting per-factor blocks in factor index order
        and slicing each by m_stride[f].

    Assumptions (same spirit as v5):
      - Binary variables, and factor parameterization uses full tables:
            m_stride[f] == 2 ** arity(f)
      - Factor indices follow canonical ordering within each instance:
            first by factor size, then lex(scope)
        and instances are concatenated by batch index.
    """

    def __init__(
        self,
        in_dim,
        n_layers=4,
        hidden_dim=64,
        dropout_p=0.25,
        normalize=False,
        bp_max_iter=10,
        bp_tol=1e-6,
        bp_damping=0.5,
        factor_sizes=[1, 2],   # include >2 here (e.g. [1,2,3,4]) if you have higher-order factors
        jk="lstm",
        norm="LayerNorm",
        act="elu",
        inference="loopy_belief_propagation",
        mapping_aggr="mean",
        normalize_theta=False,
        gnn_f="GCN",
    ):
        super().__init__()
        self.inference = inference
        self.bp_max_iter = bp_max_iter
        self.bp_tol = bp_tol
        self.bp_damping = bp_damping
        self.mapping_aggr = mapping_aggr
        self.normalize_theta = normalize_theta

        self.conv = eval(gnn_f)(
            in_dim, hidden_dim, n_layers,
            out_channels=hidden_dim,
            dropout=dropout_p,
            act=act,
            norm=norm,
            jk=jk
        )

        # One head per factor arity k: (k*H) -> (2**k)
        self.theta_heads = torch.nn.ModuleDict()
        for k in factor_sizes:
            self.theta_heads[str(int(k))] = MLP(
                int(k) * hidden_dim,
                hidden_channels=[64, 64, int(2 ** int(k))],
                activation_layer=GELU,
                norm_layer=torch.nn.LayerNorm
            )

        if mapping_aggr == "gat":
            self.mapping = GATConv(hidden_dim, hidden_dim)

    @staticmethod
    def _pack_theta_blocks(theta_blocks: torch.Tensor, m_stride: torch.Tensor) -> torch.Tensor:
        """
        theta_blocks: [n_factors, max_stride] (padded)
        m_stride:     [n_factors] true stride per factor
        Returns flat theta by taking, in factor index order, the first m_stride[f] entries of each row.
        This preserves canonical ordering as long as factor indices are canonical.
        """
        m_stride = m_stride.to(torch.long)
        max_stride = int(theta_blocks.size(1))
        mask = (torch.arange(max_stride, device=theta_blocks.device)[None, :] < m_stride[:, None])
        return theta_blocks[mask]  # [sum_f m_stride[f]]

    def forward(self, data, fg_data=None, fg_map=None, inference: str | None = None):
        inference = self.inference if inference is None else inference

        x, edge_index = data.x, data.edge_index

        from torch_geometric.transforms import AddSelfLoops
        transform = AddSelfLoops(attr="edge_weight", fill_value=1.0)
        data = transform(data)

        x = self.conv(x, edge_index)  # [n_nodes, H]

        if hasattr(data, "fg_map"):
            # Number of (approx) factors in THIS instance batch is implied by lbp_edges.m_stride
            m_stride = data.lbp_edges.m_stride.to(torch.long)     # [n_factors_total]
            n_factors = int(m_stride.numel())
            max_stride = int(m_stride.max().item())

            # --- 1) collect (var -> factor) incidences ---
            node_idx   = data.fg_map[:, 0]
            factor_idx = data.fg_map[:, 1].to(torch.long)

            # keep only variable endpoints (ignore constraints completely)
            src_is_var = (data.bipartite[node_idx] == 0)
            var_nodes  = node_idx[src_is_var].to(torch.long)      # [E_var]
            fac_of_e   = factor_idx[src_is_var].to(torch.long)    # [E_var]

            # Safety: enforce lex ordering within each factor (stable sort by (factor_idx, var_id))
            # If your variable node ids coincide with var ids, this is lex over var ids.
            if var_nodes.numel() > 0:
                max_vid = var_nodes.max().item() + 1
                key = fac_of_e * max_vid + var_nodes
                perm = torch.argsort(key, stable=True)
                var_nodes = var_nodes[perm]
                fac_of_e  = fac_of_e[perm]

            var_emb = x[var_nodes]                                 # [E_var, H]

            # Factor arity from incidence counts (more robust than log2(m_stride))
            counts = torch.bincount(fac_of_e, minlength=n_factors).to(torch.long)  # [n_factors]

            # Optional consistency check: m_stride should be 2**arity for binary full tables
            # (leave as runtime check; comment out if you don't want asserts)
            # If this fails, you need explicit factor arity / table size info per factor.
            expected = (1 << counts).to(torch.long)
            if expected.numel() == m_stride.numel():
                if not torch.all(expected == m_stride):
                    raise RuntimeError(
                        "GCN_MRF_v51: stride/arity mismatch. "
                        "Expected m_stride == 2**arity from fg_map incidence counts."
                    )

            # --- 2) group variable embeddings by factor, producing [n_factors, k_max, H] ---
            # concat_group_by pads to max group size
            grouped = concat_group_by(var_emb, fac_of_e)            # [n_factors, k_max, H]
            k_max = int(grouped.size(1))

            # --- 3) predict per-factor theta blocks ---
            theta_blocks = x.new_zeros((n_factors, max_stride))     # padded rows

            # small loop over arities (k values), NOT over factors
            for k_str, head in self.theta_heads.items():
                k = int(k_str)
                idx = (counts == k).nonzero(as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                if k > k_max:
                    raise RuntimeError(
                        f"GCN_MRF_v51: observed arity {k} but grouped only up to {k_max}. "
                        "Check fg_map / grouping."
                    )

                xk = grouped[idx, :k, :].reshape(idx.numel(), k * x.size(1))  # [n_k, kH]
                tk = head(xk)                                                 # [n_k, 2**k]
                theta_blocks[idx, :tk.size(1)] = tk

            # --- 4) pack into flat theta in canonical factor index order ---
            x = self._pack_theta_blocks(theta_blocks, m_stride)

        # --- inference ---
        if inference in (None, "none", "theta", "no_inference"):
            return x
        if self.normalize_theta:
            x = (x - x.min()) / (x.max() - x.min())

        if inference == "loopy_belief_propagation":
            lbp_res = loopy_belief_propagation(
                x, torch.zeros_like(x), data.lbp_edges,
                max_iter=self.bp_max_iter, tol=self.bp_tol, damping=self.bp_damping
            )
            return lbp_res, x

        if inference == "naive_mean_field":
            return naive_mean_field(
                x, data.nmf_edges,
                max_iter=self.bp_max_iter, tol=self.bp_tol, damping=self.bp_damping
            ), x

        return x

################ REAL Attention
class GCN_MRF_v52(torch.nn.Module):
    """
    General-arity version of Proposal 2 (constraint attention), built on top of GCN_MRF_v51.

    Differences vs v51:
      - Uses BOTH variable and constraint nodes from fg_map.
      - Builds a per-factor constraint summary h_c[f] by attention-weighted sum over
        constraint embeddings neighboring that factor in the precedence graph.
      - Theta head input for arity k becomes (k*H + H) = (k+1)H.

    Attention (generalized Proposal 2):
        q_f = AGG_{v in scope(f)} h_v          # factor context (fixed size H)
        s_{f,alpha} = MLP_att([q_f, h_alpha])  # scalar score
        beta_{f,alpha} = softmax_alpha s_{f,alpha} (per factor)
        h_c[f] = sum_alpha beta_{f,alpha} * h_alpha
    """

    def __init__(
        self,
        in_dim,
        n_layers=4,
        hidden_dim=64,
        dropout_p=0.25,
        normalize=False,
        bp_max_iter=10,
        bp_tol=1e-6,
        bp_damping=0.5,
        factor_sizes=[1, 2],   # include >2 here (e.g. [1,2,3,4]) if you have higher-order factors
        jk="lstm",
        norm="LayerNorm",
        act="elu",
        inference="loopy_belief_propagation",
        mapping_aggr="mean",   # used for factor context q_f (mean/max/min) OR "gat" (kept for compatibility)
        normalize_theta=False,
        gnn_f="GCN",
        att_hidden=64,
    ):
        super().__init__()
        self.inference = inference
        self.bp_max_iter = bp_max_iter
        self.bp_tol = bp_tol
        self.bp_damping = bp_damping
        self.mapping_aggr = mapping_aggr
        self.normalize_theta = normalize_theta

        self.conv = eval(gnn_f)(
            in_dim, hidden_dim, n_layers,
            out_channels=hidden_dim,
            dropout=dropout_p,
            act=act,
            norm=norm,
            jk=jk
        )

        # Theta heads: arity-k factor gets input (k*H + H) = (k+1)H and outputs 2**k
        self.theta_heads = torch.nn.ModuleDict()
        for k in factor_sizes:
            k = int(k)
            self.theta_heads[str(k)] = MLP(
                (k + 1) * hidden_dim,
                hidden_channels=[64, 64, int(2 ** k)],
                activation_layer=GELU,
                norm_layer=torch.nn.LayerNorm
            )

        # Attention score head: [q_f, h_alpha] -> scalar
        self.mlp_att = MLP(
            2 * hidden_dim,
            hidden_channels=[att_hidden, att_hidden, 1],
            activation_layer=GELU,
            norm_layer=torch.nn.LayerNorm
        )

        if mapping_aggr == "gat":
            self.mapping = GATConv(hidden_dim, hidden_dim)

    @staticmethod
    def _pack_theta_blocks(theta_blocks: torch.Tensor, m_stride: torch.Tensor) -> torch.Tensor:
        m_stride = m_stride.to(torch.long)
        max_stride = int(theta_blocks.size(1))
        mask = (torch.arange(max_stride, device=theta_blocks.device)[None, :] < m_stride[:, None])
        return theta_blocks[mask]  # [sum_f m_stride[f]]

    @staticmethod
    def _scatter_softmax(scores: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
        """
        Stable softmax over groups defined by 'index' (1D).
        scores: [E]
        index:  [E] in [0, dim_size)
        returns: [E] softmax(scores) per group
        """
        if scores.numel() == 0:
            return scores
        max_per = torch_scatter.scatter_max(scores, index, dim=0, dim_size=dim_size)[0]
        max_per_e = max_per.index_select(0, index)
        exp = torch.exp(scores - max_per_e)
        denom = torch_scatter.scatter_add(exp, index, dim=0, dim_size=dim_size).index_select(0, index)
        return exp / (denom + 1e-12)

    def _agg_by_factor(self, var_emb: torch.Tensor, fac_of_var: torch.Tensor, n_factors: int) -> torch.Tensor:
        """
        Computes q_f = AGG_{v in f} h_v as a fixed-size [n_factors, H] tensor.
        """
        if var_emb.numel() == 0:
            return var_emb.new_zeros((n_factors, var_emb.size(1)))

        if self.mapping_aggr == "max":
            return torch_scatter.scatter_max(var_emb, fac_of_var, dim=0, dim_size=n_factors)[0]
        if self.mapping_aggr == "min":
            return torch_scatter.scatter_min(var_emb, fac_of_var, dim=0, dim_size=n_factors)[0]
        # default / "mean"
        return torch_scatter.scatter_mean(var_emb, fac_of_var, dim=0, dim_size=n_factors)

    def forward(self, data, fg_data=None, fg_map=None, inference: str | None = None):
        inference = self.inference if inference is None else inference

        x, edge_index = data.x, data.edge_index
        from torch_geometric.transforms import AddSelfLoops
        data = AddSelfLoops(attr="edge_weight", fill_value=1.0)(data)

        # Feature extraction
        x = self.conv(x, edge_index)  # [n_nodes, H]

        if hasattr(data, "fg_map"):
            m_stride = data.lbp_edges.m_stride.to(torch.long)  # [n_factors]
            n_factors = int(m_stride.numel())
            max_stride = int(m_stride.max().item())

            node_idx   = data.fg_map[:, 0].to(torch.long)
            factor_idx = data.fg_map[:, 1].to(torch.long)

            src_is_var = (data.bipartite[node_idx] == 0)
            src_is_con = ~src_is_var

            # -----------------------------
            # 1) Variable -> factor incidences (lex ordered)
            # -----------------------------
            var_nodes = node_idx[src_is_var]
            fac_of_v  = factor_idx[src_is_var]

            if var_nodes.numel() > 0:
                max_vid = int(var_nodes.max().item()) + 1
                key = fac_of_v * max_vid + var_nodes
                perm = torch.argsort(key, stable=True)
                var_nodes = var_nodes[perm]
                fac_of_v  = fac_of_v[perm]

            var_emb = x[var_nodes]  # [E_var, H]

            # Factor arities from incidence counts
            counts = torch.bincount(fac_of_v, minlength=n_factors).to(torch.long)  # [n_factors]

            # Consistency: for binary full tables, stride == 2**arity
            expected = (1 << counts).to(torch.long)
            if expected.numel() == m_stride.numel():
                if not torch.all(expected == m_stride):
                    raise RuntimeError(
                        "GCN_MRF_v52: stride/arity mismatch. "
                        "Expected m_stride == 2**arity from fg_map incidence counts."
                    )

            # Group var embeddings by factor: [n_factors, k_max, H]
            grouped = concat_group_by(var_emb, fac_of_v)
            k_max = int(grouped.size(1))

            # -----------------------------
            # 2) Constraint attention per factor to get h_c[f] in R^H
            # -----------------------------
            # factor context q_f: [n_factors, H]
            q_f = self._agg_by_factor(var_emb, fac_of_v, n_factors)  # [n_factors, H]

            con_nodes = node_idx[src_is_con]
            fac_of_c  = factor_idx[src_is_con]
            h_alpha   = x[con_nodes]  # [E_con, H]

            if h_alpha.numel() == 0:
                h_c = x.new_zeros((n_factors, x.size(1)))
            else:
                q_e = q_f.index_select(0, fac_of_c)                 # [E_con, H]
                att_in = torch.cat([q_e, h_alpha], dim=1)           # [E_con, 2H]
                scores = self.mlp_att(att_in).squeeze(-1)           # [E_con]
                beta = self._scatter_softmax(scores, fac_of_c, dim_size=n_factors)  # [E_con]
                h_c = torch_scatter.scatter_add(
                    h_alpha * beta.unsqueeze(-1),
                    fac_of_c,
                    dim=0,
                    dim_size=n_factors
                )  # [n_factors, H]

            # -----------------------------
            # 3) Predict theta blocks per factor (loop over arities, not factors)
            # -----------------------------
            theta_blocks = x.new_zeros((n_factors, max_stride))

            for k_str, head in self.theta_heads.items():
                k = int(k_str)
                idx = (counts == k).nonzero(as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                if k > k_max:
                    raise RuntimeError(
                        f"GCN_MRF_v52: observed arity {k} but grouped only up to {k_max}. "
                        "Check fg_map / grouping."
                    )

                xk_vars = grouped[idx, :k, :].reshape(idx.numel(), k * x.size(1))  # [n_k, kH]
                xk_c    = h_c[idx]                                                 # [n_k, H]
                xk      = torch.cat([xk_vars, xk_c], dim=1)                        # [n_k, (k+1)H]
                tk      = head(xk)                                                 # [n_k, 2**k]
                theta_blocks[idx, :tk.size(1)] = tk

            # -----------------------------
            # 4) Pack into flat theta in canonical factor order
            # -----------------------------
            x = self._pack_theta_blocks(theta_blocks, m_stride)

        # -----------------------------
        # Inference
        # -----------------------------
        if inference in (None, "none", "theta", "no_inference"):
            return x

        if self.normalize_theta:
            x = (x - x.min()) / (x.max() - x.min() + 1e-12)

        if inference == "loopy_belief_propagation":
            lbp_res = loopy_belief_propagation(
                x, torch.zeros_like(x), data.lbp_edges,
                max_iter=self.bp_max_iter, tol=self.bp_tol, damping=self.bp_damping
            )
            return lbp_res, x

        if inference == "naive_mean_field":
            return naive_mean_field(
                x, data.nmf_edges,
                max_iter=self.bp_max_iter, tol=self.bp_tol, damping=self.bp_damping
            ), x

        return x
