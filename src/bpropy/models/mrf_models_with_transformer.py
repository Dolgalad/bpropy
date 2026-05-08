from typing import TYPE_CHECKING

import torch

try:
    import torch_scatter  # type: ignore
except ModuleNotFoundError:
    torch_scatter = None  # type: ignore

from torch.nn import GELU
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.models import GCN, GAT
from torchvision.ops import MLP

from bpropy.mrf.lbp import loopy_belief_propagation
from bpropy.mrf.naive_mean_field import naive_mean_field


def concat_group_by(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Group rows of x by 1D index into a dense (n_groups, max_count, ...) tensor.

    Pads with zeros to max group size and preserves stable ordering within each group
    (assuming the caller pre-sorted by (index, secondary_key)).
    """
    index = index.to(torch.long)
    if index.numel() == 0:
        # Cannot infer n_groups; caller should avoid calling in this case.
        return x.new_zeros((0, 0) + x.shape[1:])

    index_count = torch.bincount(index)
    fill_count = index_count.max() - index_count
    fill_zeros = torch.zeros_like(x[0], device=x.device).repeat(fill_count.sum(), *([1] * (len(x.shape) - 1)))
    fill_index = torch.arange(0, fill_count.shape[0], device=x.device).repeat_interleave(fill_count)
    index_ = torch.cat([index, fill_index], dim=0)
    x_ = torch.cat([x, fill_zeros], dim=0)
    x_ = x_[torch.argsort(index_, stable=True)].view(index_count.shape[0], index_count.max(), *x.shape[1:])
    return x_


def concat_group_by_1d(vals: torch.Tensor, index: torch.Tensor, fill_value: int = 0) -> torch.Tensor:
    """Group a 1D tensor vals by index into dense (n_groups, max_count) with padding."""
    index = index.to(torch.long)
    vals = vals.to(torch.long)
    if index.numel() == 0:
        return vals.new_zeros((0, 0))

    index_count = torch.bincount(index)
    max_count = int(index_count.max().item())
    fill_count = index_count.max() - index_count

    fill_vals = vals.new_full((int(fill_count.sum().item()),), fill_value)
    fill_index = torch.arange(0, fill_count.shape[0], device=vals.device).repeat_interleave(fill_count)

    index_ = torch.cat([index, fill_index], dim=0)
    vals_ = torch.cat([vals, fill_vals], dim=0)

    vals_ = vals_[torch.argsort(index_, stable=True)].view(index_count.shape[0], max_count)
    return vals_


class FactorTransformerMapper(torch.nn.Module):
    """Transformer mapping over precedence neighbors.

    Builds per-factor token sequences of shape (|F'|, M, H) and runs L transformer
    encoder layers with padding mask. Variable/constraint distinction is handled
    by learned type offsets Hv/Hc added to the token embeddings.

    Output factor embedding z_f is a masked max-pooling over the neighbor dimension.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.Hv = torch.nn.Parameter(0.01 * torch.randn(hidden_dim))
        self.Hc = torch.nn.Parameter(0.01 * torch.randn(hidden_dim))

        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_mult * hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(
        self,
        h_nodes: torch.Tensor,          # [n_nodes, H]
        nbr_ids: torch.Tensor,          # [n_factors, M]
        nbr_is_var: torch.Tensor,       # [n_factors, M] bool
        mask: torch.Tensor,             # [n_factors, M] bool
    ) -> torch.Tensor:
        # Gather token embeddings
        flat = nbr_ids.reshape(-1)
        G = h_nodes.index_select(0, flat).reshape(nbr_ids.size(0), nbr_ids.size(1), -1)  # [F, M, H]
        #print("\nin Mapper ")
        #print("\th_nodes.nans = ", torch.isnan(h_nodes).any())
        #print("\tGnans0 = ", torch.isnan(G).any())

        # Add learned type offsets
        Hv = self.Hv.view(1, 1, -1)
        Hc = self.Hc.view(1, 1, -1)
        type_offset = torch.where(nbr_is_var.unsqueeze(-1), Hv, Hc)
        G = G + type_offset

        #print("\tG.shape = ", G.shape)
        #print("\tGnans = ", torch.isnan(G).any())
        #print("\tmask.shape = ", mask.shape)
        #print("\t# all zero rows = ", (mask.sum(dim=1) == 0).sum())
        #print("\tmask row sum = ", mask.sum(dim=1))

        if (mask.sum(dim=1)==0).any():
            print("Zero row in mask")
            exit(2)

        # Zero out padding entries (so they don't leak into FFN)
        G = G * mask.unsqueeze(-1)
        #print("\tA : ", torch.isnan(G).any())

        # Transformer expects True for padding in key_padding_mask
        key_padding_mask = ~mask
        G = self.encoder(G, src_key_padding_mask=key_padding_mask)
        #print("\tA : ", torch.isnan(G).any())

        # Masked max pool over neighbor axis
        neg_inf = torch.finfo(G.dtype).min
        G_masked = G.masked_fill(~mask.unsqueeze(-1), neg_inf)
        z = G_masked.max(dim=1).values  # [F, H]

        # Handle pathological all-padding rows (shouldn't happen, but safe)
        no_tok = (mask.sum(dim=1) == 0)
        if no_tok.any():
            z = z.clone()
            z[no_tok] = 0.0

        return z


class GCN_MRF_v61_Transformer(torch.nn.Module):
    """Transformer-based mapping model for approximate factor graphs (general arity).

    Based on GCN_MRF_v51's conventions:
      - GNN feature extraction on the original bipartite graph (vars + constraints).
      - Uses data.fg_map as precedence edges from V_x (vars+constraints) -> F'.
      - Factor arity is inferred as the number of VARIABLE neighbors per factor.
      - Assumes binary full tables: m_stride[f] == 2 ** arity(f).

    Transformer mapping (per your corrected spec):
      - Build tensor G0 of shape (|F'|, M, H) by padding precedence neighbors.
      - Add learned offsets Hv/Hc to distinguish variable vs constraint tokens.
      - Apply L transformer encoder layers across the neighbor dimension.
      - Factor embedding z_f is masked max-pooling over the neighbor dimension.
      - Predict theta_f with an arity-specific head: MLP_k: R^H -> R^{2^k}.
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
        factor_sizes=(1, 2),
        jk="lstm",
        norm="LayerNorm",
        act="elu",
        inference="loopy_belief_propagation",
        normalize_theta=False,
        gnn_f="GCN",
        # transformer params
        tf_n_layers: int = 2,
        tf_n_heads: int = 4,
        tf_ff_mult: int = 4,
        tf_dropout: float = 0.1,
        tf_hidden_channel: int = 64,
        # neighbor ordering
        vars_first: bool = True,
        theta_scale: float = 0.1,
    ):
        super().__init__()

        self.inference = inference
        self.bp_max_iter = bp_max_iter
        self.bp_tol = bp_tol
        self.bp_damping = bp_damping
        self.normalize_theta = normalize_theta
        self.vars_first = vars_first
        self.theta_scale = theta_scale

        self.z_norm = torch.nn.LayerNorm(hidden_dim)

        self.conv = eval(gnn_f)(
            in_dim,
            hidden_dim,
            n_layers,
            out_channels=hidden_dim,
            dropout=dropout_p,
            act=act,
            norm=norm,
            jk=jk,
        )

        self.mapper = FactorTransformerMapper(
            hidden_dim=hidden_dim,
            n_layers=tf_n_layers,
            n_heads=tf_n_heads,
            ff_mult=tf_ff_mult,
            dropout=tf_dropout,
        )

        # One head per factor arity k: (H) -> (2**k)
        self.theta_heads = torch.nn.ModuleDict()
        for k in factor_sizes:
            k = int(k)
            self.theta_heads[str(k)] = MLP(
                hidden_dim,
                hidden_channels=[tf_hidden_channel, tf_hidden_channel, int(2 ** k)],
                activation_layer=GELU,
                norm_layer=torch.nn.LayerNorm,
            )

        if gnn_f == "GAT":
            # keep parity with other files (not strictly needed)
            self.mapping = GATConv(hidden_dim, hidden_dim)

    @staticmethod
    def _pack_theta_blocks(theta_blocks: torch.Tensor, m_stride: torch.Tensor) -> torch.Tensor:
        m_stride = m_stride.to(torch.long)
        max_stride = int(theta_blocks.size(1))
        mask = (torch.arange(max_stride, device=theta_blocks.device)[None, :] < m_stride[:, None])
        return theta_blocks[mask]

    def forward(self, data, fg_data=None, fg_map=None, inference: str | None = None):
        inference = self.inference if inference is None else inference
    
        # Use the transformed graph consistently
        from torch_geometric.transforms import AddSelfLoops
        data = AddSelfLoops(attr="edge_weight", fill_value=1.0)(data)
    
        x = torch.nan_to_num(data.x, nan=0.0)
        edge_index = data.edge_index
    
        # Feature extraction on original graph
        x = self.conv(x, edge_index)  # [n_nodes, H]
    
        # Optional debug store
        if not hasattr(data, "_debug_stats"):
            data._debug_stats = {}
    
        # Mapping + theta prediction
        #if hasattr(data, "fg_map"):
        #    m_stride = data.lbp_edges.m_stride.to(torch.long)  # [n_factors]
        #    n_factors = int(m_stride.numel())
        #    max_stride = int(m_stride.max().item())
    
        #    node_idx = data.fg_map[:, 0].to(torch.long)
        #    factor_idx = data.fg_map[:, 1].to(torch.long)
    
        #    src_is_var = (data.bipartite[node_idx] == 0)
        #    src_is_con = ~src_is_var
                # Mapping + theta prediction
        if hasattr(data, "fg_map"):
            node_idx = data.fg_map[:, 0].to(torch.long)
            factor_idx = data.fg_map[:, 1].to(torch.long)

            src_is_var = (data.bipartite[node_idx] == 0)
            src_is_con = ~src_is_var

            # This transformer is used for both LBP and NMF approximate-FG runs.
            # Do not read factor sizes from data.lbp_edges here, because NMF runs
            # may only have data.nmf_edges. The approximate factor arity is the
            # number of variable-side fg_map entries per approximate factor.
            if hasattr(data, "fg_factor_sizes"):
                n_factors = int(data.fg_factor_sizes.numel())
            else:
                n_factors = int(factor_idx.max().item()) + 1 if factor_idx.numel() else 0

            fac_of_var = factor_idx[src_is_var]
            arity = torch.bincount(fac_of_var, minlength=n_factors).to(torch.long)
            m_stride = (1 << arity).to(torch.long)
            max_stride = int(m_stride.max().item()) if m_stride.numel() else 0

            if hasattr(data, "fg_factor_sizes"):
                fg_factor_sizes = data.fg_factor_sizes.to(device=arity.device, dtype=torch.long)
                if fg_factor_sizes.numel() == arity.numel() and not torch.all(fg_factor_sizes == arity):
                    raise RuntimeError(
                        "GCN_MRF_v61_Transformer: fg_factor_sizes/fg_map mismatch. "
                        "Expected fg_factor_sizes to equal the number of variable "
                        "neighbors per approximate factor."
                    )
    
            if node_idx.numel() > 0:
                max_nid = int(node_idx.max().item()) + 1
                type_bit = src_is_con.to(torch.long) if self.vars_first else src_is_var.to(torch.long)
                key = factor_idx * (2 * max_nid) + type_bit * max_nid + node_idx
                perm = torch.argsort(key, stable=True)
                node_idx = node_idx[perm]
                factor_idx = factor_idx[perm]
                src_is_var = src_is_var[perm]
    
            nbr_ids = concat_group_by_1d(node_idx, factor_idx, fill_value=0)
            nbr_is_var = concat_group_by_1d(
                src_is_var.to(torch.long), factor_idx, fill_value=0
            ).bool()
    
            counts_all = torch.bincount(factor_idx, minlength=n_factors).to(torch.long)
            M = int(nbr_ids.size(1))
            ar = torch.arange(M, device=x.device).view(1, M)
            mask = ar < counts_all.view(-1, 1)
    
            #fac_of_var = factor_idx[src_is_var]
            #arity = torch.bincount(fac_of_var, minlength=n_factors).to(torch.long)
    
            present_arities = set(torch.unique(arity).tolist())
            available_arities = {int(k) for k in self.theta_heads.keys()}
            missing = present_arities - available_arities
            if missing:
                raise RuntimeError(
                    f"Missing theta heads for factor arities {sorted(missing)}; "
                    f"available heads are {sorted(available_arities)}"
                )
    
            expected = (1 << arity).to(torch.long)
            if expected.numel() == m_stride.numel() and not torch.all(expected == m_stride):
                raise RuntimeError(
                    "GCN_MRF_v61_Transformer: stride/arity mismatch. "
                    "Expected m_stride == 2**arity where arity is #variable neighbors in fg_map."
                )
    
            z = self.mapper(x, nbr_ids=nbr_ids, nbr_is_var=nbr_is_var, mask=mask)
            z = self.z_norm(z)
    
            theta_blocks = x.new_zeros((n_factors, max_stride))
            for k_str, head in self.theta_heads.items():
                k = int(k_str)
                idx = (arity == k).nonzero(as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                #tk = head(z.index_select(0, idx))  # [n_k, 2**k]
                tk = self.theta_scale * head(z.index_select(0, idx))  # [n_k, 2**k]
                theta_blocks[idx, :tk.size(1)] = tk
    
            theta_packed = self._pack_theta_blocks(theta_blocks, m_stride)
    
            # Debug stats before inference
            with torch.no_grad():
                data._debug_stats["theta_mean"] = float(theta_packed.mean().detach().cpu().item())
                data._debug_stats["theta_std"] = float(theta_packed.std().detach().cpu().item())
                data._debug_stats["theta_abs_mean"] = float(theta_packed.abs().mean().detach().cpu().item())
                data._debug_stats["theta_abs_max"] = float(theta_packed.abs().max().detach().cpu().item())
    
            x = theta_packed
    
        if inference in (None, "none", "theta", "no_inference"):
            return x
    
        if self.normalize_theta:
            x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    
        if inference == "loopy_belief_propagation":
            if torch.isnan(x).any():
                x = torch.nan_to_num(x, nan=0.0)
    
            lbp_res = loopy_belief_propagation(
                x,
                torch.zeros_like(x),
                data.lbp_edges,
                max_iter=self.bp_max_iter,
                tol=self.bp_tol,
                damping=self.bp_damping,
            )
    
            # Debug stats after inference
            with torch.no_grad():
                mu_v = lbp_res.mu_v
                probs = torch.softmax(mu_v, dim=-1)[..., 1]
                ent = -(probs.clamp(1e-8, 1 - 1e-8) * torch.log(probs.clamp(1e-8, 1 - 1e-8))
                        + (1 - probs).clamp(1e-8, 1 - 1e-8) * torch.log((1 - probs).clamp(1e-8, 1 - 1e-8)))
                data._debug_stats["marginal_mean"] = float(probs.mean().detach().cpu().item())
                data._debug_stats["marginal_std"] = float(probs.std().detach().cpu().item())
                data._debug_stats["entropy_mean"] = float(ent.mean().detach().cpu().item())
                data._debug_stats["frac_mid"] = float(((probs >= 0.45) & (probs <= 0.55)).float().mean().detach().cpu().item())
    
            return lbp_res, x
    
        if inference == "naive_mean_field":
            return (
                naive_mean_field(
                    x,
                    data.nmf_edges,
                    max_iter=self.bp_max_iter,
                    tol=self.bp_tol,
                    damping=self.bp_damping,
                ),
                x,
            )
    
        return x
