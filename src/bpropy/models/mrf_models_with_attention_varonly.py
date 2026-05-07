from torch_geometric.transforms import AddSelfLoops

# Assumes these are available in your project (same as other models)
from bpropy.mrf.lbp import loopy_belief_propagation
from bpropy.mrf.naive_mean_field import naive_mean_field

import torch
import torch.nn as nn
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

class GCN_MRF_Attention_VarOnly(nn.Module):
    """
    Variable-only MRF model with attention-style GNN backbone.

    Differences vs previous attention models:
      - Completely ignores constraint embeddings.
      - Factor embedding = concatenation of variable embeddings in its scope.
      - Supports arbitrary factor arity (k > 2).
      - One MLP head per factor arity.
      - Theta packed in canonical factor index order.

    Assumptions:
      - Binary variables.
      - m_stride[f] == 2 ** arity(f)  (full factor tables).
      - Factor indices follow canonical ordering per graph.
    """

    def __init__(
        self,
        in_dim,
        n_layers=4,
        hidden_dim=64,
        dropout_p=0.25,
        bp_max_iter=10,
        bp_tol=1e-6,
        bp_damping=0.5,
        factor_sizes=(1, 2),
        jk="lstm",
        norm="LayerNorm",
        act="elu",
        inference="loopy_belief_propagation",
        normalize_theta=False,
    ):
        super().__init__()

        self.inference = inference
        self.bp_max_iter = bp_max_iter
        self.bp_tol = bp_tol
        self.bp_damping = bp_damping
        self.normalize_theta = normalize_theta

        # Backbone GNN (attention-based)
        self.conv = GATConv(
            in_dim,
            hidden_dim,
            heads=1,
            dropout=dropout_p,
        )

        # One head per arity
        self.theta_heads = nn.ModuleDict()
        for k in factor_sizes:
            self.theta_heads[str(int(k))] = MLP(
                int(k) * hidden_dim,
                hidden_channels=[64, 64, int(2 ** int(k))],
                activation_layer=GELU,
                norm_layer=torch.nn.LayerNorm,
            )

    @staticmethod
    def _pack_theta_blocks(theta_blocks: torch.Tensor, m_stride: torch.Tensor):
        """
        Pack padded per-factor blocks into flat theta in factor index order.
        """
        m_stride = m_stride.long()
        max_stride = theta_blocks.size(1)
        mask = (
            torch.arange(max_stride, device=theta_blocks.device)[None, :]
            < m_stride[:, None]
        )
        return theta_blocks[mask]

    def forward(self, data, inference=None):
        inference = self.inference if inference is None else inference

        x, edge_index = data.x, data.edge_index
        data = AddSelfLoops()(data)

        # GNN
        x = self.conv(x, edge_index)

        if hasattr(data, "fg_map"):
            m_stride = data.lbp_edges.m_stride.long()
            n_factors = m_stride.numel()
            max_stride = int(m_stride.max().item())

            node_idx = data.fg_map[:, 0]
            factor_idx = data.fg_map[:, 1].long()

            # Only variable endpoints
            src_is_var = (data.bipartite[node_idx] == 0)
            var_nodes = node_idx[src_is_var].long()
            fac_of_e = factor_idx[src_is_var].long()

            # Enforce lex ordering within each factor
            if var_nodes.numel() > 0:
                max_vid = var_nodes.max().item() + 1
                key = fac_of_e * max_vid + var_nodes
                perm = torch.argsort(key, stable=True)
                var_nodes = var_nodes[perm]
                fac_of_e = fac_of_e[perm]

            var_emb = x[var_nodes]

            # Arity from incidence counts
            counts = torch.bincount(fac_of_e, minlength=n_factors).long()

            # Consistency check
            expected_stride = (1 << counts)
            if not torch.all(expected_stride == m_stride):
                raise RuntimeError(
                    "Stride/arity mismatch: expected 2**arity per factor."
                )

            grouped = concat_group_by(var_emb, fac_of_e)
            k_max = grouped.size(1)

            theta_blocks = x.new_zeros((n_factors, max_stride))

            for k_str, head in self.theta_heads.items():
                k = int(k_str)
                idx = (counts == k).nonzero(as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue

                if k > k_max:
                    raise RuntimeError(
                        f"Observed arity {k} but grouped only to {k_max}."
                    )

                xk = grouped[idx, :k, :].reshape(idx.numel(), k * x.size(1))
                tk = head(xk)
                theta_blocks[idx, :tk.size(1)] = tk

            theta = self._pack_theta_blocks(theta_blocks, m_stride)
        else:
            raise RuntimeError("fg_map missing from data object.")

        if inference in (None, "none", "theta", "no_inference"):
            return theta

        if self.normalize_theta:
            theta = (theta - theta.min()) / (theta.max() - theta.min())

        if inference == "loopy_belief_propagation":
            lbp_res = loopy_belief_propagation(
                theta,
                torch.zeros_like(theta),
                data.lbp_edges,
                max_iter=self.bp_max_iter,
                tol=self.bp_tol,
                damping=self.bp_damping,
            )
            return lbp_res, theta

        if inference == "naive_mean_field":
            nmf_res = naive_mean_field(
                theta,
                data.nmf_edges,
                max_iter=self.bp_max_iter,
                tol=self.bp_tol,
                damping=self.bp_damping,
            )
            return nmf_res, theta

        return theta
