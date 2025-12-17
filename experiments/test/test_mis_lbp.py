import os
import time
import sys
import argparse
import json
sys.path.append("../..")
import numpy as np
import torch
from torch.utils.data import DataLoader
from progress.bar import Bar
from typing import Dict, Optional, Tuple, Union, Mapping

from bpropy.models.logit_models import *
from bpropy.models.mrf_models import Hetero_GCN_MRF_v4
from bpropy.datasets.mrf_dataset_2 import MRFDatasetConfig, MRFDataset
from bpropy.datasets.prefetcher import CUDAPrefetcher
from bpropy.datasets.collator import mrf_collate
from bpropy.adapters.vc_to_data import problem_to_data
from bpropy.mrf.lbp import loopy_belief_propagation
from bpropy.mrf.mis_factor_graph import MISFactorGraph

ArrayLike = Union[np.ndarray, torch.Tensor]

MODEL_REGISTRY = {
    "GCNVarLogits": GCNVarLogits,
    "HeteroGCNVarLogits": HeteroGCNVarLogits,
    "Hetero_GCN_MRF_v4": Hetero_GCN_MRF_v4,
}

def load_checkpoint(path, model, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])

def load_model_from_logdir(path, device="cpu"):
    checkpoint_path = os.path.join(path, "checkpoints", "best.pt")
    description_path = os.path.join(path, "hparams.json")
    # load model description
    with open(description_path,"r") as f:
        hparams = json.load(f)

    model = MODEL_REGISTRY[hparams["model_class"]](**hparams["model_args"])

    load_checkpoint(checkpoint_path, model, map_location=device)
    return model


def make_ds(root, split, cfg, hetero, feature_mode, with_lbp, with_nmf, processed_dir):
    return MRFDataset(
        root, split, cfg=cfg,
        preprocess=lambda pb: problem_to_data(
            pb, hetero=hetero, feature_mode=feature_mode,
            with_lbp=with_lbp, with_nmf=with_nmf
        ),
        processed_dir=processed_dir
    )


def build_var_to_ss_indices(fg, edges_lbp):
    """
    Returns:
      idx0: List[torch.LongTensor], len = n_vars; indices where that var's bit == 0
      idx1: List[torch.LongTensor], len = n_vars; indices where that var's bit == 1
      SS_total: total length of SS vector
    Assumes your SS order per factor is little-endian (k-th var -> bit k).
    """
    import numpy as np
    SS_total = int(edges_lbp.m_stride.sum().item())
    nF = edges_lbp.m_stride.numel()
    m_stride = edges_lbp.m_stride.cpu().numpy()   # lengths = 2**arity
    starts = np.concatenate([[0], np.cumsum(m_stride[:-1])])

    n_vars = edges_lbp.n_vars.sum().to(torch.long).item()
    idx0 = [[] for _ in range(n_vars)]
    idx1 = [[] for _ in range(n_vars)]

    # Cache bit tables per arity to avoid recomputation
    bit_cache = {}  # arity -> np.ndarray [2**arity, arity] of 0/1
    def bits_for_arity(a):
        if a not in bit_cache:
            idx = np.arange(1 << a, dtype=np.uint32)[:, None]
            ks  = np.arange(a, dtype=np.uint32)[None, :]
            bit_cache[a] = ((idx >> ks) & 1).astype(np.uint8)  # [2^a, a]
        return bit_cache[a]

    # Loop once over factors (preprocessing only; runtime masking will be O(|evidence|))
    factors = list(fg.iter_factors())
    for F_id, (s, w) in enumerate(zip(starts, m_stride)):
        F = factors[F_id]
        vars_F = fg.factor_variables(F)  # list[int], stable var order
        a = len(vars_F)
        if a == 0:
            continue
        bits = bits_for_arity(a)           # [2^a, a]
        # global SS positions for this factor
        pos = np.arange(s, s + w, dtype=np.int64)  # [2^a]

        # Append positions to per-variable lists depending on bit value
        for j, v in enumerate(vars_F):
            b = bits[:, j]                 # [2^a], 0/1 at this var's bit
            idx0[v].extend(pos[b == 0])
            idx1[v].extend(pos[b == 1])

    # Turn into tensors once, on CPU (move to GPU when you build masks)
    idx0 = [torch.as_tensor(x, dtype=torch.long) if len(x) else torch.empty(0, dtype=torch.long) for x in idx0]
    idx1 = [torch.as_tensor(x, dtype=torch.long) if len(x) else torch.empty(0, dtype=torch.long) for x in idx1]
    return idx0, idx1, SS_total

def make_ss_evidence_mask_fast(evidence, idx0, idx1, SS_total, device, dtype=torch.bool):
    mask = torch.ones(SS_total, dtype=dtype, device=device)
    # Zero-out inconsistent positions
    for v, val in evidence.items():
        if val is None: 
            continue
        bad = idx0[v] if val == 1 else idx1[v]   # positions incompatible with v=val
        if bad.numel():
            mask[bad.to(device)] = False
    return mask

def make_ss_evidence_mask_dict(
    fg,
    evidence: Mapping[int, Optional[int]],
    *,
    as_torch: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bool,
    strict: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Build a mask over the over-complete sufficient-statistics vector that is True
    iff a factor configuration is consistent with the given (possibly partial) evidence.

    Args:
      fg: FactorGraph
      evidence: dict {var_id -> 0/1 or None}; missing keys are 'unknown'
      as_torch: return a torch.Tensor if True, else a np.ndarray
      device, dtype: only used when as_torch=True
      strict: if True, raise on bad var ids or values not in {0,1,None}

    Returns:
      mask of shape (sum_F 2**|F|,), in the same factor order and bit-endian
      convention as fg.factor_sufficient_statistics / your SS layout.
    """
    # Normalize evidence: keep only observed {0,1}
    obs = {}
    for k, v in evidence.items():
        if v is None:
            continue
        if v in (0, 1):
            obs[int(k)] = int(v)
        elif strict:
            raise ValueError(f"Evidence for var {k} must be 0/1/None, got {v!r}")

    blocks = []
    for F in fg.iter_factors():
        vars_F = fg.factor_variables(F)                   # sorted list of variable ids
        nF = len(vars_F)
        if nF == 0:
            # Degenerate factor: one configuration, always allowed
            blocks.append(np.array([True], dtype=bool))
            continue

        # Precompute all assignments for this factor as little-endian bits
        idx = np.arange(1 << nF, dtype=np.uint32)[:, None]     # [2^nF, 1]
        ks  = np.arange(nF, dtype=np.uint32)[None, :]          # [1, nF]
        bits = ((idx >> ks) & 1).astype(np.int64)              # [2^nF, nF], 0/1

        # Start with all allowed; constrain with observed vars in this factor
        allowed = np.ones((1 << nF), dtype=bool)
        for j, vj in enumerate(vars_F):
            if vj in obs:
                allowed &= (bits[:, j] == obs[vj])

        blocks.append(allowed)

    mask_np = np.concatenate(blocks, axis=0)  # length == fg.number_sufficient_statistics()
    if as_torch:
        out = torch.from_numpy(mask_np)
        if device is not None:
            out = out.to(device=device, dtype=dtype)
        else:
            out = out.to(dtype=dtype)
        return out
    return mask_np

def greedy_fast(pb, theta, data):
    """Greedy solver for MIS problems
    """
    fg = MISFactorGraph(pb)
    idx0, idx1, SS_total = build_var_to_ss_indices(fg, data.lbp_edges)

    v2f = torch.zeros(
        int(data.lbp_edges.n_f2v_msg.sum().item()), device=device, dtype=theta.dtype
    )

    queue = list(range(pb.nv()))
    sol = np.zeros(pb.nv())
    c = 0
    t_evid_mask = 0.0

    NEG_INF = torch.finfo(theta.dtype).min / 2  # or simply a large negative like -1e9 for float32
    neg_inf_vals = torch.full_like(theta, NEG_INF)
    mask = torch.full_like(theta, 1).to(torch.bool)
    while len(queue)>0:
        new_evid = {}
        # compute scores with evidence
        #th = theta.clone()
        #theta = torch.where(mask, theta, torch.full_like(theta, NEG_INF))
        th = torch.where(mask, theta, neg_inf_vals)
        lbp_res = loopy_belief_propagation(th, v2f, data.lbp_edges,
                                       max_iter=10, tol=1e-6, damping=0.5)

        mu_evid = lbp_res.mu_v.reshape(-1,2)[:,1].numpy()
        mu_evid[np.isnan(mu_evid)] = 0

        u = queue[np.argmax(mu_evid[np.array(queue)])]
        sol[u] = 1
        new_evid[u] = 1
        queue.remove(u)
        for v in pb.g.neighbors(u):
            if v in queue:
                sol[v] = 0
                new_evid[v] = 0
                queue.remove(v)
        c += 1

        t_ = time.time()
        tmp_mask = make_ss_evidence_mask_fast(new_evid, idx0, idx1, SS_total, device)
        mask &= tmp_mask
        t_ = time.time() - t_
        t_evid_mask += t_

    return sol

def greedy(pb, theta, data):
    """Greedy solver for MIS problems
    """
    fg = MISFactorGraph(pb)
    covered = np.zeros(pb.ne(), dtype=bool)
    x = np.zeros(pb.nv(), dtype=bool)
    v2f = torch.zeros(
        int(data.lbp_edges.n_f2v_msg.sum().item()), device=device, dtype=theta.dtype
    )

    queue = list(range(pb.nv()))
    cur_evid = {}
    sol = np.zeros(pb.nv())
    c = 0
    t_evid_mask = 0.0
    while len(queue)>0:
        # compute scores with evidence
        th = theta.clone()
        
        t_ = time.time()
        mask = make_ss_evidence_mask_dict(fg, cur_evid)
        t_ = time.time() - t_
        t_evid_mask += t_
        NEG_INF = torch.finfo(th.dtype).min / 2  # or simply a large negative like -1e9 for float32
        #theta = torch.where(mask, theta, torch.full_like(theta, NEG_INF))
        th = torch.where(mask, th, torch.full_like(th, NEG_INF))

        lbp_res = loopy_belief_propagation(th, v2f, data.lbp_edges,
                                       max_iter=10, tol=1e-6, damping=0.5)

        mu_evid = lbp_res.mu_v.reshape(-1,2)[:,1].numpy()
        #print(mu_evid)
        mu_evid[np.isnan(mu_evid)] = 0

        u = queue[np.argmax(mu_evid[np.array(queue)])]
        sol[u] = 1
        cur_evid[u] = 1
        queue.remove(u)
        for v in pb.g.neighbors(u):
            if v in queue:
                sol[v] = 0
                cur_evid[v] = 0
                queue.remove(v)
        c += 1
    print("Evidence mask creation time: ", t_evid_mask)
    return sol


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("logdir")
    p.add_argument("dataset")
    p.add_argument("processed_dir", default="processed",
                   help="subdirectory under <root>/processed/ used for cached features")
    p.add_argument("--limit", default=100)

    args = p.parse_args()

    device = "cpu"

    model = load_model_from_logdir(args.logdir)

    eval_cfg = MRFDatasetConfig(
        feature_mode="EB",
        yield_mode="both",
        refresh=False,
        return_assignment=True,
        assignment_solver="kamis",
    )
    ds_train_eval = make_ds(
        args.dataset, "test", eval_cfg,
        True, "EB", True, False,
        args.processed_dir
    )
    loader_eval = DataLoader(
        ds_train_eval, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2,
        collate_fn=mrf_collate,
    )
    
    model.eval()
    
    gaps_0, gaps_1, gaps_2 = [], [], []
    time_0, time_1, time_2 = [], [], []
    model.eval()
    L,c = 502,1
    with torch.no_grad():
        bar = Bar(f"[Test]          ", max=len(loader_eval))
        for pb, data, y in CUDAPrefetcher(loader_eval, device):
            t0 = time.time()
            lbp_res, theta= model(data)
            t_pred = time.time() - t0
    
            #x = greedy(pb[0], theta, data)
            #t_v0 = time.time() - t0
            #gap_0 = 100 * (y.sum().item() - sum(x)) / y.sum().item()
            #gaps_0.append(gap_0)
            #time_0.append(t_v0)

            #print(f"{c} GRD 0: gap = {np.mean(gaps_0):.3f} ({np.std(gaps_0):.3f}), time = {np.mean(time_0):.3f} ({np.std(time_0):.3f})")
    
            #t_v1 = time.time()
            #mu_evid = lbp_res.mu_v.reshape(-1,2)[:,1].numpy()
            #mu_evid[np.isnan(mu_evid)] = 0
            #sol_grd = pb[0].greedy(scores=mu_evid)
            #t_v1 = time.time() - t_v1 + t_pred
            #gap_1 = 100 * ( y.sum().item() - sol_grd.sum().item()) / y.sum().item()
            #gaps_1.append(gap_1)
            #time_1.append(t_v1)
            #print(f"{c} GRD 1: gap = {np.mean(gaps_1):.3f} ({np.std(gaps_1):.3f}), time = {np.mean(time_1):.3f} ({np.std(time_1):.3f})")
 
            t_v2 = time.time()
            x = greedy_fast(pb[0], theta, data)
            t_v2 = time.time() - t_v2 + t_pred
            gap_2 = 100 * (y.sum().item() - sum(x)) / y.sum().item()
            gaps_2.append(gap_2)
            time_2.append(t_v2)
            print(f"{c} GRD 2: gap = {np.mean(gaps_2):.3f} ({np.std(gaps_2):.3f}), time = {np.mean(time_2):.3f} ({np.std(time_2):.3f})")
  
            #bar.next()
            c+= 1
            if c >= L:
                break
        bar.finish()
    
    #print(f"V0 gaps: {np.mean(gaps_0):.3f}% ({np.std(gaps_0):.3f}%), time: {np.mean(time_0):.3f}s ({np.std(time_0):.3f}s)")
    #print(f"V1 gaps: {np.mean(gaps_1):.3f}% ({np.std(gaps_1):.3f}%), time: {np.mean(time_1):.3f}s ({np.std(time_1):.3f}s)")
    print(f"V2 gaps: {np.mean(gaps_2):.3f}% ({np.std(gaps_2):.3f}%), time: {np.mean(time_2):.3f}s ({np.std(time_2):.3f}s)")


