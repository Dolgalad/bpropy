import os
import time
import sys
import argparse
import json
import itertools
sys.path.append("../..")
import numpy as np
import torch
from torch.utils.data import DataLoader
from progress.bar import Bar
from typing import Dict, Optional, Tuple, Union, Mapping

from bpropy.co.mis import MIS
from bpropy.co.scp import SCP
from bpropy.datasets.mrf_dataset_2 import MRFDatasetConfig, MRFDataset
from bpropy.datasets.prefetcher import CUDAPrefetcher
from bpropy.datasets.collator import mrf_collate
from bpropy.adapters.vc_to_data import problem_to_data
from bpropy.mrf.lbp import loopy_belief_propagation
from bpropy.mrf.mis_factor_graph import MISFactorGraph
from bpropy.mrf.scp_factor_graph import SCPFactorGraph

ArrayLike = Union[np.ndarray, torch.Tensor]

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

def update_greedy_candidates(pb, u, candidates, x):
    evidence = {u: 1}
    if pb.__class__ == MIS:
        # all neighboring nodes that are still in the candidate set are removed
        neighbors = list(pb.g.neighbors(u))
        for v in neighbors:
            if v in candidates:
                evidence[v] = 0
        neighbors += [u]
        new_candidates = np.delete(candidates, np.array([v in neighbors for v in candidates], dtype=bool))
        return new_candidates, evidence
    if pb.__class__ == SCP:
        # all sets that do not cover any new elements can be removed
        # get mask of all currently covered items
        covered = np.zeros(pb.ne(), dtype=bool)
        for idx in np.argwhere(x).flatten():
            covered |= pb.adj_vec(pb.F[idx])
        covered |= pb.adj_vec(pb.F[u])
        candidate_gains = np.array([(np.logical_not(covered) & pb.adj_vec(pb.F[i])).sum() for i in candidates])
        for v in candidates[candidate_gains==0]:
            if v!=u:
                evidence[v] = 0
        new_candidates = np.delete(candidates, candidate_gains==0)
        return new_candidates, evidence






def generic_greedy(pb, theta, data, update_interval=1, bp_max_iter=10, bp_tol=1e-6, bp_damping=0.5):
    # initialize empty solution
    x = np.zeros(pb.nv(), dtype=int)
    candidates = np.arange(pb.nv())
    # keep track of the LBP result
    if pb.__class__ == MIS:
        fg = MISFactorGraph(pb)
    if pb.__class__ == SCP:
        fg = SCPFactorGraph(pb)

    device = theta.device
    idx0, idx1, SS_total = build_var_to_ss_indices(fg, data.lbp_edges)
    v2f = torch.zeros(
        int(data.lbp_edges.n_f2v_msg.sum().item()), device=device, dtype=theta.dtype
    )
    lbp_res = None
    # keep track of new evidence
    NEG_INF = torch.finfo(theta.dtype).min / 2
    neg_inf_vals = torch.full_like(theta, NEG_INF)
    mask = torch.full_like(theta, 1).to(torch.bool)
    new_evidence = {}
    # keep iterating until solution is complete
    c = 0
    while candidates.size != 0:
        # update the scores based on current evidence
        if c % update_interval == 0:
            th = torch.where(mask, theta, neg_inf_vals)
            lbp_res = loopy_belief_propagation(th, v2f, data.lbp_edges,
                                           max_iter=bp_max_iter, tol=bp_tol, damping=bp_damping)
            mu_evid = lbp_res.mu_v.reshape(-1,2)[:,1].cpu().numpy()
            mu_evid[np.isnan(mu_evid)] = 0

        # select next candidate
        idx = np.argmax(mu_evid[candidates])
        u = candidates[idx]

        # update set of candidates and evidence based on the selected candidate
        candidates, new_evidence = update_greedy_candidates(pb, u, candidates, x)
        # update solution
        for k,v in new_evidence.items():
            x[k] = v

        # update the evidence mask
        tmp_mask = make_ss_evidence_mask_fast(new_evidence, idx0, idx1, SS_total, device)
        mask &= tmp_mask

        # increment iteration counter
        c += 1
    # return solution
    return x

def greedy_fast(pb, theta, data, update_int=1):
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

    # debug
    h_min = []

    # keep lbp result
    lbp_res = None
    while len(queue)>0:
        new_evid = {}
        # compute scores with evidence
        #th = theta.clone()
        #theta = torch.where(mask, theta, torch.full_like(theta, NEG_INF))
        if c % update_int == 0:
            th = torch.where(mask, theta, neg_inf_vals)
            lbp_res = loopy_belief_propagation(th, v2f, data.lbp_edges,
                                           max_iter=10, tol=1e-6, damping=0.5)

        mu_evid = lbp_res.mu_v.reshape(-1,2)[:,1].cpu().numpy()
        mu_evid[np.isnan(mu_evid)] = 0
        #print("MU       ", mu_evid[np.array(queue)])

        #mu = lbp_res.mu_v.reshape(-1,2).numpy()
        #H = -mu[:,0] * np.log(mu[:,0]) - mu[:,1] * np.log(mu[:,1])
        #H = H[np.array(queue)]
        #print("H        ", H)
        #print("H argmin ", np.argmin(H), np.min(H))
        #print("         ", mu[np.array(queue),:][np.argmin(H)])
        #h_min.append(queue[np.argmin(H)])
        idx = np.argmax(mu_evid[np.array(queue)])
        #print("idx      ", idx, np.max(mu_evid[np.array(queue)]))
        #print(np.min(H), np.max(H))
        #print(H[idx], mu_evid[idx])
        u = queue[idx]
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

    #print("Min entropy variable values : ", sol[np.array(h_min)])
    return sol

def mis_idealized_canonical_parameters(fg, M=10.):
    theta = torch.zeros(fg.number_sufficient_statistics(), device=device, requires_grad=False)
    c = 0
    for F in fg.iter_factors():
        nF = fg.factor_size(F)
        if nF == 1:
            theta[c+1] = 1.
            theta[c] = 0.
            c += 2
        else:
            for v in range(2**nF):
                vals = np.unpackbits(np.array([v]).astype(np.uint8), count=nF, bitorder="little")
                if vals.sum()>1:
                    theta[c] = M
                else:
                    theta[c] = 0.
                c += 1
    return theta

def scp_idealized_canonical_parameters(fg, M=10.):
    theta = torch.zeros(fg.number_sufficient_statistics(), device=device, requires_grad=False)
    c = 0
    for F in fg.iter_factors():
        nF = fg.factor_size(F)
        if nF == 1:
            theta[c+1] = 0.
            theta[c] = 1.
            c += 2
        else:
            for v in range(2**nF):
                vals = np.unpackbits(np.array([v]).astype(np.uint8), count=nF, bitorder="little")
                if vals.sum()==0:
                    theta[c] = M
                else:
                    theta[c] = 0.
                c += 1
    return theta


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("processed_dir", default="processed",
                   help="subdirectory under <root>/processed/ used for cached features")
    p.add_argument("output", default="output",
                   help="output file path")

    p.add_argument("--solver", default="kamis")
    p.add_argument("--limit", default=100000, type=int)

    args = p.parse_args()

    print("Solving instance")
    print("\tDataset            : ", args.dataset)
    print("\tProcessed directory: ", args.processed_dir)
    print("\tOutput file        : ", args.output)
    print("\tSolver             : ", args.solver)
    print("\tLimit              : ", args.limit)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_cfg = MRFDatasetConfig(
        feature_mode="EB",
        yield_mode="both",
        refresh=False,
        return_assignment=True,
        assignment_solver=args.solver,
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
    
   
    update_intervals = [1,2,4,8,16,32]
    headers = ["id"] + list(itertools.chain(*[[f"gap(%, k={k})",f"t(s,k={k})"] for k in update_intervals]))
    gaps_0, gaps_1, gaps_2 = [], [], []
    time_0, time_1, time_2 = [], [], []
    L,c = args.limit,0
    with torch.no_grad():
        bar = Bar(f"[Test]          ", max=len(loader_eval))
        with open(args.output, "w") as f:
            # write headers
            print(",".join(headers), file=f)
            for pb, data, y in CUDAPrefetcher(loader_eval, device):
                t0 = time.time()
                NEG_INF = torch.finfo(torch.float32).min / 2
                if pb[0].__class__.__name__=="MIS":
                    fg = MISFactorGraph(pb[0])
                    theta = mis_idealized_canonical_parameters(fg, M=NEG_INF)
                if pb[0].__class__.__name__=="SCP":
                    fg = SCPFactorGraph(pb[0])
                    theta = scp_idealized_canonical_parameters(fg, M=NEG_INF)
                # create the sufficient statistics
                t_pred = time.time() - t0
                if pb[0].__class__ == MIS:
                    up_gaps = []
                    up_time = []
                    for up_int in update_intervals:
                        t_v3 = time.time()
                        x_ = generic_greedy(pb[0], theta, data, update_interval=up_int)
                        t_v3 = time.time() - t_v3 + t_pred
                        gap_3 = 100 * (y.sum().item() - sum(x_)) / y.sum().item()
                        up_gaps.append(gap_3)
                        up_time.append(t_v3)

                    gaps_2.append(up_gaps)
                    time_2.append(up_time)
                if pb[0].__class__ == SCP:
                    up_gaps = []
                    up_time = []
                    for up_int in update_intervals:
                        t_v3 = time.time()
                        x_ = generic_greedy(pb[0], theta, data, update_interval=up_int)
                        t_v3 = time.time() - t_v3 + t_pred
                        gap_3 = 100 * (sum(x_) - y.sum().item()) / y.sum().item()
                        up_gaps.append(gap_3)
                        up_time.append(t_v3)

                    gaps_2.append(up_gaps)
                    time_2.append(up_time)

                values = [c] + list(itertools.chain(*[[g,t] for g,t in zip(up_gaps,up_time)]))
                values = [str(it) for it in values]
                print(",".join(values), file=f)

                #print(f"{c} GRD 2: gap = {np.mean(gaps_2,axis=0).round(decimals=3)} ({np.std(gaps_2,axis=0).round(decimals=3)}), time = {np.mean(time_2, axis=0).round(decimals=3)} ({np.std(time_2, axis=0).round(decimals=3)})")

                c+= 1
                if c >= L:
                    break
        bar.finish()
    
    print(f"V2 gaps: {np.mean(gaps_2):.3f}% ({np.std(gaps_2):.3f}%), time: {np.mean(time_2):.3f}s ({np.std(time_2):.3f}s)")


