import os
import time
import sys
import json
sys.path.append("../..")
import numpy as np
import torch
from torch.utils.data import DataLoader
from progress.bar import Bar
from typing import Dict, Optional, Tuple, Union, Mapping
import argparse

from bpropy.co.mis import MIS
from bpropy.co.scp import SCP
from bpropy.models.mrf_models import Hetero_GCN_MRF_v4
from bpropy.datasets.mrf_dataset_2 import MRFDatasetConfig, MRFDataset
from bpropy.datasets.prefetcher import CUDAPrefetcher
from bpropy.datasets.collator import mrf_collate
from bpropy.adapters.vc_to_data import problem_to_data
from bpropy.mrf.lbp import loopy_belief_propagation
from bpropy.models.mrf_models import Hetero_GCN_MRF_v4
from bpropy.datasets.collator import mrf_collate
from bpropy.mrf.mis_factor_graph import MISFactorGraph
from bpropy.mrf.scp_factor_graph import SCPFactorGraph
from bpropy.graphs.random_graphs import random_mis_bounded_degree  # the bounded-degree generator

ArrayLike = Union[np.ndarray, torch.Tensor]

MODEL_REGISTRY = {
    "Hetero_GCN_MRF_v4": Hetero_GCN_MRF_v4,
}

def time_func(func, *args, **kwargs):
    t0 = time.time()
    res = func(*args, **kwargs)
    t0 = time.time() - t0
    return t0, res
def load_checkpoint(path, model, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
def load_model_description(path):
    with open(path,"r") as f:
        return json.load(f)
def load_model_from_log_dir(path, map_location="cpu"):
    # load the description
    model_desc = load_model_description(os.path.join(path,"hparams.json"))
    # initialize model
    model = MODEL_REGISTRY[model_desc["model_class"]](**model_desc["model_args"]).to(map_location)
    load_checkpoint(os.path.join(path, "checkpoints", "best.pt"), model, map_location=map_location)
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

def generic_score_greedy(pb, scores):
    # initialize empty solution
    x = np.zeros(pb.nv(), dtype=int)
    candidates = np.arange(pb.nv())
    # keep iterating until solution is complete
    c = 0
    while candidates.size != 0:
        # select next candidate
        idx = np.argmax(scores[candidates])
        u = candidates[idx]

        # update set of candidates and evidence based on the selected candidate
        candidates, new_evidence = update_greedy_candidates(pb, u, candidates, x)
        # update solution
        for k,v in new_evidence.items():
            x[k] = v

        # increment iteration counter
        c += 1
    # return solution
    return x

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


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", default="output",
                   help="<Required> Output file path")
    p.add_argument("log_dirs", nargs="+", help="<Required> List of model log directories")
    p.add_argument("--solver", default="kamis")
    p.add_argument("--limit", default=100000, type=int)
    args = p.parse_args()

    graph_sizes = [50, 100, 200, 400, 800]
    N = 10

    print("Solving instance")
    print("\tOutput file        : ", args.output)
    print("\tSolver             : ", args.solver)
    print("\tLimit              : ", args.limit)
    print("\tModels")
    print("\n".join(["\t\t"+p for p in args.log_dirs]))


    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data preprocessing function
    preprocess_f = lambda pb: mrf_collate([problem_to_data(
            pb, hetero=True, feature_mode="EB",
            with_lbp=True, with_nmf=False
        )]).to(device)
    
    # load models
    models = [load_model_from_log_dir(p, map_location=device) for p in args.log_dirs]
    for m in models:
        m.eval()

    with torch.no_grad():
        with open(args.output, "w") as f:
            headers = ["graph_size","model_id","t_data","t_solve_grd","t_total_grd","gap_grd","t_solve_gd","t_total_gd","gap_gd"]
            print(",".join(headers), file=f)
            for gs in graph_sizes:
                gs_gaps, gs_time = [], []
                for _ in range(N):
                    # generate random problem
                    nv_out, edges, w = random_mis_bounded_degree(gs, 15, p=0.15)
                    pb = MIS.from_edges(nv_out, edges, weights=w)

                    # solve the problem
                    sol = pb.solve(solver="kamis", disp=False)
                    y = sol.x

                    # create data object with LBPEdges
                    t_data, data = time_func(preprocess_f, pb)

                    # predict and infer solution
                    for m_id,m in enumerate(models):
                        lbp_res,theta = m(data)
                        t_solve_grd, x_grd = time_func(lambda pb, lbp_r: generic_score_greedy(pb, lbp_r.mu.reshape(-1,2)[:,1].cpu().numpy()), pb, lbp_res)
                        t_total_grd = t_data + t_solve_grd

                        if pb.__class__ == MIS:
                            gap_grd = 100 * (y.sum() - x_grd.sum()) / y.sum()
                        if pb.__class__ == SCP:
                            gap_grd = 100 * (x_grd.sum() - y.sum()) / y.sum()

                        # Guided Decimation
                        t_solve_gd, x_gd = time_func(generic_greedy, pb, theta, data)
                        t_total_gd = t_data + t_solve_gd
                        
                        if pb.__class__ == MIS:
                            gap_gd = 100 * (y.sum() - x_gd.sum()) / y.sum()
                        if pb.__class__ == SCP:
                            gap_gd = 100 * (x_gd.sum() - y.sum()) / y.sum()

                        print(",".join(list(map(str,[gs,m_id,t_data,t_solve_grd,t_total_grd,gap_grd,t_solve_gd,t_total_gd,gap_gd]))), file=f)
                        print(",".join(list(map(str,[gs,m_id,t_data,t_solve_grd,t_total_grd,gap_grd,t_solve_gd,t_total_gd,gap_gd]))))

