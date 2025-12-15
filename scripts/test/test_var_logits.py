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

# Import the new models:
from bpropy.models.logit_models import GCNVarLogits, HeteroGCNVarLogits
from bpropy.models.mrf_models import Hetero_GCN_MRF_v4

MODEL_REGISTRY = {
    "GCNVarLogits": GCNVarLogits,
    "HeteroGCNVarLogits": HeteroGCNVarLogits,
    "Hetero_GCN_MRF_v4": Hetero_GCN_MRF_v4,
}

ArrayLike = Union[np.ndarray, torch.Tensor]

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
        processed_dir=processed_dir
    )

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

def generic_greedy(pb, scores):
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

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", help="<Required> Dataset root path")
    p.add_argument("output", default="output",
                   help="<Required> Output file path")
    p.add_argument("log_dirs", nargs="+", help="<Required> List of model log directories")
    p.add_argument("--solver", default="kamis")
    p.add_argument("--limit", default=100000, type=int)
    p.add_argument("--processed_dir", default="processed",
                   help="<Optional> subdirectory under <root>/processed/ used for cached features")

    args = p.parse_args()

    print("Solving instance")
    print("\tDataset            : ", args.dataset)
    print("\tProcessed directory: ", args.processed_dir)
    print("\tOutput file        : ", args.output)
    print("\tSolver             : ", args.solver)
    print("\tLimit              : ", args.limit)
    print("\tModels")
    print("\n".join(["\t\t"+p for p in args.log_dirs]))


    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_cfg = MRFDatasetConfig(
        feature_mode="EB",
        yield_mode="problem",
        refresh=False,
        return_assignment=True,
        assignment_solver=args.solver,
    )
    ds_train_eval = make_ds(
        args.dataset, "test", eval_cfg,
        False, "EB", False, False,
        args.processed_dir
    )
    loader_eval = DataLoader(
        ds_train_eval, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2,
        collate_fn=mrf_collate,
    )

    # data preprocessing function
    preprocess_f = lambda pb: problem_to_data(
            pb, hetero=True, feature_mode="EB",
            with_lbp=False, with_nmf=False
        ).to(device)
    
    # load models
    models = [load_model_from_log_dir(p, map_location=device) for p in args.log_dirs]
    for m in models:
        m.eval()



    headers = ["id","solver_val","t_data(s)","model_id","t_pred(s)","t_solve(s)","t_total(s)","gap(%)"]
    L,c = args.limit,0
    with torch.no_grad():
        bar = Bar(f"[Test]          ", max=len(loader_eval))
        with open(args.output, "w") as f:
            # write headers
            print(",".join(headers), file=f)
            for pb, y in loader_eval:
                pb = pb[0]
                y = y[0]
                # create the data object
                t_data, data = time_func(preprocess_f, pb)

                # for each model predict logits and solve
                for m_id,m in enumerate(models):
                    t_pred, var_logits = time_func(lambda d: m(d).cpu().numpy(), data)

                    t_solve, x = time_func(generic_greedy, pb, var_logits)
                    
                    if pb.__class__ == MIS:
                        gap_val = 100 * (y.sum() - x.sum()) / y.sum()
                    if pb.__class__ == SCP:
                        gap_val = 100 * (x.sum() - y.sum()) / y.sum()

                    values = [c,y.sum(),t_data,m_id,t_pred,t_solve,t_data+t_pred+t_solve,gap_val]
                    values = [str(it) for it in values]
                    print(",".join(values), file=f)
                    print(",".join(values))
                c+= 1
                if c >= L:
                    break
        bar.finish()
