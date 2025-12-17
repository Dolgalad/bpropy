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

bit_cache = {}  # arity -> np.ndarray [2**arity, arity] of 0/1
def bits_for_arity(a):
    if a not in bit_cache:
        idx = np.arange(1 << a, dtype=np.uint32)[:, None]
        ks  = np.arange(a, dtype=np.uint32)[None, :]
        bit_cache[a] = ((idx >> ks) & 1).astype(np.uint8)  # [2^a, a]
    return bit_cache[a]

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", help="<Required> Dataset root path")
    p.add_argument("output", default="output",
                   help="<Required> Output file path")
    p.add_argument("log_dirs", nargs="+", help="<Required> List of model log directories")
    p.add_argument("--limit", default=100000, type=int)
    p.add_argument("--solver", default="kamis")

    p.add_argument("--processed_dir", default="processed",
                   help="<Optional> subdirectory under <root>/processed/ used for cached features")

    args = p.parse_args()

    print("Solving instance")
    print("\tDataset            : ", args.dataset)
    print("\tProcessed directory: ", args.processed_dir)
    print("\tOutput file        : ", args.output)
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
    preprocess_f = lambda pb: mrf_collate([problem_to_data(
            pb, hetero=True, feature_mode="EB",
            with_lbp=True, with_nmf=False
        )]).to(device)
    
    # load models
    models = [load_model_from_log_dir(p, map_location=device) for p in args.log_dirs]
    for m in models:
        m.eval()

    feasible_theta_values, infeasible_theta_values = [], []

    L,c = args.limit,0
    with torch.no_grad():
        bar = Bar(f"[Test]          ", max=len(loader_eval))
        for pb, y in loader_eval:
            pb = pb[0]
            y = y[0]

            # create factor graph
            if pb.__class__ == MIS:
                fg = MISFactorGraph(pb)
            if pb.__class__ == SCP:
                fg = SCPFactorGraph(pb)

            # create the data object
            t_data, data = time_func(preprocess_f, pb)


            # for each model predict logits and solve
            thetas = [m(data)[1] for m in models]
            c = 0
            for F in fg.iter_factors():
                nF = fg.factor_size(F)
                bits = bits_for_arity(nF)
                for j in range(2**nF):
                    s = bits[j]
                    if pb.__class__==MIS:
                        if s.sum() > 1:
                            # infeasible
                            for th in thetas:
                                infeasible_theta_values.append(th[c].item())
                        else:
                            # feasible
                            for th in thetas:
                                feasible_theta_values.append(th[c].item())
                    if pb.__class__==SCP:
                        if s.sum() == 0:
                            # infeasible
                            for th in thetas:
                                infeasible_theta_values.append(th[c].item())
                        else:
                            for th in thetas:
                                feasible_theta_values.append(th[c].item())
                    c += 1
            bar.next()
        bar.finish()

        import matplotlib.pyplot as plt
        #plt.rcParams.update({
        #    "text.usetex": True,
        #    "font.family": "sans-serif",
        #    "font.sans-serif": "Helvetica",
        #})
        fig = plt.figure()
        plt.violinplot([infeasible_theta_values, feasible_theta_values], positions = [0,1])
        plt.xticks([0,1], ["Infeasible", "Feasible"], rotation=30)
        plt.ylabel("log p(x_F)")
        plt.savefig(args.output+".png")
        #d = np.array(feasible_theta_values)
        #np.savetxt(args.output+".feasible", d.T, delimiter=",")
        #d = np.array(infeasible_theta_values)
        #np.savetxt(args.output+".infeasible", d.T, delimiter=",")
