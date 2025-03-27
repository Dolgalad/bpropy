"""Infer solutions to MIS problems using the "idealized" MRF
"""
import sys
sys.path.insert(0,"..")
import os
import pickle as pkl
import time
import numpy as np
import torch
from progress.bar import Bar
import pandas as pd

from bpropy.co.mis import MIS
from bpropy.mrf.mis_factor_graph import MISFactorGraph
from bpropy.mrf.lbp_edges import LBPEdges
from bpropy.mrf.lbp import loopy_belief_propagation, max_sum_loopy_belief_propagation
from bpropy.mrf.importance_sampling import mis_importance_sampling
from bpropy.mrf.sufficient_statistics import ss2x,x2ss
from bpropy.datasets.mis_data import data_to_mis

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cliques_from_data(data):
    nV = int((data.bipartite==0).sum().item())
    nF = int((data.bipartite==1).sum().item())
    cliques = []
    for f in range(nF):
        cliques.append(data.edge_index[1,data.edge_index[0,:]==f+nV].tolist())
    return cliques

def mis_idealized_canonical_parameters(fg, M=10.):
    theta = torch.zeros(fg.number_sufficient_statistics(), device=device, requires_grad=False)
    c = 0
    for F in fg.iter_factors():
        nF = fg.factor_size(F)
        if nF == 1:
            theta[c+1] = 1.
            c += 2
        else:
            for v in range(2**nF):
                vals = np.unpackbits(np.array([v]).astype(np.uint8), count=nF, bitorder="little")
                if vals.sum()>1:
                    theta[c] = -M
                c += 1

    return theta




if __name__=="__main__":
    dataset_path = sys.argv[1]
    dataset = pkl.load(open(dataset_path, "rb"))

    output_path = sys.argv[2]
    print(sys.argv)

    if os.path.exists(output_path):
        df = pd.read_csv(output_path, sep=" ")
    else:
        columns = ["dataset", "sp_greedy_gap", "sp_greedy_t",
                   "sp_is_gap", "sp_is_t",
                   "ms_greedy_gap", "ms_greedy_t",
                   "ms_is_gap", "ms_is_t"
                   ]
        df = pd.DataFrame(columns=columns)
        df.index.name = "idx"


    max_iter = 100
    
    bar = Bar("Solving dataset: ", max = len(dataset))
    for data in dataset:
        # recreate the MIS problem
        # generate MIS
        #C = cliques_from_data(data)
        pb = data_to_mis(data)
        # factor graph
        fg = MISFactorGraph(pb)
        edges = LBPEdges(fg).to(device)
        x = ss2x(data.y.numpy(), fg)
        gap_values = {"sp_greedy_gap":[], "ms_greedy_gap":[], "sp_is_gap":[], "ms_is_gap":[], "sp_greedy_t": [], "sp_is_t": [], "ms_greedy_t": [], "ms_is_t": []}
        if pb.check_solution(x):
            val = x.sum()
            # create the vector of canonical parameters
            theta = mis_idealized_canonical_parameters(fg)
            v2f = torch.zeros_like(theta)

            # Sum-product + greedy solve
            t_sp_greedy = time.time()
            mu = loopy_belief_propagation(theta, v2f, edges, max_iter=max_iter)
            x_sp_greedy = pb.greedy(mu[:2*pb.nv()].reshape(pb.nv(),2)[:,1].cpu().numpy())
            t_sp_greedy = time.time() - t_sp_greedy
            val_sp_greedy = x_sp_greedy.sum()
            sp_greedy_gap = 100 * (val - val_sp_greedy) / val
            gap_values["sp_greedy_gap"].append(sp_greedy_gap)
            gap_values["sp_greedy_t"].append(t_sp_greedy)
            #print("Sum-product greedy gap = ", sp_greedy_gap)

            # Sum-product + importance sampling solve
            t_sp_is = time.time()
            x_sp_is = mis_importance_sampling(pb, theta.clone())
            t_sp_is = time.time() - t_sp_is
            val_sp_is = x_sp_is.sum()
            sp_is_gap = 100 * (val - val_sp_is) / val
            gap_values["sp_is_gap"].append(sp_is_gap)
            gap_values["sp_is_t"].append(t_sp_is)

            #print("Sum-product IS gap     = ", sp_is_gap)

            # Max-sum + greedy solve
            t_ms_greedy = time.time()
            mu = max_sum_loopy_belief_propagation(theta, v2f, edges, max_iter=max_iter)
            x_ms_greedy = pb.greedy(mu[:2*pb.nv()].reshape(pb.nv(),2)[:,1].cpu().numpy())
            t_ms_greedy = time.time() - t_ms_greedy
            val_ms_greedy = x_ms_greedy.sum()
            ms_greedy_gap = 100 * (val - val_ms_greedy) / val
            gap_values["ms_greedy_gap"].append(ms_greedy_gap)
            gap_values["ms_greedy_t"].append(t_ms_greedy)

            #print("Max-sum greedy gap     = ", ms_greedy_gap)

            # Max-sum + importance sampling solve
            t_ms_is = time.time()
            x_ms_is = mis_importance_sampling(pb, theta.clone(), inference_f=max_sum_loopy_belief_propagation)
            t_ms_is = time.time() - t_ms_is
            val_ms_is = x_ms_is.sum()
            ms_is_gap = 100 * (val - val_ms_is) / val
            gap_values["ms_is_gap"].append(ms_is_gap)
            gap_values["ms_is_t"].append(t_ms_is)

        bar.next()
    bar.finish()
    row = {k:[np.mean(gap_values[k])] for k  in gap_values.keys()}
    row["dataset"] = os.path.basename(dataset_path)
    #row["idx"] = len(df)
    df = pd.concat([df, pd.DataFrame.from_dict(row)], ignore_index=True, axis=0, join="inner")
    df.to_csv(output_path, sep=" ", index_label="idx")



