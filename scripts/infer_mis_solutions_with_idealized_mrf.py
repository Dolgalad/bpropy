"""Infer solutions to MIS problems using the "idealized" MRF
"""
import sys
sys.path.insert(0,"..")
import pickle as pkl
import numpy as np
import torch
from progress.bar import Bar

from bpropy.co.mis import MIS
from bpropy.mrf.mis_factor_graph import MISFactorGraph
from bpropy.mrf.lbp_edges import LBPEdges
from bpropy.mrf.lbp import loopy_belief_propagation, max_sum_loopy_belief_propagation
from bpropy.mrf.importance_sampling import mis_importance_sampling
from bpropy.mrf.sufficient_statistics import ss2x,x2ss
from bpropy.datasets.mis_data import data_to_mis

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    max_iter = 100
    
    bar = Bar("Solving dataset: ", max = len(dataset))
    for data in dataset:
        # recreate the MIS problem
        # generate MIS
        pb = data_to_mis(data)
        # factor graph
        fg = MISFactorGraph(pb)
        edges = LBPEdges(fg).to(device)
        x = ss2x(data.y.numpy(), fg)
        gap_values = {"sp_greedy":[], "ms_greedy":[], "sp_is":[], "ms_is":[]}
        if pb.check_solution(x):
            val = x.sum()
            # create the vector of canonical parameters
            theta = mis_idealized_canonical_parameters(fg)
            v2f = torch.zeros_like(theta)

            # Sum-product + greedy solve
            mu = loopy_belief_propagation(theta, v2f, edges, max_iter=max_iter)
            x_sp_greedy = pb.greedy(mu[:2*pb.nv()].reshape(pb.nv(),2)[:,1].cpu().numpy())
            val_sp_greedy = x_sp_greedy.sum()
            sp_greedy_gap = 100 * (val - val_sp_greedy) / val
            gap_values["sp_greedy"].append(sp_greedy_gap)
            #print("Sum-product greedy gap = ", sp_greedy_gap)

            # Sum-product + importance sampling solve
            x_sp_is = mis_importance_sampling(pb, theta.clone())
            val_sp_is = x_sp_is.sum()
            sp_is_gap = 100 * (val - val_sp_is) / val
            gap_values["sp_is"].append(sp_is_gap)
            #print("Sum-product IS gap     = ", sp_is_gap)

            # Max-sum + greedy solve
            mu = max_sum_loopy_belief_propagation(theta, v2f, edges, max_iter=max_iter)
            x_ms_greedy = pb.greedy(mu[:2*pb.nv()].reshape(pb.nv(),2)[:,1].cpu().numpy())
            val_ms_greedy = x_ms_greedy.sum()
            ms_greedy_gap = 100 * (val - val_ms_greedy) / val
            gap_values["ms_greedy"].append(ms_greedy_gap)
            #print("Max-sum greedy gap     = ", ms_greedy_gap)

            # Max-sum + importance sampling solve
            x_ms_is = mis_importance_sampling(pb, theta.clone(), inference_f=max_sum_loopy_belief_propagation)
            val_ms_is = x_ms_is.sum()
            ms_is_gap = 100 * (val - val_ms_is) / val
            gap_values["ms_is"].append(ms_is_gap)
            #print("Max-sum IS gap         = ", ms_is_gap)
        bar.next()
    bar.finish()
    for k in gap_values:
        print(k, " = ", np.mean(gap_values[k]))



