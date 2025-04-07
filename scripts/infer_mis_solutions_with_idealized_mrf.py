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

from torch_geometric.data import Data

from bpropy.utils.set_sort import lenlex_sort
from bpropy.co.mis import MIS, cplex_model
from bpropy.datasets.mis_features import eb_features
from bpropy.vc_graph import VCGraph
from bpropy.mrf.mis_factor_graph import MISFactorGraph
from bpropy.mrf.lbp_edges import LBPEdges
from bpropy.mrf.lbp import loopy_belief_propagation, max_sum_loopy_belief_propagation
from bpropy.mrf.nmf_edges import NMFEdges
from bpropy.mrf.naive_mean_field import naive_mean_field
from bpropy.mrf.importance_sampling import mis_importance_sampling, mis_importance_sampling_nmf
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
            theta[c] = 0.
            c += 2
        else:
            for v in range(2**nF):
                vals = np.unpackbits(np.array([v]).astype(np.uint8), count=nF, bitorder="little")
                if vals.sum()>1:
                    theta[c] = -10.
                else:
                    theta[c] = 0.
                c += 1

    return theta




if __name__=="__main__":
    dataset_path = sys.argv[1]
    if dataset_path.endswith(".pkl"):
        dataset = pkl.load(open(dataset_path, "rb"))
    else:
        dataset = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    redump = False

    output_path = sys.argv[2]
    columns = ["dataset", "sp_greedy_gap", "sp_greedy_t",
               "sp_is_gap", "sp_is_t",
               "ms_greedy_gap", "ms_greedy_t",
               "ms_is_gap", "ms_is_t",
               "nmf_greedy_gap", "nmf_greedy_t",
               "nmf_is_gap", "nmf_is_t"
               ]


    max_iter = 10
    count = 0
    limit = 10
    if len(sys.argv) > 3:
        limit = int(sys.argv[3])
    gap_values = {"sp_greedy_gap":[], "ms_greedy_gap":[], 
                  "sp_is_gap":[], "ms_is_gap":[], 
                  "sp_greedy_t": [], "sp_is_t": [], 
                  "ms_greedy_t": [], "ms_is_t": [], 
                  "nmf_greedy_gap": [], "nmf_greedy_t": [],
                  "nmf_is_gap": [], "nmf_is_t": []
                  }
   
    bar = Bar("Solving dataset: ", max = len(dataset))
    with torch.no_grad():
        for k in range(len(dataset)):
            if isinstance(dataset[k], str):
                data = torch.load(dataset[k])
            else:
                data = dataset[k]
            data = data.to(device)
            # recreate the MIS problem
            # generate MIS
            #C = cliques_from_data(data)
            pb = data_to_mis(data)
            # factor graph
            fg = MISFactorGraph(pb)

            #y_ind = (data.y.cpu().numpy() > 0.5).astype(int)
            #x = (data.y[:2*pb.nv()].reshape(pb.nv(),2)[:,1].cpu().numpy() > 0.5)
            #x_ind = x2ss(x, fg)
            if not np.all(x_ind == y_ind):
                print("Recalculating edges and features")
                # features
                md = cplex_model(pb, continuous_relaxation=True, cliques=pb.C)

                C = pb.C
                _,nfeat,_,cfeat = eb_features(pb, md)
                pb.C = C

                efeat = np.ones(2 * sum(len(c) for c in pb.C))
                nfeat = nfeat - np.mean(nfeat,axis=0)
                cfeat = cfeat - np.mean(cfeat,axis=0)
                efeat = efeat - np.mean(efeat,axis=0)
                nn = nfeat.shape[1] - cfeat.shape[1]
                xx = np.concatenate((nfeat, np.pad(cfeat, ((0,0), (0,nn)))), axis=0)
            
                vcg = VCGraph(pb)
                g = vcg.to_directed()
                edge_index = torch.empty((2, g.number_of_edges()), dtype=torch.long)
                bipartite = torch.empty(g.number_of_nodes())
                for i,(src,dst) in enumerate(g.edges):
                    edge_index[0,i] = src
                    edge_index[1,i] = dst
                for n,d in g.nodes(data=True):
                    bipartite[n] = d["bipartite"]


                data1 = Data(edge_index=edge_index, 
                            bipartite=bipartite,
                            x=torch.from_numpy(xx),
                            e=torch.from_numpy(efeat),
                            y=torch.from_numpy(x2ss(x, fg))
                            )

                pb1 = data_to_mis(data1)
                # factor graph
                fg1 = MISFactorGraph(pb1)
                y_ind1 = (data1.y.cpu().numpy() > 0.5).astype(int)
                x1 = (data1.y[:2*pb1.nv()].reshape(pb1.nv(),2)[:,1].cpu().numpy() > 0.5)
                x_ind1 = x2ss(x1, fg1)
                dataset[k] = data1
                data = data1
                redump = True

            edges = LBPEdges(fg).to(device)
            x = ss2x(data.y.cpu().numpy(), fg)
            for c in pb.C:
                if x[c].sum() > 1:
                    print(c, x[c], x[c].sum())
            xx = data.y[:2*pb.nv()].reshape(pb.nv(),2)[:,1]
            print(pb.check_solution(x))
            if pb.check_solution(x):
                val = x.sum()
                # create the vector of canonical parameters
                theta = mis_idealized_canonical_parameters(fg)
                v2f = torch.zeros_like(theta)

                # Sum-product + greedy solve
                t_sp_greedy = time.time()
                mu = loopy_belief_propagation(theta.clone(), v2f.clone(), edges, max_iter=max_iter)
                x_sp_greedy = pb.greedy(mu[:2*pb.nv()].reshape(pb.nv(),2)[:,1].cpu().numpy())
                t_sp_greedy = time.time() - t_sp_greedy
                val_sp_greedy = x_sp_greedy.sum()
                sp_greedy_gap = 100 * (val - val_sp_greedy) / val
                gap_values["sp_greedy_gap"].append(sp_greedy_gap)
                gap_values["sp_greedy_t"].append(t_sp_greedy)
                print("Sum-product greedy gap          = ", sp_greedy_gap)

                # Sum-product + importance sampling solve
                t_sp_is = time.time()
                x_sp_is = mis_importance_sampling(pb, theta.clone(), max_iter=max_iter)
                t_sp_is = time.time() - t_sp_is
                val_sp_is = x_sp_is.sum()
                sp_is_gap = 100 * (val - val_sp_is) / val
                gap_values["sp_is_gap"].append(sp_is_gap)
                gap_values["sp_is_t"].append(t_sp_is)
                print("Sum-product IS gap              = ", sp_is_gap)

                # Max-sum + greedy solve
                t_ms_greedy = time.time()
                mu = max_sum_loopy_belief_propagation(theta.clone(), v2f.clone(), edges, max_iter=max_iter)
                x_ms_greedy = pb.greedy(mu[:2*pb.nv()].reshape(pb.nv(),2)[:,1].cpu().numpy())
                t_ms_greedy = time.time() - t_ms_greedy
                val_ms_greedy = x_ms_greedy.sum()
                ms_greedy_gap = 100 * (val - val_ms_greedy) / val
                gap_values["ms_greedy_gap"].append(ms_greedy_gap)
                gap_values["ms_greedy_t"].append(t_ms_greedy)
                print("Max-sum greedy gap              = ", ms_greedy_gap)

                # Max-sum + importance sampling solve
                t_ms_is = time.time()
                x_ms_is = mis_importance_sampling(pb, theta.clone(), inference_f=max_sum_loopy_belief_propagation, max_iter=max_iter)
                t_ms_is = time.time() - t_ms_is
                val_ms_is = x_ms_is.sum()
                ms_is_gap = 100 * (val - val_ms_is) / val
                gap_values["ms_is_gap"].append(ms_is_gap)
                gap_values["ms_is_t"].append(t_ms_is)
                print("Max-sum IS gap                  = ", ms_is_gap)

                # naive mean field and greedy
                nmf_edges = NMFEdges(fg)
                t_nmf_greedy = time.time()
                mu = naive_mean_field(fg, theta.clone(), nmf_edges.pmprod_edges, nmf_edges.pmprodsum_edges, max_iter=max_iter)
                x_nmf_greedy = pb.greedy(mu[:2*pb.nv()].reshape(pb.nv(),2)[:,1].cpu().numpy())
                t_nmf_greedy = time.time() - t_nmf_greedy
                val_nmf_greedy = x_nmf_greedy.sum()
                nmf_greedy_gap = 100 * (val - val_nmf_greedy) / val
                gap_values["nmf_greedy_gap"].append(nmf_greedy_gap)
                gap_values["nmf_greedy_t"].append(t_nmf_greedy)
                print("Naive Mean Field greedy gap     = ", nmf_greedy_gap)

                # naive mean field and importance sampling
                t_nmf_is = time.time()
                x_nmf_is = mis_importance_sampling_nmf(pb, theta.clone(), max_iter=max_iter, edges=nmf_edges)
                t_nmf_is = time.time() - t_nmf_is
                val_nmf_is = x_nmf_is.sum()
                nmf_is_gap = 100 * (val - val_nmf_is) / val
                gap_values["nmf_is_gap"].append(nmf_is_gap)
                gap_values["nmf_is_t"].append(t_nmf_is)
                print("Naive Mean Field IS gap         = ", nmf_is_gap)


            count += 1
            if count >= limit:
                break

            bar.next()
    bar.finish()

    #if redump:
    #    dataset = pkl.dump(dataset, open(dataset_path, "wb"))

    row = {k:"${:.3f} (\pm {:.3f})$".format(round(np.mean(gap_values[k]), 3), round(np.std(gap_values[k]), 3)) for k  in gap_values.keys()}
    row["dataset"] = os.path.basename(dataset_path)
    header = not os.path.exists(output_path)
    with open(output_path, "a") as f:
        if header:
            f.write(",".join(columns) + "\n")
        f.write(",".join([row[k] for k in columns]) + "\n")



