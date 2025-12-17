"""Infer solutions to MIS problems using the "idealized" MRF
"""
from mis_dataset_features import create_features

import sys
sys.path.insert(0,"..")
import os
import pickle as pkl
import time
import numpy as np
import torch
import pandas as pd
from filelock import FileLock
import tarfile
import tempfile

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


def get_instance_names(path, category="test"):
    instances = []
    with tarfile.open(path,"r") as f:
        for name in f.getnames():
            fields = name.split("/")
            if category in fields:
                if fields[-1].isnumeric():
                    instances.append(name)
    return sorted(instances, key=lambda p: int(p.split("/")[-1]))


if __name__=="__main__":
    archive_path = sys.argv[1]

    output_path_prefix = sys.argv[2]
    lock = FileLock(f"{output_path_prefix}.lock")

    task_id = int(sys.argv[3])
    max_tasks = int(sys.argv[4])

    dataset_size_limit = int(sys.argv[5])

    formulations = ["EB","CC","CF","EC","CP"]
    big_M_values = [1e0, 1e1, 1e2, 1e3, 1e4, 1e6]

    instance_names = get_instance_names(archive_path)
    if dataset_size_limit < len(instance_names):
        instance_names = instance_names[:dataset_size_limit]

    print("Number of instances in task : ", len(instance_names))

    n_inst = len(instance_names) // max_tasks

    if task_id < max_tasks:
        instance_names = instance_names[n_inst*(task_id-1):n_inst*task_id]
    else:
        instance_names = instance_names[n_inst*(task_id-1):]


    for formulation in formulations:
        print(f"Preparing data for formulation {formulation}")
        instance_data = []
        for k in range(len(instance_names)):
            print("Working on ", instance_names[k])
            torch.cuda.empty_cache()

            # prepare instance data
            sys.stdout.write("Extracting instance and creating features...")
            sys.stdout.flush()
            with tempfile.TemporaryDirectory() as f:
                with tarfile.open(archive_path,"r") as archive_f:
                    members = [tarinfo for tarinfo in archive_f.getmembers() if tarinfo.name.startswith(instance_names[k])]
                    archive_f.extractall(f, members=members)
                inst_path = os.path.join(f, instance_names[k])
                data = create_features((inst_path, "dummy.dat", formulation), save_to_disk=False)
                instance_data.append(data)
            sys.stdout.write("done\n")
            sys.stdout.flush()


        for big_M in big_M_values:
            print("Archive path       : ", archive_path)
            print("Output path prefix : ", output_path_prefix)
            print("Task id            : ", task_id)
            print("Max tasks          : ", max_tasks)
            print("Dataset size limit : ", dataset_size_limit)
            print("Big M              : ", big_M)
            print("Formulation        : ", formulation)
            print("N instances        : ", n_inst)



            # load output data
            columns = ["dataset", "sp_greedy_gap", "sp_greedy_t",
                       "sp_is_gap", "sp_is_t",
                       "ms_greedy_gap", "ms_greedy_t",
                       "ms_is_gap", "ms_is_t",
                       "nmf_greedy_gap", "nmf_greedy_t",
                       "nmf_is_gap", "nmf_is_t"
                       ]


            max_iter = 10
            count = 0
            limit = 10000


            gap_values = {"sp_greedy_gap":[], "ms_greedy_gap":[], 
                          "sp_is_gap":[], "ms_is_gap":[], 
                          "sp_greedy_t": [], "sp_is_t": [], 
                          "ms_greedy_t": [], "ms_is_t": [], 
                          "nmf_greedy_gap": [], "nmf_greedy_t": [],
                          "nmf_is_gap": [], "nmf_is_t": [],
                          "instance": []
                          }
            output_path = "_".join([output_path_prefix, formulation, f"M{big_M}"]) + ".pkl"
            if os.path.exists(output_path):
                with lock:
                    with open(output_path, "rb") as f:
                        gap_values = pkl.load(f)
   
            with torch.no_grad():
                for k in range(len(instance_names)):
                    print(f"Formulation {formulation}, M = {big_M}, working on ", instance_names[k])
                    torch.cuda.empty_cache()

                    device = torch.device('cuda:0')
                    free, total = torch.cuda.mem_get_info(device)
                    mem_used_MB = (total - free) / 1024 ** 2
                    print("Memory used : ", mem_used_MB, flush=True)

                    data = instance_data[k]
                    data = data.to(device)
                    # recreate the MIS problem
                    # generate MIS
                    #C = cliques_from_data(data)
                    pb = data_to_mis(data)
                    # factor graph
                    fg = MISFactorGraph(pb)


                    edges = LBPEdges(fg).to(device)
                    x = data.y.cpu().numpy()
                    data = None
                    if pb.check_solution(x):
                        val = x.sum()
                        # create the vector of canonical parameters
                        theta = mis_idealized_canonical_parameters(fg, M=big_M)
                        v2f = torch.zeros_like(theta)

                        # Sum-product + greedy solve
                        t_sp_greedy = time.time()
                        mu = loopy_belief_propagation(theta.clone(), v2f.clone(), edges, max_iter=max_iter).mu
                        x_sp_greedy = pb.greedy(mu[:2*pb.nv()].reshape(pb.nv(),2)[:,1].cpu().numpy())
                        t_sp_greedy = time.time() - t_sp_greedy
                        val_sp_greedy = x_sp_greedy.sum()
                        sp_greedy_gap = 100 * (val - val_sp_greedy) / val
                        print("Sum-product greedy gap          = ", sp_greedy_gap)

                        # Sum-product + importance sampling solve
                        t_sp_is = time.time()
                        x_sp_is = mis_importance_sampling(pb, theta.clone(), max_iter=max_iter)
                        t_sp_is = time.time() - t_sp_is
                        val_sp_is = x_sp_is.sum()
                        sp_is_gap = 100 * (val - val_sp_is) / val
                        print("Sum-product IS gap              = ", sp_is_gap)

                        # Max-sum + greedy solve
                        t_ms_greedy = time.time()
                        mu = max_sum_loopy_belief_propagation(theta.clone(), v2f.clone(), edges, max_iter=max_iter).mu
                        x_ms_greedy = pb.greedy(mu[:2*pb.nv()].reshape(pb.nv(),2)[:,1].cpu().numpy())
                        t_ms_greedy = time.time() - t_ms_greedy
                        val_ms_greedy = x_ms_greedy.sum()
                        ms_greedy_gap = 100 * (val - val_ms_greedy) / val
                        print("Max-sum greedy gap              = ", ms_greedy_gap)

                        # Max-sum + importance sampling solve
                        t_ms_is = time.time()
                        x_ms_is = mis_importance_sampling(pb, theta.clone(), inference_f=max_sum_loopy_belief_propagation, max_iter=max_iter)
                        t_ms_is = time.time() - t_ms_is
                        val_ms_is = x_ms_is.sum()
                        ms_is_gap = 100 * (val - val_ms_is) / val
                        print("Max-sum IS gap                  = ", ms_is_gap)

                        # naive mean field and greedy
                        nmf_edges = NMFEdges(fg).to(device)
                        t_nmf_greedy = time.time()
                        mu = naive_mean_field(theta.clone(), nmf_edges, max_iter=max_iter).mu
                        x_nmf_greedy = pb.greedy(mu[:2*pb.nv()].reshape(pb.nv(),2)[:,1].cpu().numpy())
                        t_nmf_greedy = time.time() - t_nmf_greedy
                        val_nmf_greedy = x_nmf_greedy.sum()
                        nmf_greedy_gap = 100 * (val - val_nmf_greedy) / val
                        print("Naive Mean Field greedy gap     = ", nmf_greedy_gap)

                        # naive mean field and importance sampling
                        t_nmf_is = time.time()
                        x_nmf_is = mis_importance_sampling_nmf(pb, theta.clone(), max_iter=max_iter, edges=nmf_edges)
                        t_nmf_is = time.time() - t_nmf_is
                        val_nmf_is = x_nmf_is.sum()
                        nmf_is_gap = 100 * (val - val_nmf_is) / val
                        print("Naive Mean Field IS gap         = ", nmf_is_gap)

                        # save data
                        with lock:
                            if os.path.exists(output_path):
                                with open(output_path, "rb") as f:
                                    gap_values = pkl.load(f)
                            gap_values["sp_greedy_gap"].append(sp_greedy_gap)
                            gap_values["sp_greedy_t"].append(t_sp_greedy)
                            gap_values["nmf_is_gap"].append(nmf_is_gap)
                            gap_values["nmf_is_t"].append(t_nmf_is)
                            gap_values["nmf_greedy_gap"].append(nmf_greedy_gap)
                            gap_values["nmf_greedy_t"].append(t_nmf_greedy)
                            gap_values["ms_is_gap"].append(ms_is_gap)
                            gap_values["ms_is_t"].append(t_ms_is)
                            gap_values["ms_greedy_gap"].append(ms_greedy_gap)
                            gap_values["ms_greedy_t"].append(t_ms_greedy)
                            gap_values["sp_is_gap"].append(sp_is_gap)
                            gap_values["sp_is_t"].append(t_sp_is)
                            gap_values["instance"].append(instance_names[k])
                            with open(output_path, "wb") as f:
                                pkl.dump(gap_values, f)

                        theta = None
                        v2f = None
                        fg = None
                        nmf_edges = None
