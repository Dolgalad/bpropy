import sys
sys.path.insert(0, "..")
import torch
import copy
import time
import argparse
import json
import hashlib
import random

from progress.bar import Bar
import networkx as nx
from torch_geometric.utils.convert import from_networkx
#from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader

from torch_geometric.data import Data

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar

import bpropy
import bpropy.co.mis
from bpropy.co.mis import *
from bpropy.mrf.importance_sampling import tgdata_to_networkx, mis_importance_sampling_nmf
from bpropy.models.mrf_models import *
from bpropy.datasets.mis_data import mis_to_data, data_to_mis
from bpropy.datasets.mrf_dataset import MRFDataset
from bpropy.datasets.mrf_dataloader import MRFDataLoader

import matplotlib.pyplot as plt

import sys
import os
import pickle as pkl

from early_stopper import EarlyStopper

parser = argparse.ArgumentParser(prog="testing_mis_solution_edge_cover.py",
                                 description="Testing MIS solution with Nair et. al. approach + Naive Mean Field + Edge Cover")
parser.add_argument("model_path")
parser.add_argument("test_dataset_path")
parser.add_argument("-M", "--model_class", default="GCN_MRF_v4")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--zero_features", action="store_true")

args = parser.parse_args()

description_file = os.path.join(args.model_path, "description.json")

if os.path.exists(description_file) and len(description_file)>0:
    with open(description_file) as f:
        description = json.load(f)
        model_args = description["model"]
        args.model_class = model_args["class"]
        del model_args["class"]
else:
    print("No description file")
    exit()

# setting seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# training description data
training_description = {
        "seed": args.seed,

        "model": {
            "class": args.model_class,
            **model_args
        },
        }



# checkpoint save path
checkpoint_dir = os.path.join(args.model_path, "checkpoints")
print("checkpoint_dir = ", checkpoint_dir)


print("Loading data")
def cliques_adjacency_matrix(pb):
    adj = np.zeros((len(pb.C), pb.nv()), dtype=bool)
    for i,c in enumerate(pb.C):
        adj[i,np.array(c, dtype=int)] = 1
    return adj
def create_simplified_factor_graph(data):
    pb = data_to_mis(data)
    pb_sT = MIS(pb.g)

    data_sT = mis_to_data(pb_sT, None, with_nmf_edges=True)

    # mapping between new constraint nodes and old constraint nodes
    adj = cliques_adjacency_matrix(pb)
    adj_sT = cliques_adjacency_matrix(pb_sT)
    mapping = []
    for i in range(pb.nv()):
        mapping.append([i,i])
    for i_sT,c_sT in enumerate(pb_sT.C):
        a = np.logical_and(adj, adj_sT[i_sT,:]).sum(axis=1)
        for v in np.argwhere(a==adj_sT[i_sT,:].sum()).flatten():
            mapping.append([pb.nv()+v,pb.nv()+i_sT])

        #for i,c in enumerate(pb.C):
        #    if all([ e in c for e in c_sT]):
        #        #mapping.append([pb.nv()+i,pb.nv()+i_sT])
        #        print("old = ", [pb.nv()+i,pb.nv()+i_sT])
    mapping = torch.tensor(mapping)

    # remove the NMF edges from data
    data_ = Data(
                edge_index=data.edge_index,
                x=data.x,
                e=data.e,
                y=data.y,
                factor_size=data.factor_size,
                bipartite=data.bipartite
            )

    return data_, data_sT, mapping


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device ", device)
#model = GCN(9, args.hidden_dim).to(device)

print("Testing model")
print(json.dumps(model_args, indent=4, sort_keys=True))
model = eval(args.model_class)(10, factor_sizes=list(range(1,10)), **model_args).to(device)


checkpoint_path = os.path.join(args.model_path, "checkpoints", "checkpoint.pt")
if os.path.exists(os.path.join(args.model_path, "checkpoints", "checkpoint.pt")):
    test_graphs_processed_filename = "_".join([args.test_dataset_path,"processed_edge_cover.pkl"])
    if not os.path.exists(test_graphs_processed_filename):
        test_graphs = pkl.load(open(args.test_dataset_path, "rb"))
        for i in range(len(test_graphs)):
            test_graphs[i] = create_simplified_factor_graph(test_graphs[i])
            print("Simplification test: ", i, " / ", len(test_graphs))
        with open(test_graphs_processed_filename,"wb") as f:
            pkl.dump(test_graphs, f)
    test_dataset = MRFDataset(test_graphs_processed_filename)
    test_loader = MRFDataLoader(test_dataset, batch_size=1, shuffle=False)

    # model 
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    
    greedy_gaps, greedy_time = [], []
    is_gaps, is_time = [], []
    bce_values, nll_values = [], []
    bar = Bar("Testing: ", max=test_dataset.len())
    with torch.no_grad():
        for (data,fg_data,fg_map) in test_loader:
            t0 = time.time()
            pb = data_to_mis(data, batch=True)

            data = data.to(device)
            if args.zero_features:
                data.x = torch.zeros_like(data.x)

            fg_data = fg_data.to(device)
            fg_data.nmf_edges = fg_data.nmf_edges.to(device)
            fg_map = torch.squeeze(fg_map,0).to(device)

            out,theta = model(data, fg_data=fg_data, fg_map=fg_map)

            unary_marginals = out[:2*pb.nv()].reshape(pb.nv(), 2)
            node_scores = unary_marginals[:,1].cpu().numpy()
            t_prep = time.time() - t0

            # solve greedy
            t_grd = time.time()
            sol = mis_greedy(pb, node_scores)
            t_grd = time.time() - t_grd

            # solve is
            t_is = time.time()
            is_sol = mis_importance_sampling_nmf(pb, theta, edges=fg_data.nmf_edges)
            t_is = time.time() - t_is


            # gap
            sol_tmp = data.y[:2*pb.nv()].reshape(pb.nv(), 2)
            opt_val = sol_tmp[:,1].sum().item()
            if opt_val == 0:
                continue
            grd_val = sol.sum()
            gap = (opt_val - grd_val) / opt_val

            is_gap = (opt_val - is_sol.sum()) / opt_val

            # loss value, NLL and BCE unary marginals
            nll_values.append(torch.nn.functional.nll_loss(unary_marginals.log(), sol_tmp[:,1].to(torch.long)).item())
            bce_values.append(torch.nn.functional.binary_cross_entropy(unary_marginals[:,1], sol_tmp[:,1].to(torch.float)).item())


            greedy_gaps.append(gap)
            greedy_time.append(t_grd)
            is_gaps.append(is_gap)
            is_time.append(t_is)
            bar.suffix = f" {np.mean(greedy_gaps)} {np.mean(is_gaps)}"
            bar.next()
        bar.finish()

    print("Test BCE   mean/std: ", np.mean(bce_values), np.std(bce_values))
    print("Test NLL   mean/std: ", np.mean(nll_values), np.std(nll_values))
    print("Greedy gap mean/std: ", np.mean(greedy_gaps), np.std(greedy_gaps))
    print("Greedy t   mean/std: ", np.mean(greedy_time), np.std(greedy_time))
    print("IS gap mean/std    : ", np.mean(is_gaps), np.std(is_gaps))
    print("IS t   mean/std    : ", np.mean(is_time), np.std(is_time))

