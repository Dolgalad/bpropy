"""Importance sampling technique, adapted for the MIS problem
"""
import os
import json
import networkx as nx
import numpy as np
import itertools

from torch_geometric.loader import DataLoader
from progress.bar import Bar

from dopy.mis import *
from bpropy.lbp import _loopy_belief_propagation

from models_mrf import *
from mis_er_mrf_dataset import MISERMRFDataset

def get_neighbors(data, u):
    return data.edge_index[1, data.edge_index[0,:]==u].numpy()
def tgdata_to_networkx(data):
    nv = int(torch.sum(data.bipartite==0).item())
    nc = int(torch.sum(data.bipartite).item())
    g = nx.Graph()
    for i in range(nv):
        g.add_node(i)
    for c in range(nc):
        neighbors = get_neighbors(data, c+nv)
        for e in itertools.product(neighbors, neighbors):
            if e[0]!=e[1]:
                g.add_edge(e[0], e[1])
    return g


def mis_importance_sampling(pb, theta, data):
    fixed_variables = np.zeros(pb.nv(), dtype=int)
    solution = np.zeros(pb.nv(), dtype=int)
    mu0 = torch.zeros_like(theta)
    v2f = torch.zeros_like(theta)
    while not np.all(fixed_variables):
        # belief propagation
        mu = _loopy_belief_propagation(data, theta, v2f, data.f2v_edges, data.r_edges, data.q_edges, data.m_edges, max_iter=10, tol=1e-6, damping=0.5, mu0=mu0)

        # vertex marginals for x_i=1
        marginals = mu[:2*pb.nv()].reshape(pb.nv(), 2)[:,1].clone().numpy()
        # set fixed vertex marginals to -inf
        marginals[fixed_variables==1] = -np.inf
        # find candidate
        u = np.argmax(marginals)
        solution[u] = 1
        fixed_variables[u] = 1
        theta[2*u] = -1000.0
        theta[2*u+1] = 1000.0
        for v in pb.g.neighbors(u):
            fixed_variables[v] = 1
            theta[2*v] = 1000.0
            theta[2*v+1] = -1000.0
        mu0 = mu.clone()

    return solution

def load_model(path):
    checkpoint_path = os.path.join(path, "checkpoints", "checkpoint.pt")
    with open(os.path.join(path, "description.json"), "r") as f:
        description = json.load(f)
    model_class = description["model"]["class"]
    del description["model"]["class"]
    model = eval(model_class)(9, **description["model"])
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return model



# model checkpoint path
def test(model_path):
    print("testing model : ", model_path)
    #model_path = "jean-zay_logs/grid_search_mrf_simple_1/mis_mrf/mis_mrf_gcn_mrf_v2_gs1_GCN_MRF_v2_MIS100_EB_aeb6e9/"
    model = load_model(model_path)
    model.eval()
    
    # dataset
    dataset_path = os.path.join(os.environ["WORK"], "datasets/MIS100_MRF_val")
    dataset_category = "MIS100_EB"
    dataset = MISERMRFDataset(dataset_path, category=dataset_category, mode="test", unravel_pool=False, solution_limit=1)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    gaps = []
    bar = Bar("Testing: ", max=dataset.len())
    with torch.no_grad():
        for data in loader:
            out = model(data)
            g = tgdata_to_networkx(data)
            pb = MIS(g)
            data_sol = data.y.numpy()
            if data_sol.sum() == 0:
                continue
            # solve greedy
            unary_marginals = out[:2*pb.nv()].reshape(pb.nv(), 2)
            node_scores = unary_marginals[:,1].cpu().numpy()
            grd_sol = mis_greedy(pb, node_scores)
    
            # optimal sol
            sol_tmp = data.y[:2*pb.nv()].reshape(pb.nv(), 2)
            opt_val = sol_tmp[:,1].sum().item()
    
            # importance sampling
            is_sol = mis_importance_sampling(pb, out, data)
    
            # gap values
            grd_gap = (opt_val - grd_sol.sum()) / opt_val
            is_gap = (opt_val - is_sol.sum()) / opt_val
    
            gaps.append([grd_gap, is_gap])
            bar.suffix = f" {np.mean(gaps, axis=0)}"
            bar.next()
        bar.finish()

import sys
for mp in sys.argv[1:]:
    test(mp)
