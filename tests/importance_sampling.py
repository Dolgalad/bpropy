"""Importance sampling technique, adapted for the MIS problem
"""
import os
import json
import time
import networkx as nx
import numpy as np
import itertools
import pandas as pd

from torch_geometric.loader import DataLoader
from progress.bar import Bar

from dopy.mis import *
from bpropy.lbp import _loopy_belief_propagation, _loopy_belief_propagation_with_evidence

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


def mis_importance_sampling(pb, theta, data, U=10.0):
    fixed_variables = np.zeros(pb.nv(), dtype=int)
    solution = np.zeros(pb.nv(), dtype=int)
    mu0 = torch.zeros_like(theta)
    v2f = torch.zeros_like(theta)
    edges = [e for e in ordered_edges(pb.g)]
    while not np.all(fixed_variables):
        # belief propagation
        mu = _loopy_belief_propagation(data, theta, v2f, data.f2v_edges, data.r_edges, data.q_edges, data.m_edges, max_iter=1000, tol=1e-6, damping=0.0, mu0=mu0)

        # vertex marginals for x_i=1
        marginals = mu[:2*pb.nv()].reshape(pb.nv(), 2)[:,1].clone().numpy()
        # set fixed vertex marginals to -inf
        marginals[fixed_variables==1] = -np.inf
        # find candidate
        u = np.argmax(marginals)
        solution[u] = 1
        fixed_variables[u] = 1
        theta[2*u] = -U
        theta[2*u+1] = U
        u_neighbors = list(pb.g.neighbors(u))
        for v in u_neighbors:
            fixed_variables[v] = 1
            theta[2*v] = U
            theta[2*v+1] = -U
            eidx = edges.index((min(u,v), max(u,v)))
            if u == min(u,v):
                theta[2*pb.nv()+4*eidx] = -U
                theta[2*pb.nv()+4*eidx+1] = U
                theta[2*pb.nv()+4*eidx+2] = -U
                theta[2*pb.nv()+4*eidx+3] = -U
            else:
                theta[2*pb.nv()+4*eidx] = -U
                theta[2*pb.nv()+4*eidx+1] = -U
                theta[2*pb.nv()+4*eidx+2] = U
                theta[2*pb.nv()+4*eidx+3] = -U

            for w in pb.g.neighbors(v):
                if w!=u:
                    vwidx = edges.index((min(w,v), max(w,v)))
                    if w in u_neighbors:
                        theta[2*pb.nv()+4*vwidx] = U
                        theta[2*pb.nv()+4*vwidx+1] = -U
                        theta[2*pb.nv()+4*vwidx+2] = -U
                        theta[2*pb.nv()+4*vwidx+3] = -U
                    else:
                        if v == min(w,v):
                            theta[2*pb.nv()+4*vwidx] = U
                            theta[2*pb.nv()+4*vwidx+1] = -U
                            theta[2*pb.nv()+4*vwidx+2] = U
                            theta[2*pb.nv()+4*vwidx+3] = -U
                        else:
                            theta[2*pb.nv()+4*vwidx] = U
                            theta[2*pb.nv()+4*vwidx+1] = U
                            theta[2*pb.nv()+4*vwidx+2] = -U
                            theta[2*pb.nv()+4*vwidx+3] = -U


        mu0 = mu.clone()

    return solution

def mis_is(pb, theta, data, U=10.0):
    # initialize solution and queue
    free_variables = np.ones(pb.nv(), dtype=bool)
    solution = np.zeros(pb.nv(), dtype=bool)
    N = pb.nv()
    # compute marginals
    v2f = torch.zeros_like(theta)
    edges = [e for e in ordered_edges(pb.g)]
    # belief propagation
    mu = _loopy_belief_propagation(data, theta, v2f, data.f2v_edges, data.r_edges, data.q_edges, data.m_edges, max_iter=1000, tol=1e-6, damping=0.0)
    while np.any(free_variables):
        marginals = mu[:2*pb.nv()].reshape(pb.nv(), 2)[:,1].clone().numpy()
        # set fixed vertex marginals to -inf
        marginals[free_variables==0] = -np.inf
        # find candidate
        u = np.argmax(marginals)
        #print("u = ", u, mu[2*u:2*u+2])
        mu[2*u:2*u+2] = torch.tensor([0,1])
        theta[2*u:2*u+2] = torch.tensor([-U,U])
        solution[u] = 1
        free_variables[u] = 0
        u_neighbors = list(pb.g.neighbors(u))
        for v in u_neighbors:
            if free_variables[v] == 0:
                continue
            solution[v] = 0
            free_variables[v] = 0
            # update marginals
            #print("v = ", v, mu[2*v:2*v+2])
            mu[2*v:2*v+2] = torch.tensor([1,0])
            theta[2*u:2*u+2] = torch.tensor([U,-U])

            for w in pb.g.neighbors(v):
                if free_variables[w]==0:
                    continue
                e = (min(v,w), max(v,w))
                eidx = edges.index(e)
                #print("\tw = ", w, mu[2*N+4*eidx:2*N+4*eidx+4])
                if w==u:
                    if v==e[0]:
                        #print("\t", mu[2*N+4*eidx] + mu[2*N+4*eidx+2])
                        mu[2*N+4*eidx]=0
                        mu[2*N+4*eidx+1]=0
                        mu[2*N+4*eidx+2]=1
                        mu[2*N+4*eidx+3]=0
                        theta[2*N+4*eidx]=-U
                        theta[2*N+4*eidx+1]=-U
                        theta[2*N+4*eidx+2]=U
                        theta[2*N+4*eidx+3]=-U
                    else:
                        #print("\t", mu[2*N+4*eidx] + mu[2*N+4*eidx+1])
                        mu[2*N+4*eidx]=0
                        mu[2*N+4*eidx+1]=1
                        mu[2*N+4*eidx+2]=0
                        mu[2*N+4*eidx+3]=0
                        theta[2*N+4*eidx]=-U
                        theta[2*N+4*eidx+1]=U
                        theta[2*N+4*eidx+2]=-U
                        theta[2*N+4*eidx+3]=-U
                else:
                    if w in u_neighbors:
                        if v==e[0]:
                            #print("\t", mu[2*N+4*eidx] + mu[2*N+4*eidx+2])
                            mu[2*N+4*eidx]=1
                            mu[2*N+4*eidx+1]=0
                            mu[2*N+4*eidx+2]=0
                            mu[2*N+4*eidx+3]=0
                            theta[2*N+4*eidx]=U
                            theta[2*N+4*eidx+1]=-U
                            theta[2*N+4*eidx+2]=-U
                            theta[2*N+4*eidx+3]=-U
                        else:
                            #print("\t", mu[2*N+4*eidx] + mu[2*N+4*eidx+1])
                            mu[2*N+4*eidx]=1
                            mu[2*N+4*eidx+1]=0
                            mu[2*N+4*eidx+2]=0
                            mu[2*N+4*eidx+3]=0
                            theta[2*N+4*eidx]=U
                            theta[2*N+4*eidx+1]=-U
                            theta[2*N+4*eidx+2]=-U
                            theta[2*N+4*eidx+3]=-U
                    else:
                        if v==e[0]:
                            #print("\t", mu[2*N+4*eidx] + mu[2*N+4*eidx+2])
                            mu[2*N+4*eidx+1]=0
                            mu[2*N+4*eidx+3]=0
                            theta[2*N+4*eidx+1]=-U
                            theta[2*N+4*eidx+3]=-U

                            theta[2*N+4*eidx]=U
                            theta[2*N+4*eidx+2]=U
                        else:
                            #print("\t", mu[2*N+4*eidx] + mu[2*N+4*eidx+1])
                            mu[2*N+4*eidx+2]=0
                            mu[2*N+4*eidx+3]=0
                            theta[2*N+4*eidx+2]=-U
                            theta[2*N+4*eidx+3]=-U

                            theta[2*N+4*eidx]=U
                            theta[2*N+4*eidx+1]=U

                        mu[2*N+4*eidx:2*N+4*eidx+4] /= mu[2*N+4*eidx:2*N+4*eidx+4].sum()
        #print("###################################")
        mu = _loopy_belief_propagation(data, theta, mu.clone(), data.f2v_edges, data.r_edges, data.q_edges, data.m_edges, max_iter=1, tol=1e-6, damping=0.0)
    return solution





def load_model(path):
    checkpoint_path = os.path.join(path, "checkpoints", "checkpoint.pt")
    with open(os.path.join(path, "description.json"), "r") as f:
        description = json.load(f)
    model_class = description["model"]["class"]
    del description["model"]["class"]
    model = eval(model_class)(9, **description["model"])
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return model, description



# model checkpoint path
def test(model_path):
    print("testing model : ", model_path)
    #model_path = "jean-zay_logs/grid_search_mrf_simple_1/mis_mrf/mis_mrf_gcn_mrf_v2_gs1_GCN_MRF_v2_MIS100_EB_aeb6e9/"
    model, desc = load_model(model_path)
    model.eval()
    
    # dataset
    dataset_path = os.path.join(os.environ["WORK"], "datasets/MIS100_MRF_val")
    dataset_category = "MIS100_EB"
    dataset = MISERMRFDataset(dataset_path, category=dataset_category, mode="test", unravel_pool=False, solution_limit=1)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    gaps = []
    bar = Bar("Testing: ", max=dataset.len())
    c = 0
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
            #t_is = time.time()
            is_sol = mis_importance_sampling(pb, out.clone(), data)
            #t_is = time.time() - t_is
            #t_is2 = time.time()
            is2_sol = mis_is(pb,  out.clone(), data)
            #t_is2 = time.time() - t_is2

            # gap values
            grd_gap = (opt_val - grd_sol.sum()) / opt_val
            is_gap = (opt_val - is_sol.sum()) / opt_val
            is2_gap = (opt_val - is2_sol.sum()) / opt_val

            #print(f"opt = {opt_val}, grd = {grd_sol.sum()}, is = {is_sol.sum()}, is2 = {is2_sol.sum()}")
    
            gaps.append([grd_gap, is_gap, is2_gap])
            print(gaps[-1])
            bar.suffix = f" {np.mean(gaps, axis=0)}"
            bar.next()
            c += 1
            if c >= 1 and False:
                exit()
                break
        bar.finish()
    return desc, np.mean([r[0] for r in gaps]), np.mean([r[1] for r in gaps]), np.mean([r[2] for r in gaps])



import sys
columns = ["model_name", "dataset", "lr", "bp_damping", "bp_max_iter", "hidden_dim", "n_layers", "seed", "grd_gap", "is_gap", "is2_gap"]
df = pd.DataFrame(columns=columns)

output = "test_is_Uedges_deb.csv"

for mp in sys.argv[1:]:
    desc, grd_gap, is_gap,is2_gap = test(mp)
    df1 = pd.DataFrame(data=[[desc["model_name"], desc["dataset"]["category"], desc["optimizer"]["lr"], desc["model"]["bp_damping"], desc["model"]["bp_max_iter"], desc["model"]["hidden_dim"], desc["model"]["n_layers"], desc["seed"], grd_gap, is_gap, is2_gap]], columns=columns)
    df = pd.concat([df, df1], ignore_index=True, axis=0)
    df.to_csv(output)

