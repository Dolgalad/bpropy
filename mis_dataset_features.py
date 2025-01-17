import os
from dopy.mis import *

import pandas as pd
import networkx as nx
import numpy as np
import time
import sys

import random
random.seed(2024)
np.random.seed(2024)

from progress.bar import Bar

from clique_covering import clique_covering_3
from bipartite_graph import *

def ordered_edges(g):
    nV = nx.number_of_nodes(g)
    adj = nx.to_numpy_array(g, nodelist=range(nV))
    if g.is_directed():
        for (s,t) in zip(*np.where(adj > 0)):
            yield (s,t)
    else:
        for (s,t) in zip(*np.where(adj > 0)):
            if s<t:
                yield (s,t)


def variable_features(pb, md):
    varlist = [md.get_var_by_name(f"x{i}") for i in range(pb.g.number_of_nodes())]
    sol = md.solve()
    x = np.array([sol.get_value(var) for var in varlist])
    if x.shape[0] != n:
        print(f"Solution shape {x.shape} does not match number of nodes {n}")

    # variable basis status
    varbasis = np.array([bstt.value for bstt in md.var_basis_statuses(varlist)])
    varbasis_onehot = np.zeros((varbasis.size, 4))
    varbasis_onehot[np.arange(varbasis.size), varbasis - 1] = 1

    # variable features
    nfeat = np.zeros((pb.nv(), 9))
    nfeat[:,0] = pb.w
    nfeat[:,1] = (x == 0)
    nfeat[:,2] = (x == 1)
    nfeat[:,3] = ( (x > 0) & (x < 1))
    nfeat[:,4:8] = varbasis_onehot
    nfeat[:,8] = np.array(sol.get_reduced_costs(varlist))
    return nfeat

def eb_features(pb, md):
    C = [list(e) for e in ordered_edges(pb.g)]

    nfeat = variable_features(pb, md)

    # constraint features
    constraints = list(md.iter_constraints())

    # edge features
    efeat = np.ones(2 * sum(len(c) for c in C))
    
    slacks = np.array(md.slack_values(constraints))
    duals = np.array(md.dual_values(constraints))
    cfeat = np.zeros((len(constraints), 4))
    cfeat[:,0] = np.cos(2) * np.ones(len(constraints))
    cfeat[:,1] = slacks
    cfeat[:,2] = (slacks == 0)
    cfeat[:,3] = duals

    return C, nfeat, efeat, cfeat

def cf_features(pb, cliques, md):
    nfeat = variable_features(pb, md)


    constraints = list(md.iter_constraints())

    slacks = np.array(md.slack_values(constraints))
    duals = np.array(md.dual_values(constraints))
    _cfeat = np.zeros((len(constraints), 4))
    _cfeat[:,0] = np.cos(2) * np.ones(len(constraints))
    _cfeat[:,1] = slacks
    _cfeat[:,2] = (slacks == 0)
    _cfeat[:,3] = duals


    edges = list([e for e in ordered_edges(pb.g)])
    efeat = np.ones(4 * len(edges))

    cfeat = np.zeros((pb.ne(), 4))
    for i in range(pb.ne()):
        a = 0
        for j,c in enumerate(cliques):
            if all(u in c for u in edges[i]):
                cfeat[i,:] += _cfeat[j,:]
                a += 1
        cfeat[i,:] /= a
    return edges,nfeat,efeat,cfeat

def cp_features(pb):
    C = clique_covering_3(pb.g, randomize=False)
    md = cplex_model(pb, continuous_relaxation=True, cliques=C)
    varlist = [md.get_var_by_name(f"x{i}") for i in range(pb.g.number_of_nodes())]
    sol = md.solve()
    x = np.array([sol.get_value(var) for var in varlist])
    _g = pb.g.copy()
    nx.set_node_attributes(_g, {i:int(1000 * x[i]) for i in range(pb.nv())}, name="weight")

    MC,W = nx.max_weight_clique(_g)
    while W > 1000:
        C.append(MC)
        md = cplex_model(pb, continuous_relaxation=True, cliques=C)
        varlist = [md.get_var_by_name(f"x{i}") for i in range(pb.g.number_of_nodes())]
        sol = md.solve()
        x = np.array([sol.get_value(var) for var in varlist])
        _g = pb.g.copy()
        nx.set_node_attributes(_g, {i:int(1000 * x[i]) for i in range(pb.nv())}, name="weight")

        MC,W = nx.max_weight_clique(_g)

    _,nfeat,_,cfeat = eb_features(pb, md)
    efeat = np.ones(2 * sum(len(c) for c in C))

    return C,nfeat,efeat,cfeat

def get_features(pb, mode):
    if mode == "EB":
        C = [list(e) for e in ordered_edges(pb.g)]
        md = cplex_model(pb, continuous_relaxation=True)
        return eb_features(pb, md)
    if mode == "CC":
        C = clique_covering_3(pb.g, randomize=False)
        md = cplex_model(pb, continuous_relaxation=True, cliques=C)
        return eb_features(pb, md)
    if mode == "CF":
        C = clique_covering_3(pb.g, randomize=False)
        md = cplex_model(pb, continuous_relaxation=True, cliques=C)
        return cf_features(pb, C, md)
    if mode == "EC":
        C = [list(e) for e in ordered_edges(pb.g)] + clique_covering_3(pb.g, randomize=False)
        md = cplex_model(pb, continuous_relaxation=True, cliques=C)
        return eb_features(pb, md)
    if mode == "CP":
        return cp_features(pb)

dataset_path = sys.argv[1]

mis_formulation = sys.argv[2]

identical_node_features = 0
identical_constraint_features = 0

instance_ids = sorted(list(os.listdir(dataset_path)))
bar = Bar("Extracting features: ", max=len(instance_ids))
for instance_id in instance_ids:
    instance_path = os.path.join(dataset_path, instance_id)
    graph_file_path = os.path.join(instance_path, "graph.adjlist")
    # load graph
    g = nx.read_adjlist(graph_file_path)
    nmap = {str(i): i for i in range(g.number_of_nodes())}
    g = nx.relabel_nodes(g, nmap, copy=False)
    n = g.number_of_nodes()

    pb = MIS(g)
    C,nfeat,efeat,cfeat = get_features(pb, mis_formulation)

    # bipartite graph
    bpg,bpg_feat = bipartite_graph(pb, cliques=C)
    if 2*bpg.number_of_edges() != efeat.shape[0]:
        print("Bad edges feature shape ")
        exit()
    bipartite_graph_file_path = os.path.join(instance_path, "bipartite.adjlist")
    nx.write_adjlist(bpg, bipartite_graph_file_path)

    with open(os.path.join(instance_path, "node_features.npy"), "wb") as f:
        np.save(f, nfeat)
    with open(os.path.join(instance_path, "edge_features.npy"), "wb") as f:
        np.save(f, efeat)
    with open(os.path.join(instance_path, "constraint_features.npy"), "wb") as f:
        np.save(f, cfeat)

    bar.next()
bar.finish()
