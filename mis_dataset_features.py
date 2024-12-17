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
    if mis_formulation == "CC":
        C = clique_covering_3(g, randomize=True)
        md = cplex_model(pb, continuous_relaxation=True, cliques=C)
    else:
        md = cplex_model(pb, continuous_relaxation=True)
    varlist = [md.get_var_by_name(f"x{i}") for i in range(g.number_of_nodes())]
    sol = md.solve()
    x = np.array([sol.get_value(var) for var in varlist])
    if x.shape[0] != n:
        print(f"Solution shape {x.shape} does not match number of nodes {n}")

    # variable basis status
    varbasis = np.array([bstt.value for bstt in md.var_basis_statuses(varlist)])
    varbasis_onehot = np.zeros((varbasis.size, 4))
    varbasis_onehot[np.arange(varbasis.size), varbasis - 1] = 1

    # variable features
    nfeat = np.zeros((g.number_of_nodes(), 9))
    nfeat[:,0] = pb.w
    nfeat[:,1] = (x == 0)
    nfeat[:,2] = (x == 1)
    nfeat[:,3] = ( (x > 0) & (x < 1))
    nfeat[:,4:8] = varbasis_onehot
    nfeat[:,8] = np.array(sol.get_reduced_costs(varlist))
    identical_node_features += np.all( (nfeat - np.mean(nfeat, axis=0)) == 0)


    # constraint features
    constraints = list(md.iter_constraints())

    # edge features
    efeat = np.ones(2 * len(constraints))

    
    slacks = np.array(md.slack_values(constraints))
    duals = np.array(md.dual_values(constraints))
    cfeat = np.zeros((len(constraints), 4))
    cfeat[:,0] = np.cos(2) * np.ones(len(constraints))
    cfeat[:,1] = slacks
    cfeat[:,2] = (slacks == 0)
    cfeat[:,3] = duals
    identical_constraint_features += np.all( np.abs(cfeat - np.mean(cfeat, axis=0)) <= 1e-8)


    with open(os.path.join(instance_path, "node_features.npy"), "wb") as f:
        np.save(f, nfeat)
    with open(os.path.join(instance_path, "edge_features.npy"), "wb") as f:
        np.save(f, efeat)
    with open(os.path.join(instance_path, "constraint_features.npy"), "wb") as f:
        np.save(f, cfeat)

    

    bar.next()
bar.finish()
