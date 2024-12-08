import os
from dopy.maxcut import *

import pandas as pd
import networkx as nx
import numpy as np
import time
import sys

from progress.bar import Bar

dataset_path = sys.argv[1]

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

    pb = MaxCut(g)
    md = cplex_model(pb, continuous_relaxation=True)
    varlist = [md.get_var_by_name(f"x{i}") for i in range(g.number_of_nodes())]
    sol = md.solve()
    df = sol.as_df()
    x = df["value"].to_numpy()

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

    # edge features
    efeat = np.ones(2 * g.number_of_edges())

    # constraint features
    constraints = list(md.iter_constraints())
    
    slacks = np.array(md.slack_values(constraints))
    duals = np.array(md.dual_values(constraints))
    cfeat = np.zeros((g.number_of_edges(), 4))
    cfeat[:,0] = np.cos(2) * np.ones(g.number_of_edges())
    cfeat[:,1] = slacks
    cfeat[:,2] = (slacks == 0)
    cfeat[:,3] = duals
    identical_constraint_features += np.all( np.abs(cfeat - np.mean(cfeat, axis=0)) <= 1e-8)

    print("Node features identical       : ", np.all( (nfeat - np.mean(nfeat, axis=0)) == 0))
    print("Constraint features identical : ", np.all( np.abs(cfeat - np.mean(cfeat, axis=0)) <= 1e-8))

    print(np.max( cfeat - np.mean(cfeat, axis=0) ) )
    print(np.mean(cfeat, axis=0))
    print(np.unique(cfeat[:,0]))
    print(np.unique(cfeat[:,1]))
    print(np.unique(cfeat[:,2]))
    print(np.unique(cfeat[:,3]))

    with open(os.path.join(instance_path, "node_features.npy"), "wb") as f:
        np.save(f, nfeat)
    with open(os.path.join(instance_path, "edge_features.npy"), "wb") as f:
        np.save(f, efeat)
    with open(os.path.join(instance_path, "constraint_features.npy"), "wb") as f:
        np.save(f, cfeat)

    

    bar.next()
bar.finish()

print("Node features identical       : ", np.all( (nfeat - np.mean(nfeat, axis=0)) == 0))
print("Constraint features identical : ", np.all( (cfeat - np.mean(cfeat, axis=0)) == 0))




g = nx.generators.random_graphs.erdos_renyi_graph(1000, 0.15)

pb = MaxCut(g)

md = cplex_model(pb, continuous_relaxation=True)

t0 = time.time()
sol = md.solve()
t0 = time.time() - t0

print("solved in ", t0)
print(sol)
