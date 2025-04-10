import os
import pandas as pd
import networkx as nx
import numpy as np
import time
import sys

import random
random.seed(2024)
np.random.seed(2024)

from progress.bar import Bar

from bpropy.co.mis import MIS
from bpropy.clique_cover import clique_cover
from bpropy.utils.graph_utils import ordered_edges
from bpropy.utils.set_sort import lenlex_sort

def variable_features(pb, sol):
    """Compute the variable features by calling a solver to solve the continuous relaxation 
    of the problem
    """
    n = pb.nv()

    if sol.x.shape[0] != n:
        print(f"Solution shape {x.shape} does not match number of nodes {n}")

    # variable features
    nfeat = np.zeros((n, 10))
    nfeat[:,0] = pb.w
    nfeat[:,1] = (sol.x == 0)
    nfeat[:,2] = (sol.x == 1)
    nfeat[:,3] = ( (sol.x > 0) & (sol.x < 1))
    nfeat[:,4:8] = sol.varbasis_onehot
    nfeat[:,8] = sol.reduced_costs
    nfeat[:,9] = sol.x
    return nfeat

def eb_features(pb, cliques = None, solver="cbc", solution=None):
    if cliques is None:
        cliques = pb.C

    if solution is None:
        sol = pb.solve(solver=solver, relax_integrality=True)
    else:
        sol = solution

    nfeat = variable_features(pb, sol)

    # constraint features
    n_constr = len(cliques)

    # edge features
    efeat = np.ones(2 * sum(len(c) for c in cliques))
    
    cfeat = np.zeros((n_constr, 4))
    cfeat[:,0] = np.cos(2) * np.ones(n_constr)
    cfeat[:,1] = sol.slacks
    cfeat[:,2] = (sol.slacks == 0)
    cfeat[:,3] = sol.duals


    return pb.C, nfeat, efeat, cfeat

def cf_features(pb, cliques, solver="cbc"):
    sol = pb.solve(solver=solver, relax_integrality=True, cliques=cliques)

    nfeat = variable_features(pb, sol)

    n_constr = len(cliques)


    _cfeat = np.zeros((n_constr, 4))
    _cfeat[:,0] = np.cos(2) * np.ones(n_constr)
    _cfeat[:,1] = sol.slacks
    _cfeat[:,2] = (sol.slacks == 0)
    _cfeat[:,3] = sol.duals


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

    pb.set_cliques(edges, sort=False)
    return edges,nfeat,efeat,cfeat

def cp_features(pb, solver="cbc"):
    C = clique_cover(pb.g, randomize=False)
    sol = pb.solve(solver=solver, relax_integrality=True, cliques=C)

    _g = pb.g.copy()
    nx.set_node_attributes(_g, {i:int(1000 * sol.x[i]) for i in range(pb.nv())}, name="weight")

    MC,W = nx.max_weight_clique(_g)
    while W > 1000:
        C.append(sorted(MC))
        C = lenlex_sort(C)
        sol = pb.solve(solver=solver, relax_integrality=True, cliques=C)
        _g = pb.g.copy()
        nx.set_node_attributes(_g, {i:int(1000 * sol.x[i]) for i in range(pb.nv())}, name="weight")

        MC,W = nx.max_weight_clique(_g)

    _,nfeat,_,cfeat = eb_features(pb, solution=sol, cliques=C)
    efeat = np.ones(2 * sum(len(c) for c in C))

    #pb.C = C
    pb.set_cliques(C, sort=False)

    return pb.C,nfeat,efeat,cfeat

def get_features(pb, mode):
    if mode == "EB":
        C = np.array([np.array(list(e), dtype=int) for e in ordered_edges(pb.g)], dtype=object)
        #pb.C = C
        pb.set_cliques(C, sort=False)
        return eb_features(pb)
    if mode == "CC":
        C = np.array([np.array(c, dtype=int) for c in clique_cover(pb.g, randomize=False)], dtype=object)
        #pb.C = C
        pb.set_cliques(C, sort=False)
        _,nfeat,_,cfeat = eb_features(pb)
        efeat = np.ones(2 * sum(len(c) for c in C))
        return C,nfeat,efeat,cfeat
    if mode == "CF":
        #C = clique_cover(pb.g, randomize=False)
        C = np.array([np.array(c, dtype=int) for c in clique_cover(pb.g, randomize=False)], dtype=object)

        return cf_features(pb, C)
    if mode == "EC":
        C = [list(e) for e in ordered_edges(pb.g)] + clique_cover(pb.g, randomize=False)
        C = lenlex_sort(C)
        C = np.array([np.array(c, dtype=int) for c in C], dtype=object)
        #pb.C = C
        pb.set_cliques(C, sort=False)
        _,nfeat,_,cfeat = eb_features(pb)
        efeat = np.ones(2 * sum(len(c) for c in C))
        return C,nfeat,efeat,cfeat
    if mode == "CP":
        return cp_features(pb)
