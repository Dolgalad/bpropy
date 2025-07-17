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

from bpropy.co.scp import SCP
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

def get_features(pb, solver="cbc", solution=None):
    if solution is None:
        import time
        t0 = time.time()
        sol = pb.solve(solver=solver, relax_integrality=True)
        print("solv t ", time.time() - t0)
    else:
        sol = solution

    nfeat = variable_features(pb, sol)

    # constraint features
    n_constr = len(pb.constraints())

    # edge features
    efeat = np.ones(2 * int(sum(c.A.sum() for c in pb.constraints())))
    
    cfeat = np.zeros((n_constr, 4))
    cfeat[:,0] = np.cos(2) * np.ones(n_constr)
    cfeat[:,1] = sol.slacks
    cfeat[:,2] = (sol.slacks == 0)
    cfeat[:,3] = sol.duals

    return nfeat, efeat, cfeat


