import torch
import numpy as np
import networkx as nx
import random
random.seed(123)
np.random.seed(123)

import matplotlib.pyplot as plt

from bpropy.lbp import *
from bpropy.trw import *
from dopy.maxcut import *

def importance_sampling(theta, g, ebr_edges, edge_idx_map2, edge_p):
    edge_idx_map = {}
    idx_edge_map = {}
    for ei,e in enumerate(g.edges):
        edge_idx_map[e[0],e[1]] = ei
        edge_idx_map[e[1],e[0]] = ei
        idx_edge_map[ei] = e
    nV = g.number_of_nodes()
    theta_ = theta.clone()
    theta_min, theta_max = theta.min().item(), theta.max().item()
    queue = np.array(list(g.nodes))
    sol = np.zeros(g.number_of_nodes(), dtype=bool)
    while not len(queue)==0:
        #lspmm = edge_based_reparameterization_logspace(theta_.clone(), g, ebr_edges, edge_idx_map2, edge_p=edge_p, max_iter=100, patience=10)
        lspmm = edge_based_reparameterization2(theta_.clone(), g, ebr_edges, edge_idx_map2, edge_p=edge_p, max_iter=100, patience=10)

        # select the highest scoring variable that has not been picked yet
        var_scores = lspmm[1:2*nV+1:2].numpy()[queue]
        selection_idx = np.argmax(var_scores)
        selection = queue[selection_idx]
        if torch.rand(1) < var_scores[selection_idx] or True:
            sol[selection] = 1
            # remove the selection and neighbors from queue, and set the corresponding parameters
            theta_[2*selection] = theta_min
            theta_[2*selection+1] = theta_max
            queue = np.delete(queue, selection_idx)
            for neighbor in g.neighbors(selection):
                sol[neighbor] = 0
                theta_[2*neighbor] = theta_max
                theta_[2*neighbor+1] = theta_min
                eidx = edge_idx_map[selection,neighbor]
                e = idx_edge_map[eidx]
                if selection == e[0]:
                    theta_[2*nV+4*ei] = theta_min
                    theta_[2*nV+4*ei+1] = theta_max
                    theta_[2*nV+4*ei+2] = theta_min
                    theta_[2*nV+4*ei+3] = theta_min
                if selection == e[1]:
                    theta_[2*nV+4*ei] = theta_min
                    theta_[2*nV+4*ei+1] = theta_min
                    theta_[2*nV+4*ei+2] = theta_max
                    theta_[2*nV+4*ei+3] = theta_min
                for u,v in itertools.product(g.neighbors(selection), g.neighbors(selection)):
                    if u != v:
                        if g.has_edge(u,v):
                            eidx = edge_idx_map[u,v]
                            theta_[2*nV+4*ei] = theta_max
                            theta_[2*nV+4*ei+1:2*nV+4*ei+4] = theta_min

                queue = np.delete(queue, queue == neighbor)
        else:
            sol[selection] = 0
            theta_[2*selection] = theta_max
            theta_[2*selection+1] = theta_min
            queue = np.delete(queue, selection_idx)
            for neighbor in g.neighbors(selection):
                eidx = edge_idx_map[selection,neighbor]
                e = idx_edge_map[eidx]
                if selection == e[0]:
                    theta_[2*nV+4*ei+1] = theta_min
                    theta_[2*nV+4*ei+3] = theta_min
                if selection == e[1]:
                    theta_[2*nV+4*ei+2] = theta_min
                    theta_[2*nV+4*ei+3] = theta_min

        #print(len(queue), np.argwhere(sol>0).flatten(), selection, np.sum(sol))
    return sol


graph_sizes = np.arange(10, 200, 10)
N = 10

gap_values = []

for n in graph_sizes:
    gap_values.append([])
    for i in range(N):
        g = nx.generators.erdos_renyi_graph(n, 0.15)

        pb = MaxCut(g)

        t_opt = time.time()
        sol = pb.solve()
        t_opt = time.time() - t_opt

        print("Graph #", i, g)
        print("\tSolve time     = ", t_opt)
        print("\tOptimal value  = ", np.sum(sol.x > 0.5))


        print("\tImportance sampling")

        fg = create_factor_graph(g)
        n_ss = sufficient_statistics_dim(fg)
        
        # create the computation edges
        f2v_idx, f2v_subsum_idx,ss_idx = f2v_index_map(fg)
        # creating message passing edges
        f2v_edges,r_edges = f2v_subsum_indexes(fg, f2v_idx, ss_idx)
        f2v_edges = torch.from_numpy(f2v_edges)
        r_edges = torch.from_numpy(r_edges)
        # scatter indices
        q_edges = torch.from_numpy(v2f_indexes(fg, f2v_idx))
        m_edges = torch.from_numpy(marginal_indexes(fg, f2v_idx, ss_idx))
        vm_edges = torch.from_numpy(variable_marginal_indexes(fg, f2v_idx, ss_idx))
        
        # stack sufficient stats and v2f messages
        v2f = torch.zeros(len(f2v_idx), dtype=torch.double)
        theta = torch.zeros(len(ss_idx), dtype=torch.double)
        c = 0
        for i in range(pb.nv()):
            theta[c+1] = 1
            c += 2
        for e in pb.g.edges:
            theta[c+1] = 1. / len(list(g.neighbors(e[0])))
            theta[c+2] = 1. / len(list(g.neighbors(e[1])))
            theta[c+3] = -n
            c += 4
        theta = (theta - theta.min()) / (theta.max() - theta.min())

        # pseudo-max-marginals
        edge_idx_map2 = {}
        for ei,e in enumerate(g.edges):
            edge_idx_map2[e[0],e[1]] = 2*ei
            edge_idx_map2[e[1],e[0]] = 2*ei+1
        ebr_edges = EBREdges(g, edge_idx_map2)

        t_is = time.time()
        is_sol_best = np.zeros(n)
        sampling_step_c = 0
        noise_val = 0.1
        while time.time() - t_is < t_opt:
            is_sol = importance_sampling(theta.clone() * (noise_val * torch.randn(n_ss)).exp(), g, ebr_edges, edge_idx_map2, edge_p=torch.ones(g.number_of_edges()))
            if np.sum(is_sol) > np.sum(is_sol_best):
                is_sol_best = is_sol

            sampling_step_c += 1
        t_is = time.time() - t_is

        print("\tSolve time     = ", t_is)
        print("\tObjective      = ", np.sum(is_sol_best))
        print("\tSampling steps = ", sampling_step_c)
        print()

        gap = 100 * (np.sum(sol.x > 0.5) - np.sum(is_sol_best)) / np.sum(sol.x > 0.5)
        gap_values[-1].append( gap )

    fig = plt.figure()
    plt.boxplot(gap_values, positions=graph_sizes[:len(gap_values)], widths=5)
    plt.ylabel("Optimality gap (%)")
    plt.xlabel("|V|")
    plt.savefig("maxcut_importance_sampling_out.png")
    plt.close()





