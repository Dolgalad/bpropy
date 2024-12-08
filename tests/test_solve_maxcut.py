import torch
import numpy as np
import networkx as nx
import random
random.seed(123)
np.random.seed(123)

from bpropy.lbp import *
from bpropy.trw import *
from dopy.maxcut import *


# support graph
n = 100
g = nx.generators.erdos_renyi_graph(n, 0.15)

# Solve ILP
pb = MaxCut(g)
sol = pb.solve()
print("Optimal solution : ", np.argwhere(sol.x>0).flatten())
print("Optimal value    : ", np.sum(sol.x))

# Solve using inference
fg = create_factor_graph(g)
n_ss = sufficient_statistics_dim(fg)

# create the computation edges
f2v_idx, f2v_subsum_idx,ss_idx = f2v_index_map(fg)
# creating message passing edges
f2v_edges,r_edges = f2v_subsum_indexes(fg, f2v_idx, ss_idx)
f2v_edges = torch.from_numpy(f2v_edges)
r_edges = torch.from_numpy(r_edges)
#f2v_edges2,r_edges2 = f2v_subsum_indexes2(fg)
#f2v_edges2 = torch.from_numpy(f2v_edges2)
#r_edges2 = torch.from_numpy(r_edges2)
# scatter indices
q_edges = torch.from_numpy(v2f_indexes(fg, f2v_idx))
#q_edges2 = torch.from_numpy(v2f_indexes2(fg))
m_edges = torch.from_numpy(marginal_indexes(fg, f2v_idx, ss_idx))
#m_edges2 = torch.from_numpy(marginal_indexes2(fg))
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
theta = (theta - theta.min())# / (theta.max() - theta.min())

beliefs = loopy_belief_propagation(theta.clone(), v2f, fg, f2v_edges, f2v_subsum_idx, r_edges, q_edges, m_edges, max_iter=100, tol=1e-6)

print(beliefs[:2*n])

# simply take the argmax on variable beliefs
vb_argmax_sol = beliefs[:2*n].reshape(n,2).argmax(axis=1)
print("Variable belief argmax solution : ", np.argwhere(vb_argmax_sol>0).flatten())
print("                          value : ", np.sum(vb_argmax_sol.numpy()))

# pseudo-max-marginals
edge_idx_map2 = {}
for ei,e in enumerate(g.edges):
    edge_idx_map2[e[0],e[1]] = 2*ei
    edge_idx_map2[e[1],e[0]] = 2*ei+1
ebr_edges = EBREdges(g, edge_idx_map2)
#lspmm = edge_based_reparameterization_logspace(theta.clone(), g, ebr_edges, edge_idx_map2, edge_p=torch.ones(g.number_of_edges()), max_iter=1000, patience=10)
lspmm = edge_based_reparameterization2(theta.clone(), g, ebr_edges, edge_idx_map2, edge_p=torch.ones(g.number_of_edges(),dtype=torch.double), max_iter=1000, patience=10)

#print(lspmm)
lspmm_sol = lspmm[:2*n].reshape(n,2).argmax(axis=1).numpy()
print("Log-pseudo-max-marginals solution : ", np.argwhere(lspmm_sol>0).flatten())
print("                          value   : ", np.sum(lspmm_sol))
print("                          valid   : ", maxcut_solution_validity(pb, lspmm_sol))

print(lspmm[1:2*n+1:2])
greedy_lspmm_sol = maxcut_greedy(pb, lspmm[1:2*n+1:2].numpy())
print(np.argwhere(greedy_lspmm_sol>0).flatten())
print(np.sum(greedy_lspmm_sol))

import matplotlib.pyplot as plt

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
        print(len(queue), np.argwhere(sol>0).flatten(), selection, np.sum(sol))
    return sol

print("Importance sampling")
is_sol = importance_sampling(theta.clone(), g, ebr_edges, edge_idx_map2, edge_p=torch.ones(g.number_of_edges()))
print("\tsolution = ", is_sol)
print("\tvalue    = ", np.sum(is_sol))
print("\tvalid    = ", maxcut_solution_validity(pb, is_sol))


gs_opt = []
gs_is_opt = []
graph_sizes=np.arange(10, 500, 10)
for n in graph_sizes:
    opt = []
    is_opt = []
    for it in range(100):
        g = nx.generators.erdos_renyi_graph(n, 0.15)
        # Solve ILP
        pb = MaxCut(g)
        sol = pb.solve()
    
        # Belief propagation
        fg = create_factor_graph(g)
        n_ss = sufficient_statistics_dim(fg)
        
        # create the computation edges
        f2v_idx, f2v_subsum_idx,ss_idx = f2v_index_map(fg)
        # creating message passing edges
        f2v_edges,r_edges = f2v_subsum_indexes(fg, f2v_idx, ss_idx)
        f2v_edges = torch.from_numpy(f2v_edges)
        r_edges = torch.from_numpy(r_edges)
        #f2v_edges2,r_edges2 = f2v_subsum_indexes2(fg)
        #f2v_edges2 = torch.from_numpy(f2v_edges2)
        #r_edges2 = torch.from_numpy(r_edges2)
        # scatter indices
        q_edges = torch.from_numpy(v2f_indexes(fg, f2v_idx))
        #q_edges2 = torch.from_numpy(v2f_indexes2(fg))
        m_edges = torch.from_numpy(marginal_indexes(fg, f2v_idx, ss_idx))
        #m_edges2 = torch.from_numpy(marginal_indexes2(fg))
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
        #lspmm = edge_based_reparameterization_logspace(theta.clone(), g, ebr_edges, edge_idx_map2, edge_p=torch.ones(g.number_of_edges()), max_iter=1000, patience=10)
        lspmm = edge_based_reparameterization2(theta.clone(), g, ebr_edges, edge_idx_map2, edge_p=torch.ones(g.number_of_edges()), max_iter=1000, patience=10)
    
    
        is_sol = importance_sampling(theta.clone(), g, ebr_edges, edge_idx_map2, edge_p=torch.ones(g.number_of_edges()))
    
    
        greedy_lspmm_sol = maxcut_greedy(pb, lspmm[1:2*n+1:2].numpy())
    
        print(it, "opt val = ", np.sum(sol.x))
        print(it, "grd val = ", np.sum(greedy_lspmm_sol))
        print(it, "ims val = ", np.sum(is_sol))
    
        nopt = 100 * (np.sum(sol.x) - np.sum(greedy_lspmm_sol)) / np.sum(sol.x)
        isopt = 100 * (np.sum(sol.x) - np.sum(is_sol)) / np.sum(sol.x)
        opt.append(nopt)
        is_opt.append(isopt)
    
        print(np.mean(opt))
        
        fig = plt.figure()
        plt.boxplot([opt, is_opt], tick_labels=["lpmm","is"])
        plt.savefig(f"importance_sampling_n{n}.png")
        plt.close()
    gs_opt.append(opt)
    gs_is_opt.append(is_opt)
    fig,ax = plt.subplots(2,1, sharex=True)
    ax[0].boxplot(gs_opt, positions=graph_sizes[:len(gs_opt)])
    ax[0].set_ylabel("LPMM opt (%)")
    ax[1].boxplot(gs_is_opt, positions=graph_sizes[:len(gs_opt)])
    ax[1].set_ylabel("IS opt (%)")
    ax[1].set_xlabel("|V|")

    plt.savefig(f"importance_sampling_gs.png")
    plt.close()


