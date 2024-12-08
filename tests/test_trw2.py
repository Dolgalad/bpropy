"""
Generate random trees, solve the MaxCut problem to optimality, with greedy heuristic, with Loopy
Belief Propagation.
"""
import os
import itertools
import random
random.seed(123)
import numpy as np
np.random.seed(123)

from bpropy.lbp import *
from bpropy.trw import *
from dopy.maxcut import *

import networkx as nx

from networkx.generators.trees import random_tree

import time

import matplotlib.pyplot as plt
plt.ion()

def maxcut_solution_to_sufficient_statistics(pb, x):
    fg = create_factor_graph(pb.g)
    nss = sufficient_statistics_dim(fg)
    ss = np.zeros(nss)
    c=0
    for i in range(pb.nv()):
        ss[c+(x[i]==1)] = 1
        c += 2
    for e in pb.g.edges:
        if x[e[0]]==0 and x[e[1]]==0:
            ss[c] = 1
        if x[e[0]]==0 and x[e[1]]==1:
            ss[c+2] = 1
        if x[e[0]]==1 and x[e[1]]==0:
            ss[c+1] = 1
        if x[e[0]]==1 and x[e[1]]==1:
            ss[c+3] = 1
        c += 4
    return ss

graph_sizes = np.arange(40, 180, 5)
#generators = [random_tree]#, nx.generators.ladder_graph]
#generators = [nx.generators.ladder_graph]
generators = [lambda n: nx.generators.erdos_renyi_graph(n, 0.15)]
N = 10

fig_c = 0
output_dir = "maxcut_tests"
os.makedirs(output_dir, exist_ok=True)


gaps = []
opt_time = []
ms_time = []
opt_speedup = []

print("hhhhhhhhhhhhhh")

for (n,func) in itertools.product(graph_sizes, generators):
    current_gaps = []
    current_ms_time = []
    current_opt_time = []
    current_opt_speedup = []
    for j in range(N):
        print(f"Random graph {j}: n = {n}, generator = {func.__name__}")
        g = func(n)
        print(g)
        print("\tnv = ", nx.number_of_nodes(g))
        print("\tne = ", nx.number_of_edges(g))

        # node and edges index maps
        node_idx_map = {n:i for i,n in enumerate(g.nodes)}
        edge_idx_map = {tuple(sorted(e)):i for i,e in enumerate(g.edges)}

        # compute familly of STs and check if all edges are covered
        trees = []
        spanning_tree_iterator = nx.algorithms.tree.mst.SpanningTreeIterator(g)
        tree_edge_counts = {tuple(sorted(e)):0 for e in g.edges}
        print("Starting computing STs")
        st_t0 = time.time()
        nx.set_edge_attributes(g, 1, "weight")
        while np.any(np.array(list(tree_edge_counts.values())) == 0):
            st = nx.algorithms.tree.mst.minimum_spanning_tree(g)
            trees.append(st)
            # update the edge counters
            for e in st.edges:
                tree_edge_counts[tuple(sorted(e))] += 1
                g[e[0]][e[1]]["weight"] += 1
        if len(trees) == 0:
            continue

        # TODO: adding same tree twice
        #trees.append(trees[0])


        st_t0 = time.time() - st_t0
        print("\tdone in ", st_t0)
        print("\tnumber of trees: ", len(trees))

        # tree edge mask
        tree_edge_mask = torch.zeros((len(trees), g.number_of_edges()), dtype=torch.bool)
        for ei,e in enumerate(g.edges):
            for ti,tree in enumerate(trees):
                if tree.has_edge(*e):
                    tree_edge_mask[ti,ei] = 1
        
        tree_edge_probability = {k:v/len(trees) for k,v in tree_edge_counts.items()}
        tree_edge_probability_ = torch.from_numpy(np.array([tree_edge_probability[e] for e in g.edges]))

        # solve ILP formulation of problem
        pb = MaxCut(g)
        t0 = time.time()
        sol = pb.solve()
        t0 = time.time() - t0
        current_opt_time.append(t0)
        t0_opt = t0
        print("\tExact solver")
        #print("\t\tSolution: ", sol.x)
        print("\t\tval: ", np.sum(sol.x))
        print("\t\tSolve t: ", t0)
        
        # Tree-reweighted max-sum loopy belief propagation
        print("\tTRW")
        trw_t0 = time.time()
        fg = create_factor_graph(g)
        n_ss = sufficient_statistics_dim(fg)
        print("\t\tnum sufficient statistics = ", n_ss)

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
        #theta = theta - theta.min()
        #theta = 10*theta + torch.rand(theta.size(0))

        # loop over trees

        tree_params = [theta.clone() for _ in trees]
        #for tidx,tree in enumerate(trees):
        #    tree_theta = theta.clone()
        #    for e in g.edges:
        #        if not tree.has_edge(*e):
        #            # set corresponding parameters to 0
        #            eidx = edge_idx_map[tuple(sorted(e))]
        #            tree_theta[2*n + 4*eidx:2*n + 4*eidx + 4] = 0
        #    tree_params.append(tree_theta)

        trw_done = False
        trw_it = 0
        sols = torch.zeros(len(trees), n, dtype=torch.int)
        trw_marginals = torch.zeros(len(trees), n_ss)
        trw_theta = theta.clone()
        tmp_trw_theta = theta.clone()
        trw_max_iter = 100
        all_time_best_val = -10
        all_time_best_sol = None

        edge_idx_map2 = {}
        for ei,e in enumerate(g.edges):
            edge_idx_map2[e[0],e[1]] = 2*ei
            edge_idx_map2[e[1],e[0]] = 2*ei+1
        ebr_edges = EBREdges(g, edge_idx_map2)

        # stack the ebr edges, one for each tree
        ebr_edges_stacked = stack_ebr_edges([EBREdges(g, edge_idx_map2) for _ in range(len(trees))])

        print("Starting TRW")

        while not trw_done:
            tree_theta_ = trw_theta.unsqueeze(0).repeat(len(trees),1)
            tem = tree_edge_mask.repeat_interleave(4,dim=1)
            tree_theta_[:,2*n:][tem.logical_not()] = 0
            tree_theta_[:,2*n:] /= tree_edge_probability_.repeat_interleave(4).unsqueeze(0).repeat(len(trees),1)
            #print("tree_theta_ = \n", tree_theta_)

            trw_m_stacked = edge_based_reparameterization2(torch.flatten(tree_theta_), g, ebr_edges_stacked, edge_idx_map2, edge_p=tree_edge_probability_.repeat(len(trees)))
            ls = edge_based_reparameterization_logspace(tree_theta_[0,:], trees[0], ebr_edges, edge_idx_map2, edge_p=tree_edge_probability_)

            print("Optimal solution : ", sol.x, np.sum(sol.x))
            print("\t", np.argwhere(sol.x>0).flatten())
            tmp_sol = maxcut_greedy(pb, sol.x)
            print("Greedy on sol    : ", tmp_sol, np.sum(tmp_sol))
            print("\t", np.argwhere(tmp_sol>0).flatten())
            print("Before greedy")
            solls = maxcut_greedy(pb, ls[:2*n].reshape(n,2)[:,1].numpy())
            print("MMM = ", solls)
            print("\t", np.argwhere(solls>0).flatten())
            print("\tval = ", np.sum(solls))


            trw_marginals = trw_m_stacked.reshape(len(trees), n_ss)

            energy_vals = []
            for tidx in range(len(trees)):
                sols[tidx,:] = trw_marginals[tidx,:2*n].reshape(n,2).argmax(axis=1).clone()
                tmp_atbs = maxcut_greedy(pb, sols[tidx,:].numpy())
                if np.sum(tmp_atbs) > all_time_best_val:
                    all_time_best_val = np.sum(tmp_atbs)
                    all_time_best_sol = tmp_atbs
                solx = maxcut_solution_to_sufficient_statistics(pb, sols[tidx,:].numpy())
                #print(trw_it, " E = ", np.dot(solx, tree_theta_[tidx,:].numpy()))
                energy_vals.append(np.dot(solx, tree_theta_[tidx,:].numpy()))
            energy_vals = torch.tensor(energy_vals)
            #print()

            if len(trees)>1:
                if np.all([sols[0,:].equal(sols[i,:]) for i in range(1,len(trees))]):
                    trw_done = True
            else:
                trw_done = True
            # update trw theta
            trw_theta[:2*n] = torch.mean((trw_marginals[:,:2*n]+1e-12).log(), axis=0)
            for ei,e in enumerate(g.edges):
                # indexes of trees containing this edge
                tree_indexes = torch.from_numpy(np.array([i for i,T in enumerate(trees) if T.has_edge(*e)]))
                tmp_trw_theta[2*n + 4*ei] = torch.sum((trw_marginals[tree_indexes,2*n+4*ei]+1e-12).log() - ((trw_marginals[tree_indexes,2*e[0]]+1e-12).log() + (trw_marginals[tree_indexes,2*e[1]]+1e-12).log())) / len(trees)
                tmp_trw_theta[2*n + 4*ei + 1] = torch.sum((trw_marginals[tree_indexes,2*n+4*ei+1]+1e-12).log() - ((trw_marginals[tree_indexes,2*e[0]+1]+1e-12).log() + (trw_marginals[tree_indexes,2*e[1]]+1e-12).log())) / len(trees)
                tmp_trw_theta[2*n + 4*ei + 2] = torch.sum((trw_marginals[tree_indexes,2*n+4*ei+2]+1e-12).log() - ((trw_marginals[tree_indexes,2*e[0]]+1e-12).log() + (trw_marginals[tree_indexes,2*e[1]+1]+1e-12).log())) / len(trees)
                tmp_trw_theta[2*n + 4*ei + 3] = torch.sum((trw_marginals[tree_indexes,2*n+4*ei+3]+1e-12).log() - ((trw_marginals[tree_indexes,2*e[0]+1]+1e-12).log() + (trw_marginals[tree_indexes,2*e[1]+1]+1e-12).log())) / len(trees)


            if torch.any(torch.isnan(trw_theta)) or torch.any(torch.isinf(trw_theta)):
                print("Nan or Inf")
                break
            if (tmp_trw_theta - trw_theta).abs().max() < 1e-6:
                print("No improvement")
                break
            if torch.all(energy_vals == energy_vals[0]):
                print("Tree agreement")
                break
            trw_it += 1
            trw_theta = tmp_trw_theta.clone()
            if trw_it >= trw_max_iter:
                print("max iter")
                break

        best_tree_val = -10
        best_sol = None
        for i in range(len(trees)):
            tree_sol = maxcut_greedy(pb, sols[i,:].numpy())
            if np.sum(tree_sol) > best_tree_val:
                best_tree_val = np.sum(tree_sol)
                best_sol = tree_sol

        trw_t0 = time.time() - trw_t0
        print("\t\tbest sol val: ", np.sum(all_time_best_sol))
        print("\t\ttime : ", trw_t0)

        gap_val = 100 * ( np.sum(sol.x) - np.sum(all_time_best_sol) ) / np.sum(sol.x)



        current_gaps.append(gap_val)
        current_ms_time.append(trw_t0)
        current_opt_speedup.append(t0_opt / trw_t0)

    gaps.append(np.mean(current_gaps))
    opt_time.append(np.mean(current_opt_time))
    ms_time.append(np.mean(current_ms_time))
    opt_speedup.append(np.mean(current_opt_speedup))

    fig,ax = plt.subplots(3,1)
    ax[0].plot(graph_sizes[:len(gaps)], gaps)
    ax[0].set_ylabel("Gap (%)")
    ax[0].set_xlabel("|V|")
    ax[1].plot(graph_sizes[:len(opt_time)], opt_time, label="opt")
    ax[1].plot(graph_sizes[:len(opt_time)], ms_time, label="ms")
    ax[1].set_ylabel("Time (s)")
    ax[1].set_xlabel("|V|")
    plt.legend()
    ax[2].plot(graph_sizes[:len(opt_time)], opt_speedup)
    plt.savefig(f"maxcut_tests/{fig_c}.png")
    plt.close()
    fig_c += 1

