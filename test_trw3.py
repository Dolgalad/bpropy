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
from dopy.maxcut import *

import networkx as nx

from networkx.generators.trees import random_tree

import time

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

graph_sizes = np.arange(10, 60, 10)
#generators = [random_tree]#, nx.generators.ladder_graph]
#generators = [nx.generators.ladder_graph]
generators = [lambda n: nx.generators.erdos_renyi_graph(n, 0.15)]
N = 10

output_dir = "maxcut_tests"
os.makedirs(output_dir, exist_ok=True)

gaps = []
opt_time = []
ms_time = []
opt_speedup = []

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
        st_t0 = time.time() - st_t0
        print("\tdone in ", st_t0)
        print("\tnumber of trees: ", len(trees))
        
        tree_edge_probability = {k:v/len(trees) for k,v in tree_edge_counts.items()}

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
        v2f = torch.zeros(len(f2v_idx))
        theta = torch.zeros(len(ss_idx), requires_grad=False)
        c = 0
        for i in range(pb.nv()):
            theta[c+1] = 1
            c += 2
        for e in pb.g.edges:
            theta[c+1] = 1. / len(list(g.neighbors(e[0])))
            theta[c+2] = 1. / len(list(g.neighbors(e[1])))
            theta[c+3] = -1
            c += 4
        #theta = theta - theta.min() + 1

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
        while not trw_done:
            #print("TRW it = ", trw_it)

            for tidx,tree in enumerate(trees):
                tree_theta = trw_theta.clone()
                for e in g.edges:
                    eidx = edge_idx_map[tuple(sorted(e))]
                    if not tree.has_edge(*e):
                        # set corresponding parameters to 0
                        tree_theta[2*n + 4*eidx:2*n + 4*eidx + 4] = 0
                    else:
                        tree_theta[2*n + 4*eidx:2*n + 4*eidx + 4] *= (1./tree_edge_probability[tuple(sorted(e))]) 

                tree_params[tidx] = tree_theta.clone()

            for tidx,tree in enumerate(trees):
                trw_marginals[tidx,:] = loopy_belief_propagation(tree_params[tidx], v2f.clone(), fg, f2v_edges, f2v_subsum_idx, r_edges, q_edges, m_edges, max_iter=100, tol=1e-6).clone()
                sols[tidx,:] = trw_marginals[tidx,:2*n].reshape(n,2).argmax(axis=1).clone()
                tmp_atbs = maxcut_greedy(pb, sols[tidx,:].numpy())
                if np.sum(tmp_atbs) > all_time_best_val:
                    all_time_best_val = np.sum(tmp_atbs)
                    all_time_best_sol = tmp_atbs


            if len(trees)>1:
                #print(sols)
                #print([np.dot(tree_params[0].numpy(), maxcut_solution_to_sufficient_statistics(pb,sols[i,:])) for i in range(len(trees))])
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



                #trw_theta[2*n + 4*ei + 1] = torch.mean((trw_marginals[:,2*n+4*ei+1]+1e-12).log() - (trw_marginals[:,2*e[0]]+1e-12).log() - (trw_marginals[:,2*e[1]+1]+1e-12).log())
                #trw_theta[2*n + 4*ei + 2] = torch.mean((trw_marginals[:,2*n+4*ei+2]+1e-12).log() - (trw_marginals[:,2*e[0]+1]+1e-12).log() - (trw_marginals[:,2*e[1]]+1e-12).log())
                #trw_theta[2*n + 4*ei + 3] = torch.mean((trw_marginals[:,2*n+4*ei+3]+1e-12).log() - (trw_marginals[:,2*e[0]+1]+1e-12).log() - (trw_marginals[:,2*e[1]+1] + 1e-12).log())
                #print(e, trw_theta[2*n+4*ei:2*n+4*ei+4])
            #trw_theta = trw_theta - trw_theta.min()

            if torch.any(torch.isnan(trw_theta)) or torch.any(torch.isinf(trw_theta)):
                print("Nan or Inf")
                break
            if (tmp_trw_theta - trw_theta).abs().max() < 1e-6:
                print("No improvement")
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

        gap_val = 100 * ( np.sum(sol.x) - np.sum(all_time_best_sol) ) / np.sum(sol.x)

        # solve ILP forlmulation by adding a lower bound value to the objective
        t00 = time.time()
        sol = pb.solve(obj_ub = -np.sum(all_time_best_sol))
        t00 = time.time() - t00
        print("Optimal solve time with objective lower bound : ", t00)
        print("Optimal solve time with objective lower bound speedup : ", t0_opt / t00)
        print("Optimal solve time with objective lower bound : ", np.sum(sol.x))
        print("Optimal solve time with objective lower bound : ", np.sum(all_time_best_sol))



        trw_t0 = time.time() - trw_t0
        current_gaps.append(gap_val)
        current_ms_time.append(trw_t0)
        current_opt_speedup.append(t0_opt / t00)

    gaps.append(np.mean(current_gaps))
    opt_time.append(np.mean(current_opt_time))
    ms_time.append(np.mean(current_ms_time))
    opt_speedup.append(np.mean(current_opt_speedup))

fig,ax = plt.subplots(3,1)
ax[0].plot(graph_sizes, gaps)
ax[0].set_ylabel("Gap (%)")
ax[0].set_xlabel("|V|")
ax[1].plot(graph_sizes, opt_time, label="opt")
ax[1].plot(graph_sizes, ms_time, label="ms")
ax[1].set_ylabel("Time (s)")
ax[1].set_xlabel("|V|")
ax[2].plot(graph_sizes, opt_speedup)
plt.legend()
plt.show()

