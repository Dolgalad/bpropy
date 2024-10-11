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
            ss[c+1] = 1
        if x[e[0]]==1 and x[e[1]]==0:
            ss[c+2] = 1
        if x[e[0]]==1 and x[e[1]]==1:
            ss[c+3] = 1
        c += 4
    return ss

graph_sizes = np.arange(10, 50, 5)
generators = [random_tree]#, nx.generators.ladder_graph]
#generators = [nx.generators.ladder_graph]
#generators = [lambda n: nx.generators.erdos_renyi_graph(n, 0.15)]
N = 20

output_dir = "maxcut_tests"
os.makedirs(output_dir, exist_ok=True)

gaps = []
opt_time = []
ms_time = []

for (n,func) in itertools.product(graph_sizes, generators):
    current_gaps = []
    current_ms_time = []
    current_opt_time = []
    for j in range(N):
        g = func(n)
        print(g)

        pb = MaxCut(g)
        t0 = time.time()
        sol = pb.solve()
        t0 = time.time() - t0
        current_opt_time.append(t0)
        print("raaaaawww ", j,n,func.__name__)
        print("\tExact solver")
        #print("\t\tSolution: ", sol.x)
        print("\t\tval: ", np.sum(sol.x))
        print("\t\tSolve t: ", t0)
        
        # Loopy belief propagation
        print("\tLoopy Belief Propagation")
        t0 = time.time()
        g = create_factor_graph(g)
        print("NSS = ", sufficient_statistics_dim(g))
        # print the factor graph
        for u,d in g.nodes(data=True):
            if d["bipartite"]==0:
                print("Variable ", u)
            else:
                print("Factor   ", u)
        for e in g.edges:
            print(e[0], " -> ", e[1])
        f2v_idx, f2v_subsum_idx,ss_idx = f2v_index_map(g)
        # creating message passing edges
        f2v_edges,r_edges = f2v_subsum_indexes(g, f2v_idx, ss_idx)
        f2v_edges = torch.from_numpy(f2v_edges)
        r_edges = torch.from_numpy(r_edges)
        f2v_edges2,r_edges2 = f2v_subsum_indexes2(g)
        f2v_edges2 = torch.from_numpy(f2v_edges2)
        r_edges2 = torch.from_numpy(r_edges2)
        
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

        theta = theta - theta.min() + 1
        
        
        # scatter indices
        q_edges = torch.from_numpy(v2f_indexes(g, f2v_idx))
        q_edges2 = torch.from_numpy(v2f_indexes2(g))
        
        m_edges = torch.from_numpy(marginal_indexes(g, f2v_idx, ss_idx))
        m_edges2 = torch.from_numpy(marginal_indexes2(g))

        vm_edges = torch.from_numpy(variable_marginal_indexes(g, f2v_idx, ss_idx))
        print(v2f.shape)
        print(vm_edges[:,1].min(), vm_edges[:,1].max())
        
        marginals = loopy_belief_propagation(theta.clone(), v2f.clone(), g, f2v_edges, f2v_subsum_idx, r_edges, q_edges, m_edges, max_iter=100, tol=1e-6)

        ms_t = time.time()
        ms_marginals = max_sum_loopy_belief_propagation(theta.clone(), v2f.clone(), g, f2v_edges, f2v_subsum_idx, r_edges, q_edges, m_edges, vm_edges, max_iter=100, tol=1e-6)
        ms_t = time.time() - ms_t
        current_ms_time.append(ms_t)
        print("n = ", n)
        print("nvm = ", number_vm_edges(g), vm_edges.shape)
        print("vm_edges[:,0] min/max ", vm_edges[:,0].min(), vm_edges[:,1].max())
        print("ms_marginal size: ", ms_marginals.shape)

        print("ms_marginals \n")
        print(ms_marginals)
        ms_sol = np.argmax(ms_marginals.numpy().reshape(int(ms_marginals.size(0)/2),2), axis=1)

        ms_sp_sol = maxcut_greedy(pb, ms_marginals[1::2])
        print(ms_sol)

        t0 = time.time() - t0
        
        mus = marginals[1:2*pb.nv()+1:2].numpy()
        #print("\t\tmarginals        : ", mus)
        #print("\t\tMax-sum marginals: \n", ms_marginals)
        
        ss = maxcut_solution_to_sufficient_statistics(pb, sol.x)
        ms_ss = maxcut_solution_to_sufficient_statistics(pb, ms_sol)
        ms_sp_sol_ss = maxcut_solution_to_sufficient_statistics(pb, ms_sp_sol)
        print("\t\tenergy: ", np.dot(ss, theta.numpy()))
        print("\t\tms energy : ", np.dot(ms_ss, theta.numpy()))
        print("\t\tms sp sol energy: ", np.dot(ms_sp_sol_ss, theta.numpy()))
        print("\t\tlbp t: ", t0)
        print("\t\tmax_sum val: ", np.sum(ms_sol))

        print("\tGreedy Sum-product scores")
        sp_sol = maxcut_greedy(pb, mus)
        #print("\t\tsol: ", sp_sol)
        print("\t\tval: ", np.sum(sp_sol))
        
        fig,ax = plt.subplots(2,2)
        pos = nx.drawing.spring_layout(pb.g)
        plot_maxcut_solution(pb, sol.x, ax=ax[0,0], pos=pos)
        ax[0,0].set_title(f"Val = {np.sum(sol.x)}")
        plot_maxcut_solution(pb, sol.x, ax=ax[0,1], pos=pos, node_color=mus, cmap=plt.cm.bwr)
        plot_maxcut_solution(pb, sp_sol, ax=ax[1,0], pos=pos)
        ax[1,0].set_title(f"Val = {np.sum(sp_sol)}")
        plot_maxcut_solution(pb, ms_sp_sol, ax=ax[1,1], pos=pos)
        ax[1,1].set_title(f"MS Val = {np.sum(ms_sp_sol)}")

        plt.savefig(os.path.join(output_dir, f"{func.__name__}_{n}_{j}.png"))
        plt.close()

        opt_val = np.sum(sol.x)
        ms_sp_val = np.sum(ms_sp_sol)
        ms_val = np.sum(ms_sol)
        current_gaps.append((opt_val - ms_val) / opt_val)
    gaps.append(np.mean(current_gaps))
    opt_time.append(np.mean(current_opt_time))
    ms_time.append(np.mean(current_ms_time))

fig,ax = plt.subplots(2,1)
ax[0].plot(graph_sizes, gaps)
ax[0].set_ylabel("Gap (%)")
ax[0].set_xlabel("|V|")
ax[1].plot(graph_sizes, opt_time, label="opt")
ax[1].plot(graph_sizes, ms_time, label="ms")
ax[1].set_ylabel("Time (s)")
ax[1].set_xlabel("|V|")
plt.legend()
plt.show()

