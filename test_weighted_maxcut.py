import os
import shutil
import torch
import numpy as np
import networkx as nx
import random
random.seed(123)
np.random.seed(123)

from bpropy.lbp import *
from bpropy.trw import *
from dopy.maxcut import *

output_prefix = "weighted_sampling_e0"
if os.path.exists(output_prefix):
    shutil.rmtree(output_prefix)
os.makedirs(output_prefix)

gs_opt = []
gs_is_opt = []
wspeedup = []
graph_sizes=np.arange(10, 100, 10)
for n in graph_sizes:
    opt = []
    is_opt = []
    wsp = []
    for it in range(10):
        g = nx.generators.erdos_renyi_graph(n, 0.15)
        # Solve ILP
        pb = MaxCut(g, weights=np.random.random_sample(n))
        t_sol = time.time()
        sol = pb.solve()
        t_sol = time.time() - t_sol
    
        # Belief propagation
        t_bp = time.time()
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
        #q_edges2 = torch.from_numpy(v2f_indexes2(fg))
        m_edges = torch.from_numpy(marginal_indexes(fg, f2v_idx, ss_idx))
        #m_edges2 = torch.from_numpy(marginal_indexes2(fg))
        vm_edges = torch.from_numpy(variable_marginal_indexes(fg, f2v_idx, ss_idx))
        
        # stack sufficient stats and v2f messages
        v2f = torch.zeros(len(f2v_idx), dtype=torch.double)
        theta = torch.zeros(len(ss_idx), dtype=torch.double)
        c = 0
        for i in range(pb.nv()):
            theta[c+1] = pb.w[i]
            c += 2
        for e in pb.g.edges:
            #theta[c+1] = pb.w[e[0]] / len(list(g.neighbors(e[0])))
            #theta[c+2] = pb.w[e[1]] / len(list(g.neighbors(e[1])))
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
        try:
            lspmm = edge_based_reparameterization2(theta.clone(), g, ebr_edges, edge_idx_map2, edge_p=torch.ones(g.number_of_edges()), max_iter=1000, patience=10)
            t_bp = time.time() - t_bp
        except:
            continue
        print(lspmm[1:2*n+1:2].numpy())
    
    
        t_greedy = time.time()
        greedy_lspmm_sol = maxcut_greedy(pb, lspmm[1:2*n+1:2].numpy())
        t_greedy = time.time() - t_greedy


        wpb = MaxCut(pb.g, weights = lspmm[1:2*n+1:2].numpy())
        t_wsol = time.time()
        solve_energy_cost_sol = wpb.solve()
        t_wsol = time.time() - t_wsol
        wsol = solve_energy_cost_sol.x > 1e-8
    
        print(it, "opt val = ", np.sum(pb.w[sol.x > 1e-8]))
        print(it, "grd val = ", np.sum(pb.w[greedy_lspmm_sol > 1e-8]))
        print(it, "wpb val = ", np.sum(pb.w[solve_energy_cost_sol.x > 1e-8]))
    
        nopt = 100 * (np.sum(pb.w[sol.x > 1e-8]) - np.sum(pb.w[greedy_lspmm_sol > 1e-8])) / np.sum(pb.w[sol.x > 1e-8])
        wopt = 100 * (np.sum(pb.w[sol.x > 1e-8]) - np.sum(pb.w[wsol > 1e-8])) / np.sum(pb.w[wsol > 1e-8] )

        if nopt<0 or wopt<0:
            continue
        opt.append(nopt)
        is_opt.append(wopt)
        wsp.append( (t_bp + t_wsol) / t_sol)
    
        fig = plt.figure()
        plt.boxplot([opt, is_opt], tick_labels=["lpmm", "ew"])
        plt.savefig(f"{output_prefix}/{output_prefix}_n{n}.png")
        plt.close()
    gs_opt.append(opt)
    gs_is_opt.append(is_opt)
    wspeedup.append(wsp)
    fig,ax = plt.subplots(3,1, sharex=True)
    ax[0].plot(graph_sizes[:len(gs_opt)], [np.mean(gg) for gg in gs_opt])
    ax[0].boxplot(gs_opt, positions=graph_sizes[:len(gs_opt)])
    ax[0].set_ylabel("LPMM opt (%)")
    ax[1].plot(graph_sizes[:len(gs_opt)], [np.mean(gg) for gg in gs_is_opt])
    ax[1].boxplot(gs_is_opt, positions=graph_sizes[:len(gs_opt)])
    ax[1].set_ylabel("EW opt (%)")
    ax[1].set_xlabel("|V|")
    ax[2].plot(graph_sizes[:len(gs_opt)], [np.mean(gg) for gg in wspeedup])
    ax[2].boxplot(wspeedup, positions=graph_sizes[:len(gs_opt)])
    ax[2].set_ylabel("speed ratio")
    ax[2].set_xlabel("|V|")

    plt.savefig(f"{output_prefix}/{output_prefix}_gs.png")
    plt.close()


