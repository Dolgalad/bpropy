import torch

from .lbp import * 
from .patience_counter import PatienceCounter

import matplotlib.pyplot as plt

import copy

def max_sum_tree_reweighted_belief_propagation(theta, v2f, g, f2v_edges, f2v_subsum_idx, r_edges, q_edges, m_edges, vm_edges, max_iter=100, tol=1e-6):
    pass

def edge_based_reparameterization(theta, g, nV, max_iter=100, edge_probability=None, eps=1e-6, patience=10, edge_p=None):
    # initialize patience counter
    patience_counter = PatienceCounter(patience)
    if edge_p is None:
        edge_p = {e:1. for e in g.edges}
    # initialize pseudo-max-marginals
    v = torch.zeros_like(theta)
    v[:2*nV] = theta[:2*nV].exp()
    for i in range(nV):
        v[2*i:2*i+2] /= v[2*i:2*i+2].sum()
    #print("Edge probability min/max: ", np.array(list(edge_p.values())).min(), np.array(list(edge_p.values())).max())

    for ei,e in enumerate(g.edges):
        v[2*nV+4*ei]   = ((theta[2*nV+4*ei]   / edge_p[e]) + theta[2*e[0]] + theta[2*e[1]]).exp()
        v[2*nV+4*ei+1] = ((theta[2*nV+4*ei+1] / edge_p[e]) + theta[2*e[0]+1] + theta[2*e[1]]).exp()
        v[2*nV+4*ei+2] = ((theta[2*nV+4*ei+2] / edge_p[e]) + theta[2*e[0]] + theta[2*e[1]+1]).exp()
        v[2*nV+4*ei+3] = ((theta[2*nV+4*ei+3] / edge_p[e]) + theta[2*e[0]+1] + theta[2*e[1]+1]).exp()
        #print(v[2*nV+4*ei:2*nV+4*ei+4], edge_p[e])
        #v[2*nV+4*ei:2*nV+4*ei+4] += eps
        v[2*nV+4*ei:2*nV+4*ei+4] /= v[2*nV+4*ei:2*nV+4*ei+4].sum()

    #print("Variable pseudo-max-marginals isnan/isinf: ", torch.any(torch.isnan(v)), torch.any(torch.isinf(v)))
    nv = torch.zeros_like(v)
    for it in range(max_iter):
        for i in range(nV):
            nv[2*i:2*i+2] = v[2*i:2*i+2]
            vs0,vs1 = [], []
            for ei,e in enumerate(g.edges):
                if e[0] == i:
                    # x_s == 0
                    vs0.append(np.power(np.max([v[2*nV+4*ei], v[2*nV+4*ei+2]])/v[2*i], edge_p[e]))
                    # x_s == 1
                    vs1.append(np.power(np.max([v[2*nV+4*ei+1], v[2*nV+4*ei+3]])/v[2*i+1], edge_p[e]))
                if e[1] == i:
                    # x_s == 0
                    vs0.append(np.power(np.max([v[2*nV+4*ei], v[2*nV+4*ei+1]])/v[2*i], edge_p[e]))
                    # x_s == 1
                    vs1.append(np.power(np.max([v[2*nV+4*ei+2], v[2*nV+4*ei+3]])/v[2*i+1], edge_p[e]))
            nv[2*i] *= np.prod(vs0)
            nv[2*i+1] *= np.prod(vs1)
            nv[2*i:2*i+2] /= nv[2*i:2*i+2].sum()

        for ei,e in enumerate(g.edges):
            nv[2*nV+4*ei] = nv[2*e[0]] * nv[2*e[1]] * v[2*nV+4*ei] / (np.max([v[2*nV+4*ei], v[2*nV+4*ei+2]]) * np.max([v[2*nV+4*ei], v[2*nV+4*ei+1]]))
            nv[2*nV+4*ei+1] = nv[2*e[0]+1] * nv[2*e[1]] * v[2*nV+4*ei+1] / (np.max([v[2*nV+4*ei+1], v[2*nV+4*ei+3]]) * np.max([v[2*nV+4*ei], v[2*nV+4*ei+1]]))
            nv[2*nV+4*ei+2] = nv[2*e[0]] * nv[2*e[1]+1] * v[2*nV+4*ei+2] / (np.max([v[2*nV+4*ei], v[2*nV+4*ei+2]]) * np.max([v[2*nV+4*ei+2], v[2*nV+4*ei+3]]))
            nv[2*nV+4*ei+3] = nv[2*e[0]+1] * nv[2*e[1]+1] * v[2*nV+4*ei+3] / (np.max([v[2*nV+4*ei+1], v[2*nV+4*ei+3]]) * np.max([v[2*nV+4*ei+2], v[2*nV+4*ei+3]]))
            nv[2*nV+4*ei:2*nV+4*ei+4] /= nv[2*nV+4*ei:2*nV+4*ei+4].sum()

        err = (v - nv).abs().max()
        print("EBR1 ", it, "\n", err)
        patience_counter.update(err)

        if patience_counter.limit_reached():
            print("Patience limit reached")
            return v

        if torch.any(torch.isnan(nv)) or torch.any(torch.isinf(nv)):
            print("Quitting bad value")
            return v

        if (v - nv).abs().max() < 1e-6:
            print("Quitting stop condition")
            return nv
        v = nv.clone()
    return v

class EBREdges:
    def __init__(self, g, edge_idx_map):
        nV = g.number_of_nodes()
        nE = g.number_of_edges()
        # pseudo-max-marginals initialization edges
        pmm_init_edges = []
        pmm_norm_edges = []
        factor_repeats = []
        for i in range(g.number_of_nodes()):
            pmm_init_edges.append([2*i, 2*i])
            pmm_init_edges.append([2*i+1, 2*i+1])
            pmm_norm_edges.append([i, 2*i])
            pmm_norm_edges.append([i, 2*i+1])
            factor_repeats.append(2)
        for ei,e in enumerate(g.edges):
            # edges can take one of 4 configurations
            for vals in range(4):
                yF = np.unpackbits(np.array([vals],dtype=np.uint8), count=2, bitorder="little")
                pmm_init_edges.append([2*nV+4*ei+vals, 2*nV+4*ei+vals])
                pmm_init_edges.append([2*nV+4*ei+vals, 2*e[0]+int(yF[0])])
                pmm_init_edges.append([2*nV+4*ei+vals, 2*e[1]+int(yF[1])])
                pmm_norm_edges.append([nV+ei, 2*nV+4*ei+vals])
            factor_repeats.append(4)
        self.pmm_init_edges = torch.from_numpy(np.array(pmm_init_edges))
        self.pmm_norm_edges = torch.from_numpy(np.array(pmm_norm_edges))
        self.factor_repeats = torch.from_numpy(np.array(factor_repeats))
        # variable pseudo-max-marginal update edges
        edge_max_xt_edges = []
        pmm_updt_prod_edges = []
        pmm_updt_subdiv_edges = []
        pmm_edge_updt_prod_edges = []
        pmm_edge_updt_max_prod_edges = []
        subdiv_gather = []
        for i in range(g.number_of_nodes()):
            for n in g.neighbors(i):
                ei = edge_idx_map[i,n]
                pmm_updt_prod_edges.append([2*i, 2*ei])
                pmm_updt_prod_edges.append([2*i+1, 2*ei+1])
        for ei,e in enumerate(g.edges):
            edge_max_xt_edges.append([4*ei, 2*nV+4*ei])     # x_s fixed to 0
            edge_max_xt_edges.append([4*ei, 2*nV+4*ei+2])   # x_s fixed to 0
            edge_max_xt_edges.append([4*ei+1, 2*nV+4*ei+1]) # x_s fixed to 1
            edge_max_xt_edges.append([4*ei+1, 2*nV+4*ei+3]) # x_s fixed to 1
            edge_max_xt_edges.append([4*ei+2, 2*nV+4*ei])   # x_t fixed to 0
            edge_max_xt_edges.append([4*ei+2, 2*nV+4*ei+1]) # x_t fixed to 0
            edge_max_xt_edges.append([4*ei+3, 2*nV+4*ei+2]) # x_t fixed to 1 
            edge_max_xt_edges.append([4*ei+3, 2*nV+4*ei+3]) # x_t fixed to 1 
            subdiv_gather.append(2*e[0])
            subdiv_gather.append(2*e[0]+1)
            subdiv_gather.append(2*e[1])
            subdiv_gather.append(2*e[1]+1)
            pmm_edge_updt_prod_edges.append([2*nV+4*ei, 2*e[0]])
            pmm_edge_updt_prod_edges.append([2*nV+4*ei, 2*e[1]])
            pmm_edge_updt_prod_edges.append([2*nV+4*ei+1, 2*e[0]+1])
            pmm_edge_updt_prod_edges.append([2*nV+4*ei+1, 2*e[1]])
            pmm_edge_updt_prod_edges.append([2*nV+4*ei+2, 2*e[0]])
            pmm_edge_updt_prod_edges.append([2*nV+4*ei+2, 2*e[1]+1])
            pmm_edge_updt_prod_edges.append([2*nV+4*ei+3, 2*e[0]+1])
            pmm_edge_updt_prod_edges.append([2*nV+4*ei+3, 2*e[1]+1])

            pmm_edge_updt_max_prod_edges.append([4*ei, 4*ei])     # max x_t,x_s=0
            pmm_edge_updt_max_prod_edges.append([4*ei, 4*ei+2])   # max x_s,x_t=0
            pmm_edge_updt_max_prod_edges.append([4*ei+1, 4*ei+1]) # max x_t,x_s=1
            pmm_edge_updt_max_prod_edges.append([4*ei+1, 4*ei+2]) # max x_s,x_t=0
            pmm_edge_updt_max_prod_edges.append([4*ei+2, 4*ei])   # max x_t,x_s=0
            pmm_edge_updt_max_prod_edges.append([4*ei+2, 4*ei+3]) # max x_s,x_t=1
            pmm_edge_updt_max_prod_edges.append([4*ei+3, 4*ei+1]) # max x_t,x_s=1
            pmm_edge_updt_max_prod_edges.append([4*ei+3, 4*ei+3]) # max x_s,x_t=1


        self.edge_max_xt_edges = torch.from_numpy(np.array(edge_max_xt_edges))
        self.pmm_updt_prod_edges = torch.from_numpy(np.array(pmm_updt_prod_edges))
        self.subdiv_gather = torch.from_numpy(np.array(subdiv_gather))
        self.pmm_edge_updt_prod_edges = torch.from_numpy(np.array(pmm_edge_updt_prod_edges))
        self.pmm_edge_updt_max_prod_edges = torch.from_numpy(np.array(pmm_edge_updt_max_prod_edges))

        self.variable_mask = torch.zeros(2*nV+4*nE, dtype=torch.bool)
        self.variable_mask[:2*nV] = 1

        self.graph_indicator = torch.zeros(2*nV+4*nE, dtype=torch.int)

        self.num_graphs = 1

def stack_ebr_edges(ebr_edge_list):
    new_ebr_edges = copy.deepcopy(ebr_edge_list[0])

    new_pmm_init_edges = [] 
    new_pmm_norm_edges = [] 
    new_factor_repeats = [] 
    new_edge_max_xt_edges            = [] 
    new_pmm_updt_prod_edges          = [] 
    new_subdiv_gather                = [] 
    new_pmm_edge_updt_prod_edges     = [] 
    new_pmm_edge_updt_max_prod_edges = [] 
    new_variable_masks = []
    new_graph_indicators = []
    # offsets
    pmm_init_edges_offset = torch.zeros(2, dtype=torch.int64)
    pmm_norm_edges_offset = torch.zeros(2, dtype=torch.int64)
    edge_max_xt_edges_offset = torch.zeros(2, dtype=torch.int64)
    pmm_updt_prod_edges_offset = torch.zeros(2, dtype=torch.int64)
    subdiv_gather_offset = torch.zeros(1, dtype=torch.int64)
    pmm_edge_updt_prod_edges_offset = torch.zeros(2, dtype=torch.int64)
    pmm_edge_updt_max_prod_edges_offset = torch.zeros(2, dtype=torch.int64)
    for i,ebr_edges in enumerate(ebr_edge_list):
        new_pmm_init_edges.append(ebr_edges.pmm_init_edges + pmm_init_edges_offset)
        pmm_init_edges_offset += ebr_edges.pmm_init_edges.max(dim=0).values+1

        new_pmm_norm_edges.append(ebr_edges.pmm_norm_edges + pmm_norm_edges_offset)
        pmm_norm_edges_offset += ebr_edges.pmm_norm_edges.max(dim=0).values+1
        print("pmm_norm_edges_offset = ", pmm_norm_edges_offset)

        new_factor_repeats.append(ebr_edges.factor_repeats)

        new_edge_max_xt_edges.append(ebr_edges.edge_max_xt_edges + edge_max_xt_edges_offset)
        edge_max_xt_edges_offset += ebr_edges.edge_max_xt_edges.max(dim=0).values+1
        print("edge_max_xt_edges_offset = ", edge_max_xt_edges_offset)

        new_pmm_updt_prod_edges.append(ebr_edges.pmm_updt_prod_edges + pmm_updt_prod_edges_offset)
        #pmm_updt_prod_edges_offset += ebr_edges.pmm_updt_prod_edges.max(dim=0).values+1
        pmm_updt_prod_edges_offset += torch.tensor([ebr_edges.variable_mask.sum().item(),
                                                    ebr_edges.pmm_updt_prod_edges[:,1].max().item()+1], dtype=torch.int64)
        #print(ebr_edges.pmm_updt_prod_edges.max(dim=0).values+1, ebr_edges.variable_mask.sum(), pmm_updt_prod_edges_offset)



        new_subdiv_gather.append(ebr_edges.subdiv_gather + subdiv_gather_offset)
        #subdiv_gather_offset += ebr_edges.subdiv_gather.max()+1
        subdiv_gather_offset += ebr_edges.variable_mask.sum()

        new_pmm_edge_updt_prod_edges.append(ebr_edges.pmm_edge_updt_prod_edges + pmm_edge_updt_prod_edges_offset)
        #pmm_edge_updt_prod_edges_offset += ebr_edges.pmm_edge_updt_prod_edges.max(dim=0).values+1
        pmm_edge_updt_prod_edges_offset += torch.tensor([ebr_edges.pmm_edge_updt_prod_edges[:,0].max().item()+1, ebr_edges.variable_mask.size(0)], dtype=torch.int64)


        new_pmm_edge_updt_max_prod_edges.append(ebr_edges.pmm_edge_updt_max_prod_edges + pmm_edge_updt_max_prod_edges_offset)
        pmm_edge_updt_max_prod_edges_offset += ebr_edges.pmm_edge_updt_max_prod_edges.max(dim=0).values+1
        print("pmm_edge_updt_max_prod_edges_offset = ", pmm_edge_updt_max_prod_edges_offset)
        new_variable_masks.append(ebr_edges.variable_mask)

        new_graph_indicators.append(ebr_edges.graph_indicator + i)

    new_ebr_edges.pmm_init_edges = torch.cat(new_pmm_init_edges, dim=0)
    new_ebr_edges.pmm_norm_edges = torch.cat(new_pmm_norm_edges, dim=0)
    new_ebr_edges.factor_repeats = torch.cat(new_factor_repeats)
    new_ebr_edges.edge_max_xt_edges = torch.cat(new_edge_max_xt_edges, dim=0)
    new_ebr_edges.pmm_updt_prod_edges = torch.cat(new_pmm_updt_prod_edges, dim=0)
    new_ebr_edges.subdiv_gather = torch.cat(new_subdiv_gather)
    new_ebr_edges.pmm_edge_updt_prod_edges = torch.cat(new_pmm_edge_updt_prod_edges, dim=0)
    new_ebr_edges.pmm_edge_updt_max_prod_edges = torch.cat(new_pmm_edge_updt_max_prod_edges, dim=0)
    new_ebr_edges.variable_mask = torch.cat(new_variable_masks)
    new_ebr_edges.graph_indicator = torch.cat(new_graph_indicators)
    new_ebr_edges.num_graphs = len(ebr_edge_list)

    return new_ebr_edges


def edge_based_reparameterization2(theta, g, edges, edge_idx_map, max_iter=100, edge_p=None, eps=1e-6, patience=10, verbose=False):
    # initialize patience counter
    patience_counter = PatienceCounter(patience, dim=edges.num_graphs)
    nV = g.number_of_nodes()
    nE = g.number_of_edges()
    if edge_p is None:
        edge_p_ = torch.from_numpy(np.ones(g.number_of_edges()))
    else:
        if isinstance(edge_p, dict):
            edge_p_ = torch.from_numpy(np.array([edge_p[e] for e in g.edges]))
        else:
            edge_p_ = edge_p.clone()
    edge_p_ = edge_p_.repeat_interleave(4)
    # make a copy of theta and divide edge thetas by the edge probability
    theta_ = theta.clone()
    if torch.any(torch.isnan(theta_)) or torch.any(torch.isinf(theta_)):
        print("Theta bad =\n", theta_)
        exit()

    #theta_[2*nV:] = theta_[2*nV:] / edge_p_
    theta_[edges.variable_mask.logical_not()] = theta_[edges.variable_mask.logical_not()] / edge_p_

    n_theta = int(theta_.size(0) / 2)

    # initialize pseudo-max-marginals
    v = torch.zeros_like(theta_).scatter_reduce_(0, edges.pmm_init_edges[:,0], theta_[edges.pmm_init_edges[:,1]], "sum", include_self=False).exp()
    v_norm_ = torch.zeros(edges.pmm_norm_edges[:,0].max().item()+1, dtype=torch.double).scatter_reduce_(0, edges.pmm_norm_edges[:,0], v[edges.pmm_norm_edges[:,1]], "sum")
    if torch.any(torch.isnan(v_norm_)) or torch.any(torch.isinf(v_norm_)):
        print("v_norm_ bad =\n", v_norm_)
        print(theta_)
        exit()

    v = v / v_norm_.repeat_interleave(edges.factor_repeats)
    v_new = v.clone()
    if torch.any(torch.isnan(v_new)) or torch.any(torch.isinf(v_new)):
        print("v_new bad =\n", v_new)
        exit()

    n_v_new = int(v_new.size(0) / 2)

    #n_neighbors = torch.from_numpy(np.array([len(list(g.neighbors(i))) for i in g.nodes]))
    nv = torch.zeros_like(v)
    for it in range(max_iter):

        max_x = torch.zeros(edges.variable_mask.logical_not().sum().item(), dtype=torch.double).scatter_reduce(0, edges.edge_max_xt_edges[:,0], v_new[edges.edge_max_xt_edges[:,1]], "amax", include_self=False)
        if torch.any(torch.isnan(max_x)):
            print(it, " max_x=\n", max_x)
            for vi in range(edges.edge_max_xt_edges[:,0].max().item()+1):
                print(vi, max_x[vi].item(), v_new[edges.edge_max_xt_edges[edges.edge_max_xt_edges[:,0]==vi,1]])
            exit()

        v_new_gathered = torch.gather(v_new[edges.variable_mask],0,edges.subdiv_gather)

        tmp_m = torch.pow(max_x / v_new_gathered, edge_p_)

        pmm_prod = v_new[edges.variable_mask].scatter_reduce(0, edges.pmm_updt_prod_edges[:,0], tmp_m[edges.pmm_updt_prod_edges[:,1]], "prod", include_self=True)

        pmm_prod_tmp = torch.zeros_like(v)
        pmm_prod_tmp[edges.variable_mask] = pmm_prod.clone()
        pmm_prod_norm_ = torch.zeros(edges.pmm_norm_edges[:,0].max().item()+1,dtype=torch.double).scatter_reduce(0, edges.pmm_norm_edges[:,0], pmm_prod_tmp[edges.pmm_norm_edges[:,1]], "sum")

        num_v_e = int(edges.variable_mask.sum()/2) + int(edges.variable_mask.logical_not().sum()/4)

        pmm_prod = pmm_prod_tmp[edges.variable_mask] / pmm_prod_norm_.repeat_interleave(edges.factor_repeats)[edges.variable_mask]

        # update edge pseudo-max-marginals
        v_tmp = v_new.clone()
        v_tmp[edges.variable_mask] = pmm_prod.clone()

        pmm_edge_prod = v_tmp.scatter_reduce(0, edges.pmm_edge_updt_prod_edges[:,0], v_tmp[edges.pmm_edge_updt_prod_edges[:,1]], "prod", include_self=True)

        max_prod = torch.zeros_like(v_tmp[edges.variable_mask.logical_not()]).scatter_reduce(0, edges.pmm_edge_updt_max_prod_edges[:,0], max_x[edges.pmm_edge_updt_max_prod_edges[:,1]], "prod", include_self=False)

        pmm_edge_prod[edges.variable_mask.logical_not()] = pmm_edge_prod[edges.variable_mask.logical_not()] / max_prod

        n_norm = int(edges.variable_mask.sum().item()/2) + int(edges.variable_mask.logical_not().sum().item() / 4)
        pmm_edge_prod_norm_ = torch.zeros(n_norm, dtype=torch.double).scatter_reduce(0, edges.pmm_norm_edges[:,0], pmm_edge_prod[edges.pmm_norm_edges[:,1]], "sum", include_self=False)

        pmm_edge_prod = pmm_edge_prod / pmm_edge_prod_norm_.repeat_interleave(edges.factor_repeats)

        graph_errs = torch.tensor([(v_new[edges.graph_indicator==i] - pmm_edge_prod[edges.graph_indicator==i]).abs().max().item() for i in range(edges.num_graphs)])

        #err = (v_new - pmm_edge_prod).abs().max()
        err = graph_errs.min().item()

        patience_counter.update(graph_errs.numpy())
        if patience_counter.limit_reached():
            if verbose:
                print("Patience limit reached ", err)
            return v_new

        if torch.any(torch.isnan(pmm_edge_prod)) or torch.any(torch.isinf(pmm_edge_prod)):
            if verbose:
                print("Bad value, quitting ", err, it)
                print("max_prod=\n",max_prod)
                print("pmm_edge_prod_norm_=\n",pmm_edge_prod_norm_)
                print("pmm_edge_prod=\n",pmm_edge_prod)
                print(graph_errs)
            return v_new

        #if (v_new - pmm_edge_prod).abs().max() < 1e-6:
        if torch.all(graph_errs < 1e-6):
            #print("Stop condition met quitting ", it)
            return pmm_edge_prod
        #v = nv.clone()
        v_new = pmm_edge_prod.clone()
    if verbose:
        print("max iter ", graph_errs)
    return v_new

def maxcut_solution_validity(sol, g):
    for u in g.nodes:
        for v in g.neighbors(u):
            if sol[u]==1 and sol[v]==1:
                return False
    return True

def edge_based_reparameterization_logspace(theta, g, edges, edge_idx_map, max_iter=10000, edge_p=None, eps=1e-6, patience=10000):
    patience_counter = PatienceCounter(patience)
    nV = g.number_of_nodes()
    # initial log-pseudo-max-marginals
    lpmm = torch.zeros_like(theta)
    lpmm[:2*nV] = theta[:2*nV]
    for ei,e in enumerate(g.edges):
        lpmm[2*nV+4*ei] = (1./edge_p[ei]) * theta[2*nV+4*ei] + theta[2*e[0]] + theta[2*e[1]]
        lpmm[2*nV+4*ei+1] = (1./edge_p[ei]) * theta[2*nV+4*ei+1] + theta[2*e[0]+1] + theta[2*e[1]]
        lpmm[2*nV+4*ei+2] = (1./edge_p[ei]) * theta[2*nV+4*ei+2] + theta[2*e[0]] + theta[2*e[1]+1]
        lpmm[2*nV+4*ei+3] = (1./edge_p[ei]) * theta[2*nV+4*ei+3] + theta[2*e[0]+1] + theta[2*e[1]+1]
    if torch.any(torch.isnan(lpmm)):
        print("DDDDDDD\n", lpmm)
        exit()

    for it in range(max_iter):
        new_lpmm = torch.zeros_like(lpmm)
        for ui,u in enumerate(g.nodes):
            new_lpmm[2*ui:2*ui+2] = lpmm[2*ui:2*ui+2]
            for ei,e in enumerate(g.edges):
                if e[1] == u:
                    new_lpmm[2*ui] += (torch.tensor([lpmm[2*nV+4*ei], lpmm[2*nV+4*ei+1]]).max() - lpmm[2*ui]) *  edge_p[ei]
                    new_lpmm[2*ui+1] += (torch.tensor([lpmm[2*nV+4*ei+2], lpmm[2*nV+4*ei+3]]).max() - lpmm[2*ui+1]) * edge_p[ei]
                if e[0] == u:
                    new_lpmm[2*ui] += (torch.tensor([lpmm[2*nV+4*ei], lpmm[2*nV+4*ei+2]]).max() - lpmm[2*ui]) * edge_p[ei]
                    new_lpmm[2*ui+1] += (torch.tensor([lpmm[2*nV+4*ei+1], lpmm[2*nV+4*ei+3]]).max() - lpmm[2*ui+1]) * edge_p[ei]
        if torch.any(torch.isnan(new_lpmm[:2*nV])):
            print("AAAAAAA\n", new_lpmm[:2*nV])
            return lpmm
            exit()

        for ei,e in enumerate(g.edges):
            new_lpmm[2*nV+4*ei] = lpmm[2*nV+4*ei] + new_lpmm[2*e[0]] + new_lpmm[2*e[1]] - torch.tensor([lpmm[2*nV+4*ei], lpmm[2*nV+4*ei+2]]).max() - torch.tensor([lpmm[2*nV+4*ei], lpmm[2*nV+4*ei+1]]).max()
            new_lpmm[2*nV+4*ei+1] = lpmm[2*nV+4*ei+1] + new_lpmm[2*e[0]+1] + new_lpmm[2*e[1]] - torch.tensor([lpmm[2*nV+4*ei+1], lpmm[2*nV+4*ei+3]]).max() - torch.tensor([lpmm[2*nV+4*ei], lpmm[2*nV+4*ei+1]]).max()
            new_lpmm[2*nV+4*ei+2] = lpmm[2*nV+4*ei+2] + new_lpmm[2*e[0]] + new_lpmm[2*e[1]+1] - torch.tensor([lpmm[2*nV+4*ei+2], lpmm[2*nV+4*ei+3]]).max() - torch.tensor([lpmm[2*nV+4*ei], lpmm[2*nV+4*ei+2]]).max()
            new_lpmm[2*nV+4*ei+3] = lpmm[2*nV+4*ei+3] + new_lpmm[2*e[0]+1] + new_lpmm[2*e[1]+1] - torch.tensor([lpmm[2*nV+4*ei+1], lpmm[2*nV+4*ei+3]]).max() - torch.tensor([lpmm[2*nV+4*ei+2], lpmm[2*nV+4*ei+3]]).max()
        if torch.any(torch.isnan(new_lpmm[2*nV:])):
            print("CCCCCCC\n", new_lpmm[2*nV:])
            return lpmm
            exit()

        for ui in range(g.number_of_nodes()):
            #print("before norm ", ui, " ", new_lpmm[2*ui:2*ui+2])
            new_lpmm[2*ui:2*ui+2] -= new_lpmm[2*ui:2*ui+2].sum() / 2
            #print("after norm ", ui, " ", new_lpmm[2*ui:2*ui+2])
        for ei in range(g.number_of_edges()):
            #print("before norm ", ei, " " , new_lpmm[2*nV+4*ei:2*nV+4*ei+4])
            new_lpmm[2*nV+4*ei:2*nV+4*ei+4] -= new_lpmm[2*nV+4*ei:2*nV+4*ei+4].sum() / 4
            #print("after norm ", ei, " " , new_lpmm[2*nV+4*ei:2*nV+4*ei+4])

        #print(it, new_lpmm)

        err = (lpmm - new_lpmm).abs().max()
        #print(it, err, patience_counter.counter)
        patience_counter.update(np.array([err.item()]))

        lpmm = .5 * (lpmm + new_lpmm).clone()
        #lpmm = new_lpmm.clone()

        if patience_counter.limit_reached():
            break
            return new_lpmm
        if err.item() < 1e-6:
            break
            return new_lpmm

    sol = lpmm[:2*nV].reshape(nV,2).argmax(axis=1)

    #print("sol = ", sol)
    #print("val = ", sol.sum())
    #print("validity = ", maxcut_solution_validity(sol, g))
    #for u in g.nodes:
    #    print(u, list(g.neighbors(u)), sol[u].item(), [sol[v].item() for v in g.neighbors(u)])
    return lpmm
           





def tree_reweighted_max_product(M, theta, edge_p, g, tol=1e-8, max_iter=100000):
    nV = g.number_of_nodes()
    nE = g.number_of_edges()
    edge_p_ = torch.from_numpy(np.array([edge_p[e] for e in g.edges]))

    msg_prod_edges = []
    msg_theta_sum_edges = []
    msg_prod_div_gather = []
    for ei,(s,t) in enumerate(g.edges):
        tsidx = edge_idx_map[t,s]
        stidx = edge_idx_map[s,t]
        msg_prod_div_gather.append(2*tsidx)
        msg_prod_div_gather.append(2*tsidx+1)
        msg_prod_div_gather.append(2*stidx)
        msg_prod_div_gather.append(2*stidx+1)

        msg_theta_sum_edges.append([8*ei, 2*nV+4*ei])
        msg_theta_sum_edges.append([8*ei, 2*s])
        msg_theta_sum_edges.append([8*ei+1, 2*nV+4*ei+1])
        msg_theta_sum_edges.append([8*ei+1, 2*s+1])
        msg_theta_sum_edges.append([8*ei+2, 2*nV+4*ei+2])
        msg_theta_sum_edges.append([8*ei+2, 2*s])
        msg_theta_sum_edges.append([8*ei+3, 2*nV+4*ei+3])
        msg_theta_sum_edges.append([8*ei+3, 2*s+1])

        msg_theta_sum_edges.append([8*ei+4, 2*nV+4*ei])
        msg_theta_sum_edges.append([8*ei+4, 2*t])
        msg_theta_sum_edges.append([8*ei+5, 2*nV+4*ei+1])
        msg_theta_sum_edges.append([8*ei+5, 2*t+1])
        msg_theta_sum_edges.append([8*ei+6, 2*nV+4*ei+2])
        msg_theta_sum_edges.append([8*ei+6, 2*t])
        msg_theta_sum_edges.append([8*ei+7, 2*nV+4*ei+3])
        msg_theta_sum_edges.append([8*ei+7, 2*t+1])

        for u in g.neighbors(s):
            if u!=t:
                usidx = edge_idx_map[u,s]
                msg_prod_edges.append([4*ei, 2*usidx])
                msg_prod_edges.append([4*ei+1, 2*usidx+1])
        for u in g.neighbors(t):
            if u!=s:
                usidx = edge_idx_map[u,t]
                msg_prod_edges.append([4*ei+2, 2*usidx])
                msg_prod_edges.append([4*ei+3, 2*usidx+1])
    msg_prod_edges = torch.from_numpy(np.array(msg_prod_edges))
    msg_prod_div_gather = torch.from_numpy(np.array(msg_prod_div_gather))
    msg_theta_sum_edges = torch.from_numpy(np.array(msg_theta_sum_edges))
    #print(msg_theta_sum_edges)

    error_values = []

    for i in range(max_iter):
        # compute first product
        msg_subprod = torch.zeros_like(M).scatter_reduce(0, msg_prod_edges[:,0], torch.pow(M, edge_p_.repeat_interleave(4))[msg_prod_edges[:,1]], "prod", include_self=False)
        msg_subprod[msg_subprod == 0] = 1
        #print("M = \n", M)
        #print("msg_subprod = \n", msg_subprod)
        #print("M gathered = \n", torch.gather(torch.pow(M, 1-edge_p_.repeat_interleave(4)), 0, msg_prod_div_gather))
        msg_subprod_div = msg_subprod / (torch.gather(torch.pow(M, 1-edge_p_.repeat_interleave(4)), 0, msg_prod_div_gather) + 1e-8)
        #print("msg_subprod_div = \n", msg_subprod_div)
        #print(theta)
        theta_sum = torch.zeros(int(msg_theta_sum_edges[:,0].max().item())+1,dtype=torch.double).scatter_reduce_(0, msg_theta_sum_edges[:,0], theta[msg_theta_sum_edges[:,1]], "sum")
        #print(theta_sum.shape)
        theta_sumprod = (theta_sum.exp() * msg_subprod_div.repeat_interleave(2)).reshape(4*nE,2)
        #print("theta_sumprod = \n", theta_sumprod)
        #print(theta_sumprod.shape)
        #print(torch.sum(theta_sumprod, axis=1).unsqueeze(-1).repeat(1,2).shape)

        #theta_sumprod /= (torch.sum(theta_sumprod, axis=1).unsqueeze(-1).repeat(1,2) + 1e-8)
        #print("theta_sumprod = \n", theta_sumprod)
        theta_max,_ = torch.max(theta_sumprod, axis=1)
        kappa = torch.sum(theta_max.reshape(2*nE, 2),axis=1)
        theta_max /= kappa.repeat_interleave(2)

        #print("theta_max = \n", theta_max)

        err = (M-theta_max).abs().max().item()
        print(i, err)
        error_values.append(err)

        if torch.any(torch.isnan(theta_max)) or torch.any(torch.isinf(theta_max)):
            print("Bad value encountered")
            break
        if err < tol:
            print("Tolerance reached")
            M = theta_max
            break

        M = theta_max.clone()

    plt.figure()
    plt.plot(error_values)
    plt.yscale("log")
    plt.show()

    return M
    



if __name__=="__main__":
    import random
    random.seed(123)
    import numpy as np
    np.random.seed(123)
    import networkx as nx
    import time
    from .lbp import *
    from networkx.generators.trees import random_tree

    n = 5
    g = nx.generators.erdos_renyi_graph(n, 0.15)
    g = random_tree(n)

    # node and edges index maps
    node_idx_map = {n:i for i,n in enumerate(g.nodes)}
    edge_idx_map = {}
    for ei,e in enumerate(g.edges):
        edge_idx_map[e[0],e[1]] = 2*ei
        edge_idx_map[e[1],e[0]] = 2*ei+1
    edge_p = {e:0.8*np.random.rand()+0.1 for e in g.edges}

    fg = create_factor_graph(g)
    n_ss = sufficient_statistics_dim(fg)
    theta = torch.zeros(n_ss, dtype=torch.double)
    c = 0
    for i in range(g.number_of_nodes()):
        theta[c+1] = 1 + np.random.rand()
        c += 2
    for e in g.edges:
        theta[c+1] = 1.# / len(list(g.neighbors(e[0])))# + np.random.rand()
        theta[c+2] = 1.# / len(list(g.neighbors(e[1])))# + np.random.rand()
        theta[c+3] = -n
        c += 4
    #theta = torch.from_numpy(np.random.random_sample(n_ss))
    t_m1 = time.time()
    m1 = edge_based_reparameterization(theta, g, n, max_iter=1000, patience=10, edge_p=edge_p)
    t_m1 = time.time() - t_m1

    ebr_edges = EBREdges(g)
    t_m2 = time.time()
    m2 = edge_based_reparameterization2(theta, g, ebr_edges, max_iter=1000, patience=10, edge_p=edge_p)
    t_m2 = time.time() - t_m2

    print(torch.all(m1 == m2))
    print("DIFF = \n", (m1 - m2).abs() < 1e-8)
    print(m1)
    print(m2)
    print("t_m1 = ", t_m1)
    print("t_m2 = ", t_m2)


    g = random_tree(100)
    fg = create_factor_graph(g)
    n_ss = sufficient_statistics_dim(fg)

    print("Edges")
    for e in g.edges:
        print(e)
    edge_idx_map = {}
    for ei,e in enumerate(g.edges):
        edge_idx_map[e[0],e[1]] = 2*ei
        edge_idx_map[e[1],e[0]] = 2*ei+1
    edge_p = {e:0.8*np.random.rand()+0.1 for e in g.edges}
    theta = torch.rand(n_ss, dtype=torch.double)
    #theta = theta - theta.min() + 1

    M = torch.rand(4*g.number_of_edges(), dtype=torch.double)
    tree_reweighted_max_product(M, theta, edge_p, g)
