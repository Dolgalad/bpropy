"""Naive Mean Field computation edges
"""
import torch
import numpy as np

class NMFEdges:
    def __init__(self, fg=None):
        self.pmprod_edges = None
        self.pmprodsum_edges = None
        self.theta_rep = None
        self.n_ss = None
        self.n_vars = None
        self.n_factors = None
        self.n_prodsum = None
        if not fg is None:
            self.fill(fg)
    @property
    def device(self):
        return self.n_ss.device
    @property
    def is_cuda(self):
        return self.n_ss.is_cuda and \
                self.n_vars.is_cuda and \
                self.n_factors.is_cuda and \
                self.n_prodsum.is_cuda and \
                self.pmprod_edges.is_cuda and \
                self.pmprodsum_edges.is_cuda and \
                self.theta_rep.is_cuda
    @property
    def num_graphs(self):
        return self.n_ss.shape[0]
    def graph_indicator(self):
        if self.num_graphs == 1:
            return torch.zeros(self.n_ss, dtype=torch.int16, device=self.device)
        return torch.arange(self.num_graphs, device=self.device).repeat_interleave(self.n_ss)

    def fill(self, fg):
        if fg.num_graphs == 1:
            self.pmprod_edges, self.pmprodsum_edges, n_prodsum = naive_mean_field_edges(fg)
            self.n_ss = torch.tensor([fg.number_sufficient_statistics()])
            self.n_vars = torch.tensor([fg.nv()])
            self.n_factors = torch.tensor([fg.nf()])
            self.n_prodsum = torch.tensor([n_prodsum])
            theta_rep = torch.tensor([fg.factor_size(F) for F in fg.iter_factors()], dtype=torch.long)
            self.theta_rep = torch.repeat_interleave(theta_rep, 2**theta_rep)
        else:
            edge_list = [NMFEdges(fg.get_subgraph(i)) for i in range(fg.num_graphs)]
            edges_batch = NMFEdges.batch(edge_list)
            self.n_ss      = edges_batch.n_ss      
            self.n_vars    = edges_batch.n_vars    
            self.n_factors = edges_batch.n_factors 
            self.n_prodsum = edges_batch.n_prodsum
            self.pmprod_edges = edges_batch.pmprod_edges
            self.pmprodsum_edges   = edges_batch.pmprodsum_edges
            self.theta_rep = edges_batch.theta_rep
    @property
    def num_graphs(self):
        return self.n_ss.shape[0]
    def clone(self):
        ne = NMFEdges()
        ne.n_ss = self.n_ss.clone()
        ne.n_vars = self.n_vars.clone()
        ne.n_factors = self.n_factors.clone()
        ne.n_prodsum = self.n_prodsum.clone()

        ne.pmprod_edges = self.pmprod_edges.clone()
        ne.pmprodsum_edges = self.pmprodsum_edges.clone()
        ne.theta_rep = self.theta_rep.clone()
        return ne
    def to(self, device, non_blocking=False, *args):
        ne = NMFEdges()
        ne.n_ss = self.n_ss.to(device, non_blocking=non_blocking)
        ne.n_vars = self.n_vars.to(device, non_blocking=non_blocking)
        ne.n_factors = self.n_factors.to(device, non_blocking=non_blocking)
        ne.n_prodsum = self.n_prodsum.to(device, non_blocking=non_blocking)
        ne.pmprod_edges = self.pmprod_edges.to(device, non_blocking=non_blocking)
        ne.pmprodsum_edges = self.pmprodsum_edges.to(device, non_blocking=non_blocking)
        ne.theta_rep = self.theta_rep.to(device, non_blocking=non_blocking)
        return ne

    def __eq__(self, other):
        if (self.n_ss.shape != other.n_ss.shape) or \
           (self.n_vars.shape != other.n_vars.shape) or \
           (self.n_factors.shape != other.n_factors.shape) or \
           (self.n_prodsum.shape != other.n_prodsum.shape) or \
           (self.pmprod_edges.shape != other.pmprod_edges.shape) or \
           (self.pmprodsum_edges.shape != other.pmprodsum_edges.shape) or \
           (self.theta_rep.shape != other.theta_rep.shape):
            return False

        return torch.all(self.n_ss == other.n_ss) and \
               torch.all(self.n_vars == other.n_vars) and \
               torch.all(self.n_factors == other.n_factors) and \
               torch.all(self.n_prodsum == other.n_prodsum) and \
               torch.all(self.pmprod_edges == other.pmprod_edges) and \
               torch.all(self.pmprodsum_edges == other.pmprodsum_edges) and \
               torch.all(self.theta_rep == other.theta_rep)
    def __getitem__(self, idx):
        if self.num_graphs==1:
            return self
        ne = NMFEdges()
        ne.n_ss = self.n_ss[idx].unsqueeze(0)
        ne.n_vars = self.n_vars[idx].unsqueeze(0)
        ne.n_factors = self.n_factors[idx].unsqueeze(0)
        ne.n_prodsum = self.n_prodsum[idx].unsqueeze(0)

        ss_offsets = torch.roll(torch.cumsum(self.n_ss,0), 1)
        ss_offsets[0] = 0

        vars_offsets = torch.roll(torch.cumsum(self.n_vars,0), 1)
        vars_offsets[0] = 0

        prodsum_offsets = torch.roll(torch.cumsum(self.n_prodsum,0), 1)
        prodsum_offsets[0] = 0

        ne.theta_rep = self.theta_rep[ss_offsets[idx]:ss_offsets[idx]+self.n_ss[idx]]

        pmprod_idx = (self.pmprod_edges[:,1] >= 2*vars_offsets[idx]) * (self.pmprod_edges[:,1] < 2*vars_offsets[idx]+2*self.n_vars[idx])
        ne.pmprod_edges = self.pmprod_edges[pmprod_idx].clone()
        ne.pmprod_edges = ne.pmprod_edges - torch.tensor([prodsum_offsets[idx], 2*vars_offsets[idx]])

        pmprodsum_idx = (self.pmprodsum_edges[:,0] >= 2*vars_offsets[idx]) * (self.pmprodsum_edges[:,0] < 2*vars_offsets[idx]+2*self.n_vars[idx])



        ne.pmprodsum_edges = self.pmprodsum_edges[pmprodsum_idx].clone()
        ne.pmprodsum_edges = ne.pmprodsum_edges - torch.tensor([2*vars_offsets[idx], prodsum_offsets[idx]])



        return ne


    @staticmethod
    def show_diff(a, b):
        print("same n_ss      = ", torch.all(a.n_ss == b.n_ss) )
        print("same n_vars    = ", torch.all(a.n_vars == b.n_vars) )
        print("same n_factors = ", torch.all(a.n_factors == b.n_factors) )
        print("same pmprod_edges.shape = ", a.pmprod_edges.shape == b.pmprod_edges.shape)
        print("same pmprodsum_edges  shape  = ", a.pmprodsum_edges.shape ==   b.pmprodsum_edges.shape) 
        print("same theta_rep shape  = ", a.theta_rep.shape ==   b.theta_rep.shape) 

        if a.pmprod_edges.shape == b.pmprod_edges.shape:
            print("same pmprod_edges = ", torch.all(a.pmprod_edges== b.pmprod_edges))
        else:
            print("a.pmprod.shape = ", a.pmprod_edges.shape, ", b.pmprodshape = ", b.pmprod_edges.shape)
        if a.pmprodsum_edges.shape == b.pmprodsum_edges.shape:
            print("same pmprodsum_edges   = ", torch.all(a.pmprodsum_edges == b.pmprodsum_edges) )
        else:
            print("a.pmprodsum_edges.shape = ", a.pmprodsum_edges.shape)
            print("b.pmprodsum_edges.shape = ", b.pmprodsum_edges.shape)

        if a.theta_rep.shape == b.theta_rep.shape:
            print("same theta_rep = ", torch.all(a.theta_rep == b.theta_rep) )
    
    @staticmethod
    def batch(edge_l):
        n_pmprod_edges, n_pmprodsum_edges, n_n_vars, n_n_factors, n_n_prodsum, n_n_ss, n_ss_tot, n_theta_rep = 0,0,0,0,0,0,0,0
        for e in edge_l:
            n_pmprod_edges += e.pmprod_edges.shape[0]
            n_pmprodsum_edges += e.pmprodsum_edges.shape[0]
            n_n_vars += e.n_vars.shape[0]
            n_n_factors += e.n_factors.shape[0]
            n_n_prodsum += e.n_prodsum.shape[0]
            n_n_ss += e.n_ss.shape[0]
            n_ss_tot += e.n_ss.sum().item()
            n_theta_rep += e.theta_rep.shape[0]

        ne = NMFEdges()
        ne.pmprod_edges = torch.empty(n_pmprod_edges,2, dtype=torch.long)
        pmprod_offset = torch.tensor([0, 0])
        pmprod_pos = 0
        ne.pmprodsum_edges = torch.empty(n_pmprodsum_edges,2, dtype=torch.long)
        pmprodsum_offset = torch.tensor([0,0])
        pmprodsum_pos = 0
        ne.n_vars = torch.empty(n_n_vars, dtype=torch.long)
        n_vars_pos = 0
        ne.n_factors = torch.empty(n_n_factors, dtype=torch.long)
        n_factors_pos = 0
        ne.n_prodsum = torch.empty(n_n_prodsum, dtype=torch.long)
        n_prodsum_pos = 0

        ne.n_ss = torch.empty(n_n_ss, dtype=torch.long)
        n_ss_pos = 0
        ne.theta_rep = torch.empty(n_theta_rep, dtype=torch.long)
        theta_rep_pos = 0
        for i,e in enumerate(edge_l):
            ne.pmprod_edges[pmprod_pos:pmprod_pos+e.pmprod_edges.shape[0]] = e.pmprod_edges.clone() + pmprod_offset
            pmprod_offset += torch.tensor([e.n_prodsum.sum(), 2*e.n_vars.sum()])

            pmprod_pos += e.pmprod_edges.shape[0]
            ne.pmprodsum_edges[pmprodsum_pos:pmprodsum_pos+e.pmprodsum_edges.shape[0]] = e.pmprodsum_edges.clone() + pmprodsum_offset


            pmprodsum_offset += torch.tensor([2*e.n_vars.sum(), e.n_prodsum.sum()])

            pmprodsum_pos += e.pmprodsum_edges.shape[0]

            ne.n_vars[n_vars_pos:n_vars_pos+e.n_vars.shape[0]] = e.n_vars.clone()
            n_vars_pos += e.n_vars.shape[0]

            ne.n_ss[n_ss_pos:n_ss_pos+e.n_ss.shape[0]] = e.n_ss.clone()
            n_ss_pos += e.n_ss.shape[0]

            ne.n_factors[n_factors_pos:n_factors_pos+e.n_factors.shape[0]] = e.n_factors.clone()
            n_factors_pos += e.n_factors.shape[0]

            ne.n_prodsum[n_prodsum_pos:n_prodsum_pos+e.n_prodsum.shape[0]] = e.n_prodsum.clone()
            n_prodsum_pos += e.n_prodsum.shape[0]

            ne.theta_rep[theta_rep_pos:theta_rep_pos+e.theta_rep.shape[0]] = e.theta_rep.clone()
            theta_rep_pos += e.theta_rep.shape[0]

        return ne



def naive_mean_field_edges_old(fg):
    nV = fg.nv()
    nF = fg.nf()

    pmprod_index_map = {}
    pmprod_edges = []

    pmprodsum_edges = []

    c,d = 0,0

    for iF,F in enumerate(fg.iter_factors()):
        neighbors = list(fg.factor_variables(F))
        nN = len(neighbors)
        for vals in range(2**nN):
            yF = np.unpackbits(np.array([vals],dtype=np.uint8), count=nN, bitorder="little")
            # loop over each neighbor
            for u,u_val in zip(neighbors,yF):
                if len(neighbors)==1:
                    pmprodsum_edges.append([2*u+u_val, d])
                pmprod_index_map[F,vals,u,u_val] = d
                for v,v_val in zip(neighbors,yF):
                    if v!=u:
                        pmprod_edges.append([d, 2*v+v_val])
                        pmprodsum_edges.append([2*u+u_val, d])
                d += 1
            c += 1
    a,b = torch.tensor(pmprod_edges),torch.tensor(pmprodsum_edges)

    return a,b,d

def naive_mean_field_edges(fg):
    pmprod_edges = []
    pmprodsum_edges = []
    d = 0  # running index over product-summand “slots” (size = sum_F 2^{nF} * nF)

    for F in fg.iter_factors():
        neighbors = list(fg.factor_variables(F))
        nN = len(neighbors)

        # enumerate all assignments to factor variables; bits: y_k = ((vals >> k) & 1)
        # use shifts to support nN > 8
        for vals in range(1 << nN):
            yF = ((vals >> np.arange(nN, dtype=np.int64)) & 1).astype(np.int32)

            # for each “center” variable u in this factor (one slot per (F, vals, u))
            for u_idx, u in enumerate(neighbors):
                u_val = int(yF[u_idx])

                # multiply contributions from all other variables v != u into slot d
                for v_idx, v in enumerate(neighbors):
                    if v_idx == u_idx:
                        continue
                    v_val = int(yF[v_idx])
                    pmprod_edges.append([d, 2*v + v_val])

                # sum this slot exactly once into the destination state of u
                pmprodsum_edges.append([2*u + u_val, d])

                d += 1

    pmprod_edges = torch.tensor(pmprod_edges, dtype=torch.long)
    pmprodsum_edges = torch.tensor(pmprodsum_edges, dtype=torch.long)

    # expected counts per factor
    exp_slots = sum( (1 << fg.factor_size(F)) * fg.factor_size(F) for F in fg.iter_factors() )
    exp_pmprod = sum( (1 << fg.factor_size(F)) * fg.factor_size(F) * max(fg.factor_size(F)-1, 0) for F in fg.iter_factors() )
    exp_pmprodsum = exp_slots
    
    assert d == exp_slots
    assert pmprod_edges.shape[0] == exp_pmprod
    assert pmprodsum_edges.shape[0] == exp_pmprodsum

    return pmprod_edges, pmprodsum_edges, d
