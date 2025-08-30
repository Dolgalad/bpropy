import numpy as np
import torch

def lexsort(e):
    x = e.numpy()
    return torch.from_numpy(x[np.lexsort((x[:,1], x[:,0]))])

"""Computing Loopy Belief Propagation computation graph edges
"""
class LBPEdges:
    def __init__(self, fg=None):
        self.n_ss = None
        self.n_vars = None
        self.n_factors = None
        self.n_f2v_ss = None
        self.n_f2v_msg = None

        self.f2v_edges = None
        self.r_edges = None
        self.q_edges = None
        self.m_edges = None
        self.vm_edges = None
        self.m_stride = None
        if not fg is None:
            self.fill(fg)
            self.sort_edges()

    def sort_edges(self):
        self.f2v_edges = lexsort(self.f2v_edges)
        self.r_edges = lexsort(self.r_edges)
        self.q_edges = lexsort(self.q_edges)
        self.m_edges = lexsort(self.m_edges)
        self.vm_edges = lexsort(self.vm_edges)



    @property
    def device(self):
        return self.n_ss.device
    @property
    def is_cuda(self):
        return self.n_ss.is_cuda and \
                self.n_vars.is_cuda and \
                self.n_factors.is_cuda and \
                self.n_f2v_ss.is_cuda and \
                self.n_f2v_msg.is_cuda and \
                self.f2v_edges.is_cuda and \
                self.r_edges.is_cuda and \
                self.q_edges.is_cuda and \
                self.m_edges.is_cuda and \
                self.vm_edges.is_cuda and \
                self.m_stride.is_cuda
    def fill(self, fg):
        if fg.num_graphs == 1:
            #import time
            #t0 = time.time()
            self.n_ss = torch.tensor([fg.number_sufficient_statistics()])
            self.n_vars = torch.tensor([fg.nv()])
            self.n_factors = torch.tensor([fg.nf()])
            # create the computation edges
            #f2v_idx, f2v_subsum_idx,ss_idx = f2v_index_map(fg)
            # creating message passing edges

            # try more efficient edge computation


            self.n_f2v_ss = torch.tensor([fg.num_r_edges()])

            self.n_f2v_msg = torch.tensor([sum(2*fg.factor_size(F) for F in fg.iter_factors())])
            self.m_stride = torch.tensor([2**len(list(fg.neighbors(F))) for F in fg.iter_factors()])
            self.f2v_edges, self.r_edges, self.q_edges, self.m_edges, self.vm_edges = make_lbp_edges(fg)
            self.f2v_edges, self.r_edges, self.q_edges, self.m_edges, self.vm_edges = make_lbp_edges_6(fg)


        else:
            edge_list = [LBPEdges(fg.get_subgraph(i)) for i in range(fg.num_graphs)]
            edges_batch = LBPEdges.batch(edge_list)
            self.n_ss      = edges_batch.n_ss      
            self.n_vars    = edges_batch.n_vars    
            self.n_factors = edges_batch.n_factors 
            self.n_f2v_ss  = edges_batch.n_f2v_ss  
            self.n_f2v_msg = edges_batch.n_f2v_msg 
            self.f2v_edges = edges_batch.f2v_edges
            self.r_edges   = edges_batch.r_edges
            self.q_edges   = edges_batch.q_edges
            self.m_edges   = edges_batch.m_edges
            self.vm_edges  = edges_batch.vm_edges

            self.m_stride  = edges_batch.m_stride  

    @property
    def num_graphs(self):
        return self.n_ss.shape[0]
    def graph_indicator(self):
        if self.num_graphs == 1:
            return torch.zeros(self.n_ss, dtype=torch.int16, device=self.device)
        return torch.arange(self.num_graphs, device=self.device).repeat_interleave(self.n_ss)
    def clone(self):
        ne = LBPEdges()
        ne.n_ss = self.n_ss.clone()
        ne.n_vars = self.n_vars.clone()
        ne.n_factors = self.n_factors.clone()
        ne.n_f2v_ss = self.n_f2v_ss.clone()
        ne.n_f2v_msg = self.n_f2v_msg.clone()

        ne.f2v_edges = self.f2v_edges.clone()
        ne.r_edges = self.r_edges.clone()
        ne.q_edges = self.q_edges.clone()
        ne.m_edges = self.m_edges.clone()
        ne.vm_edges = self.vm_edges.clone()
        ne.m_stride = self.m_stride.clone()
        return ne
    def to(self, device, non_blocking=False, *args):
        ne = LBPEdges()
        ne.n_ss = self.n_ss.to(device)
        ne.n_vars = self.n_vars.to(device)
        ne.n_factors = self.n_factors.to(device)
        ne.n_f2v_ss = self.n_f2v_ss.to(device)
        ne.n_f2v_msg = self.n_f2v_msg.to(device)

        ne.f2v_edges = self.f2v_edges.to(device)
        ne.r_edges = self.r_edges.to(device)
        ne.q_edges = self.q_edges.to(device)
        ne.m_edges = self.m_edges.to(device)
        ne.vm_edges = self.vm_edges.to(device)
        ne.m_stride = self.m_stride.to(device)
        return ne
    def __getitem__(self, idx):
        if self.num_graphs==1:
            return self

        ne = LBPEdges()
        ne.n_ss = self.n_ss[idx].unsqueeze(0)
        ne.n_vars = self.n_vars[idx].unsqueeze(0)
        ne.n_factors = self.n_factors[idx].unsqueeze(0)
        ne.n_f2v_ss = self.n_f2v_ss[idx].unsqueeze(0)
        ne.n_f2v_msg = self.n_f2v_msg[idx].unsqueeze(0)

        r_offsets = torch.roll(torch.cumsum(self.n_f2v_ss,0), 1)
        r_offsets[0] = 0

        ss_offsets = torch.roll(torch.cumsum(self.n_ss,0), 1)
        ss_offsets[0] = 0

        vars_offsets = torch.roll(torch.cumsum(self.n_vars,0), 1)
        vars_offsets[0] = 0

        factors_offsets = torch.roll(torch.cumsum(self.n_factors, 0), 1)
        factors_offsets[0] = 0


        f2v_ss_offsets = torch.roll(torch.cumsum(self.n_f2v_ss,0), 1)
        f2v_ss_offsets[0] = 0

        f2v_msg_offsets = torch.roll(torch.cumsum(self.n_f2v_msg,0), 1)
        f2v_msg_offsets[0] = 0

        aa = ss_offsets[idx].item()
        bb = (ss_offsets[idx]+self.n_ss[idx]).item()

        #cc = (self.n_ss.sum()+f2v_msg_offsets[idx]+2*self.n_vars[idx]).item() # TODO: find out why I had to add 2*self.n_vars[idx]
        #dd = (self.n_ss.sum()+f2v_msg_offsets[idx]+self.n_f2v_msg[idx]).item()

        # fix
        cc = (self.n_ss.sum() + f2v_msg_offsets[idx]).item()
        dd = (self.n_ss.sum() + f2v_msg_offsets[idx] + self.n_f2v_msg[idx]).item()

        idx_less = (aa <= self.f2v_edges[:,1]) * (self.f2v_edges[:,1] < bb)
        idx_more = (cc <= self.f2v_edges[:,1]) * (self.f2v_edges[:,1] < dd)

        less_offset = torch.tensor([f2v_ss_offsets[idx], ss_offsets[idx]], device=self.device)
        #more_offset = torch.tensor([f2v_ss_offsets[idx], self.n_ss.sum()+f2v_msg_offsets[idx]-self.n_ss[idx]], device=self.device)
        # fix
        more_offset = torch.tensor([f2v_ss_offsets[idx], self.n_ss.sum()+f2v_msg_offsets[idx]], device=self.device)

        less_offset = torch.tensor([f2v_ss_offsets[idx], ss_offsets[idx]], device=self.device)
        more_offset = torch.tensor(
            [f2v_ss_offsets[idx], self.n_ss.sum() + f2v_msg_offsets[idx] - self.n_ss[idx]],
            device=self.device
        )

        new_f2v_edges = self.f2v_edges[torch.logical_or(idx_less, idx_more)].clone()
        new_idx_less = (aa <= new_f2v_edges[:,1]) * (new_f2v_edges[:,1] < bb)
        new_idx_more = (cc <= new_f2v_edges[:,1]) * (new_f2v_edges[:,1] < dd)
        new_f2v_edges[new_idx_less] -= less_offset
        new_f2v_edges[new_idx_more] -= more_offset

        ne.f2v_edges = new_f2v_edges
        ne.r_edges = self.r_edges[r_offsets[idx]:r_offsets[idx]+self.n_f2v_ss[idx]].clone()
        ne.r_edges -= ne.r_edges.min(dim=0)[0]
        ne.m_edges = self.m_edges[r_offsets[idx]:r_offsets[idx]+self.n_f2v_ss[idx]].clone()
        ne.m_edges -= ne.m_edges.min(dim=0)[0]

        ne.q_edges = self.q_edges[(f2v_msg_offsets[idx] <= self.q_edges[:,1]) * (self.q_edges[:,1] < f2v_msg_offsets[idx]+self.n_f2v_msg[idx])].clone() - f2v_msg_offsets[idx]
        ne.vm_edges = self.vm_edges[(2*vars_offsets[idx] <= self.vm_edges[:,0]) * (self.vm_edges[:,0] < 2*(vars_offsets[idx]+self.n_vars[idx]))].clone() - torch.tensor([2*vars_offsets[idx], f2v_msg_offsets[idx]], device=self.device)
        ne.m_stride = self.m_stride[factors_offsets[idx]:factors_offsets[idx]+self.n_factors[idx]].clone()

        return ne

    @staticmethod
    def show_diff(a, b):
        print("same n_ss      = ", torch.all(a.n_ss == b.n_ss) )
        print("same n_vars    = ", torch.all(a.n_vars == b.n_vars) )
        print("same n_factors = ", torch.all(a.n_factors == b.n_factors) )
        print("same n_f2v_ss  = ", torch.all(a.n_f2v_ss == b.n_f2v_ss) )
        print("same n_f2v_msg = ", torch.all(a.n_f2v_msg == b.n_f2v_msg)) 
        print("same f2v_edges.shape = ", a.f2v_edges.shape == b.f2v_edges.shape)
        print("same r_edges  shape  = ", a.r_edges.shape ==   b.r_edges.shape) 
        print("same q_edges  shape  = ", a.q_edges.shape ==   b.q_edges.shape) 
        print("same m_edges  shape  = ", a.m_edges.shape ==   b.m_edges.shape) 
        print("same vm_edges shape  = ", a.vm_edges.shape ==  b.vm_edges.shape )
        print("same m_stride shape  = ", a.m_stride.shape ==  b.m_stride.shape)
        if a.f2v_edges.shape == b.f2v_edges.shape:
            print("same f2v_edges = ", torch.all(a.f2v_edges== b.f2v_edges))
        else:
            print("a.shape = ", a.f2v_edges.shape, ", b.shape = ", b.f2v_edges.shape)
        if a.r_edges.shape == b.r_edges.shape:
            print("same r_edges   = ", torch.all(a.r_edges == b.r_edges) )
        if a.q_edges.shape == b.q_edges.shape:
            print("same q_edges   = ", torch.all(a.q_edges == b.q_edges) )
        if a.m_edges.shape == b.m_edges.shape:
            print("same m_edges   = ", torch.all(a.m_edges == b.m_edges) )
        if a.vm_edges.shape == b.vm_edges.shape:
            print("same vm_edges  = ", torch.all(a.vm_edges == b.vm_edges ))
        if a.m_stride.shape == b.m_stride.shape:
            print("same m_stride  = ", torch.all(a.m_stride == b.m_stride))



    def __eq__(self, other):

        if (self.n_ss.shape != other.n_ss.shape) or \
           (self.n_vars.shape != other.n_vars.shape) or \
           (self.n_factors.shape != other.n_factors.shape) or \
           (self.n_f2v_ss.shape != other.n_f2v_ss.shape) or \
           (self.n_f2v_msg.shape != other.n_f2v_msg.shape) or \
           (self.f2v_edges.shape != other.f2v_edges.shape) or \
           (self.r_edges.shape != other.r_edges.shape) or \
           (self.q_edges.shape != other.q_edges.shape) or \
           (self.m_edges.shape != other.m_edges.shape) or \
           (self.vm_edges.shape != other.vm_edges.shape) or \
           (self.m_stride.shape != other.m_stride.shape):
            return False




        return torch.all(self.n_ss == other.n_ss) and \
               torch.all(self.n_vars == other.n_vars) and \
               torch.all(self.n_factors == other.n_factors) and \
               torch.all(self.n_f2v_ss == other.n_f2v_ss) and \
               torch.all(self.n_f2v_msg == other.n_f2v_msg) and \
               torch.all(self.f2v_edges == other.f2v_edges) and \
               torch.all(self.r_edges == other.r_edges) and \
               torch.all(self.q_edges == other.q_edges) and \
               torch.all(self.m_edges == other.m_edges) and \
               torch.all(self.vm_edges == other.vm_edges) and \
               torch.all(self.m_stride == other.m_stride)

    @staticmethod
    def stack(edges1, edges2):
        num_f2v_subsum = edges1.f2v_edges[:,0].max().item()+1
        num_f2v_messages_1 = edges1.n_f2v_msg.sum()
        num_ss_1 = edges1.n_ss.sum()
        num_ss_2 = edges2.n_ss.sum()

        # edge offset values
        #r_offsets = torch.tensor([num_f2v_subsum, num_f2v_messages_1])
        r_offsets = torch.tensor([num_f2v_messages_1, num_f2v_subsum])
        q_offsets = torch.tensor([num_f2v_messages_1,num_f2v_messages_1]);
        m_offsets = torch.tensor([num_ss_1, num_f2v_messages_1])
        vm_offsets = torch.tensor([2*edges1.n_vars.sum(), num_f2v_messages_1])

        edges1.r_edges = torch.concatenate((edges1.r_edges, edges2.r_edges + r_offsets), dim=0)
        edges1.q_edges = torch.concatenate((edges1.q_edges, edges2.q_edges + q_offsets), dim=0)
    
        tmp_f2v_edges_1 = edges1.f2v_edges.clone()
        idx0_more = (tmp_f2v_edges_1[:,1] >= num_ss_1)
        tmp_f2v_edges_1[idx0_more,:] = tmp_f2v_edges_1[idx0_more,:] + torch.tensor([0, num_ss_2])

        tmp_f2v_edges_2 = edges2.f2v_edges.clone()
        idx_less = (tmp_f2v_edges_2[:,1] < num_ss_2)
        idx_more = (tmp_f2v_edges_2[:,1] >= num_ss_2)
        tmp_f2v_edges_2[idx_less,:] = tmp_f2v_edges_2[idx_less,:] + torch.tensor([num_f2v_subsum, num_ss_1])


        tmp_f2v_edges_2[idx_more,:] = tmp_f2v_edges_2[idx_more,:] + torch.tensor([num_f2v_subsum, num_ss_1+num_f2v_messages_1])
        edges1.f2v_edges = torch.concatenate((tmp_f2v_edges_1, tmp_f2v_edges_2), dim=0)
    
        edges1.m_edges = torch.concatenate((edges1.m_edges, edges2.m_edges+m_offsets))
        edges1.vm_edges = torch.concatenate((edges1.vm_edges, edges2.vm_edges+vm_offsets))
        edges1.m_stride = torch.concatenate((edges1.m_stride, edges2.m_stride))
    
        edges1.n_vars = torch.concatenate((edges1.n_vars,edges2.n_vars))
        edges1.n_ss = torch.concatenate((edges1.n_ss,edges2.n_ss))
        edges1.n_factors = torch.concatenate((edges1.n_factors,edges2.n_factors))
        edges1.n_f2v_ss = torch.concatenate((edges1.n_f2v_ss, edges2.n_f2v_ss))
        edges1.n_f2v_msg = torch.concatenate((edges1.n_f2v_msg, edges2.n_f2v_msg))
    @staticmethod
    def batch_alt(edge_l):
        e = edge_l[0].clone()
        for i in range(1, len(edge_l)):
            LBPEdges.stack(e, edge_l[i])
        return e
    @staticmethod
    def batch(edge_l):
        """Prepare edges to concatenate in one operation
        """
        # compute cumulated edge sizes
        n_r_edges, n_q_edges, n_f2v_edges, n_m_edges, n_vm_edges, n_m_stride, n_n_vars, n_n_factors, n_n_ss, n_n_f2v_ss, n_n_f2v_msg, n_ss_tot = 0,0,0,0,0,0,0,0,0,0,0,0
        for e in edge_l:
            n_r_edges += e.r_edges.shape[0]
            n_q_edges += e.q_edges.shape[0]
            n_f2v_edges += e.f2v_edges.shape[0]
            n_m_edges += e.m_edges.shape[0]
            n_vm_edges += e.vm_edges.shape[0]
            n_m_stride += e.m_stride.shape[0]
            n_n_vars += e.n_vars.shape[0]
            n_n_factors += e.n_factors.shape[0]
            n_n_ss += e.n_ss.shape[0]
            n_n_f2v_ss += e.n_f2v_ss.shape[0]
            n_n_f2v_msg += e.n_f2v_msg.shape[0]
            n_ss_tot += e.n_ss.sum().item()
        ne = LBPEdges()
        ne.r_edges = torch.empty(n_r_edges,2,dtype=torch.long)
        ne.q_edges = torch.empty(n_q_edges,2,dtype=torch.long)
        ne.f2v_edges = torch.empty(n_f2v_edges,2,dtype=torch.long)
        ne.m_edges = torch.empty(n_m_edges,2,dtype=torch.long)
        ne.vm_edges = torch.empty(n_vm_edges,2,dtype=torch.long)
        ne.m_stride = torch.empty(n_m_stride,dtype=torch.long)
        ne.n_vars = torch.empty(n_n_vars, dtype=torch.long)
        n_vars_pos = 0
        ne.n_ss = torch.empty(n_n_ss, dtype=torch.long)
        n_ss_pos = 0
        ne.n_factors = torch.empty(n_n_factors, dtype=torch.long)
        n_factors_pos = 0
        ne.n_f2v_ss = torch.empty(n_n_f2v_ss, dtype=torch.long)
        n_f2v_ss_pos = 0
        ne.n_f2v_msg = torch.empty(n_n_f2v_msg, dtype=torch.long)
        n_f2v_msg_pos = 0

        r_offset = torch.tensor([0, 0])
        r_pos = 0
        q_offset = torch.tensor([0, 0])
        q_pos = 0
        f2v_pos = 0
        m_offset = torch.tensor([0, 0])
        m_pos = 0
        vm_offset = torch.tensor([0, 0])
        vm_pos = 0
        m_stride_pos = 0
        n_ss_offset = 0
        n_f2v_msg_offset = 0
        n_f2v_subsum_offset = 0
        for i,e in enumerate(edge_l):
            ne.r_edges[r_pos:r_pos+e.r_edges.shape[0]] = e.r_edges + r_offset
            r_pos += e.r_edges.shape[0]

            ne.q_edges[q_pos:q_pos+e.q_edges.shape[0]] = e.q_edges + q_offset
            q_pos += e.q_edges.shape[0]

            # update r_edges, q_edges offset value
            num_f2v_subsum = e.f2v_edges[:,0].max().item()+1
            num_f2v_messages_1 = e.n_f2v_msg.sum()


            r_offset += torch.tensor([num_f2v_messages_1, num_f2v_subsum])
            q_offset += torch.tensor([num_f2v_messages_1, num_f2v_messages_1])

            # f2v_edges
            n_ss = e.n_ss.sum()
            tmp_f2v_edges = e.f2v_edges.clone()
            idx_less = (tmp_f2v_edges[:,1] < n_ss)
            idx_more = (tmp_f2v_edges[:,1] >= n_ss)

            tmp_f2v_edges[idx_less,:] = tmp_f2v_edges[idx_less,:] + torch.tensor([n_f2v_subsum_offset, n_ss_offset])
            #tmp_f2v_edges[idx_more,:] = tmp_f2v_edges[idx_more,:] + torch.tensor([n_f2v_subsum_offset, n_ss_tot+n_f2v_msg_offset-n_ss])
            tmp_f2v_edges[idx_more,:] = tmp_f2v_edges[idx_more,:] + torch.tensor([n_f2v_subsum_offset, n_ss_tot + n_f2v_msg_offset-n_ss])

            n_ss_offset += n_ss
            n_f2v_msg_offset += num_f2v_messages_1
            n_f2v_subsum_offset += num_f2v_subsum
            ne.f2v_edges[f2v_pos:f2v_pos+tmp_f2v_edges.shape[0]] = tmp_f2v_edges
            f2v_pos += tmp_f2v_edges.shape[0]

            # m_edges, vm_edges, m_stride
            ne.m_edges[m_pos:m_pos+e.m_edges.shape[0]] = e.m_edges+m_offset
            m_pos += e.m_edges.shape[0]

            ne.vm_edges[vm_pos:vm_pos+e.vm_edges.shape[0]] = e.vm_edges+vm_offset
            vm_pos += e.vm_edges.shape[0]

            ne.m_stride[m_stride_pos:m_stride_pos+e.m_stride.shape[0]] = e.m_stride
            m_stride_pos += e.m_stride.shape[0]

            m_offset += torch.tensor([n_ss, num_f2v_messages_1])
            vm_offset += torch.tensor([2*e.n_vars.sum(), num_f2v_messages_1])

            # scalars
            ne.n_vars[n_vars_pos:n_vars_pos+e.n_vars.shape[0]] = e.n_vars
            n_vars_pos += e.n_vars.shape[0]

            ne.n_ss[n_ss_pos:n_ss_pos+e.n_ss.shape[0]] = e.n_ss
            n_ss_pos += e.n_ss.shape[0]

            ne.n_factors[n_factors_pos:n_factors_pos+e.n_factors.shape[0]] = e.n_factors
            n_factors_pos += e.n_factors.shape[0]

            ne.n_f2v_ss[n_f2v_ss_pos:n_f2v_ss_pos+e.n_f2v_ss.shape[0]] = e.n_f2v_ss
            n_f2v_ss_pos += e.n_f2v_ss.shape[0]

            #ne.n_f2v_msg[n_f2v_msg_pos:n_f2v_msg_pos+ne.n_f2v_msg.shape[0]] = e.n_f2v_msg
            #n_f2v_msg_pos += e.n_f2v_msg.shape[0]

            # fix
            ne.n_f2v_msg[n_f2v_msg_pos : n_f2v_msg_pos + e.n_f2v_msg.shape[0]] = e.n_f2v_msg
            n_f2v_msg_pos += e.n_f2v_msg.shape[0]

        return ne

    


"""Number of variable marginal edges
"""
def number_vm_edges(fg):
    return sum([2*len(list(fg.neighbors(n))) for n in fg.iter_variables()])

def f2v_messages_dim(fg):
    return sum([2*fg.factor_size(F) for F in fg.iter_factors()])

def get_edge_dimensions(fg):
    f2v_edges_n = 0
    r_edges_n = 0
    q_edges_n = 0
    vm_edges_n = 0
    for F in fg.iter_factors():
        fs = fg.factor_size(F)
        f2v_edges_n += np.power(2, fs) * np.power(fs,2)
        r_edges_n += fs * np.power(2,fs)
        for v in fg.factor_variables(F):
            q_edges_n += 2*(len(fg.containing_factors(v))-1)
    for v in fg.iter_variables():
        vm_edges_n += 2*len(fg.containing_factors(v))

    m_edges_n = r_edges_n
    return f2v_edges_n, r_edges_n, q_edges_n, m_edges_n, vm_edges_n

def make_lbp_edges(fg):
    nSS = fg.nss()
    factor_value_offsets = np.roll((2**fg.factor_sizes()).cumsum(), 1)
    factor_value_offsets[0] = 0

    factor_ss_value_offsets = np.roll((2*fg.factor_sizes()).cumsum(), 1)
    factor_ss_value_offsets[0] = 0


    f2v_edges_n, r_edges_n, q_edges_n, m_edges_n, vm_edges_n = get_edge_dimensions(fg)

    #f2v_edges_n = fg.num_f2v_edges()
    f2v_edges_c=0
    #f2v_edges = torch.zeros((f2v_edges_n,2), dtype=torch.long)
    #f2v_edges_vec = [None for _ in range(f2v_edges_n)]
    f2v_edges_vec = np.empty((f2v_edges_n, 2), dtype=int)
  
    #r_edges_n = fg.num_r_edges()
    r_edges_c=0
    #r_edges = torch.zeros((r_edges_n,2), dtype=torch.long)
    #r_edges_vec = [None for _ in range(r_edges_n)]
    r_edges_vec = np.empty((r_edges_n, 2), dtype=int)
   
    #q_edges_n = fg.num_q_edges()
    q_edges_c=0
    #q_edges = torch.zeros((q_edges_n,2), dtype=torch.long)
    #q_edges_vec = [None for _ in range(q_edges_n)]
    q_edges_vec = np.empty((q_edges_n, 2), dtype=int)

    #
    #m_edges_n = fg.num_m_edges()
    m_edges_c=0
    #m_edges = torch.zeros((m_edges_n,2), dtype=torch.long)
    #m_edges_vec = [None for _ in range(m_edges_n)]
    m_edges_vec = np.empty((m_edges_n, 2), dtype=int)

    #
    #vm_edges_n = fg.num_vm_edges()
    vm_edges_c=0
    #vm_edges = torch.zeros((vm_edges_n,2), dtype=torch.long)
    #vm_edges_vec = [None for _ in range(vm_edges_n)]
    vm_edges_vec = np.empty((vm_edges_n, 2), dtype=int)

    fmap = fg.factor_index_map()
    fvmap = fg.factor_variables_map()
    cfmap = fg.containing_factors_map()

    for iV,V in enumerate(fg.iter_variables()):
        factors = cfmap[V]
        for iF,F in enumerate(factors):
            fvn = fvmap[F]

            iidx = fvn.index(V)
            for v in [0,1]:
                vm_edges_vec[vm_edges_c] = [2*iV+v, factor_ss_value_offsets[fmap[F]]+2*iidx+v]
                vm_edges_c += 1;

    for iF,F in enumerate(fg.iter_factors()):
        factor_size = fg.factor_size(F)

        #factor_neighbors = fg.factor_variables(F)
        factor_neighbors = fvmap[F]
        for vnii,vneighbor in enumerate(factor_neighbors):
            variable_neighbors = cfmap[vneighbor]
            for vni in [0, 1]:
                for fni,fneighbor in enumerate(variable_neighbors):
                    if(F!=fneighbor):
                        fvn = fvmap[fneighbor]

                        iidx = fvn.index(vneighbor)

                        q_edges_vec[q_edges_c] = [factor_ss_value_offsets[fmap[F]]+2*vnii+vni,factor_ss_value_offsets[fmap[fneighbor]]+2*iidx+vni]
                        q_edges_c += 1;



        for v in range(2**factor_size):
            vals = np.array([(v >> k) & 1 for k in range(factor_size)], dtype=np.uint8)
            #vals = np.unpackbits(np.array([v],dtype=np.uint8), count=factor_size, bitorder="little")
            for i_n in range(factor_size):
                m_edges_vec[m_edges_c] = [factor_value_offsets[fmap[F]]+v,factor_ss_value_offsets[fmap[F]]+2*i_n+vals[i_n]]
                m_edges_c += 1


        
                for yvi in [0,1]:
                    if vals[i_n]==yvi:
                        r_edges_vec[r_edges_c] = [factor_ss_value_offsets[fmap[F]]+2*i_n+yvi,r_edges_c]


                        f2v_edges_vec[f2v_edges_c] = [r_edges_c,factor_value_offsets[fmap[F]]+v]
                        f2v_edges_c += 1


                        for iu in range(factor_size):
                            if iu!=i_n:
                                f2v_edges_vec[f2v_edges_c] = [r_edges_c,nSS+factor_ss_value_offsets[fmap[F]]+2*iu+vals[iu]]
                                f2v_edges_c += 1

        
                        r_edges_c += 1

    return torch.from_numpy(f2v_edges_vec), torch.from_numpy(r_edges_vec), torch.from_numpy(q_edges_vec), torch.from_numpy(m_edges_vec), torch.from_numpy(vm_edges_vec)

    #return torch.tensor(f2v_edges_vec, dtype=torch.long), torch.tensor(r_edges_vec, dtype=torch.long), torch.tensor(q_edges_vec, dtype=torch.long), torch.tensor(m_edges_vec, dtype=torch.long), torch.tensor(vm_edges_vec, dtype=torch.long)
    #return f2v_edges, r_edges, q_edges, m_edges, vm_edges

def make_lbp_edges_2(fg):
    """
    Vectorized builder for LBP wiring.
    Returns: f2v_edges, r_edges, q_edges, m_edges, vm_edges as torch.long.
    Conventions:
      - f2v_edges[:, 0] = r-slot id (pre–logsumexp accumulator)
      - f2v_edges[:, 1] = source in the STACK = [ SS | V2F ]
      - r_edges   [:, 0] = destination v2f message id, [:,1] = r-slot id
      - q_edges   [:, 0] = destination v2f message id, [:,1] = source v2f
      - m_edges   [:, 0] = destination SS index, [:,1] = source v2f id
      - vm_edges  [:, 0] = destination variable-belief slot (2*i + yi), [:,1] = source v2f id
    """
    #import numpy as _np
    #import torch as _torch

    # --- Factor/variable metadata ---
    factors = list(fg.iter_factors())
    fsizes  = np.array([fg.factor_size(F) for F in factors], dtype=np.int64)
    two_pow = (1 << fsizes).astype(np.int64)

    # SS offsets per factor
    ss_cum = np.roll(two_pow.cumsum(), 1); ss_cum[0] = 0
    # V2F offsets per factor
    v2f_per_factor = 2 * fsizes
    v2f_cum = np.roll(v2f_per_factor.cumsum(), 1); v2f_cum[0] = 0
    # r-slot offsets per factor
    r_per_factor = fsizes * two_pow
    r_cum = np.roll(r_per_factor.cumsum(), 1); r_cum[0] = 0

    # Variable side
    vars_list = list(fg.iter_variables())
    F_of_v = {v: list(fg.containing_factors(v)) for v in vars_list}
    var_order = {F: fg.factor_variables(F) for F in factors}

    def bit_matrix(nF: int) -> np.ndarray:
        a = np.arange(1 << nF, dtype=np.int64)[:, None]
        k = np.arange(nF, dtype=np.int64)[None, :]
        return ((a >> k) & 1).astype(np.int64)

    # ---------- Exact sizes (key fix is f2v count) ----------
    # Prefer the baseline's authoritative count if exposed:
    try:
        n_f2v_edges = int(fg.num_f2v_edges())
    except AttributeError:
        # mathematically exact: sum_F 2^{|F|} * |F|^2
        n_f2v_edges = int((two_pow * (fsizes ** 2)).sum())

    n_r_edges  = int((fsizes * two_pow).sum())  # sum_F 2^{|F|} * |F|
    # For each variable v: 2 * deg(v) * (deg(v)-1)
    n_q_edges  = int(np.sum([2 * (len(F_of_v[v]) * (len(F_of_v[v]) - 1)) for v in vars_list]))
    n_m_edges  = n_r_edges
    n_vm_edges = int(np.sum([2 * len(F_of_v[v]) for v in vars_list]))

    f2v_edges = np.empty((n_f2v_edges, 2), dtype=np.int64)
    r_edges   = np.empty((n_r_edges,    2), dtype=np.int64)
    q_edges   = np.empty((n_q_edges,    2), dtype=np.int64)
    m_edges   = np.empty((n_m_edges,    2), dtype=np.int64)
    vm_edges  = np.empty((n_vm_edges,   2), dtype=np.int64)

    # Cursors
    c_f2v = c_r = c_q = c_m = c_vm = 0

    # ---------- vm_edges ----------
    var_index = {v: i for i, v in enumerate(vars_list)}
    for v in vars_list:
        vidx = var_index[v]
        for F in F_of_v[v]:
            i_local = var_order[F].index(v)
            v2f0 = int(v2f_cum[factors.index(F)])
            vm_edges[c_vm + 0] = [2 * vidx + 0, v2f0 + 2 * i_local + 0]
            vm_edges[c_vm + 1] = [2 * vidx + 1, v2f0 + 2 * i_local + 1]
            c_vm += 2

    # ---------- q_edges ----------
    for v in vars_list:
        inc = F_of_v[v]
        if len(inc) <= 1: continue
        for Fdst in inc:
            i_dst = var_order[Fdst].index(v)
            v2f0_d = int(v2f_cum[factors.index(Fdst)])
            for Fsrc in inc:
                if Fsrc is Fdst: continue
                i_src = var_order[Fsrc].index(v)
                v2f0_s = int(v2f_cum[factors.index(Fsrc)])
                q_edges[c_q + 0] = [v2f0_d + 2 * i_dst + 0, v2f0_s + 2 * i_src + 0]
                q_edges[c_q + 1] = [v2f0_d + 2 * i_dst + 1, v2f0_s + 2 * i_src + 1]
                c_q += 2

    # ---------- Per-factor blocks: r_edges, m_edges, f2v_edges ----------
    SS_total = int(two_pow.sum())
    for fidx, F in enumerate(factors):
        nF  = int(fsizes[fidx])
        twf = int(two_pow[fidx])
        ss0, v2f0, r0 = int(ss_cum[fidx]), int(v2f_cum[fidx]), int(r_cum[fidx])

        vals = bit_matrix(nF)                        # [twf, nF]
        r_slots_grid = (r0 + np.arange(twf)[:,None]*nF + np.arange(nF)[None,:]).astype(np.int64)

        # r_edges (one per (a,i))
        dst_r = (v2f0 + 2 * np.arange(nF)[None,:] + vals).astype(np.int64)   # [twf, nF]
        n_this = twf * nF
        r_edges[c_r:c_r+n_this, 0] = dst_r.ravel(order='C')
        r_edges[c_r:c_r+n_this, 1] = r_slots_grid.ravel(order='C')
        c_r += n_this

        # m_edges (one per (a,i))
        m_edges[c_m:c_m+n_this, 0] = (ss0 + np.arange(twf)).repeat(nF)
        m_edges[c_m:c_m+n_this, 1] = (v2f0 + 2 * np.arange(nF)[None,:] + vals).ravel(order='C')
        c_m += n_this

        # f2v_edges: SS→r
        f2v_edges[c_f2v:c_f2v+n_this, 0] = r_slots_grid.ravel(order='C')
        f2v_edges[c_f2v:c_f2v+n_this, 1] = (ss0 + np.arange(twf)).repeat(nF)
        c_f2v += n_this

        # --- f2v_edges: q→r (from all j≠i, with state = y_j in assignment)
        if nF > 1:
            # v2f ids per (i, y) for this factor: shape [nF, 2]
            mid = (v2f0 + 2 * np.arange(nF)[:, None] + np.arange(2)[None, :]).astype(np.int64)

            # src_per_assign[i, a] = v2f id for variable i in assignment a
            # vals: [twf, nF] -> transpose to [nF, twf] to index mid[:, y]
            src_per_assign = mid[np.arange(nF)[:, None], vals.T]   # [nF, twf], int64

            # Build 'others' = per-destination i, all j≠i (shape [nF, nF-1])
            all_idx = np.arange(nF, dtype=np.int64)
            tiled = np.tile(all_idx, (nF, 1))                      # [nF, nF]
            mask = ~np.eye(nF, dtype=bool)                         # True where j≠i
            others = tiled[mask].reshape(nF, nF - 1)                # [nF, nF-1]

            # Gather sources: for each i, take assignments from all j≠i
            # Result: [nF, nF-1, twf] → [twf, nF, nF-1] → flatten
            src_q = src_per_assign[others, :]                       # [nF, nF-1, twf]
            src_q = src_q.transpose(2, 0, 1).reshape(-1)            # length = twf*nF*(nF-1)

            # Destinations: for each (a, i) we have (nF-1) contributors
            dst_q = np.repeat(r_slots_grid, nF - 1, axis=1).ravel(order="C")

            n_qc = src_q.shape[0]
            # sources live after SS block in the global STACK
            f2v_edges[c_f2v:c_f2v + n_qc, 0] = dst_q
            f2v_edges[c_f2v:c_f2v + n_qc, 1] = SS_total + src_q
            c_f2v += n_qc

            # sanity: per-factor expected q→r size
            expected_qc = twf * nF * (nF - 1)
            if n_qc != expected_qc:
                raise RuntimeError(
                    f"q→r count mismatch for factor {fidx}: got {n_qc}, expected {expected_qc} "
                    f"(nF={nF}, twf={twf})"
                )
            #print("fidx:", fidx, "nF:", nF, "twf:", twf,
            #      "expected_qc:", twf * nF * (nF - 1),
            #      "actual n_qc:", n_qc,
            #      "r_slots_grid.shape:", r_slots_grid.shape,
            #      "src_per_assign.shape:", src_per_assign.shape)
    # sanity
    

    assert c_r   == n_r_edges
    assert c_q   == n_q_edges
    assert c_m   == n_m_edges
    assert c_vm  == n_vm_edges
    assert c_f2v == n_f2v_edges

    return (torch.from_numpy(f2v_edges),
            torch.from_numpy(r_edges),
            torch.from_numpy(q_edges),
            torch.from_numpy(m_edges),
            torch.from_numpy(vm_edges))

def make_lbp_edges_3(fg):
    import numpy as np
    import torch

    factors = list(fg.iter_factors())
    fsizes  = np.array([fg.factor_size(F) for F in factors], dtype=np.int32)
    two_pow = (1 << fsizes).astype(np.int32)

    # Offsets
    ss_cum  = np.concatenate(([0], two_pow[:-1].cumsum())).astype(np.int32)
    v2f_pf  = (2 * fsizes).astype(np.int32)
    v2f_cum = np.concatenate(([0], v2f_pf[:-1].cumsum())).astype(np.int32)
    r_pf    = (fsizes * two_pow).astype(np.int32)
    r_cum   = np.concatenate(([0], r_pf[:-1].cumsum())).astype(np.int32)

    # Variable metadata
    vars_list = list(fg.iter_variables())
    var_index = {v: i for i, v in enumerate(vars_list)}
    fvmap = fg.factor_variables_map()
    pos_in_factor = {F: {v: i for i, v in enumerate(fvmap[F])} for F in factors}

    # Cached bit matrices
    _bit_cache = {}
    def bit_matrix(nF: int):
        m = _bit_cache.get(nF)
        if m is None:
            a = np.arange(1 << nF, dtype=np.int16)[:, None]
            k = np.arange(nF, dtype=np.int16)[None, :]
            m = ((a >> k) & 1).astype(np.int8)
            _bit_cache[nF] = m
        return m

    # Exact sizes
    try:
        n_f2v_edges = int(fg.num_f2v_edges())
    except AttributeError:
        n_f2v_edges = int((two_pow * (fsizes.astype(np.int64) ** 2)).sum())
    n_r_edges   = int((fsizes * two_pow).sum())
    # q_edges = Σ_v 2*deg(v)*(deg(v)-1)
    F_of_v = {v: list(fg.containing_factors(v)) for v in vars_list}
    n_q_edges  = int(sum(2 * len(F_of_v[v]) * max(len(F_of_v[v]) - 1, 0) for v in vars_list))
    n_m_edges  = n_r_edges
    n_vm_edges = int(sum(2 * len(F_of_v[v]) for v in vars_list))

    f2v_edges = np.empty((n_f2v_edges, 2), dtype=np.int32)
    r_edges   = np.empty((n_r_edges,   2), dtype=np.int32)
    q_edges   = np.empty((n_q_edges,   2), dtype=np.int32)
    m_edges   = np.empty((n_m_edges,   2), dtype=np.int32)
    vm_edges  = np.empty((n_vm_edges,  2), dtype=np.int32)

    c_f2v = c_r = c_q = c_m = c_vm = 0
    SS_total = int(two_pow.sum())

    # vm_edges (vectorized per (v,F))
    for v in vars_list:
        vidx = var_index[v]
        inc_idx = [factors.index(F) for F in F_of_v[v]]
        if not inc_idx: continue
        locs = np.array([pos_in_factor[factors[i]][v] for i in inc_idx], dtype=np.int32)
        bases = v2f_cum[inc_idx] + 2 * locs
        n = bases.size
        # two states
        vm_edges[c_vm:c_vm+2*n:2, 0] = 2*vidx + 0
        vm_edges[c_vm+1:c_vm+2*n:2, 0] = 2*vidx + 1
        vm_edges[c_vm:c_vm+2*n:2, 1] = bases + 0
        vm_edges[c_vm+1:c_vm+2*n:2, 1] = bases + 1
        c_vm += 2*n

    # q_edges (vectorized per variable: all ordered pairs)
    for v in vars_list:
        inc_idx = [factors.index(F) for F in F_of_v[v]]
        d = len(inc_idx)
        if d <= 1: continue
        locs  = np.array([pos_in_factor[factors[i]][v] for i in inc_idx], dtype=np.int32)
        bases = v2f_cum[inc_idx] + 2*locs
        row = np.repeat(bases, d)
        col = np.tile(bases,   d)
        mask = row != col     # off-diagonal -> ordered pairs
        r2 = row[mask]; c2 = col[mask]
        n = r2.size
        # two states
        q_edges[c_q:c_q+2*n:2, 0] = r2 + 0
        q_edges[c_q+1:c_q+2*n:2, 0] = r2 + 1
        q_edges[c_q:c_q+2*n:2, 1] = c2 + 0
        q_edges[c_q+1:c_q+2*n:2, 1] = c2 + 1
        c_q += 2*n

    # Per-factor blocks
    for fidx, F in enumerate(factors):
        nF  = int(fsizes[fidx]);  twf = int(two_pow[fidx])
        ss0 = int(ss_cum[fidx]);  v2f0 = int(v2f_cum[fidx]);  r0 = int(r_cum[fidx])

        vals = bit_matrix(nF)                         # [twf, nF] int8
        rng_twf = np.arange(twf, dtype=np.int32)
        rng_nF  = np.arange(nF,  dtype=np.int32)

        # r_slots grid [twf, nF]
        r_slots_grid = (r0 + rng_twf[:,None]*nF + rng_nF[None,:]).astype(np.int32)

        # r_edges (twf*nF)
        dst_r = (v2f0 + 2*rng_nF[None,:] + vals.astype(np.int32)).astype(np.int32)  # [twf, nF]
        n_this = twf * nF
        r_edges[c_r:c_r+n_this, 0] = dst_r.ravel('C')
        r_edges[c_r:c_r+n_this, 1] = r_slots_grid.ravel('C')
        c_r += n_this

        # m_edges (twf*nF)
        m_edges[c_m:c_m+n_this, 0] = (ss0 + rng_twf).repeat(nF)
        m_edges[c_m:c_m+n_this, 1] = (v2f0 + 2*rng_nF[None,:] + vals.astype(np.int32)).ravel('C')
        c_m += n_this

        # f2v: SS→r
        f2v_edges[c_f2v:c_f2v+n_this, 0] = r_slots_grid.ravel('C')
        f2v_edges[c_f2v:c_f2v+n_this, 1] = (ss0 + rng_twf).repeat(nF)
        c_f2v += n_this

        # f2v: q→r
        if nF > 1:
            mid = (v2f0 + 2*rng_nF[:,None] + np.array([0,1], dtype=np.int32)[None,:]).astype(np.int32)  # [nF,2]
            src_per_assign = mid[np.arange(nF)[:,None], vals.T.astype(np.int32)]                        # [nF, twf]
            # others matrix (row-wise indices of j!=i)
            all_idx = rng_nF
            tiled = np.tile(all_idx, (nF,1))
            mask  = np.ones((nF,nF), dtype=bool); np.fill_diagonal(mask, False)
            others = tiled[mask].reshape(nF, nF-1)                                                      # [nF, nF-1]
            src_q = src_per_assign[others, :]                                                           # [nF, nF-1, twf]
            src_q = src_q.transpose(2,0,1).reshape(-1)                                                  # len = twf*nF*(nF-1)
            dst_q = np.repeat(r_slots_grid, nF-1, axis=1).ravel('C')
            n_qc = src_q.size
            f2v_edges[c_f2v:c_f2v+n_qc, 0] = dst_q
            f2v_edges[c_f2v:c_f2v+n_qc, 1] = SS_total + src_q
            c_f2v += n_qc

    # invariants
    assert c_r == n_r_edges and c_q == n_q_edges and c_m == n_m_edges and c_vm == n_vm_edges and c_f2v == n_f2v_edges

    # to torch.long
    to_long = lambda a: torch.from_numpy(a.astype(np.int64))
    return to_long(f2v_edges), to_long(r_edges), to_long(q_edges), to_long(m_edges), to_long(vm_edges)


def f2v_index_map(fg):
    m,n,ss = {}, {}, {}
    c,d,e = 0,0,0
    for Fc,F in enumerate(fg.iter_factors()):
        neighbors = fg.factor_variables(F)

        nN = len(neighbors)
        for vals in range(2**nN):
            yF = np.array([(vals >> k) & 1 for k in range(nN)], dtype=np.uint8)
            #yF = np.unpackbits(np.array([vals],dtype=np.uint8), count=nN, bitorder="little")
            ss[F,vals] = e
            e += 1
            for i in neighbors:
                iidx = neighbors.index(i)
                for s,v in enumerate([0,1]):
                    if not (F,i,v) in m.keys():
                        m[F,i,v] = c
                        c += 1
                    # f2v subsum indexes
                    if yF[iidx] == v:
                        n[F,vals,i,v] = d
                        d += 1
    return m,n,ss

def f2v_subsum_indexes(fg, f2v_idx, ss_idx):
    factor_sizes = np.array([fg.factor_size(F) for F in fg.iter_factors()])

    nSS = len(ss_idx)
    edges = []
    redges = []
    c = 0
    for c_F,F in enumerate(fg.iter_factors()):
        neighbors = fg.factor_variables(F)
        nN = len(neighbors)
        for vals in range(2**nN):
            #yF = np.unpackbits(np.array([vals],dtype=np.uint8), count=nN, bitorder="little")
            yF = np.array([(vals >> k) & 1 for k in range(nN)], dtype=np.uint8)
            for c_i,i in enumerate(neighbors):
                iidx = neighbors.index(i)
                for c_v,v in enumerate([0,1]):
                    if yF[iidx] == v:
                        # add edge f2v -> rFyi
                        redges.append([f2v_idx[F,i,v], c])
                        # add edge ss -> current pos
                        edges.append([c, ss_idx[F,vals]])
                        # add one edge qyif -> current pos for each yF[j] for j!=i
                        for c_j,(j,yFj) in enumerate(zip(neighbors,yF)):
                            if j!=i:
                                edges.append([c, nSS+f2v_idx[F,j,yFj]])
                        c += 1
    return np.array(edges), np.array(redges)

def f2v_subsum_indexes2(g):
    factor_sizes = np.array([len(list(g.neighbors(F))) for F in iter_factor_nodes(g)])
    factor_value_sizes = (2**factor_sizes).cumsum() - 2**factor_sizes
    nSS = sufficient_statistics_dim(g)

    f2v_edges_n = (2**(factor_sizes) * factor_sizes**2 ).sum()
    f2v_edges = np.empty((f2v_edges_n,2))
    f2v_edges_c = 0

    redges_n = (2**factor_sizes * factor_sizes ).sum()
    redges = np.empty((redges_n,2))
    redges_c = 0
    c = 0
    for c_F,F in enumerate(iter_factor_nodes(g)):
        neighbors = list(g.neighbors(F))
        nN = len(neighbors)
        #for c_yF,yF in enumerate(itertools.product(*[[0,1] for _ in range(nN)])):
        for vals in range(2**nN):
            yF = np.array([(vals >> k) & 1 for k in range(nN)], dtype=np.uint8)
            #yF = np.unpackbits(np.array([vals],dtype=np.uint8), count=nN, bitorder="little")

            for c_i,i in enumerate(neighbors):
                iidx = neighbors.index(i)
                for c_v,v in enumerate([0,1]):
                    if yF[iidx] == v:
                        # add edge f2v -> rFyi
                        redges[redges_c] = [factor_value_sizes[c_F]+2*c_i+c_v, redges_c]

                        # add edge ss -> current pos
                        f2v_edges[f2v_edges_c] = [redges_c, factor_value_sizes[c_F]+vals]
                        f2v_edges_c += 1

                        # add one edge qyif -> current pos for each yF[j] for j!=i
                        for c_j,(j,yFj) in enumerate(zip(neighbors,yF)):
                            if j!=i:
                                f2v_edges[f2v_edges_c] = [redges_c, nSS+factor_value_sizes[c_F]+2*c_j+yFj]
                                f2v_edges_c += 1
                        redges_c += 1
    return f2v_edges, redges

def v2f_indexes(fg, f2v_idx):
    #f2v_idx,f2v_subsum_idx,ss_idx = f2v_index_map(g)
    #factor_sizes = np.array([len(list(fg.neighbors(F))) for F in fg.iter_factors()])
    factor_sizes = np.array([fg.factor_size(F) for F in fg.iter_factors()])
    factor_offset_index = (2*factor_sizes).cumsum() - 2*factor_sizes[0]
    factor_indices = {F:i for i,F in enumerate(fg.iter_factors())}
    factor_neighbor_indices = {F:{i:idx for idx,i in enumerate(fg.neighbors(F))} for F in fg.iter_factors()}
    edges = []
    for c_F,F in enumerate(fg.iter_factors()):
        neighbors = fg.factor_variables(F)
        nN = len(neighbors)
        for c_i,i in enumerate(neighbors):
            for v in [0,1]:
                for Fp in fg.containing_factors(i):
                    if Fp!=F:
                        edges.append([f2v_idx[F,i,v], f2v_idx[Fp,i,v]])
    rr = np.array(edges)
    return rr
def v2f_indexes2(g):
    f2v_dim = f2v_messages_dim(g)
    factor_indices = {F:i for i,F in enumerate(iter_factor_nodes(g))}
    factor_neighbor_indices = {F:{i:idx for idx,i in enumerate(g.neighbors(F))} for F in iter_factor_nodes(g)}

    jj=np.array([sum([len(list(g.neighbors(i)))-1 for i in g.neighbors(F)]) for F in iter_factor_nodes(g)])
    jjj = 2*jj.cumsum() - 2*jj

    factor_neighborhood_sizes = {F:np.array([len(list(g.neighbors(i)))-1 for i in g.neighbors(F)]) for F in iter_factor_nodes(g)}
    factor_neighborhood_cumsize = {k:v.cumsum()-v for k,v in factor_neighborhood_sizes.items()}


    dd = np.array([len(list(g.neighbors(i)))-1 for F in iter_factor_nodes(g) for i in g.neighbors(F)])
    # edge sources
    src = np.arange(f2v_dim).repeat(dd.repeat(2)) # VALID
    # edge destinations
    dst_idx= np.array([
            jjj[factor_indices[Fp]]+2*factor_neighborhood_cumsize[Fp][factor_neighbor_indices[Fp][i]]+v*factor_neighborhood_sizes[Fp][factor_neighbor_indices[Fp][i]]
            for F in fg.iter_factors()
            for i in g.neighbors(F)
            for v in [0,1]
            for Fp in g.neighbors(i) if Fp!=F
        ])
    dst = src[dst_idx]
    return np.stack((src,dst),axis=-1)


def marginal_indexes(fg, f2v_idx, ss_idx):
    edges = []
    for F in fg.iter_factors():
        neighbors = fg.factor_variables(F)
        nN = len(neighbors)
        for vals in range(2**nN):
            yF = np.array([(vals >> k) & 1 for k in range(nN)], dtype=np.uint8)


            #yF = np.unpackbits(np.array([vals],dtype=np.uint8), count=nN, bitorder="little")
            for (j,yFj) in zip(neighbors, yF):
                edges.append([ss_idx[F,vals], f2v_idx[F,j,yFj]])
    return np.array(edges)
def marginal_indexes2(fg):
    factor_sizes = np.array([len(list(g.neighbors(F))) for F in fg.iter_factors()])
    factor_cumsum = (2**factor_sizes).cumsum() - 2**factor_sizes

    nM = (factor_sizes * 2**factor_sizes).sum()

    dst = np.empty(nM, dtype=np.long)
    c = 0
    for c_F,F in enumerate(fg.iter_factors()):
        neighbors = list(g.neighbors(F))
        nN = len(neighbors)
        for vals in range(2**nN):
            yF = np.array([(vals >> k) & 1 for k in range(nN)], dtype=np.uint8)
            #yF = np.unpackbits(np.array([vals],dtype=np.uint8), count=nN, bitorder="little")
            for c_i,(j,yFj) in enumerate(zip(neighbors, yF)):
                dst[c] =factor_cumsum[c_F]+2*c_i+yFj
                c+=1

    factors = list(fg.iter_factors())
    factor_neighbor_sizes = np.array([len(list(g.neighbors(factors[Fidx]))) for Fidx,Fs in enumerate(factor_sizes) for _ in range(2**Fs)])
    aa = np.arange(np.sum(2**factor_sizes))
    src = aa.repeat(factor_neighbor_sizes) # VALID
    return np.stack((src, dst), axis=-1)

def variable_marginal_indexes(g, f2v_idx, ss_idx):
    vm_edges = []
    for n,d in g.nodes(data=True):
        if d["bipartite"]==0:
            for f in g.neighbors(n):
                for v in range(2):
                    vme = [2*n+v,f2v_idx[f,n,v]]
                    vm_edges.append(vme)
    return np.array(vm_edges)

def make_lbp_edges_4(fg):
    """
    Parallel (per-factor) builder for LBP wiring.
    Returns: f2v_edges, r_edges, q_edges, m_edges, vm_edges as torch.long.
    Conventions (as in baseline):
      - f2v_edges[:,0] = r-slot id (pre–logsumexp accumulator)
      - f2v_edges[:,1] = source id in STACK = [ SS | V2F ]
      - r_edges   [:,0] = destination v2f id, [:,1] = r-slot id
      - q_edges   [:,0] = destination v2f id, [:,1] = source v2f id
      - m_edges   [:,0] = destination SS id,  [:,1] = source v2f id
      - vm_edges  [:,0] = destination var-belief id (2*i + y), [:,1] = source v2f id
    """
    import numpy as np
    import torch
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

    # ---------- Factor / variable metadata ----------
    factors = list(fg.iter_factors())
    nf = len(factors)
    fsizes  = np.array([fg.factor_size(F) for F in factors], dtype=np.int32)    # nF per factor
    two_pow = (1 << fsizes).astype(np.int32)                                    # 2**nF per factor
    SS_total = int(two_pow.sum())

    # Per-factor offsets (SS, V2F, r-slots)
    ss_cum  = np.concatenate(([0], two_pow[:-1].cumsum())).astype(np.int32)
    v2f_pf  = (2 * fsizes).astype(np.int32)
    v2f_cum = np.concatenate(([0], v2f_pf[:-1].cumsum())).astype(np.int32)
    r_pf    = (fsizes * two_pow).astype(np.int32)
    r_cum   = np.concatenate(([0], r_pf[:-1].cumsum())).astype(np.int32)

    # Variable side
    vars_list  = list(fg.iter_variables())
    var_index  = {v: i for i, v in enumerate(vars_list)}
    F_of_v     = {v: list(fg.containing_factors(v)) for v in vars_list}
    fvmap      = fg.factor_variables_map()   # dict: F -> ordered list of variables in factor F

    # Cached bit matrices by arity (shared read-only across threads)
    _bit_cache = {}
    def bit_matrix(nF: int):
        m = _bit_cache.get(nF)
        if m is None:
            a = np.arange(1 << nF, dtype=np.int16)[:, None]
            k = np.arange(nF, dtype=np.int16)[None, :]
            m = ((a >> k) & 1).astype(np.int8)      # compact; cast later
            _bit_cache[nF] = m
        return m

    # ---------- Exact sizes (trust baseline if available) ----------
    try:
        n_f2v_edges = int(fg.num_f2v_edges())                   # exact Σ_F 2^{nF} * nF^2
    except AttributeError:
        n_f2v_edges = int((two_pow.astype(np.int64) * (fsizes.astype(np.int64) ** 2)).sum())

    n_r_edges   = int((fsizes * two_pow).sum())                 # Σ_F 2^{nF} * nF
    n_m_edges   = n_r_edges
    # q_edges = Σ_v 2 * deg(v) * (deg(v) - 1)
    n_q_edges   = int(sum(2 * (len(F_of_v[v]) * max(len(F_of_v[v]) - 1, 0)) for v in vars_list))
    # vm_edges = Σ_v 2 * deg(v)
    n_vm_edges  = int(sum(2 * len(F_of_v[v]) for v in vars_list))

    # Allocate final arrays (int32 during build; cast once to int64 at the end)
    f2v_edges = np.empty((n_f2v_edges, 2), dtype=np.int32)
    r_edges   = np.empty((n_r_edges,   2), dtype=np.int32)
    m_edges   = np.empty((n_m_edges,   2), dtype=np.int32)
    q_edges   = np.empty((n_q_edges,   2), dtype=np.int32)
    vm_edges  = np.empty((n_vm_edges,  2), dtype=np.int32)

    # ---------- vm_edges (vectorized per variable) ----------
    c_vm = 0
    for v in vars_list:
        vidx   = var_index[v]
        inc    = F_of_v[v]
        if not inc: continue
        inc_idx = [factors.index(F) for F in inc]
        locs    = np.array([fvmap[factors[i]].index(v) for i in inc_idx], dtype=np.int32)
        bases   = v2f_cum[inc_idx] + 2 * locs
        d = bases.size
        vm_edges[c_vm     : c_vm + 2*d : 2, 0] = 2*vidx + 0
        vm_edges[c_vm + 1 : c_vm + 2*d : 2, 0] = 2*vidx + 1
        vm_edges[c_vm     : c_vm + 2*d : 2, 1] = bases + 0
        vm_edges[c_vm + 1 : c_vm + 2*d : 2, 1] = bases + 1
        c_vm += 2*d
    assert c_vm == n_vm_edges

    # ---------- q_edges (vectorized per variable: all ordered pairs) ----------
    c_q = 0
    for v in vars_list:
        inc    = F_of_v[v]
        d = len(inc)
        if d <= 1: continue
        inc_idx = [factors.index(F) for F in inc]
        locs    = np.array([fvmap[factors[i]].index(v) for i in inc_idx], dtype=np.int32)
        bases   = v2f_cum[inc_idx] + 2 * locs                       # [d]
        row = np.repeat(bases, d)                                   # [d*d]
        col = np.tile  (bases, d)                                   # [d*d]
        mask = row != col                                           # off-diagonal pairs (ordered)
        r2, c2 = row[mask], col[mask]
        n = r2.size
        q_edges[c_q     : c_q + 2*n : 2, 0] = r2 + 0
        q_edges[c_q + 1 : c_q + 2*n : 2, 0] = r2 + 1
        q_edges[c_q     : c_q + 2*n : 2, 1] = c2 + 0
        q_edges[c_q + 1 : c_q + 2*n : 2, 1] = c2 + 1
        c_q += 2*n
    assert c_q == n_q_edges

    # ---------- Per-factor blocks (parallel) ----------
    # Offsets for final assembly
    r_off    = r_cum.copy()
    m_off    = r_cum.copy()                         # same sizes as r
    f2v_pf   = (two_pow * (fsizes ** 2)).astype(np.int32)  # per-factor count for f2v
    f2v_cum2 = np.concatenate(([0], f2v_pf[:-1].cumsum())).astype(np.int32)
    f2v_off  = f2v_cum2.copy()

    # worker builds per-factor blocks and returns them; we assemble in main thread
    def _build_factor_blocks(fidx: int):
        F   = factors[fidx]
        nF  = int(fsizes[fidx])
        twf = int(two_pow[fidx])
        ss0 = int(ss_cum[fidx])
        v2f0= int(v2f_cum[fidx])
        r0  = int(r_cum[fidx])

        vals = bit_matrix(nF).astype(np.int32)          # [twf, nF], 0/1

        rng_twf = np.arange(twf, dtype=np.int32)
        rng_nF  = np.arange(nF,  dtype=np.int32)

        # r-slots grid [twf, nF]
        r_slots_grid = (r0 + rng_twf[:,None]*nF + rng_nF[None,:]).astype(np.int32)

        # r_edges: (twf*nF, 2)
        r_blk = np.empty((twf*nF, 2), dtype=np.int32)
        dst_r = (v2f0 + 2*rng_nF[None,:] + vals).astype(np.int32)   # [twf, nF]
        r_blk[:,0] = dst_r.ravel('C')
        r_blk[:,1] = r_slots_grid.ravel('C')

        # m_edges: (twf*nF, 2)
        m_blk = np.empty((twf*nF, 2), dtype=np.int32)
        m_blk[:,0] = (ss0 + rng_twf).repeat(nF)
        m_blk[:,1] = (v2f0 + 2*rng_nF[None,:] + vals).ravel('C')

        # f2v_edges: SS→r (twf*nF) + q→r (twf*nF*(nF-1))
        f2v_blk = np.empty((twf*nF * nF, 2), dtype=np.int32)  # total per-factor count = twf*nF^2
        cf = 0

        # SS→r
        f2v_blk[cf:cf+twf*nF, 0] = r_slots_grid.ravel('C')
        f2v_blk[cf:cf+twf*nF, 1] = (ss0 + rng_twf).repeat(nF)
        cf += twf*nF

        # q→r (only if nF > 1)
        if nF > 1:
            # v2f ids per (i,y)
            mid = (v2f0 + 2*rng_nF[:,None] + np.array([0,1], dtype=np.int32)[None,:]).astype(np.int32)  # [nF,2]
            # per (i, assignment) choose message id matching y_i(a)
            src_per_assign = mid[np.arange(nF)[:,None], vals.T]   # [nF, twf]
            # others matrix: per-destination i, all j≠i  → [nF, nF-1]
            all_idx = rng_nF
            tiled   = np.tile(all_idx, (nF,1))
            mask    = np.ones((nF,nF), dtype=bool); np.fill_diagonal(mask, False)
            others  = tiled[mask].reshape(nF, nF-1)
            # gather → [nF, nF-1, twf] → [twf, nF, nF-1] → flat
            src_q = src_per_assign[others, :].transpose(2,0,1).reshape(-1)
            dst_q = np.repeat(r_slots_grid, nF-1, axis=1).ravel('C')
            n_qc  = src_q.size
            f2v_blk[cf:cf+n_qc, 0] = dst_q
            f2v_blk[cf:cf+n_qc, 1] = SS_total + src_q
            cf += n_qc

        # safety: cf == twf*nF^2
        # (twf*nF for SS→r) + (twf*nF*(nF-1) for q→r) = twf*nF^2
        # assert cf == twf*nF*nF

        return fidx, r_blk, m_blk, f2v_blk

    # Launch threads
    max_workers = min(32, (os.cpu_count() or 4) + 4)
    results = [None] * nf
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_build_factor_blocks, fidx) for fidx in range(nf)]
        for fu in as_completed(futs):
            fidx, r_blk, m_blk, f2v_blk = fu.result()
            # assemble deterministically by precomputed offsets
            ro = int(r_off[fidx]);    rn = r_blk.shape[0]
            mo = int(m_off[fidx]);    mn = m_blk.shape[0]
            fo = int(f2v_off[fidx]);  fn = f2v_blk.shape[0]
            r_edges[ro:ro+rn]   = r_blk
            m_edges[mo:mo+mn]   = m_blk
            f2v_edges[fo:fo+fn] = f2v_blk

    # ---------- Final invariants ----------
    assert r_edges.shape[0]   == n_r_edges
    assert m_edges.shape[0]   == n_m_edges
    assert f2v_edges.shape[0] == n_f2v_edges
    assert q_edges.shape[0]   == n_q_edges
    assert vm_edges.shape[0]  == n_vm_edges

    # Bounds: where ids should live
    assert f2v_edges[:,1].max() < SS_total + int((2*fsizes).sum())   # sources in [SS | V2F]
    assert r_edges[:,0].max()   < int((2*fsizes).sum())              # v2f id space
    assert m_edges[:,0].max()   < SS_total
    assert q_edges.max()        < int((2*fsizes).sum())

    # ---------- Convert to torch.long ----------
    to_long = lambda a: torch.from_numpy(a.astype(np.int64))
    return to_long(f2v_edges), to_long(r_edges), to_long(q_edges), to_long(m_edges), to_long(vm_edges)

def make_lbp_edges_5(fg, debug: bool = False):
    """
    Arity-bucket vectorized builder for LBP wiring (no threads).
    Returns: f2v_edges, r_edges, q_edges, m_edges, vm_edges as torch.long.
    Conventions (baseline-compatible):
      - f2v_edges[:,0] = r-slot id (pre–logsumexp accumulator)
      - f2v_edges[:,1] = source id in STACK = [ SS | V2F ]
      - r_edges   [:,0] = destination v2f id, [:,1] = r-slot id
      - q_edges   [:,0] = destination v2f id, [:,1] = source v2f id
      - m_edges   [:,0] = destination SS id,  [:,1] = source v2f id
      - vm_edges  [:,0] = destination var-belief id (2*i + y), [:,1] = source v2f id
    """
    import numpy as np
    import torch

    # ---------- Factors / variables ----------
    factors = list(fg.iter_factors())
    F_index = {F: i for i, F in enumerate(factors)}  # use this everywhere
    nf = len(factors)
    fsizes  = np.array([fg.factor_size(F) for F in factors], dtype=np.int32)  # nF per factor
    two_pow = (1 << fsizes).astype(np.int32)                                  # 2**nF per factor
    SS_total = int(two_pow.sum())

    # Offsets per factor
    ss_cum  = np.concatenate(([0], two_pow[:-1].cumsum())).astype(np.int32)
    v2f_pf  = (2 * fsizes).astype(np.int32)
    v2f_cum = np.concatenate(([0], v2f_pf[:-1].cumsum())).astype(np.int32)
    r_pf    = (fsizes * two_pow).astype(np.int32)
    r_cum   = np.concatenate(([0], r_pf[:-1].cumsum())).astype(np.int32)

    # Variable metadata
    vars_list = list(fg.iter_variables())
    var_index = {v: i for i, v in enumerate(vars_list)}
    F_of_v    = {v: list(fg.containing_factors(v)) for v in vars_list}
    fvmap     = fg.factor_variables_map()  # dict: F -> ordered variables inside factor

    # Cache per-arity templates
    _bit_cache = {}
    _others_cache = {}
    def bit_matrix(nF: int):
        m = _bit_cache.get(nF)
        if m is None:
            a = np.arange(1 << nF, dtype=np.int16)[:, None]
            k = np.arange(nF, dtype=np.int16)[None, :]
            m = ((a >> k) & 1).astype(np.int8)        # compact; cast later
            _bit_cache[nF] = m
        return m

    def others_mat(nF: int):
        M = _others_cache.get(nF)
        if M is None:
            rng = np.arange(nF, dtype=np.int32)
            tiled = np.tile(rng, (nF,1))
            mask = np.ones((nF,nF), dtype=bool); np.fill_diagonal(mask, False)
            M = tiled[mask].reshape(nF, nF-1).astype(np.int32)  # [nF, nF-1]
            _others_cache[nF] = M
        return M

    # ---------- Exact sizes ----------
    try:
        n_f2v_edges = int(fg.num_f2v_edges())                      # Σ_F 2^{nF} * nF^2
    except AttributeError:
        n_f2v_edges = int((two_pow.astype(np.int64) * (fsizes.astype(np.int64) ** 2)).sum())
    n_r_edges  = int((fsizes * two_pow).sum())                     # Σ_F 2^{nF} * nF
    n_m_edges  = n_r_edges
    n_q_edges  = int(sum(2 * (len(F_of_v[v]) * max(len(F_of_v[v]) - 1, 0)) for v in vars_list))
    n_vm_edges = int(sum(2 * len(F_of_v[v]) for v in vars_list))

    if debug:
        print(f"[make_lbp_edges_5] nf={nf} | SS_total={SS_total} | "
              f"n_f2v={n_f2v_edges} n_r={n_r_edges} n_q={n_q_edges} n_m={n_m_edges} n_vm={n_vm_edges}")

    f2v_edges = np.empty((n_f2v_edges, 2), dtype=np.int32)
    r_edges   = np.empty((n_r_edges,   2), dtype=np.int32)
    m_edges   = np.empty((n_m_edges,   2), dtype=np.int32)
    q_edges   = np.empty((n_q_edges,   2), dtype=np.int32)
    vm_edges  = np.empty((n_vm_edges,  2), dtype=np.int32)

    # ---------- vm_edges (vectorized per variable) ----------
    c_vm = 0
    for v in vars_list:
        vidx = var_index[v]
        inc  = F_of_v[v]
        if not inc: continue
        #inc_idx = [factors.index(F) for F in inc]
        #locs    = np.array([fvmap[factors[i]].index(v) for i in inc_idx], dtype=np.int32)
        inc_idx = np.fromiter((F_index[F] for F in inc), count=len(inc), dtype=np.int32)
        locs    = np.fromiter((fvmap[factors[i]].index(v) for i in inc_idx), count=inc_idx.size, dtype=np.int32)

        bases   = v2f_cum[inc_idx] + 2*locs
        d = bases.size
        vm_edges[c_vm     : c_vm+2*d : 2, 0] = 2*vidx + 0
        vm_edges[c_vm+1   : c_vm+2*d : 2, 0] = 2*vidx + 1
        vm_edges[c_vm     : c_vm+2*d : 2, 1] = bases + 0
        vm_edges[c_vm+1   : c_vm+2*d : 2, 1] = bases + 1
        c_vm += 2*d
    if debug:
        print(f"[vm_edges] filled={c_vm}/{n_vm_edges}")
    assert c_vm == n_vm_edges, f"vm_edges count mismatch: got {c_vm}, expected {n_vm_edges}"

    # ---------- q_edges (vectorized per variable via ordered pairs) ----------
    c_q = 0
    for v in vars_list:
        inc = F_of_v[v]
        d = len(inc)
        if d <= 1: continue
        #inc_idx = [factors.index(F) for F in inc]
        #locs    = np.array([fvmap[factors[i]].index(v) for i in inc_idx], dtype=np.int32)
        inc_idx = np.fromiter((F_index[F] for F in inc), count=len(inc), dtype=np.int32)
        locs    = np.fromiter((fvmap[factors[i]].index(v) for i in inc_idx), count=inc_idx.size, dtype=np.int32)

        bases   = v2f_cum[inc_idx] + 2*locs  # [d]
        row = np.repeat(bases, d)
        col = np.tile  (bases, d)
        mask = row != col
        r2, c2 = row[mask], col[mask]
        n = r2.size
        q_edges[c_q     : c_q+2*n : 2, 0] = r2 + 0
        q_edges[c_q+1   : c_q+2*n : 2, 0] = r2 + 1
        q_edges[c_q     : c_q+2*n : 2, 1] = c2 + 0
        q_edges[c_q+1   : c_q+2*n : 2, 1] = c2 + 1
        c_q += 2*n
    if debug:
        print(f"[q_edges] filled={c_q}/{n_q_edges}")
    assert c_q == n_q_edges, f"q_edges count mismatch: got {c_q}, expected {n_q_edges}"

    # ---------- Per-arity bulk vectorization for r_edges, m_edges, f2v_edges ----------
    # Precompute per-factor final offsets
    r_off   = r_cum.copy()
    m_off   = r_cum.copy()
    f2v_pf  = (two_pow * (fsizes ** 2)).astype(np.int32)
    f2v_off = np.concatenate(([0], f2v_pf[:-1].cumsum())).astype(np.int32)

    # Group factor indices by arity nF
    from collections import defaultdict
    buckets = defaultdict(list)
    for fidx, nF in enumerate(fsizes.tolist()):
        buckets[int(nF)].append(fidx)

    for nF, idxs in buckets.items():
        idxs = np.array(idxs, dtype=np.int32)
        twf  = 1 << nF
        if nF == 0:
            continue
        vals = bit_matrix(nF).astype(np.int32)              # [twf, nF]
        others = others_mat(nF)                             # [nF, nF-1]
        rng_twf = np.arange(twf, dtype=np.int32)
        rng_nF  = np.arange(nF,  dtype=np.int32)

        # Gather per-factor offsets for this bucket
        ss0s  = ss_cum[idxs]    # [B]
        v2f0s = v2f_cum[idxs]   # [B]
        r0s   = r_cum[idxs]     # [B]
        B = idxs.size

        if debug:
            print(f"[bucket nF={nF}] B={B}, twf={twf}")

        # ---- r_edges block for the whole bucket ----
        r_slots_grid_f = (r0s[:,None,None] + rng_twf[None,:,None]*nF + rng_nF[None,None,:]).astype(np.int32)  # [B,twf,nF]
        dst_r_f        = (v2f0s[:,None,None] + 2*rng_nF[None,None,:] + vals[None,:,:]).astype(np.int32)       # [B,twf,nF]

        for bi, fidx in enumerate(idxs):
            ro = int(r_off[fidx]); rn = twf * nF
            r_edges[ro:ro+rn, 0] = dst_r_f[bi].ravel('C')
            r_edges[ro:ro+rn, 1] = r_slots_grid_f[bi].ravel('C')
            if debug and bi < 2:
                print(f"  [r_edges f={fidx}] slice=({ro}:{ro+rn})")

        # ---- m_edges block for the whole bucket ----
        # m_dest_f: [B, twf] (SS ids per assignment)
        m_dest_f = (ss0s[:,None] + rng_twf[None,:]).astype(np.int32)   # [B, twf]
        m_src_f  = dst_r_f                                             # [B, twf, nF] (same expression)

        for bi, fidx in enumerate(idxs):
            mo = int(m_off[fidx]); mn = twf * nF
            # dest: for each assignment, repeat SS id nF times
            dest = np.repeat(m_dest_f[bi], nF)                         # [twf*nF]
            m_edges[mo:mo+mn, 0] = dest
            m_edges[mo:mo+mn, 1] = m_src_f[bi].ravel('C')
            if debug and bi < 2:
                print(f"  [m_edges f={fidx}] slice=({mo}:{mo+mn}) dest.shape={dest.shape}")

        # ---- f2v_edges block (SS→r then q→r) for the whole bucket ----
        for bi, fidx in enumerate(idxs):
            fo   = int(f2v_off[fidx])
            ss0  = int(ss0s[bi]); v2f0 = int(v2f0s[bi])

            # SS→r
            n_ss = twf * nF
            f2v_edges[fo:fo+n_ss, 0] = r_slots_grid_f[bi].ravel('C')
            f2v_edges[fo:fo+n_ss, 1] = (ss0 + rng_twf).repeat(nF)
            cf = fo + n_ss
            if debug and bi < 2:
                print(f"  [f2v SS→r f={fidx}] slice=({fo}:{fo+n_ss})")

            # q→r (only if nF > 1)
            if nF > 1:
                # mid: [nF,2]
                mid = (v2f0 + 2*rng_nF[:,None] + np.array([0,1], dtype=np.int32)[None,:]).astype(np.int32)
                src_per_assign = mid[np.arange(nF)[:,None], vals.T]        # [nF, twf]
                src_q = src_per_assign[others, :].transpose(2,0,1).reshape(-1)                     # [twf*nF*(nF-1)]
                dst_q = np.repeat(r_slots_grid_f[bi], nF-1, axis=1).ravel('C')
                n_qc  = src_q.size
                f2v_edges[cf:cf+n_qc, 0] = dst_q
                f2v_edges[cf:cf+n_qc, 1] = SS_total + src_q
                if debug and bi < 2:
                    print(f"  [f2v q→r f={fidx}] slice=({cf}:{cf+n_qc}) n_qc={n_qc}")
                # optional: per-factor sanity
                assert n_qc == twf * nF * max(nF - 1, 0), \
                       f"q→r count mismatch for factor {fidx}: got {n_qc}, expected {twf*nF*max(nF-1,0)}"
                cf += n_qc

            # optional: assert cf == fo + twf * nF * nF

    # ---------- Final invariants ----------
    if debug:
        print("[final shapes]",
              f2v_edges.shape, r_edges.shape, q_edges.shape, m_edges.shape, vm_edges.shape)

    assert r_edges.shape[0]   == n_r_edges,  f"r_edges size mismatch: {r_edges.shape[0]} vs {n_r_edges}"
    assert m_edges.shape[0]   == n_m_edges,  f"m_edges size mismatch: {m_edges.shape[0]} vs {n_m_edges}"
    assert f2v_edges.shape[0] == n_f2v_edges, f"f2v_edges size mismatch: {f2v_edges.shape[0]} vs {n_f2v_edges}"
    assert q_edges.shape[0]   == n_q_edges,  f"q_edges size mismatch: {q_edges.shape[0]} vs {n_q_edges}"
    assert vm_edges.shape[0]  == n_vm_edges, f"vm_edges size mismatch: {vm_edges.shape[0]} vs {n_vm_edges}"

    # Bounds
    V2F_total = int((2 * fsizes).sum())
    assert f2v_edges[:,1].max() < SS_total + V2F_total, "f2v source out of STACK bounds"
    assert r_edges[:,0].max()   < V2F_total,            "r_edges dst (v2f id) out of bounds"
    assert m_edges[:,0].max()   < SS_total,             "m_edges dst (SS id) out of bounds"
    assert q_edges.max()        < V2F_total,            "q_edges id out of bounds"

    # ---------- Convert to torch.long ----------
    to_long = lambda a: torch.from_numpy(a.astype(np.int64))
    return to_long(f2v_edges), to_long(r_edges), to_long(q_edges), to_long(m_edges), to_long(vm_edges)

def make_lbp_edges_6(fg, debug: bool = False):
    """
    Arity-bucket vectorized builder for LBP wiring (no threads).
    Returns: f2v_edges, r_edges, q_edges, m_edges, vm_edges as torch.long.
    Conventions (baseline-compatible):
      - f2v_edges[:,0] = r-slot id (pre–logsumexp accumulator)
      - f2v_edges[:,1] = source id in STACK = [ SS | V2F ]
      - r_edges   [:,0] = destination v2f id, [:,1] = r-slot id
      - q_edges   [:,0] = destination v2f id, [:,1] = source v2f id
      - m_edges   [:,0] = destination SS id,  [:,1] = source v2f id
      - vm_edges  [:,0] = destination var-belief id (2*i + y), [:,1] = source v2f id
    """
    import numpy as np
    import torch

    # ---------- Factors / variables ----------
    factors = list(fg.iter_factors())
    F_index = {F: i for i, F in enumerate(factors)}  # use this everywhere
    nf = len(factors)
    fsizes  = np.array([fg.factor_size(F) for F in factors], dtype=np.int32)  # nF per factor
    two_pow = (1 << fsizes).astype(np.int32)                                  # 2**nF per factor
    SS_total = int(two_pow.sum())

    # Offsets per factor
    ss_cum  = np.concatenate(([0], two_pow[:-1].cumsum())).astype(np.int32)
    v2f_pf  = (2 * fsizes).astype(np.int32)
    v2f_cum = np.concatenate(([0], v2f_pf[:-1].cumsum())).astype(np.int32)
    r_pf    = (fsizes * two_pow).astype(np.int32)
    r_cum   = np.concatenate(([0], r_pf[:-1].cumsum())).astype(np.int32)

    # Variable metadata
    vars_list = list(fg.iter_variables())
    var_index = {v: i for i, v in enumerate(vars_list)}
    F_of_v    = {v: list(fg.containing_factors(v)) for v in vars_list}
    fvmap     = fg.factor_variables_map()  # dict: F -> ordered variables inside factor

    # Cache per-arity templates
    _bit_cache = {}
    _others_cache = {}
    def bit_matrix(nF: int):
        m = _bit_cache.get(nF)
        if m is None:
            a = np.arange(1 << nF, dtype=np.int16)[:, None]
            k = np.arange(nF, dtype=np.int16)[None, :]
            m = ((a >> k) & 1).astype(np.int8)        # compact; cast later
            _bit_cache[nF] = m
        return m

    def others_mat(nF: int):
        M = _others_cache.get(nF)
        if M is None:
            rng = np.arange(nF, dtype=np.int32)
            tiled = np.tile(rng, (nF,1))
            mask = np.ones((nF,nF), dtype=bool); np.fill_diagonal(mask, False)
            M = tiled[mask].reshape(nF, nF-1).astype(np.int32)  # [nF, nF-1]
            _others_cache[nF] = M
        return M

    # ---------- Exact sizes ----------
    try:
        n_f2v_edges = int(fg.num_f2v_edges())                      # Σ_F 2^{nF} * nF^2
    except AttributeError:
        n_f2v_edges = int((two_pow.astype(np.int64) * (fsizes.astype(np.int64) ** 2)).sum())
    n_r_edges  = int((fsizes * two_pow).sum())                     # Σ_F 2^{nF} * nF
    n_m_edges  = n_r_edges
    n_q_edges  = int(sum(2 * (len(F_of_v[v]) * max(len(F_of_v[v]) - 1, 0)) for v in vars_list))
    n_vm_edges = int(sum(2 * len(F_of_v[v]) for v in vars_list))

    if debug:
        print(f"[make_lbp_edges_5] nf={nf} | SS_total={SS_total} | "
              f"n_f2v={n_f2v_edges} n_r={n_r_edges} n_q={n_q_edges} n_m={n_m_edges} n_vm={n_vm_edges}")

    f2v_edges = np.empty((n_f2v_edges, 2), dtype=np.int32)
    r_edges   = np.empty((n_r_edges,   2), dtype=np.int32)
    m_edges   = np.empty((n_m_edges,   2), dtype=np.int32)
    q_edges   = np.empty((n_q_edges,   2), dtype=np.int32)
    vm_edges  = np.empty((n_vm_edges,  2), dtype=np.int32)

    # ---------- vm_edges (vectorized per variable) ----------
    c_vm = 0
    # ---------- q_edges (vectorized per variable via ordered pairs) ----------
    c_q = 0

    for v in vars_list:
        vidx = var_index[v]
        inc  = F_of_v[v]
        if not inc: continue
        #inc_idx = [factors.index(F) for F in inc]
        #locs    = np.array([fvmap[factors[i]].index(v) for i in inc_idx], dtype=np.int32)
        inc_idx = np.fromiter((F_index[F] for F in inc), count=len(inc), dtype=np.int32)
        locs    = np.fromiter((fvmap[factors[i]].index(v) for i in inc_idx), count=inc_idx.size, dtype=np.int32)

        bases   = v2f_cum[inc_idx] + 2*locs
        d = bases.size
        vm_edges[c_vm     : c_vm+2*d : 2, 0] = 2*vidx + 0
        vm_edges[c_vm+1   : c_vm+2*d : 2, 0] = 2*vidx + 1
        vm_edges[c_vm     : c_vm+2*d : 2, 1] = bases + 0
        vm_edges[c_vm+1   : c_vm+2*d : 2, 1] = bases + 1
        c_vm += 2*d

        inc = F_of_v[v]
        d = len(inc)
        if d <= 1: continue
        #inc_idx = [factors.index(F) for F in inc]
        #locs    = np.array([fvmap[factors[i]].index(v) for i in inc_idx], dtype=np.int32)
        inc_idx = np.fromiter((F_index[F] for F in inc), count=len(inc), dtype=np.int32)
        locs    = np.fromiter((fvmap[factors[i]].index(v) for i in inc_idx), count=inc_idx.size, dtype=np.int32)

        bases   = v2f_cum[inc_idx] + 2*locs  # [d]
        row = np.repeat(bases, d)
        col = np.tile  (bases, d)
        mask = row != col
        r2, c2 = row[mask], col[mask]
        n = r2.size
        q_edges[c_q     : c_q+2*n : 2, 0] = r2 + 0
        q_edges[c_q+1   : c_q+2*n : 2, 0] = r2 + 1
        q_edges[c_q     : c_q+2*n : 2, 1] = c2 + 0
        q_edges[c_q+1   : c_q+2*n : 2, 1] = c2 + 1
        c_q += 2*n

    if debug:
        print(f"[vm_edges] filled={c_vm}/{n_vm_edges}")
    assert c_vm == n_vm_edges, f"vm_edges count mismatch: got {c_vm}, expected {n_vm_edges}"

    if debug:
        print(f"[q_edges] filled={c_q}/{n_q_edges}")
    assert c_q == n_q_edges, f"q_edges count mismatch: got {c_q}, expected {n_q_edges}"

    # ---------- Per-arity bulk vectorization for r_edges, m_edges, f2v_edges ----------
    # Precompute per-factor final offsets
    r_off   = r_cum.copy()
    m_off   = r_cum.copy()
    f2v_pf  = (two_pow * (fsizes ** 2)).astype(np.int32)
    f2v_off = np.concatenate(([0], f2v_pf[:-1].cumsum())).astype(np.int32)

    # Group factor indices by arity nF
    from collections import defaultdict
    buckets = defaultdict(list)
    for fidx, nF in enumerate(fsizes.tolist()):
        buckets[int(nF)].append(fidx)

    for nF, idxs in buckets.items():
        idxs = np.array(idxs, dtype=np.int32)
        twf  = 1 << nF
        if nF == 0:
            continue
        vals = bit_matrix(nF).astype(np.int32)              # [twf, nF]
        others = others_mat(nF)                             # [nF, nF-1]
        rng_twf = np.arange(twf, dtype=np.int32)
        rng_nF  = np.arange(nF,  dtype=np.int32)

        # Gather per-factor offsets for this bucket
        ss0s  = ss_cum[idxs]    # [B]
        v2f0s = v2f_cum[idxs]   # [B]
        r0s   = r_cum[idxs]     # [B]
        B = idxs.size

        if debug:
            print(f"[bucket nF={nF}] B={B}, twf={twf}")

        # ---- r_edges block for the whole bucket ----
        r_slots_grid_f = (r0s[:,None,None] + rng_twf[None,:,None]*nF + rng_nF[None,None,:]).astype(np.int32)  # [B,twf,nF]
        dst_r_f        = (v2f0s[:,None,None] + 2*rng_nF[None,None,:] + vals[None,:,:]).astype(np.int32)       # [B,twf,nF]

        for bi, fidx in enumerate(idxs):
            ro = int(r_off[fidx]); rn = twf * nF
            r_edges[ro:ro+rn, 0] = dst_r_f[bi].ravel('C')
            r_edges[ro:ro+rn, 1] = r_slots_grid_f[bi].ravel('C')
            if debug and bi < 2:
                print(f"  [r_edges f={fidx}] slice=({ro}:{ro+rn})")

        # ---- m_edges block for the whole bucket ----
        # m_dest_f: [B, twf] (SS ids per assignment)
        m_dest_f = (ss0s[:,None] + rng_twf[None,:]).astype(np.int32)   # [B, twf]
        m_src_f  = dst_r_f                                             # [B, twf, nF] (same expression)

        for bi, fidx in enumerate(idxs):
            mo = int(m_off[fidx]); mn = twf * nF
            # dest: for each assignment, repeat SS id nF times
            dest = np.repeat(m_dest_f[bi], nF)                         # [twf*nF]
            m_edges[mo:mo+mn, 0] = dest
            m_edges[mo:mo+mn, 1] = m_src_f[bi].ravel('C')
            if debug and bi < 2:
                print(f"  [m_edges f={fidx}] slice=({mo}:{mo+mn}) dest.shape={dest.shape}")

            # ---- f2v_edges block (SS→r then q→r) for the whole bucket ----
            fo   = int(f2v_off[fidx])
            ss0  = int(ss0s[bi]); v2f0 = int(v2f0s[bi])

            # SS→r
            n_ss = twf * nF
            f2v_edges[fo:fo+n_ss, 0] = r_slots_grid_f[bi].ravel('C')
            f2v_edges[fo:fo+n_ss, 1] = (ss0 + rng_twf).repeat(nF)
            cf = fo + n_ss
            if debug and bi < 2:
                print(f"  [f2v SS→r f={fidx}] slice=({fo}:{fo+n_ss})")

            # q→r (only if nF > 1)
            if nF > 1:
                # mid: [nF,2]
                mid = (v2f0 + 2*rng_nF[:,None] + np.array([0,1], dtype=np.int32)[None,:]).astype(np.int32)
                src_per_assign = mid[np.arange(nF)[:,None], vals.T]        # [nF, twf]
                src_q = src_per_assign[others, :].transpose(2,0,1).reshape(-1)                     # [twf*nF*(nF-1)]
                dst_q = np.repeat(r_slots_grid_f[bi], nF-1, axis=1).ravel('C')
                n_qc  = src_q.size
                f2v_edges[cf:cf+n_qc, 0] = dst_q
                f2v_edges[cf:cf+n_qc, 1] = SS_total + src_q
                if debug and bi < 2:
                    print(f"  [f2v q→r f={fidx}] slice=({cf}:{cf+n_qc}) n_qc={n_qc}")
                # optional: per-factor sanity
                assert n_qc == twf * nF * max(nF - 1, 0), \
                       f"q→r count mismatch for factor {fidx}: got {n_qc}, expected {twf*nF*max(nF-1,0)}"
                cf += n_qc

            # optional: assert cf == fo + twf * nF * nF

    # ---------- Final invariants ----------
    if debug:
        print("[final shapes]",
              f2v_edges.shape, r_edges.shape, q_edges.shape, m_edges.shape, vm_edges.shape)

    assert r_edges.shape[0]   == n_r_edges,  f"r_edges size mismatch: {r_edges.shape[0]} vs {n_r_edges}"
    assert m_edges.shape[0]   == n_m_edges,  f"m_edges size mismatch: {m_edges.shape[0]} vs {n_m_edges}"
    assert f2v_edges.shape[0] == n_f2v_edges, f"f2v_edges size mismatch: {f2v_edges.shape[0]} vs {n_f2v_edges}"
    assert q_edges.shape[0]   == n_q_edges,  f"q_edges size mismatch: {q_edges.shape[0]} vs {n_q_edges}"
    assert vm_edges.shape[0]  == n_vm_edges, f"vm_edges size mismatch: {vm_edges.shape[0]} vs {n_vm_edges}"

    # Bounds
    V2F_total = int((2 * fsizes).sum())
    assert f2v_edges[:,1].max() < SS_total + V2F_total, "f2v source out of STACK bounds"
    assert r_edges[:,0].max()   < V2F_total,            "r_edges dst (v2f id) out of bounds"
    assert m_edges[:,0].max()   < SS_total,             "m_edges dst (SS id) out of bounds"
    assert q_edges.max()        < V2F_total,            "q_edges id out of bounds"

    # ---------- Convert to torch.long ----------
    to_long = lambda a: torch.from_numpy(a.astype(np.int64))
    return to_long(f2v_edges), to_long(r_edges), to_long(q_edges), to_long(m_edges), to_long(vm_edges)

def make_lbp_edges_7(fg, debug: bool = False):
    """
    Arity-bucket vectorized builder for LBP wiring (no threads).
    Returns: f2v_edges, r_edges, q_edges, m_edges, vm_edges as torch.long.
    Conventions (baseline-compatible):
      - f2v_edges[:,0] = r-slot id (pre–logsumexp accumulator)
      - f2v_edges[:,1] = source id in STACK = [ SS | V2F ]
      - r_edges   [:,0] = destination v2f id, [:,1] = r-slot id
      - q_edges   [:,0] = destination v2f id, [:,1] = source v2f id
      - m_edges   [:,0] = destination SS id,  [:,1] = source v2f id
      - vm_edges  [:,0] = destination var-belief id (2*i + y), [:,1] = source v2f id
    """
    import numpy as np
    import torch

    # ---------- Factors / variables ----------
    factors = list(fg.iter_factors())
    F_index = {F: i for i, F in enumerate(factors)}  # use this everywhere
    nf = len(factors)
    fsizes  = np.array([fg.factor_size(F) for F in factors], dtype=np.int32)  # nF per factor
    two_pow = (1 << fsizes).astype(np.int32)                                  # 2**nF per factor
    SS_total = int(two_pow.sum())

    # Offsets per factor
    ss_cum  = np.concatenate(([0], two_pow[:-1].cumsum())).astype(np.int32)
    v2f_pf  = (2 * fsizes).astype(np.int32)
    v2f_cum = np.concatenate(([0], v2f_pf[:-1].cumsum())).astype(np.int32)
    r_pf    = (fsizes * two_pow).astype(np.int32)
    r_cum   = np.concatenate(([0], r_pf[:-1].cumsum())).astype(np.int32)

    # Variable metadata
    vars_list = list(fg.iter_variables())
    var_index = {v: i for i, v in enumerate(vars_list)}
    F_of_v    = {v: list(fg.containing_factors(v)) for v in vars_list}
    fvmap     = fg.factor_variables_map()  # dict: F -> ordered variables inside factor

    # Cache per-arity templates
    _bit_cache = {}
    _others_cache = {}
    def bit_matrix(nF: int):
        m = _bit_cache.get(nF)
        if m is None:
            a = np.arange(1 << nF, dtype=np.int16)[:, None]
            k = np.arange(nF, dtype=np.int16)[None, :]
            m = ((a >> k) & 1).astype(np.int8)        # compact; cast later
            _bit_cache[nF] = m
        return m

    def others_mat(nF: int):
        M = _others_cache.get(nF)
        if M is None:
            rng = np.arange(nF, dtype=np.int32)
            tiled = np.tile(rng, (nF,1))
            mask = np.ones((nF,nF), dtype=bool); np.fill_diagonal(mask, False)
            M = tiled[mask].reshape(nF, nF-1).astype(np.int32)  # [nF, nF-1]
            _others_cache[nF] = M
        return M

    # ---------- Exact sizes ----------
    try:
        n_f2v_edges = int(fg.num_f2v_edges())                      # Σ_F 2^{nF} * nF^2
    except AttributeError:
        n_f2v_edges = int((two_pow.astype(np.int64) * (fsizes.astype(np.int64) ** 2)).sum())
    n_r_edges  = int((fsizes * two_pow).sum())                     # Σ_F 2^{nF} * nF
    n_m_edges  = n_r_edges
    n_q_edges  = int(sum(2 * (len(F_of_v[v]) * max(len(F_of_v[v]) - 1, 0)) for v in vars_list))
    n_vm_edges = int(sum(2 * len(F_of_v[v]) for v in vars_list))

    if debug:
        print(f"[make_lbp_edges_5] nf={nf} | SS_total={SS_total} | "
              f"n_f2v={n_f2v_edges} n_r={n_r_edges} n_q={n_q_edges} n_m={n_m_edges} n_vm={n_vm_edges}")

    f2v_edges = np.empty((n_f2v_edges, 2), dtype=np.int32)
    r_edges   = np.empty((n_r_edges,   2), dtype=np.int32)
    m_edges   = np.empty((n_m_edges,   2), dtype=np.int32)
    q_edges   = np.empty((n_q_edges,   2), dtype=np.int32)
    vm_edges  = np.empty((n_vm_edges,  2), dtype=np.int32)

    # ---------- vm_edges (vectorized per variable) ----------
    c_vm = 0
    # ---------- q_edges (vectorized per variable via ordered pairs) ----------
    c_q = 0

    _deg_pairs_cache = {}  # deg -> (row_idx, col_idx) for ordered off-diagonal pairs
    def deg_pairs(deg: int):
        tpl = _deg_pairs_cache.get(deg)
        if tpl is None:
            r = np.repeat(np.arange(deg, dtype=np.int32), deg)
            c = np.tile  (np.arange(deg, dtype=np.int32), deg)
            m = (r != c)
            _deg_pairs_cache[deg] = (r[m], c[m])   # each length = deg*(deg-1)
        return _deg_pairs_cache[deg]

    # bucket variables by degree
    from collections import defaultdict
    by_deg = defaultdict(list)
    for v in vars_list:
        by_deg[len(F_of_v[v])].append(v)
    
    c_q = 0
    for deg, vs in by_deg.items():
        if deg <= 1:  # nothing to do
            continue
        r_idx, c_idx = deg_pairs(deg)  # cached template
        # process variables of this degree in a tight loop
        for v in vs:
            inc = F_of_v[v]
            inc_idx = np.fromiter((F_index[F] for F in inc), count=deg, dtype=np.int32)
            locs    = np.fromiter((fvmap[factors[i]].index(v) for i in inc_idx), count=deg, dtype=np.int32)
            bases   = v2f_cum[inc_idx] + 2*locs   # [deg]
            dst_b   = bases[r_idx]  # [deg*(deg-1)]
            src_b   = bases[c_idx]
            n       = dst_b.size
            # two states (interleave 0/1)
            q_edges[c_q    : c_q+2*n : 2, 0] = dst_b + 0
            q_edges[c_q+1  : c_q+2*n : 2, 0] = dst_b + 1
            q_edges[c_q    : c_q+2*n : 2, 1] = src_b + 0
            q_edges[c_q+1  : c_q+2*n : 2, 1] = src_b + 1
            c_q += 2*n

            vidx   = var_index[v]
            inc    = F_of_v[v]
            inc_idx= np.fromiter((F_index[F] for F in inc), count=deg, dtype=np.int32)
            locs   = np.fromiter((fvmap[factors[i]].index(v) for i in inc_idx), count=deg, dtype=np.int32)
            bases  = v2f_cum[inc_idx] + 2*locs
            vm_edges[c_vm    : c_vm+2*deg : 2, 0] = 2*vidx + 0
            vm_edges[c_vm+1  : c_vm+2*deg : 2, 0] = 2*vidx + 1
            vm_edges[c_vm    : c_vm+2*deg : 2, 1] = bases + 0
            vm_edges[c_vm+1  : c_vm+2*deg : 2, 1] = bases + 1
            c_vm += 2*deg


    if debug:
        print(f"[vm_edges] filled={c_vm}/{n_vm_edges}")
    assert c_vm == n_vm_edges, f"vm_edges count mismatch: got {c_vm}, expected {n_vm_edges}"

    if debug:
        print(f"[q_edges] filled={c_q}/{n_q_edges}")
    assert c_q == n_q_edges, f"q_edges count mismatch: got {c_q}, expected {n_q_edges}"

    # ---------- Per-arity bulk vectorization for r_edges, m_edges, f2v_edges ----------
    # Precompute per-factor final offsets
    r_off   = r_cum.copy()
    m_off   = r_cum.copy()
    f2v_pf  = (two_pow * (fsizes ** 2)).astype(np.int32)
    f2v_off = np.concatenate(([0], f2v_pf[:-1].cumsum())).astype(np.int32)

    # Group factor indices by arity nF
    from collections import defaultdict
    buckets = defaultdict(list)
    for fidx, nF in enumerate(fsizes.tolist()):
        buckets[int(nF)].append(fidx)

    for nF, idxs in buckets.items():
        idxs = np.array(idxs, dtype=np.int32)
        twf  = 1 << nF
        if nF == 0:
            continue
        vals = bit_matrix(nF).astype(np.int32)              # [twf, nF]
        others = others_mat(nF)                             # [nF, nF-1]
        rng_twf = np.arange(twf, dtype=np.int32)
        rng_nF  = np.arange(nF,  dtype=np.int32)

        # Gather per-factor offsets for this bucket
        ss0s  = ss_cum[idxs]    # [B]
        v2f0s = v2f_cum[idxs]   # [B]
        r0s   = r_cum[idxs]     # [B]
        B = idxs.size

        if debug:
            print(f"[bucket nF={nF}] B={B}, twf={twf}")

        # ---- r_edges block for the whole bucket ----
        r_slots_grid_f = (r0s[:,None,None] + rng_twf[None,:,None]*nF + rng_nF[None,None,:]).astype(np.int32)  # [B,twf,nF]
        dst_r_f        = (v2f0s[:,None,None] + 2*rng_nF[None,None,:] + vals[None,:,:]).astype(np.int32)       # [B,twf,nF]

        for bi, fidx in enumerate(idxs):
            ro = int(r_off[fidx]); rn = twf * nF
            r_edges[ro:ro+rn, 0] = dst_r_f[bi].ravel('C')
            r_edges[ro:ro+rn, 1] = r_slots_grid_f[bi].ravel('C')
            if debug and bi < 2:
                print(f"  [r_edges f={fidx}] slice=({ro}:{ro+rn})")

        # ---- m_edges block for the whole bucket ----
        # m_dest_f: [B, twf] (SS ids per assignment)
        m_dest_f = (ss0s[:,None] + rng_twf[None,:]).astype(np.int32)   # [B, twf]
        m_src_f  = dst_r_f                                             # [B, twf, nF] (same expression)

        for bi, fidx in enumerate(idxs):
            mo = int(m_off[fidx]); mn = twf * nF
            # dest: for each assignment, repeat SS id nF times
            dest = np.repeat(m_dest_f[bi], nF)                         # [twf*nF]
            m_edges[mo:mo+mn, 0] = dest
            m_edges[mo:mo+mn, 1] = m_src_f[bi].ravel('C')
            if debug and bi < 2:
                print(f"  [m_edges f={fidx}] slice=({mo}:{mo+mn}) dest.shape={dest.shape}")

            # ---- f2v_edges block (SS→r then q→r) for the whole bucket ----
            fo   = int(f2v_off[fidx])
            ss0  = int(ss0s[bi]); v2f0 = int(v2f0s[bi])

            # SS→r
            n_ss = twf * nF
            f2v_edges[fo:fo+n_ss, 0] = r_slots_grid_f[bi].ravel('C')
            f2v_edges[fo:fo+n_ss, 1] = (ss0 + rng_twf).repeat(nF)
            cf = fo + n_ss
            if debug and bi < 2:
                print(f"  [f2v SS→r f={fidx}] slice=({fo}:{fo+n_ss})")

            # q→r (only if nF > 1)
            if nF > 1:
                # mid: [nF,2]
                mid = (v2f0 + 2*rng_nF[:,None] + np.array([0,1], dtype=np.int32)[None,:]).astype(np.int32)
                src_per_assign = mid[np.arange(nF)[:,None], vals.T]        # [nF, twf]
                src_q = src_per_assign[others, :].transpose(2,0,1).reshape(-1)                     # [twf*nF*(nF-1)]
                dst_q = np.repeat(r_slots_grid_f[bi], nF-1, axis=1).ravel('C')
                n_qc  = src_q.size
                f2v_edges[cf:cf+n_qc, 0] = dst_q
                f2v_edges[cf:cf+n_qc, 1] = SS_total + src_q
                if debug and bi < 2:
                    print(f"  [f2v q→r f={fidx}] slice=({cf}:{cf+n_qc}) n_qc={n_qc}")
                # optional: per-factor sanity
                assert n_qc == twf * nF * max(nF - 1, 0), \
                       f"q→r count mismatch for factor {fidx}: got {n_qc}, expected {twf*nF*max(nF-1,0)}"
                cf += n_qc

            # optional: assert cf == fo + twf * nF * nF

    # ---------- Final invariants ----------
    if debug:
        print("[final shapes]",
              f2v_edges.shape, r_edges.shape, q_edges.shape, m_edges.shape, vm_edges.shape)

    assert r_edges.shape[0]   == n_r_edges,  f"r_edges size mismatch: {r_edges.shape[0]} vs {n_r_edges}"
    assert m_edges.shape[0]   == n_m_edges,  f"m_edges size mismatch: {m_edges.shape[0]} vs {n_m_edges}"
    assert f2v_edges.shape[0] == n_f2v_edges, f"f2v_edges size mismatch: {f2v_edges.shape[0]} vs {n_f2v_edges}"
    assert q_edges.shape[0]   == n_q_edges,  f"q_edges size mismatch: {q_edges.shape[0]} vs {n_q_edges}"
    assert vm_edges.shape[0]  == n_vm_edges, f"vm_edges size mismatch: {vm_edges.shape[0]} vs {n_vm_edges}"

    # Bounds
    V2F_total = int((2 * fsizes).sum())
    assert f2v_edges[:,1].max() < SS_total + V2F_total, "f2v source out of STACK bounds"
    assert r_edges[:,0].max()   < V2F_total,            "r_edges dst (v2f id) out of bounds"
    assert m_edges[:,0].max()   < SS_total,             "m_edges dst (SS id) out of bounds"
    assert q_edges.max()        < V2F_total,            "q_edges id out of bounds"

    # ---------- Convert to torch.long ----------
    to_long = lambda a: torch.from_numpy(a.astype(np.int64))
    return to_long(f2v_edges), to_long(r_edges), to_long(q_edges), to_long(m_edges), to_long(vm_edges)

