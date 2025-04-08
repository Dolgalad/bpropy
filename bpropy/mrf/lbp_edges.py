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
    def fill(self, fg):
        if fg.num_graphs == 1:
            self.n_ss = torch.tensor([fg.number_sufficient_statistics()])
            self.n_vars = torch.tensor([fg.nv()])
            self.n_factors = torch.tensor([fg.nf()])
            # create the computation edges
            f2v_idx, f2v_subsum_idx,ss_idx = f2v_index_map(fg)
            # creating message passing edges
            f2v_edges,r_edges = f2v_subsum_indexes(fg, f2v_idx, ss_idx)

            self.f2v_edges = lexsort(torch.from_numpy(f2v_edges))
            self.r_edges = lexsort(torch.from_numpy(r_edges))
            # scatter indices
            self.q_edges = lexsort(torch.from_numpy(v2f_indexes(fg, f2v_idx)))
            self.m_edges = lexsort(torch.from_numpy(marginal_indexes(fg, f2v_idx, ss_idx)))
            self.vm_edges = lexsort(torch.from_numpy(variable_marginal_indexes(fg, f2v_idx, ss_idx)))

            self.n_f2v_ss = torch.tensor([len(f2v_subsum_idx)])
            self.n_f2v_msg = torch.tensor([sum(2*fg.factor_size(F) for F in fg.iter_factors())])
            self.m_stride = torch.tensor([2**len(list(fg.neighbors(F))) for F in fg.iter_factors()])
        else:
            edge_list = [LBPEdges(fg.get_subgraph(i)) for i in range(fg.num_graphs)]
            edges_batch = LBPEdges.batch(edge_list)
            self.n_ss      = edges_batch.n_ss      
            self.n_vars    = edges_batch.n_vars    
            self.n_factors = edges_batch.n_factors 
            self.n_f2v_ss  = edges_batch.n_f2v_ss  
            self.n_f2v_msg = edges_batch.n_f2v_msg 
            self.f2v_edges = lexsort(edges_batch.f2v_edges)
            self.r_edges   = lexsort(edges_batch.r_edges)
            self.q_edges   = lexsort(edges_batch.q_edges)
            self.m_edges   = lexsort(edges_batch.m_edges)
            self.vm_edges  = lexsort(edges_batch.vm_edges)
            self.m_stride  = edges_batch.m_stride  


    def num_graphs(self):
        return self.n_ss.shape[0]
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
    def to(self, device):
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

    def __eq__(self, other):
        #print(torch.all(self.n_ss == other.n_ss) )
        #print(torch.all(self.n_vars == other.n_vars) )
        #print(torch.all(self.n_factors == other.n_factors) )
        #print(torch.all(self.n_f2v_ss == other.n_f2v_ss) )
        #print(torch.all(self.n_f2v_msg == other.n_f2v_msg) )
        #print("same f2v_edges = ", torch.all(self.f2v_edges == other.f2v_edges) )
        #print("same r_edges  = ", torch.all(self.r_edges == other.r_edges)) 
        #print("same q_edges  = ", torch.all(self.q_edges == other.q_edges)) 
        #print("same m_edges  = ", torch.all(self.m_edges == other.m_edges)) 
        #print("same vm_edges = ", torch.all(self.vm_edges == other.vm_edges) )
        #print("same m_stride = ", torch.all(self.m_stride == other.m_stride))


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
            tmp_f2v_edges[idx_more,:] = tmp_f2v_edges[idx_more,:] + torch.tensor([n_f2v_subsum_offset, n_ss_tot+n_f2v_msg_offset-n_ss])
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

            ne.n_f2v_msg[n_f2v_msg_pos:n_f2v_msg_pos+ne.n_f2v_msg.shape[0]] = e.n_f2v_msg
            n_f2v_msg_pos += e.n_f2v_msg.shape[0]
        return ne

    


"""Number of variable marginal edges
"""
def number_vm_edges(fg):
    return sum([2*len(list(fg.neighbors(n))) for n in fg.iter_variables()])

def f2v_messages_dim(fg):
    return sum([2*fg.factor_size(F) for F in fg.iter_factors()])

def f2v_index_map(fg):
    m,n,ss = {}, {}, {}
    c,d,e = 0,0,0
    for Fc,F in enumerate(fg.iter_factors()):
        neighbors = fg.factor_variables(F)

        nN = len(neighbors)
        for vals in range(2**nN):
            yF = np.unpackbits(np.array([vals],dtype=np.uint8), count=nN, bitorder="little")
            ss[F,vals] = e
            e += 1
            for i in neighbors:
                iidx = neighbors.index(i)
                for s,v in enumerate([0,1]):
                    #if not fi_value_set[i][s]:
                    if not (F,i,v) in m.keys():
                        m[F,i,v] = c
                        c += 1
                    # f2v subsum indexes
                    if yF[iidx] == v:
                        n[F,vals,i,v] = d
                        d += 1
    return m,n,ss

def f2v_subsum_indexes(fg, f2v_idx,ss_idx):
    #factor_sizes = np.array([len(list(fg.neighbors(F))) for F in fg.iter_factors()])
    factor_sizes = np.array([fg.factor_size(F) for F in fg.iter_factors()])

    nSS = len(ss_idx)
    edges = []
    redges = []
    c = 0
    for c_F,F in enumerate(fg.iter_factors()):
        neighbors = fg.factor_variables(F)
        nN = len(neighbors)
        for vals in range(2**nN):
            yF = np.unpackbits(np.array([vals],dtype=np.uint8), count=nN, bitorder="little")
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
            yF = np.unpackbits(np.array([vals],dtype=np.uint8), count=nN, bitorder="little")

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
            yF = np.unpackbits(np.array([vals],dtype=np.uint8), count=nN, bitorder="little")
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
            yF = np.unpackbits(np.array([vals],dtype=np.uint8), count=nN, bitorder="little")
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


