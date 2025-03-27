"""Naive Mean Field computation edges
"""

class NMFEdges:
    def __init__(self, fg=None):
        self.pmprod_edges = None
        self.pmprodsum_edges = None
        self.n_ss = None
        self.n_vars = None
        self.n_factors = None
        if not fg is None:
            self.fill(fg)
    def fill(self, fg):
        self.pmprod_edges, self.pmprodsum_edges = naive_mean_field_edges(fg)
        self.n_ss = torch.tensor([fg.number_sufficient_statistics()])
        self.n_vars = torch.tensor([fg.nv()])
        self.n_factors = torch.tensor([fg.nf()])
    def num_graphs(self):
        return self.n_ss.shape[0]
    def clone(self):
        ne = NMFEdges()
        ne.n_ss = self.n_ss.clone()
        ne.n_vars = self.n_vars.clone()
        ne.n_factors = self.n_factors.clone()

        ne.pmprod_edges = self.pmprod_edges.clone()
        ne.pmprodsum_edges = self.pmprodsum_edges.clone()
        return ne
    def to(self, device):
        ne = LBPEdges()
        ne.n_ss = self.n_ss.to(device)
        ne.n_vars = self.n_vars.to(device)
        ne.n_factors = self.n_factors.to(device)

        ne.pmprod_edges = self.pmprod_edges.to(device)
        ne.pmprodsum_edges = self.pmprodsum_edges.to(device)
        return ne



def naive_mean_field_edges(fg):
    nV = number_of_variables(fg)
    nF = number_of_factors(fg)

    pmprod_index_map = {}
    pmprod_edges = []

    pmprodsum_edges = []

    c,d = 0,0

    for iF,F in enumerate(iter_factor_nodes(fg)):
        neighbors = list(fg.neighbors(F))
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
    return torch.tensor(pmprod_edges),torch.tensor(pmprodsum_edges)

