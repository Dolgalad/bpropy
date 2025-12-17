import networkx as nx
import sys
import time

from bpropy.lbp import *
from dopy.mis import *

# load graph
g = nx.read_adjlist(sys.argv[1])
nmap = {str(i):i for i in range(g.number_of_nodes())}
g = nx.relabel_nodes(g, nmap, copy=False)

# create the factor graph
t0 = time.time()
pb = MIS(g)
fg = create_factor_graph(pb.g)
t0 = time.time() - t0
print("Factor graph creation time : ", t0)

# BP edges
t0 = time.time()
n_ss = sufficient_statistics_dim(fg)
# create the computation edges
f2v_idx, f2v_subsum_idx,ss_idx = f2v_index_map(fg)
# creating message passing edges
f2v_edges,r_edges = f2v_subsum_indexes(fg, f2v_idx, ss_idx)
f2v_edges = torch.from_numpy(f2v_edges)
r_edges = torch.from_numpy(r_edges)
# scatter indices
q_edges = torch.from_numpy(v2f_indexes(fg, f2v_idx))
m_edges = torch.from_numpy(marginal_indexes(fg, f2v_idx, ss_idx))
vm_edges = torch.from_numpy(variable_marginal_indexes(fg, f2v_idx, ss_idx))
t0 = time.time() - t0

print("BP edges creation time : ", t0)

