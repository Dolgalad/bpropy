import sys
sys.path.insert(0,"..")
import time
import torch
import numpy as np
import networkx as nx

from bpropy.co.mis import MIS
from bpropy.mrf.mis_factor_graph import MISFactorGraph
from bpropy.mrf.lbp_edges import LBPEdges
from bpropy.mrf.lbp import loopy_belief_propagation
from bpropy.test.test_mis import random_mis

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mis_idealized_canonical_parameters(fg, M=10.):
    theta = torch.zeros(fg.number_sufficient_statistics(), device=device, requires_grad=False)
    c = 0
    for F in fg.iter_factors():
        nF = fg.factor_size(F)
        if nF == 1:
            theta[c+1] = 1.
            theta[c] = 0.
            c += 2
        else:
            for v in range(2**nF):
                vals = np.unpackbits(np.array([v]).astype(np.uint8), count=nF, bitorder="little")
                if vals.sum()>1:
                    theta[c] = -10.
                else:
                    theta[c] = 0.
                c += 1
    return theta

def time_f(func, *args, **kwargs):
    t = time.time()
    r = func(*args, **kwargs)
    t = time.time() - t
    return t, r


N = 10
bp_max_iter = 10

# generate a list of instances
pbs = [random_mis(100, 0.15) for _ in range(N)]

# solve instances one at a time
t_fg,t_edges,t_theta,t_v2f,t_lbp,t_grd=0,0,0,0,0,0
t_tot = time.time()
for pb in pbs:
    _t_fg,fg = time_f(MISFactorGraph,pb)
    _t_edges,edges = time_f(lambda _fg: LBPEdges(_fg).to(device), fg)

    _t_theta,theta = time_f(mis_idealized_canonical_parameters,fg)
    _t_v2f,v2f = time_f(torch.zeros_like,theta)

    _t_lbp,mu = time_f(loopy_belief_propagation,theta, v2f, edges, max_iter=bp_max_iter)
    _t_grd,x_sp_greedy = time_f(pb.greedy,mu[:2*pb.nv()].reshape(pb.nv(),2)[:,1].cpu().numpy())
    t_fg += _t_fg
    t_edges += _t_edges
    t_theta += _t_theta
    t_v2f += _t_v2f
    t_lbp += _t_lbp
    t_grd += _t_grd

t_tot = time.time() - t_tot

print("One problem at a time = ", t_tot)
print("\tfg    = ", t_fg)
print("\tedges = ", t_edges)
print("\ttheta = ", t_theta)
print("\tv2f   = ", t_v2f)
print("\tlbp   = ", t_lbp)
print("\tgrd   = ", t_grd)



# solve instances by creating one big batch
t0 = time.time()
t_batch,batch_pb = time_f(MIS.batch,pbs)
t_fg,batch_fg = time_f(MISFactorGraph,batch_pb)
t_edges,batch_edges = time_f(lambda fg: LBPEdges(fg).to(device), batch_fg)

t_theta,batch_theta = time_f(mis_idealized_canonical_parameters,batch_fg)
t_v2f,batch_v2f = time_f(torch.zeros_like,batch_theta)

t_lbp,batch_mu = time_f(loopy_belief_propagation,batch_theta, batch_v2f, batch_edges, max_iter=bp_max_iter)
t_grd,x_sp_greedy = time_f(batch_pb.greedy,batch_mu[:2*batch_pb.nv()].reshape(batch_pb.nv(),2)[:,1].cpu().numpy())
t0 = time.time() - t0

print("One big batch = ", t0)
print("\tbatch = ", t_batch)
print("\tfg    = ", t_fg)
print("\tedges = ", t_edges)
print("\ttheta = ", t_theta)
print("\tv2f   = ", t_v2f)
print("\tlbp   = ", t_lbp)
print("\tgrd   = ", t_grd)

