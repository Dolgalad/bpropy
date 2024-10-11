from bpropy.lbp import *
from bpropy.trw import *
from dopy.maxcut import *

import time
import networkx as nx
import itertools
import matplotlib.pyplot as plt

import random
random.seed(123)
import numpy as np
np.random.seed(123)

n = 5

g = nx.generators.erdos_renyi_graph(n, 0.5)
print("Graph")
for e in g.edges:
    print("\t", e)
edge_list = [e for e in g.edges]

# Solve ILP
pb = MaxCut(g)
t_sol = time.time()
ilp_sol = pb.solve()
t_sol = time.time() - t_sol
ilp_val = np.sum(ilp_sol.x > 0.5)

# create factor graph
fg = create_factor_graph(g)
print("Factor graph")
for e in fg.edges:
    print("\t", e)
n_ss = sufficient_statistics_dim(fg)

# stack sufficient stats and v2f messages
#theta = np.zeros(n_ss)
#c = 0
#for i in range(pb.nv()):
#    #theta[c] = pb.w[i]
#    theta[c+1] = pb.w[i]
#    c += 2
#for e in pb.g.edges:
#    theta[c+3] = -n+1
#    c += 4

theta = np.zeros(n_ss)
c = 0
for i in range(pb.nv()):
    theta[c+1] = 1
    c += 2
for e in pb.g.edges:
    #theta[c+1] = 1. / len(list(g.neighbors(e[0])))
    #theta[c+2] = 1. / len(list(g.neighbors(e[1])))
    theta[c+3] = -n-1
    c += 4

theta  = theta - np.min(theta) + 1
print("theta = \n", theta)

# initialize messages
v2f_dim = f2v_messages_dim(fg)
v2f_msg = np.zeros(v2f_dim)
f2v_msg = np.zeros(v2f_dim)

def update_variable_to_factor_messages(fg, f2v_msg, index_map):
    c = 0
    v2f_msg = np.zeros_like(f2v_msg)
    for i,d in fg.nodes(data=True):
        if d["bipartite"]==0:
            for F in fg.neighbors(i):
                for yi in [0,1]:
                    for Fp in fg.neighbors(i):
                        if Fp!=F:
                            print(i, " -> ", F, "yi=", yi, " add r ", Fp, " -> ", i, " = ",f2v_msg[index_map[Fp,i,yi]])

                            v2f_msg[c] += f2v_msg[index_map[Fp,i,yi]]
                    print("v2f ", i, "->", F, " = ", v2f_msg[c])
                    c += 1
    #print("v2f before norm = \n", v2f_msg)
    #for i in range(int(v2f_msg.shape[0] / 2)):
    #    delta = np.log(np.sum(np.exp(v2f_msg[2*i:2*i+2])))
    #    print("delta = ", delta, v2f_msg[2*i:2*i+2])
    #    print(np.sum(np.exp(v2f_msg[2*i:2*i+2])))
    #    if delta != 0:
    #        v2f_msg[2*i:2*i+2] = np.exp(v2f_msg[2*i:2*i+2] - delta)
    #print("v2f after norm = \n", v2f_msg)
    return v2f_msg

def update_factor_to_variable_messages(fg, theta, v2f_msg, ss_index_map, v2f_index_map):
    c = 0
    f2v_msg = np.zeros_like(v2f_msg)
    for F,d in fg.nodes(data=True):
        if d["bipartite"]==1:
            F_neighbors = list(fg.neighbors(F))
            nF = len(F_neighbors)
            for i in fg.neighbors(F):
                i_idx = F_neighbors.index(i)
                for yi in [0,1]:
                    for vals in range(2**nF):
                        yF = np.unpackbits(np.array([vals],dtype=np.uint8), count=nF, bitorder="little")
                        if yF[i_idx]==yi:
                            print(theta[ss_index_map[F,vals]])
                            q = theta[ss_index_map[F,vals]].copy()
                            for j in fg.neighbors(F):
                                if j!=i:
                                    q += v2f_msg[v2f_index_map[F,j,yF[i_idx]]].copy()
                            print(theta[ss_index_map[F,vals]])
                            print()

                            f2v_msg[c] += np.exp(q)
                    f2v_msg[c] = np.log(f2v_msg[c])
                    c += 1
    print("f2v_msg = \n", f2v_msg)
    return f2v_msg

def compute_factor_marginals(fg, theta, v2f, v2f_index_map):
    muF = theta.copy()
    c = 0
    for F,d in fg.nodes(data=True):
        if d["bipartite"]==1:
            neighbors = list(fg.neighbors(F))
            nF = len(neighbors)
            for vals in range(2**nF):
                yF = np.unpackbits(np.array([vals],dtype=np.uint8), count=nF, bitorder="little")
                for jidx,j in enumerate(neighbors):
                    muF[c] += v2f[v2f_index_map[F,j,yF[jidx]]]
                c += 1
    c = 0
    for F,d in fg.nodes(data=True):
        if d["bipartite"]==1:
            neighbors = list(fg.neighbors(F))
            nF = len(neighbors)
            zF = np.log(np.sum(np.exp(muF[c:c+2**nF])))
            muF[c:c+2**nF] = np.exp(muF[c:c+2**nF] - zF)
            c += 2**nF


    return muF


v2f_index_map,_,ss_index_map = f2v_index_map(fg)

max_iter = 1000
tol = 1e-8

# initialize messages
v2f_dim = f2v_messages_dim(fg)
v2f_msg = np.zeros(v2f_dim)
f2v_msg = np.zeros(v2f_dim)

mu = np.zeros_like(theta)
for it in range(max_iter):
    f2v_msg = update_factor_to_variable_messages(fg, theta, v2f_msg, ss_index_map, v2f_index_map)
    print("f2v =")
    print(f2v_msg)
    v2f_msg = update_variable_to_factor_messages(fg, f2v_msg, v2f_index_map)
    print("v2f =")
    print(v2f_msg)
    muF = compute_factor_marginals(fg, theta, v2f_msg, v2f_index_map)
    if np.any(np.isinf(muF)) or np.any(np.isinf(v2f_msg)) or np.any(np.isinf(f2v_msg)):
        break

    print(muF[:2*n].reshape(n,2))
    print(np.argmax(muF[:2*n].reshape(n,2), axis=1))
    u = np.max(np.abs(muF - mu))
    print(it, u)
    ans = input("?")
    if u<tol:
        break

    mu = muF

print(mu[:2*n].reshape(n,2))
print(np.argmax(mu[:2*n].reshape(n,2), axis=1))
print(ilp_sol.x)
print(ilp_val)

