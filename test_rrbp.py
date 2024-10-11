from dopy.maxcut import *
import time
import networkx as nx
import itertools
import matplotlib.pyplot as plt


def phi_ij(i, j, x_i, msg, pb, eim):
    r = np.exp(x_i)
    for k in pb.g.neighbors(i):
        if k!=j:
            e = (min(i,k), max(i,k))
            r *= msg[eim[e, i, x_i]]
    return r

def bp_update_messages(pb, msg,eim):
    new_msg = msg.copy()
    for ei,e in enumerate(pb.g.edges):
        new_msg[4*ei]   = phi_ij(e[1],e[0],0,msg,pb,eim) +             phi_ij(e[1],e[0],1,msg,pb,eim)
        new_msg[4*ei+1] = phi_ij(e[1],e[0],0,msg,pb,eim) + np.exp(-n)*phi_ij(e[1],e[0],1,msg,pb,eim)
        a = new_msg[4*ei] + new_msg[4*ei+1]
        if a==0:
            new_msg[4*ei] = 0.5
            new_msg[4*ei+1] = 0.5
        else:
            new_msg[4*ei] /= a
            new_msg[4*ei+1] /= a

        new_msg[4*ei+2] = phi_ij(e[0],e[1],0,msg,pb,eim) +             phi_ij(e[0],e[1],1,msg,pb,eim)
        new_msg[4*ei+3] = phi_ij(e[0],e[1],0,msg,pb,eim) + np.exp(-n)*phi_ij(e[0],e[1],1,msg,pb,eim)
        a = new_msg[4*ei+2] + new_msg[4*ei+3]
        if a==0:
            new_msg[4*ei+2] = 0.5
            new_msg[4*ei+3] = 0.5
        else:
            new_msg[4*ei+2] /= a
            new_msg[4*ei+3] /= a

    return new_msg

# marginals computation from messages
def pair_marginals_from_messages(pb, msg, eim):
    edges = []
    assignments = []
    marginals = []
    for ei,e in enumerate(pb.g.edges):
        for x in itertools.product([0,1],[0,1]):
            marginals.append(np.exp(-n*x[0]*x[1])*phi_ij(e[0],e[1],x[0],msg,pb,eim)*phi_ij(e[1],e[0],x[1],msg,pb,eim))
            edges.append(e)
            assignments.append(x)
    marginals = np.array(marginals)

    c=0
    for ei,e in enumerate(pb.g.edges):
        normc = np.sum(marginals[4*ei:4*ei+4])
        for x in itertools.product([0,1],[0,1]):
            marginals[c] /= normc
            c += 1

    return edges, assignments, marginals

def unary_marginals(marginals, edges, pb):
    m = []
    for u in range(n):
        pxu0 = 0.0
        pxu1 = 0.0
        for v in pb.g.neighbors(u):
            e = (min(u,v), max(u,v))
            eidx = edges.index(e)
            if u<v:
                pxu0 += marginals[eidx]+marginals[eidx+1]
                pxu1 += marginals[eidx+2]+marginals[eidx+3]
            else:
                pxu0 += marginals[eidx]+marginals[eidx+2]
                pxu1 += marginals[eidx+1]+marginals[eidx+3]
        #print(f"P[x_{u}=0] = ", pxu0)
        #print(f"P[x_{u}=1] = ", pxu1)
        m.append(pxu1)
    return np.array(m)

def maxcut_greedy_with_pair_marginals(pb, edges, assignments, marginals):
    # solve greedy with these marginals
    n = pb.g.number_of_nodes()
    queue = list(range(n))
    sol = np.zeros(n, dtype=int)
    while not len(queue)==0:
        um = unary_marginals(marginals, edges, pb)
        u = queue[np.argmax(um[np.array(queue)])]
    
        sol[u] = 1
        queue.remove(u)
        for v in pb.g.neighbors(u):
            sol[v] = 0
            if v in queue:
                queue.remove(v)
    
        # update pair marginals
        for i in range(len(marginals)):
            e = edges[i]
            x = assignments[i]
            margi = marginals[i]
            if not e[0] in queue:
                if x[0] != sol[e[0]]:
                    marginals[i] = 0
            if not e[1] in queue:
                if x[1] != sol[e[1]]:
                    marginals[i] = 0
            if (not e[0] in queue) and (not e[1] in queue):
                if (x[0]==sol[e[0]]) and (x[1]==sol[e[1]]):
                    marginals[i] = 1
                else:
                    marginals[i] = 0
    return sol

# number of nodes
n = 65

# maximum BP iterations
max_iter = 1000
convergence_tol = 1e-3

# Randomly generate a graph
g = nx.generators.erdos_renyi_graph(n, 0.15)
edge_list = [e for e in g.edges]


# Solve ILP
w = np.random.random_sample(n)
pb = MaxCut(g, weights=w)
t_sol = time.time()
ilp_sol = pb.solve()
t_sol = time.time() - t_sol
ilp_val = np.dot(w, ilp_sol.x > 0.5)


# edge index map
eim = {}
c = 0
for ei,e in enumerate(pb.g.edges):
    eim[e,e[0],0] = c
    eim[e,e[0],1] = c+1
    eim[e,e[1],0] = c+2
    eim[e,e[1],1] = c+3
    c += 4


opts = []
resi = []
rand_opts = []
rand_resi = []

for ii in range(10):
    # message passing
    msg = np.random.random_sample(4*pb.g.number_of_edges())
    r_msg = msg.copy()
    residuals = np.zeros(4*pb.g.number_of_edges())

    iopts, iresi, irand_opts, irand_resi = [], [], [], []
    
    
    for it in range(max_iter):
        new_msg = bp_update_messages(pb, msg, eim)
        r_new_msg = bp_update_messages(pb, r_msg, eim)
    
        # residual belief propagation
        mt = int(np.floor(np.argmax(residuals) / 4))
        # direction of the edge
        #dir_idx = int(np.floor(np.argmax(residuals[4*mt:4*mt+4]) / 2))
    
        nr_new_msg = r_msg.copy()
        nr_new_msg[4*mt:4*mt+4] = r_new_msg[4*mt:4*mt+4]
        #nr_new_msg[4*mt+2*dir_idx:4*mt+2*dir_idx+2] = r_new_msg[4*mt+2*dir_idx:4*mt+2*dir_idx+2]
    
        re = edge_list[mt]
        for ei,e in enumerate(pb.g.edges):
            if e!=re and ((re[0] in e) or (re[1] in e)):
                nr_new_msg[4*ei:4*ei+4] = r_new_msg[4*ei:4*ei+4]
    
        residuals = np.abs(nr_new_msg - r_msg)
    
        u = np.max(np.abs(new_msg - msg))
        v = np.max(np.abs(r_new_msg - r_msg))
    
    
        # solve the problem greedily
        edges, assignments, marginals = pair_marginals_from_messages(pb, nr_new_msg, eim)
        grd_sol = maxcut_greedy_with_pair_marginals(pb, edges, assignments, marginals)
        grd_val = np.dot(w, grd_sol)
        gap = (ilp_val - grd_val) / ilp_val
        
        um = unary_marginals(marginals, edges, pb)
        wpb = MaxCut(pb.g, weights=um)
        wilp_sol = wpb.solve()
        wilp_val = np.dot(w, wilp_sol.x > 0.5)
        gap = (ilp_val - wilp_val) / ilp_val

        print(ii, "classic", it, u, v, mt, gap)
       

    
        iopts.append(gap)
        iresi.append(v)
    
        #if u < convergence_tol:
        #    msg = new_msg
        #    break
        #if v < convergence_tol:
        #    msg = r_new_msg
        #    break
        #if gap == 0:
        #    break
        msg = new_msg
        r_msg = nr_new_msg
    
    # message passing
    msg = np.random.random_sample(4*pb.g.number_of_edges())
    r_msg = msg.copy()
    residuals = np.zeros(4*pb.g.number_of_edges())
    
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))
    
    for it in range(max_iter):
        new_msg = bp_update_messages(pb, msg, eim)
        r_new_msg = bp_update_messages(pb, r_msg, eim)
    
        # randomized residual belief propagation
        mt = int(np.floor(np.argmax(residuals) / 4))
        a = residuals.shape[0]
        mt = np.random.choice(range(int(a/4)), p=softmax(np.max(residuals.reshape(int(a/4), 4), axis=1)))
        # direction of the edge
        #dir_idx = int(np.floor(np.argmax(residuals[4*mt:4*mt+4]) / 2))
    
        nr_new_msg = r_msg.copy()
        nr_new_msg[4*mt:4*mt+4] = r_new_msg[4*mt:4*mt+4]
        #nr_new_msg[4*mt+2*dir_idx:4*mt+2*dir_idx+2] = r_new_msg[4*mt+2*dir_idx:4*mt+2*dir_idx+2]
    
        re = edge_list[mt]
        for ei,e in enumerate(pb.g.edges):
            if e!=re and ((re[0] in e) or (re[1] in e)):
                nr_new_msg[4*ei:4*ei+4] = r_new_msg[4*ei:4*ei+4]
    
        residuals = np.abs(nr_new_msg - r_msg)
    
        u = np.max(np.abs(new_msg - msg))
        v = np.max(np.abs(r_new_msg - r_msg))
    
    
        # solve the problem greedily
        edges, assignments, marginals = pair_marginals_from_messages(pb, nr_new_msg, eim)
        grd_sol = maxcut_greedy_with_pair_marginals(pb, edges, assignments, marginals)
        grd_val = np.dot(w, grd_sol)
        gap = (ilp_val - grd_val) / ilp_val

        um = unary_marginals(marginals, edges, pb)
        wpb = MaxCut(pb.g, weights=um)
        wilp_sol = wpb.solve()
        wilp_val = np.dot(w, wilp_sol.x > 0.5)
        gap = (ilp_val - wilp_val) / ilp_val

        print(ii, "rand", it, u, v, mt, gap)

        irand_opts.append(gap)
        irand_resi.append(v)
    
        #if u < convergence_tol:
        #    msg = new_msg
        #    break
        #if v < convergence_tol:
        #    msg = r_new_msg
        #    break
        #if gap == 0:
        #    break
        msg = new_msg
        r_msg = nr_new_msg

    opts.append(iopts)
    resi.append(iresi)
    rand_opts.append(irand_opts)
    rand_resi.append(irand_resi)

opts = np.array(opts)
resi = np.array(resi)
rand_opts = np.array(rand_opts)
rand_resi = np.array(rand_resi)

print(opts.shape)

fig,ax = plt.subplots(2,1)
ax[0].plot(np.mean(resi,axis=0), label="max")
ax[0].plot(np.mean(rand_resi,axis=0), label="rand")
ax[0].set_ylabel("Residual")
ax[1].plot(np.mean(opts,axis=0), label="max")
ax[1].plot(np.mean(rand_opts,axis=0),label="rand")

ax[1].set_ylabel("Gap")
plt.legend()

plt.savefig(f"test_rrbp_n{n}.png")
plt.close()


