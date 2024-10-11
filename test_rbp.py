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
n = 100


# maximum BP iterations
max_iter = 1000
convergence_tol = 1e-6

# store the optimality gap
opts = []
resi = []

for j in range(100):
    
    # Randomly generate a graph
    g = nx.generators.erdos_renyi_graph(n, 0.15)
    edge_list = [e for e in g.edges]
   
    
    # Solve ILP
    pb = MaxCut(g)
    t_sol = time.time()
    ilp_sol = pb.solve()
    t_sol = time.time() - t_sol
    ilp_val = np.sum(ilp_sol.x > 0.5)
    
    
    # edge index map
    eim = {}
    c = 0
    for ei,e in enumerate(pb.g.edges):
        eim[e,e[0],0] = c
        eim[e,e[0],1] = c+1
        eim[e,e[1],0] = c+2
        eim[e,e[1],1] = c+3
        c += 4
    
    # message passing
    msg = np.random.random_sample(4*pb.g.number_of_edges())
    r_msg = msg.copy()
    residuals = np.zeros(4*pb.g.number_of_edges())
    
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
            #if e!=re and dir_idx==0 and re[0] in e:
            #    if re[0]==e[0]:
            #        nr_new_msg[4*ei+2:4*ei+4] = r_new_msg[4*ei+2:4*ei+4]
            #    if re[0]==e[1]:
            #        nr_new_msg[4*ei:4*ei+2] = r_new_msg[4*ei:4*ei+2]
            #if e!=re and dir_idx==1 and re[0] in e:
            #    if re[0]==e[0]:
            #        nr_new_msg[4*ei:4*ei+2] = r_new_msg[4*ei:4*ei+2]
            #    if re[0]==e[1]:
            #        nr_new_msg[4*ei+2:4*ei+4] = r_new_msg[4*ei+2:4*ei+4]
 
    
        residuals = np.abs(nr_new_msg - r_msg)
    
        u = np.max(np.abs(new_msg - msg))
        v = np.max(np.abs(r_new_msg - r_msg))
    
        # solve the problem greedily
        edges, assignments, marginals = pair_marginals_from_messages(pb, nr_new_msg, eim)
        grd_sol = maxcut_greedy_with_pair_marginals(pb, edges, assignments, marginals)
        grd_val = np.sum(grd_sol)
        gap = (ilp_val - grd_val) / ilp_val

        # solve the problem with unary marginals as weights
        um = unary_marginals(marginals, edges, pb)
        wpb = MaxCut(pb.g, weights=np.log(um+1e-8))
        t_wsol = time.time()
        wilp_sol = wpb.solve()
        t_wsol = time.time() - t_wsol
        wilp_val = np.sum(wilp_sol.x > 0.5)
        print("\t", t_sol, t_wsol, ilp_val, wilp_val, grd_val)
        print("\t", np.where(np.array(ilp_sol.x > 0.5, dtype=int))[0])
        print("\t", np.where(np.array(wilp_sol.x > 0.5, dtype=int))[0])
        print("\t", np.where(np.array(grd_sol > 0.5, dtype=int))[0])


        opts.append(gap)
        resi.append(v)
        print(n, j, it, u, v, mt, gap)
    
        plt.figure()
        plt.scatter(resi, opts,alpha=0.3)
        x, y = zip(*sorted((xVal, np.mean([yVal for a, yVal in zip(resi, opts) if np.abs(xVal-a)<0.05*a])) for xVal in set(resi)))
        plt.plot(x,y,c="r")
        plt.ylabel("Optimality gap")
        plt.xlabel("BP Residual")
        #plt.xscale("log")
        #plt.yscale("log")
        plt.savefig(f"test_rbp_n{n}.png")
        plt.close()
    
        #if u < convergence_tol:
        #    msg = new_msg
        #    break
        if v < convergence_tol:
            msg = r_new_msg
            break
        #if gap == 0:
        #    break
        msg = new_msg
        r_msg = nr_new_msg

