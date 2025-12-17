from dopy.maxcut import *
import time
import networkx as nx
import itertools
import matplotlib.pyplot as plt

graph_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,170, 180, 190,200]
N = 10

opts = []

for n in graph_sizes:
    max_iter = 1000

    opt = []

    for j in range(N):
        g = nx.generators.erdos_renyi_graph(n, 0.15)
        #print("Graph: ", g)
        #for ei,e in enumerate(g.edges):
        #    print("\t", ei, e)
        
        
        # Solve ILP
        pb = MaxCut(g)
        t_sol = time.time()
        ilp_sol = pb.solve()
        t_sol = time.time() - t_sol
        
        #print("ILP solution = ", ilp_sol.x)
        
        
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
        msg = np.ones(4*pb.g.number_of_edges())
        r_msg = msg.copy()
        residuals = np.zeros(4*pb.g.number_of_edges())
        
        for it in range(max_iter):
            new_msg = bp_update_messages(pb, msg, eim)
            r_new_msg = bp_update_messages(pb, r_msg, eim)

            # residual belief propagation
            mt = int(np.floor(np.argmax(residuals) / 4))
            nr_new_msg = r_msg.copy()
            nr_new_msg[4*mt:4*mt+4] = r_new_msg[4*mt:4*mt+4]
            edge_list = [e for e in pb.g.edges]
            re = edge_list[mt]
            for ei,e in enumerate(pb.g.edges):
                if e!=re and ((re[0] in e) or (re[1] in e)):
                    nr_new_msg[4*ei:4*ei+4] = r_new_msg[4*ei:4*ei+4]


            residuals = np.abs(nr_new_msg - r_msg)
            #print(residuals)

            u = np.max(np.abs(new_msg - msg))
            v = np.max(np.abs(r_new_msg - r_msg))
            print(n, j, it, u, v, mt)
            #print("\t", new_msg - msg)
        
            if u < 1e-8:
                msg = new_msg
                break
            if v < 1e-8:
                msg = r_new_msg
                break
            msg = new_msg
            r_msg = nr_new_msg
        msg = r_new_msg
        
        # marginals
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
                #print(ei, e, x, marginals[c])
                c += 1
        
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
        
        # solve greedy with these marginals
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
        
        
        print("n = ", n)
        print("Solution     = ", sol, np.sum(sol > 0.5))
        ilps = np.array(ilp_sol.x > 0.5, dtype=int)
        print("ILP Solution = ", ilps, np.sum(ilps > 0.5))

        gap = 100 * (np.sum(ilps > 0.5)  - np.sum(sol > 0.5)) / np.sum(ilps > 0.5)

        opt.append( gap)
    opts.append(opt)
    print(len(opts))
    plt.figure()
    plt.boxplot(opts, positions=graph_sizes[:len(opts)], widths=5)
    plt.savefig("residual_maxcut.png")
    plt.close()


plt.figure()
plt.boxplot(opts, positions=graph_sizes, widths=5)
plt.show()


