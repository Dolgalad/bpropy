import networkx as nx
import numpy as np

# create the bipartite graph representing an instance
def bipartite_graph(pb):
    g = nx.Graph()
    features = []
    for i in range(pb.nv()):
        g.add_node(i, bipartite=0, x=[pb.w[i]])
        features.append(pb.w[i])
    ci = 0
    constraints = pb.constraints()
    for cj,c in enumerate(constraints):
        for i in range(c.A.shape[0]):
            if c.lb[i]==c.ub[i]:
                # equality constraint
                g.add_node(pb.nv() + ci, bipartite=1, x=[c.lb[i]])
                features.append(c.lb[i])
                for j in np.where(c.A[i,:] != 0)[0]:
                    g.add_edge(j, pb.nv() + ci, e=[c.A[i,j]])
                ci += 1
            else:
                if c.lb[i] != -np.inf:
                    g.add_node(pb.nv() + ci, bipartite=1, x=[c.lb[i]])
                    features.append(c.lb[i])
                    for j in np.where(c.A[i,:] != 0)[0]:
                        g.add_edge(j, pb.nv() + ci, e=[c.A[i,j]])
                    ci += 1
                if c.ub[i] != np.inf:
                    g.add_node(pb.nv() + ci, bipartite=1, x=[c.ub[i]])
                    features.append(c.ub[i])
                    for j in np.where(c.A[i,:] != 0)[0]:
                        g.add_edge(j, pb.nv() + ci, e=[c.A[i,j]])
                    ci += 1

    return g,np.array(features)

if __name__=="__main__":
    from dopy.mis import *
    g = nx.generators.random_graphs.erdos_renyi_graph(1000, 0.15)
    pb = MIS(g)

    bpg, bpg_feat = bipartite_graph(pb)

    print("Nodes ok : ", bpg.number_of_nodes() == (g.number_of_nodes() + g.number_of_edges()))
    print("Edges ok : ", bpg.number_of_edges() == 2*g.number_of_edges())
