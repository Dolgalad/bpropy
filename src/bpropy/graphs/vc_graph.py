import networkx as nx
import numpy as np

class VCGraph(nx.Graph):
    def __init__(self, pb, cliques=None):
        super().__init__()
        self.graph["problem_type"] = pb.__class__.__name__  # "MIS" | "SCP" | ...
        _,self.features = variable_constraint_graph(pb, cliques=cliques, g=self)
    def iter_variables(self):
        """Iterate over variable nodes, always iterates in increasing label value
        """
        for i in range(self.number_of_nodes()):
            if self.nodes[i]["bipartite"]==0:
                yield i
    def iter_constraints(self):
        """Iterate over constraint nodes, always iterates in increasing label value
        """
        for i in range(self.number_of_nodes()):
            if self.nodes[i]["bipartite"]==1:
                yield i
    def constraint_variables(self, C):
        """Return list of variable nodes adjacent to constraint node `C`. Always returns a sorted list.
        """
        return sorted(list(self.neighbors(C)))
    def constraint_size(self, C):
        """Return number of variable nodes adjacent to constraint node `C`
        """
        return len(self.constraint_variables(C))
    def number_of_variables(self):
        """Return number of variables
        """
        return len(list(self.iter_variables()))
    def nv(self):
        """Convenience function for getting number of variables
        """
        return self.number_of_variables()
    def number_of_constraints(self):
        """Returns number of constraint nodes
        """
        return len(list(self.iter_constraints()))
    def nc(self):
        """Convenience function for getting number of constraint nodes
        """
        return self.number_of_constraints()


def variable_constraint_graph(pb, cliques=None, g=None):
    """Create variable-constraint bipartite graph from LP formulation 
    """
    W = pb.weights()
    if g is None:
        g = nx.Graph()
    features = []
    for i in range(pb.nv()):
        g.add_node(i, bipartite=0, x=[W[i]])
        features.append(W[i])
    ci,ce = 0,0
    #constraints = pb.constraints(cliques=cliques)
    constraints = pb.constraints()

    for cj,c in enumerate(constraints):
        for i in range(c.A.shape[0]):
            if c.lb[i]==c.ub[i]:
                # equality constraint
                g.add_node(pb.nv() + ci, bipartite=1, x=[c.lb[i]])
                features.append(c.lb[i])
                for j in np.where(c.A[i,:] != 0)[0]:
                    g.add_edge(j, pb.nv() + ci, e=[c.A[i,j]])
                    ce += 1
                ci += 1
            else:
                if c.lb[i] != -np.inf:
                    g.add_node(pb.nv() + ci, bipartite=1, x=[c.lb[i]])
                    features.append(c.lb[i])
                    for j in np.where(c.A[i,:] != 0)[0]:
                        g.add_edge(j, pb.nv() + ci, e=[c.A[i,j]])
                        ce += 1
                    ci += 1
                if c.ub[i] != np.inf:
                    g.add_node(pb.nv() + ci, bipartite=1, x=[c.ub[i]])
                    features.append(c.ub[i])
                    for j in np.where(c.A[i,:] != 0)[0]:
                        g.add_edge(j, pb.nv() + ci, e=[c.A[i,j]])
                        ce += 1
                    ci += 1

    return g,np.array(features)

if __name__=="__main__":
    from dopy.mis import *
    g = nx.generators.random_graphs.erdos_renyi_graph(1000, 0.15)
    pb = MIS(g)

    bpg, bpg_feat = bipartite_graph(pb)

    print("Nodes ok : ", bpg.number_of_nodes() == (g.number_of_nodes() + g.number_of_edges()))
    print("Edges ok : ", bpg.number_of_edges() == 2*g.number_of_edges())
