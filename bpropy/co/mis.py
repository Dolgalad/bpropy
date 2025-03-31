"""
MIS solving routines
"""

import networkx as nx
import numpy as np
import itertools
from scipy.optimize import milp, LinearConstraint, Bounds
import matplotlib.pyplot as plt

import importlib.util
docplex_spec = importlib.util.find_spec("docplex")
if docplex_spec is not None:
    from docplex.mp.model import Model

from bpropy.utils.graph_utils import ordered_edges
from bpropy.clique_cover import is_edge_cover
from bpropy.utils.set_sort import lenlex_sort

from bpropy.vc_graph import VCGraph

class MIS:
    def __init__(self, g, weights=None, cliques=None):
        self.g = g
        if weights is None:
            self.w = np.ones(nx.number_of_nodes(g))
        else:
            self.w = weights
        if cliques is None:
            self.C = ordered_edges(self.g)
        else:
            self.C = lenlex_sort(cliques)
        if not is_edge_cover(self.C, self.g):
            raise Exception("MIS: cliques do not cover all edges of graph")
    def nv(self):
        """Get number of variables
        """
        return nx.number_of_nodes(self.g)
    def ne(self):
        """Get number of edges
        """
        return nx.number_of_edges(self.g)
    def constraints(self, cliques=None):
        """Get list of scipy.LinearConstraint objects. If provided `cliques` should be a list of cliques covering all edges of the graph.
        """
        c = []
        if cliques is None:
            cliques = self.C
        else:
            cliques = lenlex_sort(cliques)
            if not is_edge_cover(cliques, self.g):
                raise Exception("MIS: cliques do not cover all edges of graph")
        for clq in cliques:
            row = np.zeros(self.nv())
            for u in clq:
                row[u] = 1
            c.append(LinearConstraint(row, ub=1))
        return c
    def solve(self, disp=False, cliques=None):
        """Solve MIS problem
        """
        return milp(
                    -self.w,
                    integrality=np.ones(self.nv()),
                    constraints=self.constraints(cliques=cliques),
                    bounds=Bounds(np.zeros(self.nv()), np.ones(self.nv())),
                    options={"disp":disp}
                )
    def greedy(self, scores=None):
        """Solve the problem geedily
        """
        if scores is None:
            scores = np.array([-self.g.degree[n] for n in range(self.nv())])
        return mis_greedy(self, scores)
    def check_solution(self, x):
        """Check if solution is feasible
        """
        return solution_feasible(self, x)
    @staticmethod
    def from_vcgraph(vcg):
        """Construct MIS instance from a VCGraph representation
        """
        g = nx.Graph()
        g.add_nodes_from(vcg.iter_variables())
        cliques = []
        for C in vcg.iter_constraints():
            c_vars = vcg.constraint_variables(C)
            cliques.append(c_vars)
            for e in itertools.product(c_vars, c_vars):
                if e[0]!=e[1]:
                    g.add_edge(*e)
        return MIS(g, cliques=cliques)

def mis_equal(pb1, pb2):
    """Check if two MIS problems are defined on the same graph
    """
    if pb1.nv() != pb2.nv():
        return False
    if pb1.ne() != pb2.ne():
        return False
    for e in pb1.g.edges:
        if not pb2.g.has_edge(*e):
            return False
    return True
    

def plot_maxcut_solution(pb, x, ax=None, pos=None, node_color=None, cmap=None):
    if pos is None:
        pos = nx.drawing.spring_layout(pb.g)
    if node_color is None:
        node_color = []
        for i in range(pb.nv()):
            if x[i]==0:
                node_color.append("#1f78b4")
            else:
                node_color.append("red")
    nx.draw_networkx(pb.g, pos, node_color=node_color, ax=ax, cmap=cmap)

def mis_greedy(pb, scores):
    """Greedy solver for MIS problems
    """
    queue = list(range(pb.nv()))
    sol = np.zeros(pb.nv())
    c = 0
    while len(queue)>0:
        u = queue[np.argmax(scores[np.array(queue)])]
        sol[u] = 1
        queue.remove(u)
        for v in pb.g.neighbors(u):
            if v in queue:
                sol[v] = 0
                queue.remove(v)
        c += 1
    return sol

def solution_feasible(pb, sol):
    """Check if solution `sol` is a valid independent set for `pb`
    """
    for u in pb.g.nodes:
        for v in pb.g.neighbors(u):
            if sol[u]==1 and sol[v]==1:
                return False
    return True

def cplex_model(pb, continuous_relaxation=False, cliques=None):
    model = Model()
    # add the variables
    if continuous_relaxation:
        x = np.array([model.continuous_var(lb=0.0, ub=1.0, name=f"x{i}") for i in range(pb.nv())])
    else:
        x = np.array([model.binary_var(f"x{i}") for i in range(pb.nv())])

    # add the constraints
    if cliques is None:
        cliques = pb.C
        #for e in ordered_edges(pb.g):
        #    model.add_constraint( x[e[0]] + x[e[1]] <= 1 )
    #else:
    for clq in cliques:
        model.add_constraint(sum([x[u] for u in clq]) <= 1)
    # add objective
    model.maximize( np.dot( x, pb.w) )
    return model

def get_solution_pool(pb, absgap=None, relgap=None, limit=None, timelimit=None, threads=None):
    model = cplex_model(pb)
    cpx = model.get_cplex()
    cpx.parameters.mip.pool.intensity.set(4)
    if not absgap is None:
        cpx.parameters.mip.pool.absgap.set(absgap)
    if not relgap is None:
        cpx.parameters.mip.pool.relgap.set(relgap)
    if not limit is None:
        cpx.parameters.mip.limits.populate.set(limit)
    if not timelimit is None:
        cpx.parameters.timelimit.set(timelimit)
    if not threads is None:
        cpx.parameters.threads.set(threads)

    cpx.populate_solution_pool()

    n_sol = cpx.solution.pool.get_num()
    solutions = []
    for i in range(n_sol):
        solutions.append(np.array(cpx.solution.pool.get_values(i)))
    return np.array(solutions)

if __name__=="__main__":
    from networkx.generators.trees import random_tree

    g = random_tree(5)
    pb = MIS(g)
    sol = pb.solve()
    print("Solution: ", sol.x)

    # plot solution
    plot_maxcut_solution(pb, sol.x)
    plt.show()

    # solving with CPLEX
    print("Solving CPLEX Model")
    model = cplex_model(pb)
    cplex_sol = model.solve()
    print(cplex_sol)

    # getting solution pool
    pool = get_solution_pool(pb)
    for i in range(pool.shape[0]):
        print(pool[i,:], np.dot(pool[i,:], pb.w))


