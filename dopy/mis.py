"""
MIS solving routines
"""

import networkx as nx
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import matplotlib.pyplot as plt

import importlib.util
docplex_spec = importlib.util.find_spec("docplex")
if docplex_spec is not None:
    from docplex.mp.model import Model

class MIS:
    def __init__(self, g, weights=None):
        self.g = g
        if weights is None:
            self.w = np.ones(nx.number_of_nodes(g))
        else:
            self.w = weights
    def nv(self):
        return nx.number_of_nodes(self.g)
    def ne(self):
        return nx.number_of_edges(self.g)

    def constraints(self, obj_ub=None, cliques=None):
        c = []
        if cliques is None:
            for e in self.g.edges:
                row = np.zeros(self.nv())
                row[e[0]] = 1
                row[e[1]] = 1
                c.append(LinearConstraint(row, ub=1))
        else:
            for clq in cliques:
                row = np.zeros(self.nv())
                for u in clq:
                    row[u] = 1
                c.append(LinearConstraint(row, ub=1))

        if not obj_ub is None:
            print("Adding UB constraint ", obj_ub)
            c.append(LinearConstraint(-self.w, ub=obj_ub))
        return c
    def solve(self, obj_ub=None, disp=False):
        return milp(
                    -self.w,
                    integrality=np.ones(self.nv()),
                    constraints=self.constraints(obj_ub=obj_ub),
                    bounds=Bounds(np.zeros(self.nv()), np.ones(self.nv())),
                    options={"disp":disp}
                )

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

def mis_greedy(pb, node_scores):
    queue = list(range(pb.nv()))
    sol = np.zeros(pb.nv())
    c = 0
    while len(queue)>0:
        u = queue[np.argmax(node_scores[np.array(queue)])]
        print("---------")
        print(c, np.argmax(node_scores[np.array(queue)]), u, list(pb.g.neighbors(u)))
        print("\t", queue)
        print("\t", [len(list(pb.g.neighbors(n))) for n in queue])
        print("\t", node_scores[np.array(queue)])
        sol[u] = 1
        queue.remove(u)
        for v in pb.g.neighbors(u):
            if v in queue:
                sol[v] = 0
                queue.remove(v)
        c += 1
    return sol

#def maxcut_greedy(pb, marginals):
#    queue = list(range(pb.nv()))
#    sol = np.zeros(pb.nv())
#    while len(queue)>0:
#        if np.sum(sol)==0:
#            u = queue[np.argmax(node_scores[np.array(queue)])]
#        else:
#            # get neighbors of current solution
#            neighborhood = [v in queue if np.any(gT
#        sol[u] = 1
#        queue.remove(u)
#        for v in pb.g.neighbors(u):
#            if v in queue:
#                sol[v] = 0
#                queue.remove(v)
#    return sol


def maxcut_solution_validity(pb, sol):
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
        for e in pb.g.edges:
            model.add_constraint( x[e[0]] + x[e[1]] <= 1 )
    else:
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


