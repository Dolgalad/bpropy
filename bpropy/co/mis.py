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

import mip
from mip import xsum, maximize, BINARY

from bpropy.utils.graph_utils import ordered_edges
from bpropy.clique_cover import is_edge_cover, clique_cover
from bpropy.utils.set_sort import lenlex_sort

from bpropy.vc_graph import VCGraph
from bpropy.co.mis_solution import MISSolution

class MIS:
    def __init__(self, g, weights=None, cliques=None, num_graphs=1, graph_sizes=None):
        self.g = g
        if weights is None:
            self.w = np.ones(nx.number_of_nodes(g))
        else:
            self.w = weights
        if cliques is None:
            self.C = np.array([np.array(e) for e in ordered_edges(self.g)], dtype=object)
        else:
            self.C = np.array([np.array(c) for c in lenlex_sort(cliques)], dtype=object)
        if not is_edge_cover(self.C, self.g):
            raise Exception("MIS: cliques do not cover all edges of graph")
        # batch information
        self.num_graphs = num_graphs
        if graph_sizes is None:
            self.graph_sizes = np.array([g.number_of_nodes()])
        else:
            self.graph_sizes = graph_sizes
        self.clique_indicator = self.get_clique_indicators()
    def set_cliques(self, cliques, sort=False):
        if sort:
            self.C = np.array([np.array(c) for c in lenlex_sort(cliques)], dtype=object)
        else:
            self.C = np.array([np.array(c) for c in cliques], dtype=object)
        self.clique_indicator = self.get_clique_indicators()


    def get_clique_indicators(self):
        var_offsets = np.roll(self.graph_sizes.cumsum(), 1)
        var_offsets[0] = 0
        var_offsets = np.flip(var_offsets)
        ind = np.zeros(len(self.C), dtype=int)
        for i,c in enumerate(self.C):
            ind[i] = self.num_graphs - 1 - np.argmax(var_offsets <= c.min())
        return ind




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
    def solve(self, disp=False, cliques=None, solver="cbc", relax_integrality=False):
        """Solve MIS problem
        """
        if cliques is None:
            cliques = self.C
        if solver=="cplex":
            if docplex_spec is None:
                raise Exception("CPLEX solver not found. Missing package 'docplex'")
            md = cplex_model(self, continuous_relaxation=relax_integrality, cliques=cliques)
            sol = md.solve()
            return MISSolution.from_cplex(sol, md)
        if solver=="scipy":
            if relax_integrality:
                integrality = np.zeros(self.nv())
            else:
                integrality = np.ones(self.nv())
            return MISSolution.from_scipy_milp(milp(
                    -self.w,
                    integrality=integrality,
                    constraints=self.constraints(cliques=cliques),
                    bounds=Bounds(np.zeros(self.nv()), np.ones(self.nv())),
                    options={"disp":disp}
                    ))
        # default is using CBC
        md = mip.Model("mis")
        x = [md.add_var(lb=0.0, ub=1.0, var_type=BINARY) for i in range(self.nv())]
        md.objective = mip.maximize(xsum(x))
        for c in cliques:
            md += mip.xsum(x[u] for u in c) <= 1
        md.verbose = 0
        status = md.optimize(relax=relax_integrality)
        return MISSolution.from_python_mip(x, md)
    def greedy(self, scores=None):
        """Solve the problem geedily
        """
        if scores is None:
            scores = np.array([-self.g.degree[n] for n in range(self.nv())])
        return mis_greedy(self, scores)
    def rgreedy(self, scores):
        """Solve the problem geedily
        """
        return mis_random_greedy(self, scores)

    def check_solution(self, x):
        """Check if solution is feasible
        """
        return solution_feasible(self, x)
    def get_problem(self, idx):
        """Get `idx`-th problem
        """
        if idx>=self.num_graphs:
            raise Exception(f"Subproblem {idx} greater than number of graphs {self.num_graphs}")
        offset = np.roll(self.graph_sizes.cumsum(),1)
        nV = self.graph_sizes[idx]
        offset[0] = 0
        g = nx.subgraph(self.g, [n for n in range(offset[idx], offset[idx]+nV)]).copy()
        nmap = {n+offset[idx]:n for n in range(nV)}
        g = nx.relabel_nodes(g, nmap)
        #C = []
        #for c in self.C:
        #    if np.all([u>=offset[idx] and u<offset[idx]+nV for u in c]):
        #        C.append([u-offset[idx] for u in c])
        C = self.C[self.clique_indicator == idx] - offset[idx]
        return MIS(g, cliques=C)
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
    @staticmethod
    def batch(pbs):
        return mis_batch(pbs)

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
    # check cliques
    return np.all([np.all(c1==c2) for c1,c2 in zip(pb1.C,pb2.C)])
    

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
def np_softmax(x):
    return np.exp(x-np.max(x)) / np.exp(x - np.max(x)).sum()
def mis_random_greedy(pb, scores):
    """Greedy solver for MIS problems
    """
    queue = list(range(pb.nv()))
    sol = np.zeros(pb.nv())
    c = 0
    while len(queue)>0:
        u = np.random.choice(queue, p=np_softmax(scores[np.array(queue)]))
        #u = queue[np.argmax(scores[np.array(queue)])]
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

def offset_clique(c, o):
    return [u+o for u in c]
def mis_batch(pbs):
    g = nx.Graph()
    C = []
    graph_sizes = []
    offset = 0
    for i,pb in enumerate(pbs):
        graph_sizes.append(pb.g.number_of_nodes())
        C += [offset_clique(c,offset) for c in pb.C]
        g.add_nodes_from([offset+n for n in pb.g.nodes])
        g.add_edges_from([[e[0]+offset, e[1]+offset] for e in pb.g.edges])
        offset += pb.g.number_of_nodes()
    pb = MIS(g, cliques=C, num_graphs=len(pbs), graph_sizes=np.array(graph_sizes))
    #pb.C = C
    return pb

def random_mis(n=100, p=0.15, cliques=False):
    g = nx.generators.random_graphs.erdos_renyi_graph(n, p)
    if cliques:
        cc = clique_cover(g)
        pb = MIS(g, cliques=cc)
    else:
        pb = MIS(g)
    while pb.g.number_of_edges() == 0:
        g = nx.generators.random_graphs.erdos_renyi_graph(n, p)
        if cliques:
            cc = clique_cover(g)
            pb = MIS(g, cliques=cc)
        else:
            pb = MIS(g)
    return pb

