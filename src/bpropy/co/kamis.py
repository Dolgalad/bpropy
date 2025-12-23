import os
import time
import networkx as nx
import numpy as np

from bpropy.co.solution import Solution

def write_metis_graph(filename, g):
    # make graph undirected
    g = nx.to_undirected(g)

    n,m = g.number_of_nodes(), g.number_of_edges()

    with open(filename, "w") as f:
        # first line
        f.write(f"{n} {m}\n")
        for n in g.nodes:
            neighbors = list(g.neighbors(n))
            neighbors.sort()
            f.write(" ".join([str(i+1) for i in neighbors]))
            f.write("\n")

def kamis_solve(graph_path, solution_path, timelimit=1000, seed=0, executable="redumis", verbose=False):
    kamis_root = os.getenv("KAMIS_ROOT")
    if kamis_root is None:
        raise Exception("KAMIS_ROOT environment variable not set")
    cmd = f"{os.path.join(kamis_root,'deploy',executable)} {graph_path} --output={solution_path} --seed={seed} --time_limit={timelimit}"
    if not verbose:
        cmd += " > /dev/null 2>&1"
    t0 = time.time()
    os.system(cmd)
    t0 = time.time() - t0
    return t0

def load_kamis_solution(path):
    with open(path, "r") as f:
        return np.array(list(map(int, filter(None, f.read().split("\n")))))

def solution_from_kamis(path):
    x = load_kamis_solution(path)
    return Solution(x=x, solver="kamis")

