import os
import time
import networkx as nx
from dopy.mis import *

kamis_root = "/data/aschulz/Documents2023/KaMIS/deploy"

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


def kamis_solve(graph_path, solution_path, timelimit=1000, seed=0, executable="redumis"):
    cmd = f"{os.path.join(kamis_root,executable)} {graph_path} --output={solution_path} --seed={seed} --time_limit={timelimit} > /dev/null 2>&1"
    t0 = time.time()
    os.system(cmd)
    t0 = time.time() - t0
    return t0


if __name__=="__main__":
    g = nx.generators.random_graphs.erdos_renyi_graph(50, 0.15)

    pb = MIS(g)
    # solve with cplex
    sol = pb.solve()
    print(sol)

    # save metis graph
    write_metis_graph("g.metis", g)
    for executable in ["redumis", "online_mis", "weighted_branch_reduce"]:
        t = kamis_solve("g.metis", f"{executable}.sol", executable=executable)
        print(f"solve with {executable} in {t}s")

    print("getting solution pool")
    t = time.time()
    pool = get_solution_pool(pb, timelimit=5.)
    t = time.time() - t
    print(f"done in {t}s")
    print(pool.shape)
    
