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

import networkx as nx

def load_metis_graph(filename):
    G = nx.Graph()

    with open(filename, "r") as f:
        # Read first line
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError("Empty file")

        parts = first_line.split()
        print(parts)
        if len(parts) < 2:
            raise ValueError("First line must contain at least n and m")

        n = int(parts[0])
        # m = int(parts[1])  # not strictly needed

        # Add nodes (0-based internally)
        G.add_nodes_from(range(n))

        # Read adjacency lists
        for i in range(n):
            line = f.readline()
            if line is None:
                raise ValueError(f"Expected {n} lines of adjacency, got fewer")

            line = line.strip()
            if not line:
                continue  # node with no neighbors

            neighbors = list(map(int, line.split()))

            for j in neighbors:
                # Convert from 1-based to 0-based
                u = i
                v = j - 1

                if v < 0 or v >= n:
                    raise ValueError(f"Invalid neighbor index {j} at line {i+2}")

                # Add edge (networkx will ignore duplicates)
                G.add_edge(u, v)

    return G

import networkx as nx

def load_dimacs_graph(filename):
    G = nx.Graph()
    n = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("c"):
                continue  # comment

            parts = line.split()

            if parts[0] == "p":
                # format: p edge n m
                if len(parts) < 4:
                    raise ValueError("Invalid problem line")

                n = int(parts[2])
                G.add_nodes_from(range(n))  # 0-based

            elif parts[0] == "e":
                u = int(parts[1]) - 1  # convert to 0-based
                v = int(parts[2]) - 1
                G.add_edge(u, v)

            else:
                raise ValueError(f"Unknown line type: {parts[0]}")

    if n is None:
        raise ValueError("No problem line found")

    return G
