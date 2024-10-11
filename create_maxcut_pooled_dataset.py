
import networkx as nx
import numpy as np
from progress.bar import Bar
import os
import sys
import pickle as pkl

from dopy.maxcut import *

# create dataset
def create_dataset(n, n_nodes, edge_p, weights="unit"):
    bar = Bar(f"Creating dataset (n_nodes={n_nodes}, edge_p={edge_p}: ", max=n)
    instances = []
    solutions = []
    for i in range(n):
        # Randomly generate a graph
        g = nx.generators.erdos_renyi_graph(n_nodes, edge_p)
        
        # Solve ILP
        if weights=="uniform":
            w = np.random.random_sample(n_nodes)
        if weights=="unit":
            w = np.ones(n_nodes)
        pb = MaxCut(g, weights=w)

        instances.append(pb)
        solutions.append(get_solution_pool(pb))
        bar.next()
    bar.finish()
    return instances, solutions

if __name__=="__main__":
    output_path = sys.argv[1]
    n_nodes = int(sys.argv[2])
    edge_p = float(sys.argv[3])
    n_train = int(sys.argv[4])
    n_test = int(sys.argv[5])
    weights = sys.argv[6]

    # train set
    train_instances, train_solutions = create_dataset(n_train, n_nodes, edge_p, weights)
    train_output_path = os.path.join(output_path, "train")

    os.makedirs(train_output_path, exist_ok=True)
    for i,(inst,sol) in enumerate(zip(train_instances, train_solutions)):
        pkl.dump(inst.g, open(os.path.join(train_output_path, f"{i}_graph.pkl"), "wb"))
        pkl.dump(inst.w, open(os.path.join(train_output_path, f"{i}_weights.pkl"), "wb"))
        pkl.dump(sol, open(os.path.join(train_output_path, f"{i}_sol.pkl"), "wb"))

    # test set
    test_instances, test_solutions = create_dataset(n_test, n_nodes, edge_p, weights)
    test_output_path = os.path.join(output_path, "test")

    os.makedirs(test_output_path, exist_ok=True)
    for i,(inst,sol) in enumerate(zip(test_instances, test_solutions)):
        pkl.dump(inst.g, open(os.path.join(test_output_path, f"{i}_graph.pkl"), "wb"))
        pkl.dump(inst.w, open(os.path.join(test_output_path, f"{i}_weights.pkl"), "wb"))
        pkl.dump(sol, open(os.path.join(test_output_path, f"{i}_sol.pkl"), "wb"))

