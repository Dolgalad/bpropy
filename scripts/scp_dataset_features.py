import os
import sys
sys.path.insert(0,"..")
import torch
import argparse
import networkx as nx
#from progress.bar import Bar
from multiprocessing import Pool

from bpropy.co.scp import SCP, adjacency_matrix_to_sets
from bpropy.datasets.scp_data import scp_to_data

import numpy as np
np.random.seed(2025)
import random
random.seed(2025)


parser = argparse.ArgumentParser("scp_dataset_features.py")
parser.add_argument("dataset_path", help="Dataset path")
parser.add_argument("output_path", help="Path in which to save the features")
parser.add_argument("--task_id", type=int, default=0)
parser.add_argument("--num_tasks", type=int, default=0)

def load_scp(path):
    adj_m = np.load(path)
    n = adj_m.shape[0]
    F = adjacency_matrix_to_sets(adj_m)
    return SCP(list(range(n)), F)

def load_scp_solution(path):
    return np.load(path)

def create_features(instance_file, solution_file, output,  save_to_disk=True):
    print(f"Instance {instance_file}, {solution_file}, output = {output}", flush=True)
     # load problem
    pb = load_scp(instance_file)
    y = load_scp_solution(solution_file)
    # solve continuous relaxation
    data = scp_to_data(pb, with_lbp_edges=True, y=y)
    data.y = torch.from_numpy(y)
    if save_to_disk:
        return torch.save(data, output)
    else:
        return data

if __name__=="__main__":
    args = parser.parse_args()

    instances = [int(f.split(".")[0]) for f in os.listdir(args.dataset_path) if not "_" in f]
    instances = sorted(instances)

    offset = 0
    max_N = len(instances)
    if args.task_id!=0 and args.num_tasks!=0:
        offset = (args.task_id-1) * (max_N // args.num_tasks)
        max_N = args.task_id * (max_N // args.num_tasks)

    # create output director
    os.makedirs(args.output_path, exist_ok=True)

    for instance_id in instances[offset:max_N]:
        instance_file = os.path.join(args.dataset_path, f"{instance_id}.npy")
        sf = f"{instance_id}_sol.npy"
        solution_file = os.path.join(args.dataset_path, sf)
        print("instance file : ", instance_file, os.path.exists(instance_file))
        print("solution file : ", solution_file, os.path.exists(solution_file))

        output = f"{instance_id}.dat"
        output_file = os.path.join(args.output_path, output)
        print("output file: ", output_file)

        create_features(instance_file, solution_file, output_file)
