import os
import sys
sys.path.insert(0,"..")
import torch
import argparse
import networkx as nx
from progress.bar import Bar
from multiprocessing import Pool

from bpropy.co.mis import MIS
from bpropy.co.mis_solution import load_kamis_solution
from bpropy.datasets.mis_data import mis_to_data

import tqdm

parser = argparse.ArgumentParser("mis_dataset_features.sh")
parser.add_argument("dataset_path", help="Dataset path")
parser.add_argument("output_path", help="Path in which to save the features")
parser.add_argument("category", help="MIS feature category (EB,CC,CF,EC,CP)")
parser.add_argument("-n", "--num_process", default=1, help="Number of processes")
parser.add_argument("-o", "--offset", default=0, help="Instance number offset")
parser.add_argument("-N", "--number", default=-1, help="Number of instances")

def load_mis(path):
    g = nx.read_adjlist(os.path.join(path, "graph.adjlist"), nodetype=int)
    return MIS(g)

def create_features(args):
    instance_path, output, category = args
    print(f"Instance {instance_path}, output = {output}, category = {category}")
     # load problem
    pb = load_mis(instance_path)
    y = load_kamis_solution(os.path.join(instance_path, "redumis.sol"))
    # solve continuous relaxation
    data = mis_to_data(pb, category, with_lbp_edges=True)
    data.y = torch.from_numpy(y)
    torch.save(data, output)

if __name__=="__main__":
    args = parser.parse_args()

    instance_list = sorted(list(os.listdir(args.dataset_path)))
    if args.number == -1:
        args.number = len(instance_list) - args.offset
    instance_list = instance_list[args.offset:args.offset+args.number]

    arg_list = [(os.path.join(args.dataset_path, f), os.path.join(args.output_path, f"{f}.dat"), args.category) for f in instance_list]
    N = len(instance_list)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    pool = Pool(processes=args.num_process)
    for _ in tqdm.tqdm(pool.imap_unordered(create_features, arg_list), total=len(arg_list)):
        pass

    exit()
