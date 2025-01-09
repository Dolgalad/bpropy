import sys
import os
import argparse
import itertools
import json
import numpy as np
from progress.bar import Bar
import pandas as pd

import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.utils.tensorboard import SummaryWriter

from dopy.mis import *
from mis_er_dataset import MISERDataset
from models import *

parser = argparse.ArgumentParser(prog="test_trained_mis_nair.py",
                                 description="Test trained model following Nair et. al.")
parser.add_argument("dataset_path")
parser.add_argument("dataset_category")
parser.add_argument("model_log_dir", nargs="+")

args = parser.parse_args()

def get_neighbors(data, u):
    return data.edge_index[1, data.edge_index[0,:]==u].numpy()
def tgdata_to_networkx(data):
    nv = int(torch.sum(data.bipartite==0).item())
    nc = int(torch.sum(data.bipartite).item())
    g = nx.Graph()
    for i in range(nv):
        g.add_node(i)
    for c in range(nc):
        neighbors = get_neighbors(data, c+nv)
        for e in itertools.product(neighbors, neighbors):
            if e[0]!=e[1]:
                g.add_edge(e[0], e[1])
    return g

def test_model(dataset_path, dataset_category, model_log_dir, verbose=True):
    # load training run description
    with open(os.path.join(model_log_dir, "description.json"), "r") as f:
        description = json.load(f)
    
    # summary
    if verbose:
        print(json.dumps(description, sort_keys=True, indent=4))
    
    model_class = description["model"]["class"]
    model_desc = description["model"]
    del model_desc["class"]
    
    # checkpoint path
    checkpoint_path = os.path.join(model_log_dir, "checkpoints/checkpoint.pt")
    
    
    if verbose:
        print("Dataset path = ", dataset_path)
        print("Dataset category = ", dataset_category)
    
    dataset = MISERDataset(dataset_path, category=dataset_category, mode="test", unravel_pool=False, solution_limit=1, verbose=verbose)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # model 
    model = eval(model_class)(9, **model_desc)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    
    gaps = []
    bar = Bar("Testing: ", max=dataset.len())
    with torch.no_grad():
        for data in loader:
            out = model(data)
            pred = out[data.bipartite==0]
            g = tgdata_to_networkx(data)
            pb = MIS(g)
            data_sol = data.y.numpy()
            if data_sol.sum() == 0:
                continue
            # solve greedy
            sol = mis_greedy(pb, pred.numpy())
            gap = (data.y.numpy().sum() - sol.sum()) / data.y.numpy().sum()
    
            gaps.append(gap)
            bar.suffix = f" {np.mean(gaps)}"
            bar.next()
        bar.finish()
    return np.mean(gaps),description

df = pd.DataFrame(columns=["model", "test_gap"])
for model_log_dir in args.model_log_dir:
    print("Testing ", model_log_dir)
    mg,desc = test_model(args.dataset_path, args.dataset_category, model_log_dir, verbose=False)
    df.loc[len(df)] = [desc["model_name"], mg]

print(df)

