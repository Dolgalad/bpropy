import sys
import os
import argparse
import itertools
import numpy as np
from progress.bar import Bar

import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.utils.tensorboard import SummaryWriter

from dopy.mis import *
from mis_er_dataset import MISERDataset


parser = argparse.ArgumentParser(prog="test_mis_solution_nair.py",
                                 description="Test trained model following Nair et. al.")
parser.add_argument("checkpoint_path")
parser.add_argument("dataset_path")
parser.add_argument("dataset_category")
parser.add_argument("-a", "--hidden_dim", default=64, type=int)
parser.add_argument("-d", "--dropout_p", default=0.25, type=float)

args = parser.parse_args()


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(9, hidden_dim, normalize=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv3 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv4 = GCNConv(hidden_dim, 1, normalize=False)
        self.sig = torch.nn.LogSigmoid()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=args.dropout_p, training=self.training)

        x = x + self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=args.dropout_p, training=self.training)

        x = x + self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=args.dropout_p, training=self.training)

        x = self.conv4(x, edge_index)

        return x #torch.sigmoid(x)

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


print("Dataset path = ", args.dataset_path)
print("Dataset category = ", args.dataset_category)

dataset = MISERDataset(args.dataset_path, category=args.dataset_category, mode="test", unravel_pool=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# model 
model = GCN(9, args.hidden_dim)
model.load_state_dict(torch.load(args.checkpoint_path, weights_only=True))

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

print("Mean : ", np.mean(gaps))

