import torch

from progress.bar import Bar
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.utils.tensorboard import SummaryWriter

from bpropy.lbp import *
from dopy.maxcut import *
from progress.bar import Bar

import matplotlib.pyplot as plt

import sys
import os
import pickle as pkl

# dataset path 
dataset_path = sys.argv[1]

# create the bipartite graph representing an instance
def bipartite_graph(pb):
    g = nx.Graph()
    features = []
    for i in range(pb.nv()):
        g.add_node(i, bipartite=0, x=[pb.w[i]])
        features.append(pb.w[i])
    ci = 0
    constraints = pb.constraints()
    for cj,c in enumerate(constraints):
        for i in range(c.A.shape[0]):
            if c.lb[i]==c.ub[i]:
                # equality constraint
                g.add_node(pb.nv() + ci, bipartite=1, x=[c.lb[i]])
                features.append(c.lb[i])
                for j in np.where(c.A[i,:] != 0)[0]:
                    g.add_edge(j, ci, e=[c.A[i,j]])
                ci += 1
            else:
                if c.lb[i] != -np.inf:
                    g.add_node(pb.nv() + ci, bipartite=1, x=[c.lb[i]])
                    features.append(c.lb[i])
                    for j in np.where(c.A[i,:] != 0)[0]:
                        g.add_edge(j, ci, e=[c.A[i,j]])
                    ci += 1
                if c.ub[i] != np.inf:
                    g.add_node(pb.nv() + ci, bipartite=1, x=[c.ub[i]])
                    features.append(c.ub[i])
                    for j in np.where(c.A[i,:] != 0)[0]:
                        g.add_edge(j, ci, e=[c.A[i,j]])
                    ci += 1
    return g,np.array(features)

def solution_pool_to_target(pb, pool):
    pool_values = np.array([np.dot(pb.w, s) for s in pool])
    Z = np.sum(np.exp(pool_values))
    target = np.stack([np.exp(np.dot(pb.w,s))*s for s in pool])
    target = np.sum(target, axis=0) /Z
    target[target<=0] = 1e-8
    return target

def instances_to_gnn_data(instances):
    graphs = []
    for pb,spool in zip(*instances):
        bg,feat = bipartite_graph(pb)
        gnn_data = from_networkx(bg)
        gnn_data.y = torch.from_numpy(np.pad(solution_pool_to_target(pb, spool), (0,pb.ne()))).to(torch.float)
        gnn_data.x = gnn_data.x.to(torch.float)
        gnn_data.e = gnn_data.e.to(torch.float)
        gnn_data.bipartite = gnn_data.bipartite.to(torch.bool)
    
        graphs.append(gnn_data)
    return graphs

def load_dataset(path):
    instances, solutions = [], []
    # 3 files per instances
    n = int(len(list(os.listdir(path))) / 3)
    for i in range(n):
        g = pkl.load(open(os.path.join(path, f"{i}_graph.pkl"), "rb"))
        w = pkl.load(open(os.path.join(path, f"{i}_weights.pkl"), "rb"))
        sol = pkl.load(open(os.path.join(path, f"{i}_sol.pkl"), "rb"))
        instances.append(MaxCut(g, weights=w))
        solutions.append(sol)
    return instances, solutions


train_instances = load_dataset(os.path.join(dataset_path,"train"))
train_split = 0.8
train_n = int(0.8 * len(train_instances[0]))

train_instances = (train_instances[0][:train_n], train_instances[1][:train_n])
val_instances = (train_instances[0][train_n:], train_instances[1][train_n:])


train_graphs = instances_to_gnn_data(train_instances)
val_graphs = instances_to_gnn_data(val_instances)

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=1, shuffle=False)


class GCN(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(1, hidden_dim, normalize=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv3 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv4 = GCNConv(hidden_dim, 1, normalize=False)
        self.sig = torch.nn.LogSigmoid()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)


        x = self.conv4(x, edge_index)

        return torch.sigmoid(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
max_epoch = 10000
writer = SummaryWriter()

train_loss_values  =[]
val_loss_values = []
val_gap_values = []
for epoch in range(max_epoch):
    # train
    model.train()
    epoch_train_loss = []
    epoch_val_loss = []
    epoch_val_gap = []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.binary_cross_entropy(torch.flatten(out[data.bipartite==0]), data.y[data.bipartite==0])
        epoch_train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        model.eval()
        for data,pb,pool in zip(val_loader,*val_instances):
            data = data.to(device)
            out = model(data)
            loss = F.binary_cross_entropy(torch.flatten(out[data.bipartite==0]), data.y[data.bipartite==0])
            if loss.item() < 0:
                print(loss.item())
                print(torch.flatten(out[data.bipartite==0]))
                print(data.y[data.bipartite==0])
                print(data.y[data.bipartite==0].log())
                print(torch.flatten(out[data.bipartite==0]).log())
                print(data.y[data.bipartite==0] * (data.y[data.bipartite==0].log() - torch.flatten(out[data.bipartite==0]).log()))
                exit()
            epoch_val_loss.append(loss.item())

            sol = maxcut_greedy(pb, out[data.bipartite==0].cpu().numpy())
            val = np.dot(pb.w, sol)
            opt_val = np.max([np.dot(pb.w, s) for s in pool])
            gap = (opt_val - val) / opt_val
            epoch_val_gap.append(gap)

    mtrainloss = np.mean(epoch_train_loss)
    mvalloss = np.mean(epoch_val_loss)
    mvalgap = np.mean(epoch_val_gap)
    train_loss_values.append(mtrainloss)
    val_loss_values.append(mvalloss)
    val_gap_values.append(mvalgap)

    print(f"epoch {epoch}, loss = {mtrainloss}, val = {mvalloss}, gap = {mvalgap}")
    writer.add_scalars("loss", {"train":mtrainloss, "val":mvalloss}, epoch)
    writer.add_scalar("gap/val", mvalgap, epoch)
    #writer.flush()
    #fig,ax = plt.subplots(2,1,sharex=True)
    #ax[0].plot(train_loss_values, label="train")
    #ax[0].plot(val_loss_values, label="val")
    #ax[0].set_ylabel("loss")
    #ax[0].set_yscale("log")
    #ax[0].legend()
    #ax[1].plot(val_gap_values)
    #ax[1].set_ylabel("Gap")
    #ax[1].set_xlabel("epoch")

    #plt.savefig("training_test1.png")
    #plt.close()
