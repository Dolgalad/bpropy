import torch
import copy

from progress.bar import Bar
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar

#from bpropy.lbp import *
from dopy.mis import *
from mis_er_dataset import MISERDataset

import matplotlib.pyplot as plt

import sys
import os
import pickle as pkl

batch_size = 2
lr = 0.0001

# dataset path 
dataset_path = sys.argv[1]
if dataset_path.endswith("/"):
    dataset_name = os.path.basename(dataset_path[:-1])
else:
    dataset_name = os.path.basename(dataset_path)

# checkpoint save path
checkpoint_dir = sys.argv[2]
if os.path.exists(checkpoint_dir):
    os.system(f"rm -rf {checkpoint_dir}")
os.makedirs(checkpoint_dir)

print("Loading data")
full_dataset = MISERDataset(os.path.join(dataset_path,"train"), os.path.join(dataset_path,"train"))
train_split = 0.8

val_dataset = full_dataset[train_split:]
train_dataset = full_dataset[:train_split]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print("done")


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
        x = F.dropout(x, training=self.training)

        x = x + self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)

        x = x + self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv4(x, edge_index)

        return x #torch.sigmoid(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(9).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
max_epoch = 10000
writer = SummaryWriter(log_dir=f"logs_bs{batch_size}_{dataset_name}")

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
        no_red_loss = F.binary_cross_entropy_with_logits(torch.flatten(out[data.bipartite==0]), data.y, reduction="none")

        loss = torch.mean(data.w * no_red_loss)
        epoch_train_loss.append(torch.mean(no_red_loss).item())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        model.eval()
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            no_red_loss = F.binary_cross_entropy_with_logits(torch.flatten(out[data.bipartite==0]), data.y, reduction="none")

            loss = torch.mean(data.w * no_red_loss)
            epoch_val_loss.append(torch.mean(no_red_loss).item())


    mtrainloss = np.mean(epoch_train_loss)
    mvalloss = np.mean(epoch_val_loss)
    print(f"epoch {epoch}, loss = {mtrainloss}, val = {mvalloss}")
    writer.add_scalars("loss", {"train":mtrainloss, "val":mvalloss}, epoch)

    # save checkpoint
    checkpoint_filename = os.path.join(checkpoint_dir, f"{epoch}.pt")
    _model = copy.deepcopy(model).cpu()
    torch.save(_model.state_dict(), checkpoint_filename)
