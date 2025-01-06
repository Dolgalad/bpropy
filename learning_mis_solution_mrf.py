import torch
import copy
import time
import argparse
import json

from progress.bar import Bar
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar

from bpropy.lbp import _loopy_belief_propagation
from dopy.mis import *
from mis_er_mrf_dataset import MISERMRFDataset, tgdata_to_networkx

import matplotlib.pyplot as plt

import sys
import os
import pickle as pkl

from early_stopper import EarlyStopper

parser = argparse.ArgumentParser(prog="learning_mis_solution_nair.py",
                                 description="Learning MIS solution with Nair et. al. approach and MRFs")
parser.add_argument("train_dataset_path")
parser.add_argument("validation_dataset_path")
parser.add_argument("dataset_category")
parser.add_argument("checkpoint_dir")
parser.add_argument("log_dir")
parser.add_argument("-b", "--batch_size", default=1, type=int)
parser.add_argument("-l", "--learning_rate", default=0.0001, type=float)
parser.add_argument("-a", "--hidden_dim", default=64, type=int)
parser.add_argument("-E", "--max_epochs", default=10000, type=int)
parser.add_argument("-p", "--es_patience", default=10, type=int)
parser.add_argument("-d", "--dropout_p", default=0.25, type=float)
parser.add_argument("-m", "--model_name_prefix", default="mis_mrf")
parser.add_argument("-o", "--optimizer", default="torch.optim.Adam")
parser.add_argument("-O", "--optimizer_args", default="{}")
parser.add_argument("-s", "--seed", default=None)
parser.add_argument("-g", "--gap_limit", default=-1)
parser.add_argument("-i", "--bp_max_iter", default=10, type=int)

args = parser.parse_args()
if float(args.gap_limit) < 0:
    gap_limit = None
else:
    gap_limit = float(args.gap_limit)

# setting seed

# dataset path 
if args.train_dataset_path.endswith("/"):
    dataset_name = os.path.basename(args.train_dataset_path[:-1])
else:
    dataset_name = os.path.basename(args.train_dataset_path)

# model name
def get_model_name(_dataset_name, _args):
    lr = _args.learning_rate
    optimizer_args = json.loads(args.optimizer_args)
    if "lr" in optimizer_args:
        lr = optimizer_args["lr"]
    name = "_".join([_args.model_name_prefix,
                    args.dataset_category,
                    f"b{_args.batch_size}",
                    f"l{lr}",
                    f"a{_args.hidden_dim}",
                    f"E{_args.max_epochs}",
                    f"p{_args.es_patience}",
                    f"d{_args.dropout_p}"
                    ])
    test_name = name
    c = 1
    while os.path.exists(os.path.join(_args.checkpoint_dir, test_name)) or os.path.exists(os.path.join(_args.log_dir, test_name)):
        test_name = f"{name}_{c}"
        c += 1
    return test_name
model_name = get_model_name(dataset_name, args)

# checkpoint save path
checkpoint_dir = os.path.join(args.checkpoint_dir, model_name)
os.makedirs(checkpoint_dir)

# tensorboard log dir
log_dir = os.path.join(args.log_dir, model_name)

writer = SummaryWriter(log_dir=log_dir)

# early stopping
es = EarlyStopper(patience=args.es_patience, min_delta=1e-6)

# optimizer arguments
print("Optimizer args : ", args.optimizer_args)
optimizer_args = json.loads(args.optimizer_args)
if not "lr" in optimizer_args:
    optimizer_args["lr"] = args.learning_rate

print("Loading data")
train_dataset = MISERMRFDataset(args.train_dataset_path, category=args.dataset_category, mode="train", gap_limit=gap_limit)
val_dataset = MISERMRFDataset(args.validation_dataset_path, category=args.dataset_category, mode="validation", unravel_pool=False, gap_limit=gap_limit)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
print("done")

# training description data
training_description = {
        "dataset_category": args.dataset_category,
        "dataset_gap_limit": args.gap_limit,
        "train_dataset": args.train_dataset_path,
        "validation_dataset": args.validation_dataset_path,
        "dataset_name": dataset_name,
        "checkpoint_dir": checkpoint_dir,
        "log_dir": log_dir,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "max_epoch": args.max_epochs,
        "early_stopping": {
            "es_patience": args.es_patience,
        },
        "optimizer": {
            "class": args.optimizer,
            **optimizer_args
        },
        "dropout_p": args.dropout_p,
        "model_name": model_name,
        "bp_max_iter": args.bp_max_iter
        }

print("Summary")
print(json.dumps(training_description, sort_keys=True, indent=4))

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(9, hidden_dim, normalize=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv3 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv4 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.sig = torch.nn.LogSigmoid()

        # canonical parameter predictors
        self.unary_predictor = torch.nn.Linear(hidden_dim, 2)
        self.binary_predictor = torch.nn.Linear(hidden_dim, 4)

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

        theta = torch.concatenate((
            torch.flatten(self.unary_predictor(x[data.bipartite==0])),
            torch.flatten(self.binary_predictor(x[data.bipartite==1]))))
        theta = torch.nn.functional.normalize(theta, p=1, dim=0)
        print("theta.device ", theta.device)
        v2f = torch.zeros_like(theta, device=device)
        print("v2f.device ", v2f.device)
        mu = _loopy_belief_propagation(data, theta, v2f, data.f2v_edges, data.r_edges, data.q_edges, data.m_edges, max_iter=args.bp_max_iter, tol=1e-6)
        print("mu.device ", mu.device)
        return mu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(9, args.hidden_dim).to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
optimizer = eval(args.optimizer)(model.parameters(), **optimizer_args)

training_description["start_time"] = time.time()
with open(os.path.join(log_dir, "description.json"), "w") as f:
    f.write(json.dumps(training_description, sort_keys=True, indent=4))

train_loss_values  =[]
val_loss_values = []
val_gap_values = []
for epoch in range(args.max_epochs):
    # train
    model.train()
    epoch_train_loss = []
    epoch_val_loss = []
    epoch_val_gap = []
    bar = Bar(f"Epoch {epoch} training:", max = train_dataset.len() // args.batch_size)
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        no_red_loss = F.binary_cross_entropy_with_logits(torch.flatten(out[data.bipartite==0]), data.y, reduction="none")

        loss = torch.mean(data.w * no_red_loss)
        epoch_train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        bar.next()
    bar.finish()

    with torch.no_grad():
        bar = Bar(f"Epoch {epoch} validation:", max = val_dataset.len() // args.batch_size)
        model.eval()
        for data in val_loader:
            # retrieve graph
            g =  tgdata_to_networkx(data)
            pb = MIS(g)

            data = data.to(device)
            out = model(data)[data.bipartite==0].cpu().numpy()
            unary_marginals = out[:2*pb.nv()].reshape(pb.nv(), 2)

            node_scores = unary_marginals[:,1]
            #node_scores = torch.flatten(out[data.bipartite==0]).cpu().numpy()

            # solve greedy
            sol = mis_greedy(pb, node_scores)

            # gap
            opt_val = data.y.sum().item()
            grd_val = sol.sum()
            gap = (opt_val - grd_val) / opt_val

            #no_red_loss = F.binary_cross_entropy_with_logits(torch.flatten(out[data.bipartite==0]), data.y, reduction="none")

            #loss = torch.mean(data.w * no_red_loss)
            #epoch_val_loss.append(loss.item())
            epoch_val_loss.append(gap)
            bar.next()
        bar.finish()


    mtrainloss = np.mean(epoch_train_loss)
    mvalloss = np.mean(epoch_val_loss)
    print(f"epoch {epoch}, loss = {mtrainloss}, val_gap = {mvalloss}")
    writer.add_scalar("loss_train", mtrainloss, epoch)
    writer.add_scalar("gap_val", mvalloss, epoch)

    # save checkpoint
    checkpoint_filename = os.path.join(checkpoint_dir, f"{epoch}.pt")
    _model = copy.deepcopy(model).cpu()
    torch.save(_model.state_dict(), checkpoint_filename)

    # early stopping
    if es.early_stop(epoch, mvalloss):
        print("Stopping early")
        break

training_description["stop_time"] = time.time()
training_description["min_validation_gap"] = es.min_validation_loss
training_description["min_validation_gap_epoch"] = es.min_validation_loss_epoch
training_description["epochs"] = epoch
with open(os.path.join(log_dir, "description.json"), "w") as f:
    f.write(json.dumps(training_description, sort_keys=True, indent=4))

