import torch
import copy
import random

from progress.bar import Bar
import networkx as nx
from torch_geometric.utils.convert import from_networkx
#from torch_geometric.loader import DataLoader
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

device="cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Training device = ", device)

# dataset path 
dataset_path = sys.argv[1]
if dataset_path.endswith("/"):
    dataset_name = os.path.basename(dataset_path[:-1])
else:
    dataset_name = os.path.basename(dataset_path)

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
    bar = Bar("Instanc2GNNData: ", max=len(instances[0]))
    for pb,spool in zip(*instances):
        bg,feat = bipartite_graph(pb)
        pool_values = np.array([np.dot(pb.w, s) for s in spool])
        Z = np.sum(np.exp(pool_values))
        gnn_data = from_networkx(bg)
        gnn_data.x = gnn_data.x.to(torch.float)
        gnn_data.e = gnn_data.e.to(torch.float)
        gnn_data.bipartite = gnn_data.bipartite.to(torch.bool)

        # create the factor graph
        fg = create_factor_graph(pb.g)
        n_ss = sufficient_statistics_dim(fg)
        
        # create the computation edges
        f2v_idx, f2v_subsum_idx,ss_idx = f2v_index_map(fg)
        # creating message passing edges
        f2v_edges,r_edges = f2v_subsum_indexes(fg, f2v_idx, ss_idx)
        f2v_edges = torch.from_numpy(f2v_edges).to(device)
        r_edges = torch.from_numpy(r_edges).to(device)
        #f2v_edges2,r_edges2 = f2v_subsum_indexes2(fg)
        #f2v_edges2 = torch.from_numpy(f2v_edges2)
        #r_edges2 = torch.from_numpy(r_edges2)
        # scatter indices
        q_edges = torch.from_numpy(v2f_indexes(fg, f2v_idx)).to(device)
        #q_edges2 = torch.from_numpy(v2f_indexes2(fg))
        m_edges = torch.from_numpy(marginal_indexes(fg, f2v_idx, ss_idx)).to(device)
        #m_edges2 = torch.from_numpy(marginal_indexes2(fg))
        vm_edges = torch.from_numpy(variable_marginal_indexes(fg, f2v_idx, ss_idx)).to(device)

        # compute the mean target over entire solution pool
        target = np.stack([np.exp(np.dot(pb.w,s))*s for s in spool])
        target = np.sum(target, axis=0)/Z
        target = torch.from_numpy(target).to(torch.float)
        gnn_data.y = torch.from_numpy(np.pad(target, (0,pb.ne()))).to(torch.float)
        gnn_data.weight = torch.ones_like(gnn_data.y)
        gnn_data.f2v_edges = f2v_edges.clone()
        gnn_data.r_edges = r_edges.clone()
        gnn_data.q_edges = q_edges
        gnn_data.m_edges = m_edges
        graphs.append((gnn_data, copy.deepcopy(fg), copy.deepcopy(f2v_subsum_idx)))
        bar.next()
        continue


        for sol in spool:
            tgnn = gnn_data.clone()
            tgnn.y = torch.from_numpy(np.pad(sol, (0,pb.ne()))).to(torch.float)
            tgnn.weight = (np.exp(np.dot(pb.w, sol)) / Z) * torch.ones_like(tgnn.y)
            tgnn.f2v_edges = f2v_edges.clone()
            tgnn.r_edges = r_edges.clone()
            tgnn.q_edges = q_edges
            tgnn.m_edges = m_edges
            graphs.append((tgnn, copy.deepcopy(fg), copy.deepcopy(f2v_subsum_idx)))
        bar.next()
    bar.finish()
    return graphs

def load_dataset(path):
    instances, solutions = [], []
    # 3 files per instances
    n = int(len(list(os.listdir(path))) / 3)
    bar = Bar("Loading dataset", max=n)
    for i in range(n):
        g = pkl.load(open(os.path.join(path, f"{i}_graph.pkl"), "rb"))
        w = pkl.load(open(os.path.join(path, f"{i}_weights.pkl"), "rb"))
        sol = pkl.load(open(os.path.join(path, f"{i}_sol.pkl"), "rb"))
        instances.append(MaxCut(g, weights=w))
        solutions.append(sol)
        bar.next()
    bar.finish()
    return instances, solutions

print("Loading data")
train_instances = load_dataset(os.path.join(dataset_path,"train"))
train_split = 0.8
train_n = int(0.8 * len(train_instances[0]))

val_instances = (train_instances[0][train_n:], train_instances[1][train_n:])
train_instances = (train_instances[0][:train_n], train_instances[1][:train_n])

train_graphs = instances_to_gnn_data(train_instances)
val_graphs = instances_to_gnn_data(val_instances)

#train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
#val_loader = DataLoader(val_graphs, batch_size=1, shuffle=False)
print("done")


class GCN(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(1, hidden_dim, normalize=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv3 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv4 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.sig = torch.nn.LogSigmoid()

        # canonical parameter predictors
        self.unary_predictor = torch.nn.Linear(hidden_dim, 2)
        self.binary_predictor = torch.nn.Linear(hidden_dim, 4)


    def forward(self, data, fg, ss_idx):
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

        theta = torch.concatenate((
            torch.flatten(self.unary_predictor(x[data.bipartite==0])),
            torch.flatten(self.binary_predictor(x[data.bipartite==1]))))
        theta = torch.nn.functional.normalize(theta, p=1, dim=0)
        #print("theta = ", theta)
        v2f = torch.zeros_like(theta)
        mu = residual_loopy_belief_propagation(theta, v2f, fg, data.f2v_edges, ss_idx, data.r_edges, data.q_edges, data.m_edges, max_iter=10, tol=1e-6)
        n_var = torch.sum(data.bipartite==0).item()
        #print("mu small : ", torch.all(mu < 1e-8))
        #print(torch.all(mu[1:2*int(n_var):2] < 1e-8))
        #print(mu[1:2*int(n_var):2].min(), mu[1:2*int(n_var):2].max())



        return mu[1:2*int(n_var):2]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
max_epoch = 10000
writer = SummaryWriter(log_dir=f"logs_mrf_{dataset_name}")

train_loss_values  =[]
val_loss_values = []
val_gap_values = []
for epoch in range(max_epoch):
    # train
    model.train()
    epoch_train_loss = []
    epoch_val_loss = []
    epoch_val_gap = []
    epoch_pred_time = []
    random.shuffle(train_graphs)
    for data,fg,ss_idx in train_graphs:
        data = data.to(device)
        optimizer.zero_grad()
        t0 = time.time()
        out = model(data,fg,ss_idx)
        torch.cuda.current_stream().synchronize()
        t0 = time.time() - t0
        epoch_pred_time.append(t0)
        #no_red_loss = data.weight[data.bipartite==0] * F.binary_cross_entropy_with_logits(torch.flatten(out[data.bipartite==0]), data.y[data.bipartite==0], reduction="none")
        no_red_loss = data.weight[data.bipartite==0] * F.binary_cross_entropy_with_logits(torch.flatten(out), data.y[data.bipartite==0], reduction="none")


        loss = torch.mean(no_red_loss)
        epoch_train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        model.eval()
        for (data,fg,ss_idx),pb,pool in zip(val_graphs,*val_instances):
            data = data.to(device)
            out = model(data,fg,ss_idx)
            #no_red_loss = data.weight[data.bipartite==0] * F.binary_cross_entropy_with_logits(torch.flatten(out[data.bipartite==0]), data.y[data.bipartite==0], reduction="none")
            no_red_loss = data.weight[data.bipartite==0] * F.binary_cross_entropy_with_logits(torch.flatten(out), data.y[data.bipartite==0], reduction="none")


            loss = torch.mean(no_red_loss)
            #loss = data.weight * F.binary_cross_entropy(torch.flatten(out[data.bipartite==0]), data.y[data.bipartite==0])
            if loss.item() < 0:
                print(loss.item())
                print(torch.flatten(out[data.bipartite==0]))
                print(data.y[data.bipartite==0])
                print(data.y[data.bipartite==0].log())
                print(torch.flatten(out[data.bipartite==0]).log())
                print(data.y[data.bipartite==0] * (data.y[data.bipartite==0].log() - torch.flatten(out[data.bipartite==0]).log()))
                exit()
            epoch_val_loss.append(loss.item())

            sol = maxcut_greedy(pb, out.cpu().numpy())
            val = np.dot(pb.w, sol)
            opt_val = np.max([np.dot(pb.w, s) for s in pool])
            gap = (opt_val - val) / opt_val
            epoch_val_gap.append(gap)

    mtrainloss = np.mean(epoch_train_loss)
    mvalloss = np.mean(epoch_val_loss)
    mvalgap = np.mean(epoch_val_gap)
    mpredt = np.mean(epoch_pred_time)
    train_loss_values.append(mtrainloss)
    val_loss_values.append(mvalloss)
    val_gap_values.append(mvalgap)

    print(f"epoch {epoch}, loss = {mtrainloss}, val = {mvalloss}, gap = {mvalgap}, pred_t = {mpredt}")
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
