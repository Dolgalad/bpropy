import sys
sys.path.insert(0, "..")
import torch
import copy
import time
import argparse
import json
import hashlib
import random

from progress.bar import Bar
import networkx as nx
from torch_geometric.utils.convert import from_networkx
#from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader

from torch_geometric.data import Data

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar

import bpropy
import bpropy.co.mis
from bpropy.co.mis import *
from bpropy.mrf.importance_sampling import tgdata_to_networkx
from bpropy.models.mrf_models import *
from bpropy.datasets.mis_data import mis_to_data, data_to_mis

import matplotlib.pyplot as plt

import sys
import os
import pickle as pkl

from early_stopper import EarlyStopper

parser = argparse.ArgumentParser(prog="learning_mis_solution_tree_cover.py",
                                 description="Learning MIS solution with Nair et. al. approach + Naive Mean Field + Tree Cover")
parser.add_argument("train_dataset_path")
parser.add_argument("val_dataset_path")
parser.add_argument("test_dataset_path")
parser.add_argument("log_dir")
parser.add_argument("-b", "--batch_size", default=1, type=int)
parser.add_argument("-l", "--learning_rate", default=0.0001, type=float)
parser.add_argument("-a", "--hidden_dim", default=64, type=int)
parser.add_argument("-E", "--max_epochs", default=10000, type=int)
parser.add_argument("-p", "--es_patience", default=100, type=int)
parser.add_argument("-d", "--dropout_p", default=0.25, type=float)
parser.add_argument("-m", "--model_name_prefix", default="mis_nmf_tree_cover")
parser.add_argument("-o", "--optimizer", default="torch.optim.RAdam")
parser.add_argument("-O", "--optimizer_args", default="{\"lr\":0.0001, \"decoupled_weight_decay\": true, \"weight_decay\": 5e-4}")
parser.add_argument("-s", "--seed", default=0, type=int)
parser.add_argument("-g", "--gap_limit", default=-1, type=float)
parser.add_argument("-M", "--model_class", default="GCN_MRF_v4")
parser.add_argument("-A", "--model_args", default="{}")
parser.add_argument("-D", "--dataset_args", default="{}")
parser.add_argument("-V", "--validation_dataset_args", default='{"unravel_pool":false, "solution_limit":1, "aggregate":false, "gap_limit": 0.0}')
parser.add_argument("-j", "--description_file", default="", type=str)
parser.add_argument("--debug", action="store_true")
parser.add_argument("-c", "--bp_damping", default=0.5, type=float)
parser.add_argument("-e", "--bp_max_iter", default=10, type=int)

args = parser.parse_args()

if os.path.exists(args.description_file) and len(args.description_file)>0:
    with open(args.description_file) as f:
        description = json.load(f)
        optimizer_args = description["optimizer"]
        args.optimizer = description["optimizer"]["class"]
        del optimizer_args["class"]
        dataset_args = description["dataset"]
        del dataset_args["class"]
        if "validation_dataset_args" in description:
            validation_dataset_args = description["validation_dataset_args"]
        else:
            validation_dataset_args = json.loads(args.validation_dataset_args)
            #validation_dataset_args["category"] =  args.dataset_category
        args.seed = description["seed"]
        model_args = description["model"]
        args.model_class = model_args["class"]
        del model_args["class"]
else:
    # optimizer arguments
    print("Optimizer args : ", args.optimizer_args)
    optimizer_args = json.loads(args.optimizer_args)
    if not "lr" in optimizer_args:
        optimizer_args["lr"] = args.learning_rate
    
    # dataset arguments
    dataset_args = json.loads(args.dataset_args)
    #dataset_args["category"] =  args.dataset_category
    if not "gap_limit" in dataset_args:
        if args.gap_limit < 0:
            args.gap_limit = None
        dataset_args["gap_limit"] = args.gap_limit
    
    validation_dataset_args = json.loads(args.validation_dataset_args)
    #validation_dataset_args["category"] =  args.dataset_category
    
    # model arguments
    model_args = json.loads(args.model_args)
    if not "hidden_dim" in model_args:
        model_args["hidden_dim"] = args.hidden_dim
    if not "dropout_p" in model_args:
        model_args["dropout_p"] = args.dropout_p
    if not "bp_damping" in model_args:
        model_args["bp_damping"] = args.bp_damping
    if not "bp_max_iter" in model_args:
        model_args["bp_max_iter"] = args.bp_max_iter

# setting seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# training description data
training_description = {
        "seed": args.seed,
        "dataset": {
            "class": "MISERMRFDataset",
            **dataset_args
        },
        "validation_dataset": {
            "class": "MISERMRFDataset",
            **validation_dataset_args

        },
        "train_dataset": args.train_dataset_path,
        "validation_dataset": args.val_dataset_path,
        #"dataset_name": dataset_name,
        "batch_size": args.batch_size,
        "model": {
            "class": args.model_class,
            **model_args
        },
        "max_epoch": args.max_epochs,
        "early_stopping": {
            "es_patience": args.es_patience,
        },
        "optimizer": {
            "class": args.optimizer,
            **optimizer_args
        },
        }



# model name
def get_model_name(prefix, desc, _args):
    model_hash = hashlib.md5(json.dumps(desc, sort_keys=True).encode("utf8")).hexdigest()[:6]
    name = "_".join([prefix,
                    desc["model"]["class"],
                    #desc["dataset"]["category"],
                    model_hash
                    ])
    test_name = name
    c = 1
    while os.path.exists(os.path.join(_args.log_dir, test_name)) or os.path.exists(os.path.join(_args.log_dir, test_name)):
        test_name = f"{name}_{c}"
        c += 1
    return test_name

model_name = get_model_name(args.model_name_prefix, training_description, args)
training_description["model_name"] = model_name


# checkpoint save path
checkpoint_dir = os.path.join(args.log_dir, model_name, "checkpoints")
print("checkpoint_dir = ", checkpoint_dir)
os.makedirs(checkpoint_dir, mode=0o777)
training_description["checkpoint_dir"] = checkpoint_dir

# tensorboard log dir
log_dir = os.path.join(args.log_dir, model_name)
writer = SummaryWriter(log_dir=log_dir)
training_description["log_dir"] = log_dir

# early stopping
es = EarlyStopper(patience=args.es_patience, min_delta=1e-6)



print("Loading data")
def cliques_adjacency_matrix(pb):
    adj = np.zeros((len(pb.C), pb.nv()), dtype=bool)
    for i,c in enumerate(pb.C):
        adj[i,np.array(c, dtype=int)] = 1
    return adj
def create_simplified_factor_graph(data):
    pb = data_to_mis(data)
    sT = nx.algorithms.tree.mst.minimum_spanning_tree(pb.g)
    pb_sT = MIS(sT)

    data_sT = mis_to_data(pb_sT, None, with_nmf_edges=True)

    # mapping between new constraint nodes and old constraint nodes
    adj = cliques_adjacency_matrix(pb)
    adj_sT = cliques_adjacency_matrix(pb_sT)
    mapping = []
    for i in range(pb.nv()):
        mapping.append([i,i])
    for i_sT,c_sT in enumerate(pb_sT.C):
        a = np.logical_and(adj, adj_sT[i_sT,:]).sum(axis=1)
        for v in np.argwhere(a==adj_sT[i_sT,:].sum()).flatten():
            mapping.append([pb.nv()+v,pb.nv()+i_sT])

        #for i,c in enumerate(pb.C):
        #    if all([ e in c for e in c_sT]):
        #        #mapping.append([pb.nv()+i,pb.nv()+i_sT])
        #        print("old = ", [pb.nv()+i,pb.nv()+i_sT])
    mapping = torch.tensor(mapping)

    # remove the NMF edges from data
    data_ = Data(
                edge_index=data.edge_index,
                x=data.x,
                e=data.e,
                y=data.y,
                factor_size=data.factor_size,
                bipartite=data.bipartite
            )

    return data_, data_sT, mapping



train_graphs_processed_filename = "_".join([args.train_dataset_path, "processed_tree_cover.pkl"])
if not os.path.exists(train_graphs_processed_filename):
    train_graphs = pkl.load(open(args.train_dataset_path, "rb"))
    for i in range(len(train_graphs)):
        train_graphs[i] = create_simplified_factor_graph(train_graphs[i])
        print("simplification train: ", i, " / ", len(train_graphs))
    with open(train_graphs_processed_filename, "wb") as f:
        pkl.dump(train_graphs, f)
else:
    train_graphs = pkl.load(open(train_graphs_processed_filename,"rb"))

val_graphs_processed_filename = "_".join([args.val_dataset_path, "processed_tree_cover.pkl"])
if not os.path.exists(val_graphs_processed_filename):
    val_graphs = pkl.load(open(args.val_dataset_path, "rb"))
    for i in range(len(val_graphs)):
        val_graphs[i] = create_simplified_factor_graph(val_graphs[i])
        print("simplification val: ", i, " / ", len(val_graphs))
    with open(val_graphs_processed_filename, "wb") as f:
        pkl.dump(val_graphs, f)
else:
    val_graphs = pkl.load(open(val_graphs_processed_filename,"rb"))


train_loader = DataLoader(train_graphs, shuffle=True, batch_size=None, batch_sampler=None)
val_loader = DataLoader(val_graphs, shuffle=False, batch_size=None, batch_sampler=None)


print("done")


print("Summary")
print(json.dumps(training_description, sort_keys=True, indent=4))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device ", device)
#model = GCN(9, args.hidden_dim).to(device)
model = eval(args.model_class)(10, factor_sizes=list(range(1,10)), inference="naive_mean_field", **model_args).to(device)

optimizer = eval(args.optimizer)(model.parameters(), **optimizer_args)

training_description["start_time"] = time.time()
training_description["status"] = "running"
with open(os.path.join(log_dir, "description.json"), "w") as f:
    f.write(json.dumps(training_description, sort_keys=True, indent=4))


train_loss_values  =[]
val_loss_values = []
val_gap_values = []
best_val_gap = np.inf
print("Started training")
for epoch in range(args.max_epochs):
    # train
    model.train()
    epoch_train_loss = []
    epoch_val_loss = []
    epoch_val_gap = []
    bar = Bar(f"Epoch {epoch} training:", max = len(train_graphs) // args.batch_size)
    cc = 0 # simulate batching
    for data, fg_data, fg_map in train_loader:
        data = data.to(device)
        fg_data = fg_data.to(device)
        fg_data.nmf_edges = fg_data.nmf_edges.to(device)
        fg_map = torch.squeeze(fg_map,0).to(device)

        if cc == 0:
            optimizer.zero_grad()

        out,theta = model(data, fg_data=fg_data, fg_map=fg_map)
        if torch.any(torch.isnan(out)) or torch.any(torch.isinf(out)):
            print("Encountered bad value in output")
            training_description["status"] = "failed"
            break

        # loss unary factors
        nV = int((data.bipartite==0).sum().item())
        nC = int((data.bipartite==1).sum().item())
        factor_labels = data.y[:2 * nV].reshape(-1, 2).argmax(dim=1)

        loss = F.nll_loss((out.reshape(nV,2)+1e-8).log(), factor_labels) + 0.0001 * torch.linalg.norm(theta)

        if torch.isnan(loss) or torch.isinf(loss):
            print()
            print("Loss invalid")
            print(loss)
            exit()
        #print("Loss   : ", loss)

        #no_red_loss = F.binary_cross_entropy_with_logits(out, data.y, reduction="none")

        #loss = torch.mean(data.w * no_red_loss)
        epoch_train_loss.append(loss.item())
        loss.backward()
        cc += 1
        if cc == args.batch_size:
            optimizer.step()
            bar.suffix = f" {np.mean(epoch_train_loss)}"
            bar.next()
            cc = 0
    bar.finish()

    positive_scores = None
    negative_scores = None
    with torch.no_grad():
        bar = Bar(f"Epoch {epoch} validation:", max = len(val_graphs))
        model.eval()
        for i,(data,fg_data,fg_map) in enumerate(val_loader):
            # retrieve graph
            pb = data_to_mis(data)

            data = data.to(device)
            fg_data = fg_data.to(device)
            fg_data.nmf_edges = fg_data.nmf_edges.to(device)
            fg_map = torch.squeeze(fg_map,0).to(device)

            out,theta = model(data, fg_data=fg_data, fg_map=fg_map)

            if torch.any(torch.isnan(out)) or torch.any(torch.isinf(out)):
                print("Encountered bad value in output")
                training_description["status"] = "failed"
                break

            unary_marginals = out[:2*pb.nv()].reshape(pb.nv(), 2)
            node_scores = unary_marginals[:,1].cpu().numpy()

            # solve greedy
            sol = mis_greedy(pb, node_scores)

            # gap
            sol_tmp = data.y[:2*pb.nv()].reshape(pb.nv(), 2)
            opt_val = sol_tmp[:,1].sum().item()
            grd_val = sol.sum()
            gap = (opt_val - grd_val) / opt_val

            # loss unary factors
            factor_labels = data.y[:2 * nV].reshape(-1, 2).argmax(dim=1)
            loss = F.nll_loss((out.reshape(nV,2)+1e-8).log(), factor_labels) + 0.0001 * torch.linalg.norm(theta)

            #no_red_loss = F.binary_cross_entropy_with_logits(out, data.y, reduction="none")

            #loss = torch.mean(data.w * no_red_loss)

            epoch_val_loss.append(loss.item())
            epoch_val_gap.append(gap)
            
            labels = sol_tmp[:,1].cpu().numpy()
            if positive_scores is None:
                positive_scores = node_scores[labels == 1]
                negative_scores = node_scores[labels == 0]
            else:
                positive_scores = np.concatenate((positive_scores, node_scores[labels==1]))
                negative_scores = np.concatenate((negative_scores, node_scores[labels==0]))

            bar.next()
            if args.debug and i>10:
                break
        bar.finish()
    if training_description["status"] == "failed":
        break


    mtrainloss = np.mean(epoch_train_loss)
    mvalloss = np.mean(epoch_val_loss)
    mvalgap = np.mean(epoch_val_gap)
    print(f"epoch {epoch}, loss = {mtrainloss}, val_gap = {mvalgap}, val_loss = {mvalloss}")
    writer.add_scalars("loss", {"train": mtrainloss, "val": mvalloss}, epoch)
    writer.add_scalars("gap", {"val": mvalgap}, epoch)
    writer.add_scalars("scores", {"pos": np.mean(positive_scores), "neg": np.mean(negative_scores)}, epoch)
    for tag, value in model.named_parameters():
        writer.add_histogram(tag, value.cpu(), epoch)
        if value.grad is not None:
            writer.add_histogram(tag + "/grad", value.grad.cpu(), epoch)

    fig = plt.figure()
    plt.boxplot([negative_scores, positive_scores], labels=["0", "1"])
    writer.add_figure("scores", fig, global_step=epoch)
    plt.close()

    # save checkpoint if gap is better
    if mvalgap <= best_val_gap:
        best_val_gap = mvalgap
        checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint.pt")
        _model = copy.deepcopy(model).cpu()
        checkpoint = {
                "epoch": epoch,
                "model": _model.state_dict(),
                "optimizer": optimizer.state_dict()
                }
        torch.save(checkpoint, checkpoint_filename)

    # early stopping
    if es.early_stop(epoch, mvalloss):
        print("Stopping early")
        break

checkpoint_path = os.path.join(log_dir, "checkpoints", "checkpoint.pt")
if os.path.exists(os.path.join(log_dir, "checkpoints", "checkpoint.pt")):
    #test_dataset = MISERDataset(test_dataset_path, category=args.dataset_category, mode="test", unravel_pool=False, solution_limit=1, verbose=verbose)
    test_graphs_processed_filename = "_".join([args.test_dataset_path, "processed_tree_cover.pkl"])
    if not os.path.exists(test_graphs_processed_filename):
        test_graphs = pkl.load(open(args.test_dataset_path, "rb"))
        for i in range(len(test_graphs)):
            test_graphs[i] = create_simplified_factor_graph(test_graphs[i])
            print("simplification test: ", i, " / ", len(test_graphs))
        with open(test_graphs_processed_filename, "wb") as f:
            pkl.dump(test_graphs, f)
    else:
        test_graphs = pkl.load(open(test_graphs_processed_filename,"rb"))

    test_loader = DataLoader(test_graphs, shuffle=False, batch_size=None, batch_sampler=None)

    # model 
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    
    gaps = []
    bar = Bar("Testing: ", max=len(test_graphs))
    with torch.no_grad():
        for (data,fg_data,fg_map) in test_loader:
            pb = data_to_mis(data)

            data = data.to(device)
            fg_data = fg_data.to(device)
            fg_data.nmf_edges = fg_data.nmf_edges.to(device)
            fg_map = torch.squeeze(fg_map,0).to(device)

            out,theta = model(data, fg_data=fg_data, fg_map=fg_map)

            unary_marginals = out[:2*pb.nv()].reshape(pb.nv(), 2)
            node_scores = unary_marginals[:,1].cpu().numpy()

            # solve greedy
            sol = mis_greedy(pb, node_scores)

            # gap
            sol_tmp = data.y[:2*pb.nv()].reshape(pb.nv(), 2)
            opt_val = sol_tmp[:,1].sum().item()
            if opt_val == 0:
                continue
            grd_val = sol.sum()
            gap = (opt_val - grd_val) / opt_val

            gaps.append(gap)
            bar.suffix = f" {np.mean(gaps)}"
            bar.next()
        bar.finish()
    training_description["test_gap"] = np.mean(gaps)


training_description["stop_time"] = time.time()
training_description["min_validation_gap"] = es.min_validation_loss
training_description["min_validation_gap_epoch"] = es.min_validation_loss_epoch
training_description["epochs"] = epoch
training_description["status"] = "success"
with open(os.path.join(log_dir, "description.json"), "w") as f:
    f.write(json.dumps(training_description, sort_keys=True, indent=4))

