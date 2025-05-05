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
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar

from dopy.mis import *
from mis_er_mrf_dataset import tgdata_to_networkx
from models_nmf import *

import matplotlib.pyplot as plt

import sys
import os
import pickle as pkl

from early_stopper import EarlyStopper

parser = argparse.ArgumentParser(prog="learning_mis_solution_nmf.py",
                                 description="Learning MIS solution with Nair et. al. approach + Naive Mean Field")
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
parser.add_argument("-m", "--model_name_prefix", default="mis_mrf")
parser.add_argument("-o", "--optimizer", default="torch.optim.Adam")
parser.add_argument("-O", "--optimizer_args", default="{}")
parser.add_argument("-s", "--seed", default=0, type=int)
parser.add_argument("-g", "--gap_limit", default=-1, type=float)
parser.add_argument("-M", "--model_class", default="GCN_NMF_v4")
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

# dataset path 
#dataset_name = args.dataset_category+"_mrf_"+hashlib.md5(json.dumps(dataset_args, sort_keys=True).encode("utf8")).hexdigest()[:6]
#validation_dataset_name = args.dataset_category+"_mrf_"+hashlib.md5(json.dumps(validation_dataset_args, sort_keys=True).encode("utf8")).hexdigest()[:6]
#test_dataset_name = args.dataset_category+"_"+hashlib.md5(json.dumps(validation_dataset_args, sort_keys=True).encode("utf8")).hexdigest()[:6]
#train_dataset_path = os.path.join(args.dataset_path, dataset_name)
#validation_dataset_path = os.path.join(args.dataset_path, validation_dataset_name)
#test_dataset_path = os.path.join(args.dataset_path, test_dataset_name)

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
#train_dataset = MISERMRFDataset(train_dataset_path, mode="train", **dataset_args)
#val_dataset = MISERMRFDataset(validation_dataset_path, mode="validation", **validation_dataset_args)
#
#train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

train_graphs = pkl.load(open(args.train_dataset_path, "rb"))
val_graphs = pkl.load(open(args.val_dataset_path, "rb"))

train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=1, shuffle=False)
print("done")


print("Summary")
print(json.dumps(training_description, sort_keys=True, indent=4))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device ", device)
#model = GCN(9, args.hidden_dim).to(device)
model = eval(args.model_class)(10, factor_sizes=list(range(1,10)), **model_args).to(device)

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
    for data in train_loader:
        data = data.to(device)
        if cc == 0:
            optimizer.zero_grad()

        out,theta = model(data)
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
        for i,data in enumerate(val_loader):
            # retrieve graph
            g =  tgdata_to_networkx(data)
            pb = MIS(g)

            data = data.to(device)
            out,theta = model(data)

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
    test_graphs = pkl.load(open(args.test_dataset_path, "rb"))
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
    
    # model 
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    
    gaps = []
    bar = Bar("Testing: ", max=len(test_graphs))
    with torch.no_grad():
        for data in loader:
            out,theta = model(data.to(device))
            g = tgdata_to_networkx(data)
            pb = MIS(g)

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

