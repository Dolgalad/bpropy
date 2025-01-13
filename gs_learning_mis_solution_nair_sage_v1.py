import os
import itertools
import argparse
import time
import numpy as np


parser = argparse.ArgumentParser(prog="gs_learning_mis_solution_nair_sage_v1.py",
                                 description="Grid search Learning MIS solution with Nair et. al. approach, model SAGE_v1")
parser.add_argument("dataset_path")
parser.add_argument("log_dir")
parser.add_argument("index", type=int)
args = parser.parse_args()

seeds = [2022,2023,2024]
datasets = ["MIS100_EB"]
batch_sizes = [100]
learning_rates = [1e-2,1e-4,1e-6]
hidden_dims = [64, 128]
optimizers = ["torch.optim.RAdam"]
patience = [100]
n_layers = [8]
model_prefix = "mis_nair_sage_v1_gs"
model_class = "SAGE_v1"

param_list = list(itertools.product(seeds,datasets, batch_sizes, learning_rates, hidden_dims, optimizers, patience, n_layers))

def optimizer_str(opt, lr):
    if opt=="torch.optim.RAdam":
        return "-o "+opt+" -O '{\"lr\":"+str(lr)+", \"weight_decay\": 5e-4, \"decoupled_weight_decay\":true}'"
    if opt=="torch.optim.Adam":
        return "-o "+opt+" -O '{\"lr\":"+str(lr)+", \"weight_decay\": 5e-4}'"

def model_str(m, h, nl):
    return f"-M {m} -A "+"'{\"n_layers\":"+str(nl)+", \"hidden_dim\":"+str(h)+"}'"
def dataset_str():
    return "-D '{\"solution_limit\": 1, \"aggregate\": false, \"unravel_pool\": false, \"gap_limit\": 0.0}'"

def command_line(args, params):
    [seed,dataset,bs,lr,h,opt,p,nl] = params
    opt_str = optimizer_str(opt, lr)
    m_str = model_str(model_class, h, nl)
    data_str = dataset_str()
    train_dataset_path = args.dataset_path
    validation_dataset_path = args.dataset_path

    if opt == "torch.optim.RAdam":
        return f"python learning_mis_solution_nair.py {train_dataset_path} {dataset} {args.log_dir} -b {bs} -a {h} -p {p} {opt_str} -m {model_prefix}{args.index} -s {seed} {m_str} {data_str}"
    if opt == "torch.optim.Adam":
        return f"python learning_mis_solution_nair.py {train_dataset_path} {validation_dataset_path} {dataset} {args.checkpoint_dir} {args.log_dir} -b {bs} -a {h} -p {p} {opt_str} -m {model_prefix}{args.index} -s {seed}"

cmd = command_line(args, param_list[args.index - 1])
print("CMD = ", cmd)
print("Index = ", args.index)

if args.index > 1:
    # sleep random time
    rt = 60 + np.random.rand() * 300
    print(f"Sleep for {rt} seconds")
    time.sleep(rt)
    print("Waking up")

os.system(cmd)
