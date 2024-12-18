import os
import itertools
import argparse

parser = argparse.ArgumentParser(prog="gs_learning_mis_solution_nair.py",
                                 description="Grid search Learning MIS solution with Nair et. al. approach")
parser.add_argument("train_dataset_path")
parser.add_argument("validation_dataset_path")
parser.add_argument("dataset_category")
parser.add_argument("checkpoint_dir")
parser.add_argument("log_dir")
parser.add_argument("index", type=int)
args = parser.parse_args()

batch_sizes = [10, 100]
learning_rates = [1e-6, 1e-5, 1e-4]
hidden_dims = [64, 128]
optimizers = ["torch.optim.Adam", "torch.optim.RAdam"]
patience = [100, 500]

param_list = list(itertools.product(batch_sizes, learning_rates, hidden_dims, optimizers, patience))

def optimizer_str(opt, lr):
    if opt=="torch.optim.RAdam":
        return "-o "+opt+"-O '{\"lr\":"+str(lr)+", \"weight_decay\": 5e-4, \"decoupled_weight_decay\":true}'"
    if opt=="torch.optim.Adam":
        return "-o "+opt+"-O '{\"lr\":"+str(lr)+", \"weight_decay\": 5e-4}'"


def command_line(args, params):
    [bs,lr,h,opt,p] = params
    opt_str = optimizer_str(opt, lr)
    if opt == "torch.optim.RAdam":
        return f"python learning_mis_solution_nair.py {args.train_dataset_path} {args.validation_dataset_path} {args.dataset_category} {args.checkpoint_dir} {args.log_dir} -b {bs} -a {h} -p {p} {opt_str} -m mis_nair_gs{args.index}"
    if opt == "torch.optim.Adam":
        return f"python learning_mis_solution_nair.py {args.train_dataset_path} {args.validation_dataset_path} {args.dataset_category} {args.checkpoint_dir} {args.log_dir} -b {bs} -a {h} -p {p} {opt_str} -m mis_nair_gs{args.index}"

cmd = command_line(args, param_list[args.index])
print("CMD = ", cmd)
os.system(cmd)
