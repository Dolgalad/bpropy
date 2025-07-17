"""Generate train/validation/test datasets of Set Covering Problems
"""
import sys
sys.path.append("..")
import os

import time
from progress.bar import Bar

import numpy as np
from bpropy.co.scp import SCP, fixed_factor_scp
import argparse

parser = argparse.ArgumentParser(prog="scp_dataset_generator.py",
                                 description="SCP dataset generation")
parser.add_argument("output_path")
parser.add_argument("ncols", type=int)
parser.add_argument("nrows", type=int)
parser.add_argument("factor_size", type=int)
parser.add_argument("N", type=int)
parser.add_argument("--solve", action="store_true")

args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)

def time_f(func, *args, **kwargs):
    t0 = time.time()
    r = func(*args, **kwargs)
    return time.time() - t0, r

bar = Bar("Generating SCPs: ", max=args.N)
gaps = []
for i in range(args.N):
    # generate a random instance, regenerate instances if it contains empty sets
    t_gen = time.time()
    pb = fixed_factor_scp(args.nrows, args.ncols, args.factor_size)
    while any([len(f)==0 for f in pb.F]):
        pb = fixed_factor_scp(args.nrows, args.ncols, args.factor_size)
    t_gen = time.time() - t_gen
    # solve
    t_sol, sol = time_f(pb.solve)
    #print("Solve time: ", t_sol)
    #print("\t", sol)
    #print("\t", sol.x.sum())

    # solve with greedy
    t_grd, sol_grd = time_f(pb.greedy)
    #print("Greedy time: ", t_grd)
    #print("\t", sol_grd.sum())

    gap = 100 * (sol_grd.sum() - sol.x.sum()) / sol.x.sum()
    gaps.append(gap)

    # save instance
    instance_path = os.path.join(args.output_path, f"{i}.npy")
    np.save(instance_path, pb.adj_matrix())

    # save solution
    solution_path = os.path.join(args.output_path, f"{i}_sol.npy")
    np.save(solution_path, sol.x)
    bar.suffix = f" {i+1}/{args.N} {np.mean(gaps)}"
    bar.next()
bar.finish()

