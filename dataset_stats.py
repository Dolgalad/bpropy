import os
import numpy as np
import sys


def load_redumis_solution(path):
    with open(path, "r") as f:
        return np.array(list(map(int, filter(None, f.read().split("\n")))))
def load_pool(path):
    with open(path, "rb") as f:
        return np.load(f)
def load_solution_pool(path):
    # redumis solution
    y = load_redumis_solution(os.path.join(path, "redumis.sol"))
    # pool
    p = load_pool(os.path.join(path, "pool.npy"))
    return np.concatenate((np.expand_dims(y,axis=0),p), axis=0)

if __name__=="__main__":
    dataset_path = sys.argv[1]

    instance_c = 0
    total_samples_c = 0
    pool_sizes = []
    average_gap_values = []
    for f in os.listdir(dataset_path):
        # load solution pool
        pool = load_solution_pool(os.path.join(dataset_path, f))

        # solution values
        solution_values = pool.sum(axis=1)
        best_val = solution_values.max()

        instance_c += 1
        total_samples_c += pool.shape[0]
        pool_sizes.append(pool.shape[0])
        average_gap_values.append( np.mean( (best_val - solution_values) / best_val))

    print("Number of instances : ", instance_c)
    print("Total number of samples : ", total_samples_c)
    print("Average pool size   : ", np.mean(pool_sizes))
    print("Average average gap values : ", np.mean(average_gap_values))



