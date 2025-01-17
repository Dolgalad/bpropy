import os
import matplotlib.pyplot as plt

from mis_er_dataset import MISERDataset

root_dir = "/media/aschulz/1fe12cc4-1650-4d25-b2fb-00786a9d1e6f"

gap_limits = [0.0, 0.05, 0.1, 0.2, 0.5]
dataset_sizes = []

for gl in [0.0, 0.05, 0.1, 0.2, 0.5]:
    dataset = MISERDataset(os.path.join(root_dir, f"mis100_test_gl{gl}"), mode="test", gap_limit=gl)
    dataset_sizes.append(dataset.len())
    print(gl, dataset.len())

plt.figure()
plt.scatter(gap_limits, dataset_sizes)
plt.xlabel("Optimality gap limit")
plt.ylabel("Dataset size")
plt.show()

