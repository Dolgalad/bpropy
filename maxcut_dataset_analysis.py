import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
import numpy as np
import pickle as pkl
import sys
import os
import networkx as nx

from dopy.maxcut import *

dataset_path = sys.argv[1]

def load_dataset(path):
    instances, solutions = [], []
    # 3 files per instances
    n = int(len(list(os.listdir(path))) / 3)
    for i in range(n):
        g = pkl.load(open(os.path.join(path, f"{i}_graph.pkl"), "rb"))
        w = pkl.load(open(os.path.join(path, f"{i}_weights.pkl"), "rb"))
        sol = pkl.load(open(os.path.join(path, f"{i}_sol.pkl"), "rb"))
        instances.append(MaxCut(g, weights=w))
        solutions.append(sol)
    return instances, solutions

dataset = load_dataset(dataset_path)

# distribution of costs, gaps
costs = []
gaps = []
probability_targets = []
props = []
weights = []
for pb,pool in zip(*dataset):
    values = np.array([np.dot(s,pb.w) for s in pool])
    opt_v = np.max(values)
    Z = np.sum(np.exp(values))
    target = np.stack([np.exp(np.dot(pb.w,s))*s for s in pool])
    target = np.sum(target, axis=0) / Z
    pro = np.stack([s for s in pool])
    pro = np.sum(pro, axis=0) / pool.shape[0]
    for v in values:
        costs.append(v)
        gaps.append(100 * (opt_v - v) / opt_v)
    for p in target:
        probability_targets.append(p)
    for w in pb.w:
        weights.append(w)
    for p in pro:
        props.append(p)

fig, ax = plt.subplots(2,1)
plt.suptitle(f"Dataset: {dataset_path}")
ax[0].hist(costs, bins=int(np.max(costs)))
ax[0].set_ylabel("#")
ax[0].set_xlabel("cost")
ax[1].hist(gaps, bins=int(np.max(costs)))
ax[1].set_ylabel("#")
ax[1].set_xlabel("gap (%)")
plt.tight_layout()

fig, ax = plt.subplots(1,1)
plt.suptitle(f"Dataset: {dataset_path}")
ax.hist(probability_targets, bins=int(np.max(costs)))
ax.set_ylabel("#")
ax.set_xlabel("p")
plt.tight_layout()

import pandas as pd

fig, ax = plt.subplots(1,1)
ax.set_ylabel("p[i]")
ax.set_xlabel("w[i]")
ax.plot(weights, weights, c="red")
ax.scatter(weights, probability_targets, marker="+", alpha=0.01)
df = pd.DataFrame({"x":weights, "y":probability_targets})
df.sort_values(by="x", inplace=True)
round_by = 2
bins = df.groupby(df.x.round(round_by)).mean()
ax.plot(bins.index, bins.y, "g--", lw=3)

fig, ax = plt.subplots(1,1)
ax.set_ylabel("proportion[i]")
ax.set_xlabel("w[i]")
ax.plot(weights, weights, c="red")
ax.scatter(weights, props, marker="+", alpha=0.01)
df = pd.DataFrame({"x":weights, "y":props})
df.sort_values(by="x", inplace=True)
round_by = 2
bins = df.groupby(df.x.round(round_by)).mean()
ax.plot(bins.index, bins.y, "g--", lw=3)


# plot network optimal solution and target probabilities
inst, pool = dataset[0][0], dataset[1][0]
pos = nx.drawing.layout.spring_layout(inst.g)
sol_node_colors = []
grd_node_colors = []
values = np.array([np.dot(s > 0.5,inst.w) for s in pool])
print("max val = ", np.max(values))
sol = pool[np.argmax(values)] > 0.5
print("sol = ", sol)
print("dot = ", np.dot(sol, inst.w))
Z = np.sum(np.exp(values))
target = np.stack([np.exp(np.dot(inst.w,s))*s for s in pool])
target = np.sum(target, axis=0) / Z
grd_sol = maxcut_greedy(inst, target)
for i in range(inst.nv()):
    if sol[i]:
        sol_node_colors.append("blue")
    else:
        sol_node_colors.append("red")

    if grd_sol[i]:
        grd_node_colors.append("blue")
    else:
        grd_node_colors.append("red")

fig, ax = plt.subplots(1,4)
opt_val = np.max(values)
ax[0].set_title(f"Optimal, val={opt_val}")
nx.drawing.draw_networkx(inst.g, pos, node_color=sol_node_colors, alpha=0.1, with_labels=False, node_size=50, ax=ax[0])
nx.drawing.draw_networkx_nodes(inst.g, pos, node_color=sol_node_colors,  node_size=50, ax=ax[0])

ax[1].set_title("Targets")
nx.drawing.draw_networkx(inst.g, pos, node_color=sol_node_colors, alpha=0.1, with_labels=False, node_size=50, ax=ax[1])
cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
nx.drawing.draw_networkx_nodes(inst.g, pos, node_color=target,  node_size=50, ax=ax[1], vmin=np.min(target), vmax=np.max(target), cmap=cm1)

grd_val = np.dot(grd_sol, inst.w)

print(sol, np.dot(sol, inst.w))
print(grd_sol, np.dot(grd_sol, inst.w))
ax[2].set_title(f"Greedy, val={grd_val}")
nx.drawing.draw_networkx(inst.g, pos, node_color=sol_node_colors, alpha=0.1, with_labels=False, node_size=50, ax=ax[2])
nx.drawing.draw_networkx_nodes(inst.g, pos, node_color=grd_node_colors,  node_size=50, ax=ax[2])

ax[3].set_title(f"Diff")
diff_nodes = np.where(grd_sol!=sol)[0]
diff_node_colors = []
for n in diff_nodes:
    if grd_sol[n]:
        diff_node_colors.append("red")
    else:
        diff_node_colors.append("blue")

diff_g = inst.g.subgraph(diff_nodes)
diff_pos = {n:pos[n] for n in diff_nodes}
nx.drawing.draw_networkx(diff_g, diff_pos, node_size=50, ax=ax[3], node_color=diff_node_colors, with_labels=False)
ax[3].set_xlim(ax[0].get_xlim())
ax[3].set_ylim(ax[0].get_ylim())


plt.show()

