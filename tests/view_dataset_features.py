import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from progress.bar import Bar

dataset_path = sys.argv[1]

node_feature_labels = ["Objective coef", "Sol is at LB", "Sol is at UB", "Sol value", "In basis", "Non-basic at lower", "Non-basic at upper", "Free", "Reduced cost"]
constraint_feature_labels = ["Cosine similarity", "Slacks", "Is tight", "Duals"]

instance_ids = list(os.listdir(dataset_path))
bar = Bar("Loading features: ", max=len(instance_ids))

full_node_features = None
full_constraint_features = None
c = 0
for instance_id in instance_ids:
    node_feature_path = os.path.join(dataset_path, instance_id, "node_features.npy")
    edge_feature_path = os.path.join(dataset_path, instance_id, "edge_features.npy")
    constraint_feature_path = os.path.join(dataset_path, instance_id, "constraint_features.npy")
    # load features
    with open(node_feature_path,"rb") as f:
        nfeat = np.load(f)
    with open(constraint_feature_path,"rb") as f:
        cfeat = np.load(f)

    # node features
    if full_node_features is None:
        full_node_features = nfeat - np.mean(nfeat,axis=0)
        full_constraint_features = cfeat - np.mean(cfeat, axis=0)
    else:
        full_node_features = np.concatenate((full_node_features, nfeat - np.mean(nfeat,axis=0)), axis=0)
        full_constraint_features = np.concatenate((full_constraint_features, cfeat - np.mean(cfeat,axis=0)), axis=0)

    bar.next()
    c+= 1
    if c >= 1000:
        break
bar.finish()

fig, ax = plt.subplots(3,3,figsize=(10,10))
print("Node features")
for i in range(full_node_features.shape[1]):
    min_val, max_val = full_node_features[:,i].min(), full_node_features[:,i].max()
    print(node_feature_labels[i], min_val, max_val)
    margin = 0.2 * (max_val - min_val)
    if margin == 0:
        margin = 0.2
    ax[i//3, i%3].hist(full_node_features[:,i], bins=50)
    ax[i//3, i%3].set_xlabel(node_feature_labels[i])
    ax[i//3, i%3].set_xlim(min_val - margin, max_val + margin)
plt.suptitle("Node features (normalized)")
plt.tight_layout()
plt.savefig("view_nfeat.png")

fig, ax = plt.subplots(2,2,figsize=(10,10))
print("Constraint features")
for i in range(full_constraint_features.shape[1]):
    min_val, max_val = full_constraint_features[:,i].min(), full_constraint_features[:,i].max()
    margin = 0.2 * (max_val - min_val)
    if margin == 0:
        margin = 0.2
    print(constraint_feature_labels[i], min_val, max_val)

    ax[i//2, i%2].hist(full_constraint_features[:,i], bins=50)
    ax[i//2, i%2].set_xlabel(constraint_feature_labels[i])
    ax[i//2, i%2].set_xlim(min_val - margin, max_val + margin)
plt.suptitle("Constraint features (normalized)")
plt.tight_layout()
plt.savefig("view_cfeat.png")


