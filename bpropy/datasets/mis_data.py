import itertools
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data

from bpropy.vc_graph import VCGraph
from bpropy.co.mis import MIS

from bpropy.datasets.mis_features import get_features

def mis_to_data(pb, feature_category="EB"):
    """Create a torch_geometric.data.Data object from MIS problem data
    """
    vcg = VCGraph(pb)
    g = vcg.to_directed()
    edge_index = torch.empty((2, g.number_of_edges()), dtype=torch.long)
    bipartite = torch.empty(g.number_of_nodes())
    for i,(src,dst) in enumerate(g.edges):
        edge_index[0,i] = src
        edge_index[1,i] = dst
    for n,d in g.nodes(data=True):
        bipartite[n] = d["bipartite"]
    # features
    C, nfeat, efeat, cfeat = get_features(pb, feature_category)
    nfeat = nfeat - np.mean(nfeat,axis=0)
    cfeat = cfeat - np.mean(cfeat,axis=0)
    efeat = efeat - np.mean(efeat,axis=0)
    nn = nfeat.shape[1] - cfeat.shape[1]
    x = np.concatenate((nfeat, np.pad(cfeat, ((0,0), (0,nn)))), axis=0)
    #data.x = torch.from_numpy(x)
    #data.e = torch.from_numpy(np.concatenate((efeat,efeat)))

    return Data(edge_index=edge_index, 
                bipartite=bipartite,
                x=torch.from_numpy(x),
                e=torch.from_numpy(efeat),
                )

def get_neighbors(data, u):
    return data.edge_index[1, data.edge_index[0,:]==u].tolist()

def data_to_mis(data):
    """Create a MIS problem from torch_geometric.data.Data object
    """
    if hasattr(data, "num_graphs"):
        return [data_to_mis(data.get_example(i)) for i in range(data.num_graphs)]
    else:
        nv = int(torch.sum(data.bipartite==0).item())
        nc = int(torch.sum(data.bipartite).item())
        g = nx.Graph()
        for i in range(nv):
            g.add_node(i)
        C = []
        for c in range(nc):
            neighbors = get_neighbors(data, c+nv)
            for e in itertools.product(neighbors, neighbors):
                if e[0]!=e[1]:
                    g.add_edge(e[0], e[1])
            C.append(neighbors)
        return MIS(g, cliques=C)
