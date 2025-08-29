import itertools
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data, HeteroData

from bpropy.utils.graph_utils import get_edge_index

from bpropy.vc_graph import VCGraph
from bpropy.co.mis import MIS, mis_batch

from bpropy.mrf.mis_factor_graph import MISFactorGraph
from bpropy.mrf.lbp_edges import LBPEdges
from bpropy.mrf.nmf_edges import NMFEdges
from bpropy.mrf.sufficient_statistics import x2ss

from bpropy.datasets.mis_features import get_features

def mis_to_data(pb, feature_category="EB", with_lbp_edges=False, with_nmf_edges=False, y=None):
    """Create a torch_geometric.data.Data object from MIS problem data
    """
    # features
    if not feature_category is None:
        C, nfeat, efeat, cfeat = get_features(pb, feature_category)
        nfeat = nfeat - np.mean(nfeat,axis=0)
        cfeat = cfeat - np.mean(cfeat,axis=0)
        efeat = efeat - np.mean(efeat,axis=0)
        nn = nfeat.shape[1] - cfeat.shape[1]
        x = np.concatenate((nfeat, np.pad(cfeat, ((0,0), (0,nn)))), axis=0)

    vcg = VCGraph(pb)
    #g = vcg.to_directed()
    edge_index = get_edge_index(vcg)
    bipartite = torch.zeros(vcg.number_of_nodes())
    bipartite[[n for n,d in vcg.nodes(data=True) if d["bipartite"]]] = 1
    factor_sizes = torch.ones(vcg.number_of_nodes(), dtype=torch.int)
    neighbors = torch.zeros(vcg.number_of_nodes(), dtype=torch.int)
    for n,d in vcg.nodes(data=True):
        if d["bipartite"]==1:
            factor_sizes[n] = len(list(vcg.neighbors(n)))
        neighbors[n] = len(list(vcg.neighbors(n)))

    data = Data(edge_index=edge_index, 
                bipartite=bipartite,
                factor_sizes=factor_sizes,
                neighbors=neighbors,
                )
    if not feature_category is None:
        data.x=torch.from_numpy(x).to(dtype=torch.float32)
        data.e=torch.from_numpy(efeat).to(dtype=torch.float32)

    if with_lbp_edges or with_nmf_edges:
        fg = MISFactorGraph(pb)
        if with_lbp_edges:
            data.lbp_edges = LBPEdges(fg)
        if with_nmf_edges:
            data.nmf_edges = NMFEdges(fg)
        if not y is None:
            data.y = y
            data.y_ss = torch.from_numpy(x2ss(y.to(torch.long).numpy(), fg)).to(torch.float)

    return data

def get_vc_edge_index(g, direction="v2c"):
    nV = len(list([n for n,d in g.nodes(data=True) if d["bipartite"] == 0]))
    edge_index = torch.empty((2, g.number_of_edges()), dtype=torch.long)
    for i,(src,dst) in enumerate(g.edges):
        if (g.nodes[src]["bipartite"]==1 and direction=="v2c") or (g.nodes[src]["bipartite"]==0 and direction=="c2v"):
            src,dst = dst,src
        if src>=nV:
            src -= nV
        if dst>=nV:
            dst -= nV
        edge_index[0,i] = src
        edge_index[1,i] = dst
    return edge_index


def mis_to_hetero_data(pb, feature_category="EB", with_lbp_edges=False, with_nmf_edges=False, y=None):
    """Create a torch_geometric.data.Data object from MIS problem data
    """
    # features
    if not feature_category is None:
        C, nfeat, efeat, cfeat = get_features(pb, feature_category)
        nfeat = nfeat - np.mean(nfeat,axis=0)
        cfeat = cfeat - np.mean(cfeat,axis=0)
        efeat = efeat - np.mean(efeat,axis=0)
        nn = nfeat.shape[1] - cfeat.shape[1]
        cfeat = np.pad(cfeat, ((0,0), (0,nn)))
        #x = np.concatenate((nfeat, np.pad(cfeat, ((0,0), (0,nn)))), axis=0)

    vcg = VCGraph(pb)
    #g = vcg.to_directed()
    v2c_edge_index = get_vc_edge_index(vcg, direction="v2c")
    c2v_edge_index = get_vc_edge_index(vcg, direction="c2v")

    #bipartite = torch.zeros(vcg.number_of_nodes())
    #bipartite[[n for n,d in vcg.nodes(data=True) if d["bipartite"]]] = 1
    factor_sizes = torch.ones(vcg.nc(), dtype=torch.int)
    v_neighbors = torch.zeros(vcg.nv(), dtype=torch.int)
    c_neighbors = torch.zeros(vcg.nc(), dtype=torch.int)
    for n,d in vcg.nodes(data=True):
        if d["bipartite"]==1:
            factor_sizes[n-pb.nv()] = len(list(vcg.neighbors(n)))
            c_neighbors[n-pb.nv()] = len(list(vcg.neighbors(n)))
        else:
            v_neighbors[n] = len(list(vcg.neighbors(n)))

    data = HeteroData()
    data["variable","v2c","constraint"].edge_index = v2c_edge_index
    data["constraint","c2v","variable"].edge_index = c2v_edge_index

    if not feature_category is None:
        data["variable"].x=torch.from_numpy(nfeat).to(dtype=torch.float32)
        data["constraint"].x=torch.from_numpy(cfeat).to(dtype=torch.float32)
        data["constraint"].factor_sizes = factor_sizes
        data["variable"].neighbors = v_neighbors
        data["constraint"].neighbors = c_neighbors
        #data.e=torch.from_numpy(efeat).to(dtype=torch.float32), # not used for now

    if with_lbp_edges or with_nmf_edges:
        fg = MISFactorGraph(pb)
        if with_lbp_edges:
            data.lbp_edges = LBPEdges(fg)
        if with_nmf_edges:
            data.nmf_edges = NMFEdges(fg)
        if not y is None:
            data.y = y
            data.y_ss = torch.from_numpy(x2ss(y.to(torch.long).numpy(), fg)).to(torch.float)

    return data

def get_neighbors(data, u):
    if isinstance(data, HeteroData):
        return sorted(data["constraint","c2v","variable"].edge_index[1, data["constraint","c2v","variable"].edge_index[0,:]==u].tolist() )
    else:
        return sorted(data.edge_index[1, data.edge_index[0,:]==u].tolist())

def data_to_mis(data, batch=False):
    """Create a MIS problem from torch_geometric.data.Data object
    """
    if hasattr(data, "num_graphs"):
        if batch:
            return mis_batch([data_to_mis(data.get_example(i)) for i in range(data.num_graphs)])
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

def hetero_data_to_mis(data, batch=False):
    """Create a MIS problem from torch_geometric.data.HeteroData object
    """
    if hasattr(data, "num_graphs"):
        #if batch:
        #    return mis_batch([data_to_mis(data.get_example(i)) for i in range(data.num_graphs)])
        if data.num_graphs==1:
            return hetero_data_to_mis(data.get_example(0))
        return [hetero_data_to_mis(data.get_example(i)) for i in range(data.num_graphs)]
    else:
        #nv = int(torch.sum(data.bipartite==0).item())
        #nc = int(torch.sum(data.bipartite).item())
        nv = data["variable"].x.shape[0]
        nc = data["constraint"].x.shape[0]

        g = nx.Graph()
        for i in range(nv):
            g.add_node(i)
        C = []
        for c in range(nc):
            neighbors = get_neighbors(data, c)
            for e in itertools.product(neighbors, neighbors):
                if e[0]!=e[1]:
                    g.add_edge(e[0], e[1])
            C.append(neighbors)
        return MIS(g, cliques=C)
