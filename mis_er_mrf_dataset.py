"""
Definition of the MISERMRF dataset
"""
import os
import os.path as osp
import torch
from torch_geometric.data import Dataset
from torch_geometric.utils.convert import from_networkx
import networkx as nx
import numpy as np
from progress.bar import Bar
from filelock import FileLock

from bpropy.lbp import *
from dopy.mis import *
from bipartite_graph import *


def solution_to_indicators(x, fg):
    n_ss = sufficient_statistics_dim(fg)
    ind = np.zeros(n_ss)
    fact_offset = 0
    for i,F in enumerate(iter_factor_nodes(fg)):
        neighbors = list(fg.neighbors(F))
        nF = len(neighbors)
        for v in range(2**nF):
            yF = np.unpackbits(np.array([v],dtype=np.uint8), count=nF, bitorder="little")
            xF = np.array([int(x[j] > 0.5) for j in neighbors])
            if np.all(yF == xF):
                ind[fact_offset + v] = 1
        fact_offset += 2**nF
    return ind



class MISERMRFDataset(Dataset):
    def __init__(self, root, data_path, transform=None, pre_transform=None, pre_filter=None, unravel_pool=True):
        self.data_path = data_path
        self.unravel_pool = unravel_pool
        self._processed_file_names = []
        super().__init__(root, transform, pre_transform, pre_filter)
    @property
    def raw_file_names(self):
        r = list(filter(lambda x: x.isnumeric() , os.listdir(self.data_path)))
        return r
    @property
    def processed_file_names(self):
        if len(self._processed_file_names):
            return self._processed_file_names
        if not self.unravel_pool:
            self._processed_file_names = [f"{os.path.splitext(p)[0]}.pt" for p in self.raw_file_names]
            return self._processed_file_names
        self._processed_file_names = []
        bar = Bar("Loading processed file names: ", max=len(self.raw_file_names))
        for p in self.raw_paths:
            # load solution pool
            pool = self.load_solution_pool(p)
            for i in range(pool.shape[0]):
                self._processed_file_names.append(f"{os.path.basename(p)}_{i}.pt")
            bar.next()
        bar.finish()
        return self._processed_file_names
    def download(self):
        # copy files from data_path to raw_dir
        print("Downloading")
        for f in os.listdir(self.data_path):
            if f.isnumeric():
                os.system(f"cp -r {os.path.join(self.data_path, f)} {self.raw_dir}")
        print("Download finished")
    def load_redumis_solution(self, path):
        with open(path, "r") as f:
            return np.array(list(map(int, filter(None, f.read().split("\n")))))
    def load_pool(self, path):
        with open(path, "rb") as f:
            return np.load(f)
    def load_solution_pool(self, path):
        # redumis solution
        y = self.load_redumis_solution(os.path.join(path, "redumis.sol"))
        # pool
        p = self.load_pool(os.path.join(path, "pool.npy"))
        return np.concatenate((np.expand_dims(y,axis=0),p), axis=0)
    def load_raw_graph_features(self, path):
        with open(os.path.join(path, "node_features.npy"), "rb") as f:
            nfeat = np.load(f)
        with open(os.path.join(path, "constraint_features.npy"), "rb") as f:
            cfeat = np.load(f)
        with open(os.path.join(path, "edge_features.npy"), "rb") as f:
            efeat = np.load(f)
        return nfeat,cfeat,efeat
    def process(self):
        # create lock to avoid processing by multiple training runs
        lock = FileLock(osp.join(self.processed_dir, "processing.lock"))
        print("Acquire lock")
        lock.acquire()
        print("Lock acquired")
        idx = 0
        bar = Bar("Processing: ", max=len(self.raw_paths))
        for raw_path in self.raw_paths:
            instance_id = os.path.basename(raw_path)
            processed_names = list(filter(lambda x: x.startswith(f"{instance_id}_"), self.processed_file_names))
            processed_paths = [osp.join(self.processed_dir, p) for p in processed_names]
            if not all([os.path.exists(p) for p in processed_paths]):
                g = nx.read_adjlist(os.path.join(raw_path, "graph.adjlist"))
                nmap = {str(i):i for i in range(g.number_of_nodes())}
                g = nx.relabel_nodes(g, nmap, copy=False)
                pb = MIS(g)
                bpg,bpg_feat = bipartite_graph(pb)
                data = from_networkx(bpg)

                # create the factor graph
                fg = create_factor_graph(pb.g)
                n_ss = sufficient_statistics_dim(fg)
                # create the computation edges
                f2v_idx, f2v_subsum_idx,ss_idx = f2v_index_map(fg)
                # creating message passing edges
                f2v_edges,r_edges = f2v_subsum_indexes(fg, f2v_idx, ss_idx)
                f2v_edges = torch.from_numpy(f2v_edges)
                r_edges = torch.from_numpy(r_edges)
                # scatter indices
                q_edges = torch.from_numpy(v2f_indexes(fg, f2v_idx))
                m_edges = torch.from_numpy(marginal_indexes(fg, f2v_idx, ss_idx))
                vm_edges = torch.from_numpy(variable_marginal_indexes(fg, f2v_idx, ss_idx))

                # node features
                nfeat, cfeat, efeat = self.load_raw_graph_features(raw_path)
                nfeat = nfeat - np.mean(nfeat,axis=0)
                cfeat = cfeat - np.mean(cfeat,axis=0)
                efeat = efeat - np.mean(efeat,axis=0)
                x = np.concatenate((nfeat, np.pad(cfeat, ((0,0), (0,5)))), axis=0)
                data.x = torch.from_numpy(x)
                data.e = torch.from_numpy(np.concatenate((efeat,efeat)))
                data.f2v_edges = f2v_edges.clone()
                data.r_edges = r_edges.clone()
                data.q_edges = q_edges
                data.m_edges = m_edges
                data.ss_dim = sufficient_statistics_dim(fg)
                data.f2v_ss_dim = len(f2v_subsum_idx)
                data.m_stride = torch.tensor([2**len(list(fg.neighbors(F))) for F in iter_factor_nodes(fg)])

                # solutions
                pool = self.load_solution_pool(raw_path)
                pool_values = np.sum(pool, axis=1)
                Z = np.exp(pool_values).sum()
                indicators = [solution_to_indicators(pool[i,:],fg) for i in range(pool.shape[0])]
                for i in range(pool.shape[0]):
                    save_path = osp.join(self.processed_dir, f'{os.path.basename(raw_path)}_{i}.pt')
                    _data = data.clone()
                    _data.y = torch.from_numpy(indicators[i])
                    _data.w = torch.ones_like(_data.y) * np.exp(pool_values[i]) / Z
                    torch.save(_data, save_path)
            bar.next()
        bar.finish()
        lock.release()
    def len(self):
        return len(self.processed_file_names)
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, self._processed_file_names[idx]))
        data.x = data.x.to(torch.float32)
        data.e = data.e.to(torch.float32)
        data.w = data.w.to(torch.float32)
        data.y = data.y.to(torch.float32)
        data.f2v_edges = data.f2v_edges.to(torch.int64)
        data.r_edges   = data.r_edges.to(torch.int64)
        data.q_edges   = data.q_edges.to(torch.int64)
        data.m_edges   = data.m_edges.to(torch.int64)
        return data

if __name__=="__main__":
    dataset_path = "./datasets/mis_er_n700-800_dummy_mrf/train"

    dataset = MISERMRFDataset("./datasets/mis_er_n700-800_dummy_mrf/train", dataset_path)

    print("Dataset length: ", dataset.len())
    print("Data[0] = ")
    print(dataset[0])

    from torch_geometric.loader import DataLoader

    train_dataset = dataset[0:10]
    print(train_dataset)
    loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for data in loader:
        print(torch.unique(data.batch))



