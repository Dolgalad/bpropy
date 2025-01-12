"""
Definition of the MISER dataset
"""
import os
import sys
import os.path as osp
import itertools
import torch
import json
from torch_geometric.data import Dataset, Data
from torch_geometric.utils.convert import from_networkx
import networkx as nx
import numpy as np
from progress.bar import Bar
from filelock import FileLock
import requests
import tarfile
#import xml.etree.ElementTree as ET
from lxml import html
from lxml.cssselect import CSSSelector
from distutils.dir_util import copy_tree
import tempfile

from bpropy.lbp import *
from dopy.mis import *
from bipartite_graph import *

# dataset data urls
dataset_urls = {
        "MIS100_EB": "https://cloud.lipn.univ-paris13.fr/index.php/s/RGdPmTfj83DmMdP",
        "MIS100_CC": "https://cloud.lipn.univ-paris13.fr/index.php/s/f3545KoiRb7x4iW"
        }
dataset_archives = {
        "MIS100_EB": "mis_er_n100-100_p0.15_tl1000.0_pooltl1000.0_eb.tar.gz",
        "MIS100_CC": "mis_er_n100-100_p0.15_tl1000.0_pooltl1000.0_cc.tar.gz"
        }

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
    def __init__(self, root, category="MIS100_EB", mode="train", transform=None, pre_transform=None, pre_filter=None, unravel_pool=True, gap_limit=None, solution_limit=None, aggregate=False, verbose=True):
        self.category = category
        self.mode = mode
        self.unravel_pool = unravel_pool
        self.solution_limit = solution_limit
        self.aggregate = aggregate
        self.verbose = verbose
        if solution_limit is None:
            self.solution_limit = np.inf
        self._processed_file_names = []
        self.gap_limit = gap_limit
        # write description
        description = {"category": category, "unravel_pool": unravel_pool, "gap_limit": gap_limit, "solution_limit": solution_limit, "aggregate": aggregate}

        lock_path = osp.join(root, "processing.lock")
        lock = FileLock(lock_path, thread_local=False)
        if self.verbose:
            print("Waiting on lock: ", lock_path)
        lock.acquire()
        if self.verbose:
            print(f"Lock {lock_path} acquired")
        with open(os.path.join(root, "description.json"), "w") as f:
            f.write(json.dumps(description, sort_keys=True, indent=4))
        super().__init__(root, transform, pre_transform, pre_filter)
        if self.verbose:
            print(f"Lock {lock_path} released")
    @property
    def raw_file_names(self):
        if self.mode=="train":
            return list([str(i) for i in range(1,3001)])
        if self.mode=="validation":
            return list([str(i) for i in range(3001,4001)])
        if self.mode=="test":
            return list([str(i) for i in range(4001,5001)])

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
            if self.unravel_pool:
                pool = self.load_solution_pool(p)
                # best solution val
                pool_vals = np.sum(pool, axis=1)
                best_val = np.max(pool_vals)
                gaps = (best_val - pool_vals) / best_val
                for i in range(pool.shape[0]):
                    val = np.sum(pool[i,:])
                    if (self.gap_limit is None) or (gaps[i] <= self.gap_limit):
                        self._processed_file_names.append(f"{os.path.basename(p)}_{i}.pt")
            bar.next()
        bar.finish()
        return self._processed_file_names
    def download(self):
        if all([os.path.exists(os.path.join(self.raw_dir, rf)) for rf in self.raw_file_names]):
            return 

        with tempfile.TemporaryDirectory(dir=os.environ["WORK"]) as tmpdir:
            archive_path = os.path.join(os.environ["WORK"], f"datasets_archive/{dataset_archives[self.category]}")

            ## download data archive
            #req = requests.get(dataset_urls[self.category])
            #tree = html.fromstring(req.text)
            #selector = CSSSelector("#downloadFile")
            #elements_with_id = selector(tree)
            #if len(elements_with_id)==0 and self.verbose:
            #    print("Cannot find download link")
            #url = elements_with_id[0].get("href")
            #if self.verbose:
            #    print("url = ", url)
            #r = requests.get(url, stream=True)
            #with open(path, 'wb') as f:
            #    total_length = int(r.headers.get('content-length'))
            #    bar = Bar("Downloading: ", max=(total_length/1024) + 1)
            #    for chunk in r.iter_content(chunk_size=1024):
            #        if chunk:
            #            f.write(chunk)
            #            f.flush()
            #            bar.next()
            #    bar.finish()

            # extract data
            if self.verbose:
                sys.stdout.write("Extracting...")
                sys.stdout.flush()
            file = tarfile.open(archive_path)
            file.extractall(tmpdir)
            file.close()
            if self.verbose:
              sys.stdout.write("done\n")
              sys.stdout.flush()
       
            # remove archive
            if self.verbose:
                sys.stdout.write("Cleaning up...")
                sys.stdout.flush()
            # move the instance directories to raw_dir
            data_dir = os.path.join(tmpdir, os.listdir(tmpdir)[0])
            for rf in self.raw_file_names:
                dest_dir = os.path.join(self.raw_dir, rf)
                if not os.path.exists(dest_dir):
                    orig_dir = os.path.join(data_dir, self.mode, rf)
                    copy_tree(orig_dir, dest_dir)



            sys.stdout.write("done\n")
            sys.stdout.flush()

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
        pool = np.concatenate((np.expand_dims(y,axis=0),p), axis=0)
        # sort pool by descending value
        pool_vals = np.sum(pool, axis=1)
        pool = pool[np.flip(np.argsort(pool_vals))]
        if self.solution_limit is None or np.isinf(self.solution_limit):
            return pool
        return pool[:self.solution_limit]

    def load_raw_graph_features(self, path):
        with open(os.path.join(path, "node_features.npy"), "rb") as f:
            nfeat = np.load(f)
        with open(os.path.join(path, "constraint_features.npy"), "rb") as f:
            cfeat = np.load(f)
        with open(os.path.join(path, "edge_features.npy"), "rb") as f:
            efeat = np.load(f)
        return nfeat,cfeat,efeat
    def ordered_edges(self, g):
        nV = nx.number_of_nodes(g)
        adj = nx.to_numpy_array(g, nodelist=range(nV))
        if g.is_directed():
            for (s,t) in zip(*np.where(adj > 0)):
                yield (s,t)
        else:
            for (s,t) in zip(*np.where(adj > 0)):
                if s<t:
                    yield (s,t)
    def networkx_to_tgdata(self, g):
        g = g.to_directed()
        edge_index = torch.empty((2, g.number_of_edges()), dtype=torch.long)
        bipartite = torch.empty(g.number_of_nodes())
        #for i,(src,dst) in enumerate(g.edges):
        for i,(src,dst) in enumerate(self.ordered_edges(g)):
            edge_index[0,i] = src
            edge_index[1,i] = dst
        for n,d in g.nodes(data=True):
            bipartite[n] = d["bipartite"]
        return Data(edge_index=edge_index, bipartite=bipartite)
    def process(self):
        # create lock to avoid processing by multiple training runs
        idx = 0
        bar = Bar("Processing: ", max=len(self.raw_paths))
        for raw_path in self.raw_paths:
            instance_id = os.path.basename(raw_path)
            if self.unravel_pool:
                processed_names = list(filter(lambda x: x.startswith(f"{instance_id}_"), self.processed_file_names))
                processed_paths = [osp.join(self.processed_dir, p) for p in processed_names]
            else:
                processed_names = list(filter(lambda x: x.startswith(f"{instance_id}"), self.processed_file_names))
                processed_paths = [osp.join(self.processed_dir, p) for p in processed_names]

            if not all([os.path.exists(p) for p in processed_paths]):
                g = nx.read_adjlist(os.path.join(raw_path, "graph.adjlist"))
                nmap = {str(i):i for i in range(g.number_of_nodes())}
                g = nx.relabel_nodes(g, nmap, copy=False)
                pb = MIS(g)
                
                # bipartite graph
                bpg_file_name = os.path.join(raw_path, "bipartite.adjlist")
                if os.path.exists(bpg_file_name):
                    bpg = nx.read_adjlist(bpg_file_name)
                    nmap = {str(i):i for i in range(bpg.number_of_nodes())}
                    bpg = nx.relabel_nodes(bpg, nmap, copy=False)
                    # set bipartite attributes
                    attrs = {i:{"bipartite":int(i>=g.number_of_nodes())} for i in bpg.nodes}
                    nx.set_node_attributes(bpg, attrs)
                else:
                    bpg,_ = bipartite_graph(pb)
                #data = from_networkx(bpg)
                data = self.networkx_to_tgdata(bpg)

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

                # best solution val
                pool_vals = np.sum(pool, axis=1)
                best_val = np.max(pool_vals)
                gaps = (best_val - pool_vals) / best_val

                if self.unravel_pool:
                    for i in range(pool.shape[0]):
                        if (self.gap_limit is None) or (gaps[i] <= self.gap_limit):
                            save_path = osp.join(self.processed_dir, f'{os.path.basename(raw_path)}_{i}.pt')
                            _data = data.clone()
                            _data.y = torch.from_numpy(solution_to_indicators(pool[i,:], fg))
                            _data.w = torch.ones_like(_data.y) * np.exp(pool_values[i]) / Z
                            torch.save(_data, save_path)
                else:
                    save_path = osp.join(self.processed_dir, f'{os.path.basename(raw_path)}.pt')
                    if self.aggregate:
                        data.y = torch.from_numpy(self.aggregate_pool(pool, fg))
                        data.w = torch.ones_like(data.y)
                    else:
                        # best solution
                        data.y = torch.from_numpy(solution_to_indicators(pool[0], fg))
                        data.w = torch.ones_like(data.y)
                    torch.save(data, save_path)

            bar.next()
        bar.finish()
    def aggregate_pool(self, pool, fg):
        indicator_pool = np.array([solution_to_indicators(pool[i,:], fg) for i in range(pool.shape[0])])
        pool_vals = np.sum(pool, axis=1).reshape(pool.shape[0],1)
        Z = np.exp(pool_vals).sum()
        w = np.tile(np.exp(pool_vals), (1, indicator_pool.shape[1]))
        r = (w * pool).sum(axis=0) / Z
        return r


    def len(self):
        return len(self.processed_file_names)
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, self._processed_file_names[idx]), weights_only=False)
        data.x = data.x.to(torch.float32)
        data.e = data.e.to(torch.float32)
        if hasattr(data, "w"):
            data.w = data.w.to(torch.float32)
        data.y = data.y.to(torch.float32)
        data.bipartite = data.bipartite.to(torch.bool)
        return data

def get_neighbors(data, u):
    return data.edge_index[1, data.edge_index[0,:]==u].numpy()
def tgdata_to_networkx(data):
    nv = int(torch.sum(data.bipartite==0).item())
    nc = int(torch.sum(data.bipartite).item())
    g = nx.Graph()
    for i in range(nv):
        g.add_node(i)
    for c in range(nc):
        neighbors = get_neighbors(data, c+nv)
        for e in itertools.product(neighbors, neighbors):
            if e[0]!=e[1]:
                g.add_edge(e[0], e[1])
    return g


if __name__=="__main__":
    dataset = MISERDataset("./datasets/mis_er_n100-100_dummy_val", category="MIS100_EB", mode="validation")

    print("Dataset length: ", dataset.len())
    print("Data[0] = ")
    print(dataset[0])

    from torch_geometric.loader import DataLoader

    train_dataset = dataset[0:10]
    print(train_dataset)
    loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    print(len(dataset))
    print(len(loader))
    for data in loader:
        print(torch.unique(data.batch))



