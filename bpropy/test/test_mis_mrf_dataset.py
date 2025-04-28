import os
import unittest
import networkx as nx
import numpy as np
import pickle as pkl
import tempfile
import torch
import atexit
import shutil

from torch_geometric.data.batch import Batch

from bpropy.co.mis import MIS, random_mis
from bpropy.datasets.mis_data import mis_to_data
from bpropy.mrf.lbp_edges import LBPEdges, lexsort
from bpropy.datasets.mrf_dataset import MRFDataset
from bpropy.datasets.mrf_dataloader import MRFDataLoader, MRFDataCollater

def create_in_memory_dataset(N=10, n=100, p=0.15, formulation="EB", with_lbp_edges=True, with_nmf_edges=False):
        # create a small dataset of problems
        if callable(n):
            pbs = [random_mis(n(), p) for _ in range(N)]
        elif hasattr(n, "__iter__"):
            pbs = [random_mis(v, p) for v in n]
        else:
            pbs = [random_mis(n,p) for _ in range(N)]
        # create the data objects
        data = [mis_to_data(pb, formulation, with_lbp_edges=with_lbp_edges, with_nmf_edges=with_nmf_edges) for pb in pbs]
        # save to temporary pickle file
        data_file = tempfile.NamedTemporaryFile("w+b", delete=False)
        pkl.dump(data, data_file, protocol=pkl.HIGHEST_PROTOCOL)
        data_file.close()
        return data_file.name, data

def create_on_disk_dataset(N=10, n=100, p=0.15, with_lbp_edges=True, with_nmf_edges=False):
        # create a small dataset of problems
        pbs = [random_mis(n,p) for _ in range(N)]
        # create the data objects
        data = [mis_to_data(pb, "EB", with_lbp_edges=with_lbp_edges, with_nmf_edges=with_nmf_edges) for pb in pbs]
        # save to temporary pickle file
        data_dir = tempfile.mkdtemp()
        for i,d in enumerate(data):
            with open(os.path.join(data_dir, f"{i}.dat"), "wb") as f:
                pkl.dump(d, f, protocol=pkl.HIGHEST_PROTOCOL)
        return data_dir, data

class TestMISMRFDataset(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # create simple in memory dataset
        self.n = 10
        min_nodes,max_nodes = 15,31
        self.in_memory_dataset_file_name, self.in_memory_data = create_in_memory_dataset(self.n, lambda: np.random.choice(range(min_nodes,max_nodes)))
        self.in_memory_cc_dataset_file_name, self.in_memory_cc_data = create_in_memory_dataset(self.n, lambda: np.random.choice(range(min_nodes, max_nodes)), formulation="CC")
        self.in_memory_cf_dataset_file_name, self.in_memory_cf_data = create_in_memory_dataset(self.n, lambda: np.random.choice(range(min_nodes, max_nodes)), formulation="CF")
        self.in_memory_ec_dataset_file_name, self.in_memory_ec_data = create_in_memory_dataset(self.n, lambda: np.random.choice(range(min_nodes, max_nodes)), formulation="EC")
        self.in_memory_cp_dataset_file_name, self.in_memory_cp_data = create_in_memory_dataset(self.n, lambda: np.random.choice(range(min_nodes, max_nodes)), formulation="CP")

        self.on_disk_dataset_dir, self.on_disk_data = create_on_disk_dataset(self.n)
        atexit.register(self.cleanup)
    def cleanup(self):
        os.remove(self.in_memory_dataset_file_name)
        os.remove(self.in_memory_cc_dataset_file_name)
        os.remove(self.in_memory_cf_dataset_file_name)
        os.remove(self.in_memory_ec_dataset_file_name)
        os.remove(self.in_memory_cp_dataset_file_name)

        if os.path.exists(self.on_disk_dataset_dir):
            shutil.rmtree(self.on_disk_dataset_dir)

    def test_in_memory_mis_mrf_dataset(self):
        # instantiate the dataset
        dataset = MRFDataset(self.in_memory_dataset_file_name, in_memory=True)
    
        self.assertTrue(dataset.len() == self.n)

        for i in range(self.n):
            d = dataset.get(i)
            self.assertTrue(torch.all(d.x == self.in_memory_data[i].x))
            self.assertTrue(torch.all(d.edge_index == self.in_memory_data[i].edge_index))
            self.assertTrue(torch.all(d.e == self.in_memory_data[i].e))
            self.assertTrue(d.lbp_edges == self.in_memory_data[i].lbp_edges)

    def test_on_disk_mis_mrf_dataset(self):
        # instantiate the dataset
        dataset = MRFDataset(self.on_disk_dataset_dir, in_memory=False)
    
        self.assertTrue(dataset.len() == self.n)

        for i in range(self.n):
            d = dataset.get(i)
            self.assertTrue(torch.all(d.x == self.on_disk_data[i].x))
            self.assertTrue(torch.all(d.edge_index == self.on_disk_data[i].edge_index))
            self.assertTrue(torch.all(d.e == self.on_disk_data[i].e))
            self.assertTrue(d.lbp_edges == self.on_disk_data[i].lbp_edges)

        # test dataloader
        dataloader = MRFDataLoader(dataset, batch_size=2, shuffle=False)
        batch_data = next(iter(dataloader))
        self.assertTrue(batch_data.num_graphs == 2)
        self.assertTrue(batch_data.lbp_edges.num_graphs == 2)
        self.assertTrue(batch_data.lbp_edges == LBPEdges.batch([self.on_disk_data[0].lbp_edges, self.on_disk_data[1].lbp_edges]))

    def test_mis_mrf_data_batch_separate(self):
        dataset = MRFDataset(self.in_memory_dataset_file_name, in_memory=True)
        data = self.in_memory_data
        dataloader = MRFDataLoader(dataset, batch_size=self.n, shuffle=False)

        batch = next(iter(dataloader))

        for i in range(batch.num_graphs):
            _data = batch.get_example(i)
            self.assertTrue(torch.all(_data.x == data[i].x))
            self.assertTrue(torch.all(_data.e == data[i].e))
            self.assertTrue(torch.all(_data.edge_index == data[i].edge_index))
            self.assertTrue(_data.lbp_edges == data[i].lbp_edges)
            self.assertTrue(torch.all(_data.bipartite == data[i].bipartite))
            self.assertTrue(torch.all(_data.factor_sizes == data[i].factor_sizes))


    def test_mis_cc_mrf_data_batch_separate(self):
        dataset = MRFDataset(self.in_memory_cc_dataset_file_name, in_memory=True)
        data = self.in_memory_cc_data
        dataloader = MRFDataLoader(dataset, batch_size=self.n, shuffle=False)

        batch = next(iter(dataloader))

        for i in range(batch.num_graphs):
            _data = batch.get_example(i)
            self.assertTrue(torch.all(_data.x == data[i].x))
            self.assertTrue(torch.all(_data.e == data[i].e))
            self.assertTrue(torch.all(_data.edge_index == data[i].edge_index))
            self.assertTrue(torch.all(_data.bipartite == data[i].bipartite))
            self.assertTrue(torch.all(_data.factor_sizes == data[i].factor_sizes))

            a = _data.lbp_edges == data[i].lbp_edges
            if not a:
                LBPEdges.show_diff(_data.lbp_edges, data[i].lbp_edges)

            self.assertTrue(_data.lbp_edges == data[i].lbp_edges)

    def test_mis_cf_mrf_data_batch_separate(self):
        dataset = MRFDataset(self.in_memory_cf_dataset_file_name, in_memory=True)
        data = self.in_memory_cf_data
        dataloader = MRFDataLoader(dataset, batch_size=self.n, shuffle=False)

        batch = next(iter(dataloader))

        for i in range(batch.num_graphs):
            _data = batch.get_example(i)
            self.assertTrue(torch.all(_data.x == data[i].x))
            self.assertTrue(torch.all(_data.e == data[i].e))
            self.assertTrue(torch.all(_data.edge_index == data[i].edge_index))
            self.assertTrue(torch.all(_data.bipartite == data[i].bipartite))
            self.assertTrue(torch.all(_data.factor_sizes == data[i].factor_sizes))

            a = _data.lbp_edges == data[i].lbp_edges
            if not a:
                LBPEdges.show_diff(_data.lbp_edges, data[i].lbp_edges)

            self.assertTrue(_data.lbp_edges == data[i].lbp_edges)
    def test_mis_ec_mrf_data_batch_separate(self):
        dataset = MRFDataset(self.in_memory_ec_dataset_file_name, in_memory=True)
        data = self.in_memory_ec_data
        dataloader = MRFDataLoader(dataset, batch_size=self.n, shuffle=False)

        batch = next(iter(dataloader))

        for i in range(batch.num_graphs):
            _data = batch.get_example(i)
            self.assertTrue(torch.all(_data.x == data[i].x))
            self.assertTrue(torch.all(_data.e == data[i].e))
            self.assertTrue(torch.all(_data.edge_index == data[i].edge_index))
            self.assertTrue(torch.all(_data.bipartite == data[i].bipartite))
            self.assertTrue(torch.all(_data.factor_sizes == data[i].factor_sizes))

            a = _data.lbp_edges == data[i].lbp_edges
            if not a:
                LBPEdges.show_diff(_data.lbp_edges, data[i].lbp_edges)

            self.assertTrue(_data.lbp_edges == data[i].lbp_edges)

    def test_mis_cp_mrf_data_batch_separate(self):
        dataset = MRFDataset(self.in_memory_cp_dataset_file_name, in_memory=True)
        data = self.in_memory_cp_data
        dataloader = MRFDataLoader(dataset, batch_size=self.n, shuffle=False)

        batch = next(iter(dataloader))

        for i in range(batch.num_graphs):
            _data = batch.get_example(i)
            self.assertTrue(torch.all(_data.x == data[i].x))
            self.assertTrue(torch.all(_data.e == data[i].e))
            self.assertTrue(torch.all(_data.edge_index == data[i].edge_index))
            self.assertTrue(torch.all(_data.bipartite == data[i].bipartite))
            self.assertTrue(torch.all(_data.factor_sizes == data[i].factor_sizes))

            a = _data.lbp_edges == data[i].lbp_edges
            if not a:
                LBPEdges.show_diff(_data.lbp_edges, data[i].lbp_edges)

            self.assertTrue(_data.lbp_edges == data[i].lbp_edges)



if __name__=="__main__":
    unittest.main(verbosity=2)
