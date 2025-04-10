import os
import torch
import unittest
import atexit
import numpy as np

from bpropy.test.test_mis_mrf_dataset import random_mis, create_in_memory_dataset
from bpropy.datasets.mis_data import data_to_mis
from bpropy.mrf.mis_factor_graph import MISFactorGraph
from bpropy.datasets.mrf_dataset import MRFDataset
from bpropy.datasets.mrf_dataloader import MRFDataLoader
from bpropy.models.mrf_models import GCN_MRF_v4

class TestMISMRFModels(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # create simple in memory dataset
        self.n = 10
        min_nodes, max_nodes = 15, 31
        self.dataset_file_name, self.data = create_in_memory_dataset(self.n)
        self.dataset = MRFDataset(self.dataset_file_name)
        self.cc_dataset_file_name, self.cc_data = create_in_memory_dataset(self.n, lambda: np.random.choice(range(min_nodes, max_nodes)),  formulation="CC")
        self.cc_dataset = MRFDataset(self.cc_dataset_file_name)

        self.cf_dataset_file_name, self.cf_data = create_in_memory_dataset(self.n, lambda: np.random.choice(range(min_nodes, max_nodes)), formulation="CF")
        self.cf_dataset = MRFDataset(self.cf_dataset_file_name)

        self.ec_dataset_file_name, self.ec_data = create_in_memory_dataset(self.n, lambda: np.random.choice(range(min_nodes, max_nodes)),formulation="EC")
        self.ec_dataset = MRFDataset(self.ec_dataset_file_name)

        self.cp_dataset_file_name, self.cp_data = create_in_memory_dataset(self.n, lambda: np.random.choice(range(min_nodes, max_nodes)),formulation="CP")
        self.cp_dataset = MRFDataset(self.cp_dataset_file_name)

        # TODO: do cpu and gpu tests separately
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = GCN_MRF_v4(10, n_layers=2, factor_sizes=list(range(1,10)), jk=None, norm=None, act=None, bp_max_iter=2)
        self.model.to(self.device)


        atexit.register(self.cleanup)
    def cleanup(self):
        os.remove(self.dataset_file_name)
        os.remove(self.cc_dataset_file_name)
        os.remove(self.cf_dataset_file_name)
        os.remove(self.ec_dataset_file_name)
        os.remove(self.cp_dataset_file_name)

    def test_mis_mrf_model_GCN_MRF_v4(self):
        dataloader = MRFDataLoader(self.dataset, batch_size=1, shuffle=False)
        dataloader_bs2 = MRFDataLoader(self.dataset, batch_size=2, shuffle=False)

        self.model.eval()
        with torch.no_grad():
            for d in dataloader:
                pb = data_to_mis(d, batch=True)
                fg = MISFactorGraph(pb)
                self.assertTrue(fg.nss() == d.lbp_edges.n_ss.sum())
                x = self.model(d.to(self.device))
                self.assertTrue(x.shape[0] == d.lbp_edges.n_ss.sum())
            for d in dataloader_bs2:
                pb = data_to_mis(d, batch=True)
                fg = MISFactorGraph(pb)
                self.assertTrue(fg.nss() == d.lbp_edges.n_ss.sum())
                x = self.model(d.to(self.device))
                self.assertTrue(x.shape[0] == d.lbp_edges.n_ss.sum())

                y = torch.concatenate([self.model(d.get_example(i).to(self.device)) for i in range(d.num_graphs)], dim=0)
                self.assertTrue(x.shape == y.shape)
                self.assertTrue(torch.all(torch.isclose(x, y, rtol=1e-5, atol=1e-5)))
    def test_cc_mis_mrf_model_GCN_MRF_v4(self):
        dataloader = MRFDataLoader(self.cc_dataset, batch_size=1, shuffle=False)
        dataloader_bs2 = MRFDataLoader(self.cc_dataset, batch_size=2, shuffle=False)

        self.model.eval()
        with torch.no_grad():
            for d in dataloader:
                pb = data_to_mis(d, batch=True)
                fg = MISFactorGraph(pb)
                self.assertTrue(fg.nss() == d.lbp_edges.n_ss.sum())
                x = self.model(d.to(self.device))
                self.assertTrue(x.shape[0] == d.lbp_edges.n_ss.sum())
            for d in dataloader_bs2:
                pb = data_to_mis(d, batch=True)
                fg = MISFactorGraph(pb)
                self.assertTrue(fg.nss() == d.lbp_edges.n_ss.sum())
                x = self.model(d.to(self.device))
                self.assertTrue(x.shape[0] == d.lbp_edges.n_ss.sum())

                y = torch.concatenate([self.model(d.get_example(i).to(self.device)) for i in range(d.num_graphs)], dim=0)
                self.assertTrue(x.shape == y.shape)
                self.assertTrue(torch.all(torch.isclose(x, y, rtol=1e-5, atol=1e-5)))

    def test_cf_mis_mrf_model_GCN_MRF_v4(self):
        dataloader = MRFDataLoader(self.cf_dataset, batch_size=1, shuffle=False)
        dataloader_bs2 = MRFDataLoader(self.cf_dataset, batch_size=2, shuffle=False)

        self.model.eval()
        with torch.no_grad():
            for d in dataloader:
                pb = data_to_mis(d, batch=True)
                fg = MISFactorGraph(pb)
                self.assertTrue(fg.nss() == d.lbp_edges.n_ss.sum())
                x = self.model(d.to(self.device))
                self.assertTrue(x.shape[0] == d.lbp_edges.n_ss.sum())
            for d in dataloader_bs2:
                pb = data_to_mis(d, batch=True)
                fg = MISFactorGraph(pb)
                self.assertTrue(fg.nss() == d.lbp_edges.n_ss.sum())
                x = self.model(d.to(self.device))
                self.assertTrue(x.shape[0] == d.lbp_edges.n_ss.sum())

                y = torch.concatenate([self.model(d.get_example(i).to(self.device)) for i in range(d.num_graphs)], dim=0)
                self.assertTrue(x.shape == y.shape)
                self.assertTrue(torch.all(torch.isclose(x, y, rtol=1e-5, atol=1e-5)))

    def test_ec_mis_mrf_model_GCN_MRF_v4(self):
        dataloader = MRFDataLoader(self.ec_dataset, batch_size=1, shuffle=False)
        dataloader_bs2 = MRFDataLoader(self.ec_dataset, batch_size=2, shuffle=False)

        self.model.eval()
        with torch.no_grad():
            for d in dataloader:
                pb = data_to_mis(d, batch=True)
                fg = MISFactorGraph(pb)
                self.assertTrue(fg.nss() == d.lbp_edges.n_ss.sum())
                x = self.model(d.to(self.device))
                self.assertTrue(x.shape[0] == d.lbp_edges.n_ss.sum())
            for d in dataloader_bs2:
                pb = data_to_mis(d, batch=True)
                fg = MISFactorGraph(pb)
                self.assertTrue(fg.nss() == d.lbp_edges.n_ss.sum())
                x = self.model(d.to(self.device))
                self.assertTrue(x.shape[0] == d.lbp_edges.n_ss.sum())

                y = torch.concatenate([self.model(d.get_example(i).to(self.device)) for i in range(d.num_graphs)], dim=0)
                self.assertTrue(x.shape == y.shape)
                self.assertTrue(torch.all(torch.isclose(x, y, rtol=1e-5, atol=1e-5)))

    def test_cp_mis_mrf_model_GCN_MRF_v4(self):
        dataloader = MRFDataLoader(self.cp_dataset, batch_size=1, shuffle=False)
        dataloader_bs2 = MRFDataLoader(self.cp_dataset, batch_size=2, shuffle=False)

        self.model.eval()
        with torch.no_grad():
            for d in dataloader:
                pb = data_to_mis(d, batch=True)
                fg = MISFactorGraph(pb)
                self.assertTrue(fg.nss() == d.lbp_edges.n_ss.sum())
                x = self.model(d.to(self.device))
                self.assertTrue(x.shape[0] == d.lbp_edges.n_ss.sum())
            for d in dataloader_bs2:
                pb = data_to_mis(d, batch=True)
                fg = MISFactorGraph(pb)
                self.assertTrue(fg.nss() == d.lbp_edges.n_ss.sum())
                x = self.model(d.to(self.device))
                self.assertTrue(x.shape[0] == d.lbp_edges.n_ss.sum())

                y = torch.concatenate([self.model(d.get_example(i).to(self.device)) for i in range(d.num_graphs)], dim=0)
                self.assertTrue(x.shape == y.shape)
                self.assertTrue(torch.all(torch.isclose(x, y, rtol=1e-5, atol=1e-5)))



if __name__=="__main__":
    unittest.main(verbosity=2)
