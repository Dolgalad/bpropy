import unittest
import torch
import time
import numpy as np

from bpropy.co.mis import MIS
from bpropy.mrf.mis_factor_graph import MISFactorGraph
from bpropy.mrf.nmf_edges import NMFEdges
from bpropy.mrf.naive_mean_field import naive_mean_field
from bpropy.mrf.sufficient_statistics import x2ss

from bpropy.test.test_mis import random_mis, simple_mis, very_simple_mis

class TestMISNMF(unittest.TestCase):
    def test_mis_nmf(self):
        pb = simple_mis()
        x = pb.greedy()
        fg = MISFactorGraph(pb)
        edges = NMFEdges(fg)

        self.assertTrue(edges.n_ss == fg.number_sufficient_statistics())

        # naive mean field
        theta = torch.from_numpy(x2ss(x.astype(bool), fg)).to(torch.float)
        v2f = torch.zeros_like(theta)
        mu = naive_mean_field(theta, edges, max_iter=10, tol=1e-6).mu

        self.assertTrue(mu.shape[0] == 2*edges.n_vars)
        
        offset = 0
        for v in fg.iter_variables():
            self.assertTrue(mu[offset:offset+2].sum() == 1)
            offset += 2
    def test_large_mis_nmf(self):
        pb = random_mis(100)
        x = pb.greedy()
        fg = MISFactorGraph(pb)
        edges = NMFEdges(fg)

        self.assertTrue(edges.n_ss == fg.nss())
        self.assertTrue(edges.n_vars == fg.nv())

        # naive mean field
        theta = torch.from_numpy(x2ss(x.astype(bool), fg)).to(torch.float)
        mu = naive_mean_field(theta, edges, max_iter=10, tol=1e-6).mu

        self.assertTrue(mu.shape[0] == 2*edges.n_vars)
        
        offset = 0
        for v in fg.iter_variables():
            self.assertTrue(torch.isclose(mu[offset:offset+2].sum(),torch.tensor(1.)))
            offset += 2
    def test_stack_nmfedges(self):
        max_iter = 1
        batch_size = 4
        # prepare problems and solve
        pbs = []
        mus = []
        batch_edges = []
        batch_thetas = []
        for _ in range(batch_size):
            pb = random_mis(10)
            fg = MISFactorGraph(pb)
            edges = NMFEdges(fg)
            theta = torch.randn(fg.number_sufficient_statistics())
            mu = naive_mean_field(theta.clone(), edges.clone(), max_iter=max_iter, tol=1e-6).mu
            mus.append(mu)
            batch_edges.append(edges)
            batch_thetas.append(theta)

        # solve batched graph
        batch = NMFEdges.batch(batch_edges)
        batch_theta = torch.concatenate(batch_thetas, dim=0)
        batch_mus = naive_mean_field(batch_theta.clone(), batch.clone(), max_iter=max_iter, tol=1e-6).mu
        
        end_offset = (2 * batch.n_vars).cumsum(dim=0)
        beg_offset = end_offset.roll(1)
        beg_offset[0] = 0
        for i in range(batch_size):
            self.assertTrue(torch.all(torch.isclose(mus[i], batch_mus[beg_offset[i]:end_offset[i]])))

    def test_mis_cc_batch_nmf(self):
        # create a list of problems
        batch_size = 10
        max_iter = 1
        #pbs = [random_mis(15, p=0.10, cliques=True) for _ in range(batch_size)]
        pbs = [random_mis(np.random.choice(range(10,20)), p=0.10, cliques=False) for _ in range(batch_size)]

        # batch mis problems
        batch_pb = MIS.batch(pbs)
        batch_fg = MISFactorGraph(batch_pb)
        batch_edges = NMFEdges(batch_fg)
        batch_theta = torch.randn(batch_fg.nss())
        batch_mu = naive_mean_field(batch_theta, batch_edges, max_iter=max_iter, tol=1e-6).mu

        ss_offset, mu_offset = 0,0
        for i,pb in enumerate(pbs):
            fg = MISFactorGraph(pb)
            edges = NMFEdges(fg)
            theta = batch_theta[ss_offset:ss_offset+fg.nss()]
            mu = naive_mean_field(theta, edges, max_iter=max_iter, tol=1e-6).mu
            self.assertTrue(torch.all(torch.isclose(mu, batch_mu[mu_offset:mu_offset+2*fg.nv()])))
            mu_offset += 2*fg.nv()
            ss_offset += fg.nss()

if __name__=="__main__":
    unittest.main()
