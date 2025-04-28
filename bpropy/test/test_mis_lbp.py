import unittest
import torch
import time
import numpy as np

from bpropy.co.mis import MIS
from bpropy.mrf.mis_factor_graph import MISFactorGraph
from bpropy.mrf.lbp_edges import LBPEdges
from bpropy.mrf.lbp import loopy_belief_propagation, max_sum_loopy_belief_propagation
from bpropy.mrf.sufficient_statistics import x2ss

from bpropy.test.test_mis import random_mis, simple_mis, very_simple_mis


class TestMISLBP(unittest.TestCase):
    def test_mis_factor_graph(self):
        pb = simple_mis()
        fg = MISFactorGraph(pb)

        self.assertTrue(fg.nv() == pb.nv())
        self.assertTrue(fg.nf() == pb.nv()+pb.ne())
        self.assertTrue(fg.number_sufficient_statistics() == 2*pb.nv() + 4*pb.ne())
    def test_mis_lbp(self):
        pb = simple_mis()
        x = pb.greedy()
        fg = MISFactorGraph(pb)
        edges = LBPEdges(fg)

        self.assertTrue(edges.n_ss == fg.number_sufficient_statistics())

        # loopy belief propagation
        
        #theta = torch.randn(edges.n_ss)
        theta = torch.from_numpy(x2ss(x.astype(bool), fg)).to(torch.float)
        v2f = torch.zeros_like(theta)
        mu = loopy_belief_propagation(theta, v2f, edges, max_iter=10, tol=1e-6, damping=0.).mu

        self.assertTrue(mu.shape[0] == edges.n_ss)
        
        offset = 0
        for F in fg.iter_factors():
            nF = fg.factor_size(F)
            self.assertTrue(mu[offset:offset+2**nF].sum() == 1)
            offset += 2**nF
    def test_large_mis_lbp(self):
        pb = random_mis(100)
        x = pb.greedy()
        fg = MISFactorGraph(pb)
        edges = LBPEdges(fg)

        self.assertTrue(edges.n_ss == fg.number_sufficient_statistics())

        # loopy belief propagation
        
        #theta = torch.randn(edges.n_ss)
        theta = torch.from_numpy(x2ss(x.astype(bool), fg)).to(torch.float)
        v2f = torch.zeros_like(theta)
        mu = loopy_belief_propagation(theta, v2f, edges, max_iter=10, tol=1e-6, damping=0.).mu

        self.assertTrue(mu.shape[0] == edges.n_ss)
        
        offset = 0
        for F in fg.iter_factors():
            nF = fg.factor_size(F)
            self.assertTrue(torch.isclose(mu[offset:offset+2**nF].sum(),torch.tensor(1.)))
            offset += 2**nF

        # max sum lbp
        mu = max_sum_loopy_belief_propagation(theta, v2f, edges, max_iter=10, tol=1e-6, damping=0.).mu
        self.assertTrue(mu.shape[0] == 2*edges.n_vars.sum())

    def test_stack_lbpedges(self):
        max_iter = 10
        batch_size = 4
        # prepare problems and solve
        pbs = []
        mus = []
        batch_edges = []
        batch_thetas = []
        for _ in range(batch_size):
            pb = random_mis(10)
            fg = MISFactorGraph(pb)
            edges = LBPEdges(fg)
            theta = torch.randn(fg.number_sufficient_statistics())
            v2f = torch.zeros_like(theta)
            mu = loopy_belief_propagation(theta, v2f, edges, max_iter=max_iter, tol=1e-6, damping=0.).mu
            mus.append(mu)
            batch_edges.append(edges)
            batch_thetas.append(theta)

        # solve batched graph
        batch = LBPEdges.batch(batch_edges)
        batch_theta = torch.concatenate(batch_thetas, dim=0)
        batch_v2f = torch.zeros_like(batch_theta)
        batch_mus = loopy_belief_propagation(batch_theta, batch_v2f, batch, max_iter=max_iter, tol=1e-6, damping=0.).mu
        
        end_offset = batch.n_ss.cumsum(dim=0)
        beg_offset = end_offset.roll(1)
        beg_offset[0] = 0
        for i in range(batch_size):
            self.assertTrue(torch.all(torch.isclose(mus[i], batch_mus[beg_offset[i]:end_offset[i]])))
    def test_batch_maxsum(self):
        max_iter = 10
        batch_size = 4
        # prepare problems and solve
        pbs = []
        mus = []
        batch_edges = []
        batch_thetas = []
        for _ in range(batch_size):
            pb = random_mis(10)
            fg = MISFactorGraph(pb)
            edges = LBPEdges(fg)
            theta = torch.randn(fg.number_sufficient_statistics())
            v2f = torch.zeros_like(theta)
            mu = max_sum_loopy_belief_propagation(theta, v2f, edges, max_iter=max_iter, tol=1e-6, damping=0.).mu
            mus.append(mu)
            batch_edges.append(edges)
            batch_thetas.append(theta)

        # solve batched graph
        batch = LBPEdges.batch(batch_edges)
        batch_theta = torch.concatenate(batch_thetas, dim=0)
        batch_v2f = torch.zeros_like(batch_theta)
        batch_mus = max_sum_loopy_belief_propagation(batch_theta, batch_v2f, batch, max_iter=max_iter, tol=1e-6, damping=0.).mu
        
        end_offset = (2*batch.n_vars).cumsum(dim=0)
        beg_offset = end_offset.roll(1)
        beg_offset[0] = 0
        for i in range(batch_size):
            self.assertTrue(torch.all(torch.isclose(mus[i], batch_mus[beg_offset[i]:end_offset[i]])))

    def test_batch_1(self):
        # prepare problems and solve
        batch_size = 100
        edges_l = [LBPEdges(MISFactorGraph(random_mis(np.random.choice(range(10,20))))) for _ in range(batch_size)]

        t0 = time.time()
        batch = LBPEdges.batch(edges_l)
        t0 = time.time() - t0

        # alternate version
        t1 = time.time()
        batch1 = LBPEdges.batch_alt(edges_l)
        t1 = time.time() - t1
        self.assertTrue(torch.all(batch.r_edges == batch1.r_edges))
        self.assertTrue(torch.all(batch.q_edges == batch1.q_edges))
        self.assertTrue(torch.all(batch.f2v_edges == batch1.f2v_edges))
        self.assertTrue(torch.all(batch.m_edges == batch1.m_edges))
        self.assertTrue(torch.all(batch.vm_edges == batch1.vm_edges))
        self.assertTrue(torch.all(batch.m_stride == batch1.m_stride))

        self.assertTrue(torch.all(batch.n_vars == batch1.n_vars))
        self.assertTrue(torch.all(batch.n_ss == batch1.n_ss))
        self.assertTrue(torch.all(batch.n_factors == batch1.n_factors))
        self.assertTrue(torch.all(batch.n_f2v_ss == batch1.n_f2v_ss))
        self.assertTrue(torch.all(batch.n_f2v_msg == batch1.n_f2v_msg))

    def test_mis_batch_lbp(self):
        # create a list of problems
        batch_size = 10
        max_iter = 10
        pbs = [random_mis(np.random.choice(range(10,20))) for _ in range(batch_size)]
        # batch mis problems
        batch_pb = MIS.batch(pbs)
        batch_fg = MISFactorGraph(batch_pb)
        batch_edges = LBPEdges(batch_fg)
        batch_theta = torch.randn(batch_fg.nss())
        batch_v2f = torch.zeros_like(batch_theta)
        batch_mu = loopy_belief_propagation(batch_theta, batch_v2f, batch_edges, max_iter=max_iter, tol=1e-6, damping=0.).mu

        offset = 0
        for i,pb in enumerate(pbs):
            fg = MISFactorGraph(pb)
            edges = LBPEdges(fg)
            theta = batch_theta[offset:offset+fg.nss()]
            v2f = torch.zeros_like(theta)
            mu = loopy_belief_propagation(theta, v2f, edges, max_iter=max_iter, tol=1e-6, damping=0.).mu

            self.assertTrue(torch.all(torch.isclose(mu, batch_mu[offset:offset+fg.nss()])))
            offset += fg.nss()

    def test_mis_cc_batch_lbp(self):
        # create a list of problems
        batch_size = 4
        max_iter = 5
        #pbs = [random_mis(15, p=0.10, cliques=True) for _ in range(batch_size)]
        pbs = [random_mis(np.random.choice(range(20,40)), p=0.10, cliques=True) for _ in range(batch_size)]

        # batch mis problems
        batch_pb = MIS.batch(pbs)
        batch_fg = MISFactorGraph(batch_pb)
        batch_edges = LBPEdges(batch_fg)
        batch_theta = torch.randn(batch_fg.nss())
        batch_v2f = torch.zeros_like(batch_theta)
        batch_mu = loopy_belief_propagation(batch_theta, batch_v2f, batch_edges, max_iter=max_iter, tol=1e-6, damping=0.).mu

        offset = 0
        for i,pb in enumerate(pbs):
            fg = MISFactorGraph(pb)
            edges = LBPEdges(fg)
            theta = batch_theta[offset:offset+fg.nss()]
            v2f = torch.zeros_like(theta)
            mu = loopy_belief_propagation(theta, v2f, edges, max_iter=max_iter, tol=1e-6, damping=0.).mu

            self.assertTrue(torch.all(torch.isclose(mu, batch_mu[offset:offset+fg.nss()])))
            offset += fg.nss()




if __name__=="__main__":
    unittest.main()
