import unittest
import networkx as nx
import numpy as np

from bpropy.co.mis import MIS, mis_batch, mis_equal
from bpropy.co.generators.mis import random_mis
from bpropy.clique_cover import clique_cover
from bpropy.mrf.lbp_edges import LBPEdges
from bpropy.mrf.mis_factor_graph import MISFactorGraph
from bpropy.mrf.factor_graph import FactorGraph

class TestMISBatch(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_pb_1 = random_mis(100)
        self.random_pb_2 = random_mis(100)
        self.random_pb_cc = random_mis()
        self.random_pb_cc.C = clique_cover(self.random_pb_cc.g)

    def test_mis_batch(self):
        pb = mis_batch([self.random_pb_1, self.random_pb_2, self.random_pb_cc])
        pb1 = pb.get_problem(0)
        pb2 = pb.get_problem(1)
        pb3 = pb.get_problem(2)
        self.assertTrue(mis_equal(pb1, self.random_pb_1))
        self.assertTrue(mis_equal(pb2, self.random_pb_2))
        self.assertTrue(mis_equal(pb3, self.random_pb_cc))
        x = pb.greedy()

        offset = np.roll(pb.var_sizes.cumsum(),1)
        offset[0] = 0

        self.assertTrue(pb1.check_solution(x.x[offset[0]:offset[0]+pb.var_sizes[0]]))
        self.assertTrue(pb2.check_solution(x.x[offset[1]:offset[1]+pb.var_sizes[1]]))
        self.assertTrue(pb3.check_solution(x.x[offset[2]:offset[2]+pb.var_sizes[2]]))
    def test_mis_big_batch(self):
        pbs = [random_mis(np.random.choice(range(10,100))) for _ in range(10)]
        pb = mis_batch(pbs)
        x = pb.greedy()
        offset = np.roll(pb.var_sizes.cumsum(),1)
        offset[0] = 0

        for i in range(10):
            pb_ = pb.get_problem(i)
            self.assertTrue(mis_equal(pb_, pbs[i]))
            self.assertTrue(pbs[i].check_solution(x.x[offset[i]:offset[i]+pb.var_sizes[i]]))

    def test_mis_batch_factor_graph(self):
        #pbs = [random_mis(2,1.0), random_mis(3,1.0), random_mis(4,1.0)]
        pbs = [random_mis(np.random.choice(range(10,100))) for _ in range(10)]

        pb_batch = mis_batch(pbs)

        fg_batch = FactorGraph(pb_batch)
        edges = []
        for i in range(len(pbs)):
            ofg = FactorGraph(pbs[i])

            pb = pb_batch.get_problem(i)
            fg_pb = FactorGraph(pb)

            sg = fg_batch.get_subgraph(i)
            self.assertTrue(nx.utils.misc.graphs_equal(fg_pb, fg_batch.get_subgraph(i)))
            self.assertTrue(nx.utils.misc.graphs_equal(fg_pb, ofg))


    def test_mis_batch_lbp_edges(self):
        pbs = [random_mis(np.random.choice(range(10,50)), .15, cliques=True) for _ in range(3)]

        pb_batch = mis_batch(pbs)

        fg_batch = FactorGraph(pb_batch)
        edges_batch = LBPEdges(fg_batch)

        edges = []
        for i in range(len(pbs)):
            ofg = FactorGraph(pbs[i])
            oed = LBPEdges(ofg)
            pb = pb_batch.get_problem(i)


            fg_pb = FactorGraph(pb)

            fg_sg = fg_batch.get_subgraph(i)
            esg = LBPEdges(fg_sg)
            edges.append(LBPEdges(fg_pb))
            self.assertTrue(nx.utils.misc.graphs_equal(ofg, fg_pb))
            self.assertTrue(nx.utils.misc.graphs_equal(fg_sg, fg_pb))
            self.assertTrue(nx.utils.misc.graphs_equal(fg_sg, ofg))

            self.assertTrue(edges[-1] == oed)

        edges_sg = [LBPEdges(fg_batch.get_subgraph(i)) for i in range(len(pbs))]
        edges_b = LBPEdges.batch(edges)
        self.assertTrue(edges_batch == edges_b)



if __name__=="__main__":
    unittest.main()

