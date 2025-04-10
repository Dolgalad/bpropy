import unittest
import torch
import numpy as np

from torch_geometric.data import Batch

from bpropy.co.mis import mis_equal
from bpropy.datasets.mis_data import mis_to_data, data_to_mis
from bpropy.mrf.sufficient_statistics import x2ss, ss2x
from bpropy.mrf.mis_factor_graph import MISFactorGraph

from bpropy.test.test_mis import simple_mis, random_mis

class TestMISData(unittest.TestCase):
    def test_mis_to_data_and_back(self):
        pb = random_mis(100)
        data = mis_to_data(pb)
        self.assertTrue(data.edge_index.max().item()+1 == pb.nv()+pb.ne())
        self.assertTrue(data.edge_index.shape[1] == 4*pb.ne())
        nV = (data.bipartite == 0).sum().item()
        nF = (data.bipartite == 1).sum().item()
        self.assertTrue(nV == pb.nv())
        self.assertTrue(nF == pb.ne())
        self.assertTrue(data.edge_index.shape[1] == data.e.shape[0])
        
        pb1 = data_to_mis(data)
        self.assertTrue(pb.nv() == pb1.nv())
        self.assertTrue(pb.ne() == pb1.ne())
        for e in pb.g.edges:
            self.assertTrue(pb1.g.has_edge(*e))

    def test_mis_data_batch(self):
        pb1 = random_mis(100)
        pb2 = random_mis(150)
        pbs = [pb1, pb2]
        data1 = mis_to_data(pb1)
        data1.x = torch.zeros(pb1.nv()+pb1.ne(), 3)
        data2 = mis_to_data(pb2)
        data2.x = torch.zeros(pb2.nv()+pb2.ne(), 3)

        batch = Batch.from_data_list([data1,data2])
        self.assertTrue(batch.num_graphs == 2)
        self.assertTrue(batch.bipartite.sum().item() == pb1.ne()+pb2.ne())

        batch_pbs = data_to_mis(batch)
        self.assertTrue(len(batch_pbs) == batch.num_graphs)
        for i in range(batch.num_graphs):
            self.assertTrue(mis_equal(pbs[i], batch_pbs[i]))
    def test_mis_data_solution(self):
        pb = random_mis(50)
        x = pb.greedy()
        data = mis_to_data(pb)
        pb1 = data_to_mis(data)
        self.assertTrue(pb1.check_solution(x))

        # CC features
        for cat in ["EB","CC","CF","EC","CP"]:
            data_cat = mis_to_data(pb, feature_category=cat)
            pb_cat = data_to_mis(data_cat)
            self.assertTrue(pb_cat.check_solution(x))
            # create factor graph
            fg = MISFactorGraph(pb_cat)
            x_ind = x2ss(x.astype(int), fg)
            y = ss2x(x_ind, fg)
            self.assertTrue(np.all(y == x))
            self.assertTrue(pb.check_solution(y))
            self.assertTrue(pb_cat.check_solution(y))
            self.assertTrue(len(pb.C) == fg.nf() - pb.nv())
            fg_factors = [fg.factor_variables(F) for F in fg.iter_factors()]
            self.assertTrue(np.all([np.all(c1==c2) for c1,c2 in zip(pb.C,fg_factors[pb.nv():])]))
            off = 0
            for F in fg.iter_factors():
                nF = fg.factor_size(F)
                self.assertTrue(x_ind[off:off+2**nF].sum() == 1)
                self.assertTrue(y[fg.factor_variables(F)].sum() <= 1)
                off += 2**nF
    def test_mis_data_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            pb = random_mis(50)
            x = pb.greedy()
            data = mis_to_data(pb)
            data = data.to(device)
            for k,v in data.to_dict().items():
                self.assertTrue(v.is_cuda)

            data = mis_to_data(pb, with_lbp_edges=True)
            data = data.to(device)
            for k,v in data.to_dict().items():
                self.assertTrue(v.is_cuda)




if __name__=="__main__":
    unittest.main()


