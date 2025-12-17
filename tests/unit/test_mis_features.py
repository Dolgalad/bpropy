import unittest

from bpropy.co.mis import random_mis
from bpropy.datasets.mis_features import *
from bpropy.clique_cover import is_edge_cover

class TestMISFeatures(unittest.TestCase):
    def test_eb_features(self):
        pb = random_mis(50)
        C,nfeat,efeat,cfeat = get_features(pb, "EB")

        self.assertTrue(len(C) == pb.ne())
        self.assertTrue(is_edge_cover(C, pb.g))
        #self.assertTrue(C == pb.C)
        self.assertTrue(np.all([np.all(c1==c2) for c1,c2 in zip(C, pb.C)]))
        self.assertTrue(nfeat.shape[0] == pb.nv())
        self.assertTrue(efeat.shape[0] == 4*pb.ne())
        self.assertTrue(cfeat.shape[0] == pb.ne())

    def test_cc_features(self):
        pb = random_mis(50)
        C,nfeat,efeat,cfeat = get_features(pb, "CC")

        self.assertTrue(is_edge_cover(C, pb.g))
        #self.assertTrue(C == pb.C)
        self.assertTrue(np.all([np.all(c1==c2) for c1,c2 in zip(C, pb.C)]))
        self.assertTrue(nfeat.shape[0] == pb.nv())
        self.assertTrue(efeat.shape[0] == 2*sum(len(c) for c in C))
        self.assertTrue(cfeat.shape[0] == len(C))

    def test_cf_features(self):
        pb = random_mis(50)
        C,nfeat,efeat,cfeat = get_features(pb, "CF")

        self.assertTrue(is_edge_cover(C, pb.g))
        self.assertTrue(np.all([np.all(c1==c2) for c1,c2 in zip(C, pb.C)]))
        self.assertTrue(nfeat.shape[0] == pb.nv())
        self.assertTrue(efeat.shape[0] == 4*pb.ne())
        self.assertTrue(cfeat.shape[0] == len(C))

    def test_ec_features(self):
        pb = random_mis(50)
        C,nfeat,efeat,cfeat = get_features(pb, "EC")

        self.assertTrue(np.all([np.all(c1==c2) for c1,c2 in zip(C, pb.C)]))
        self.assertTrue(is_edge_cover(C, pb.g))
        self.assertTrue(nfeat.shape[0] == pb.nv())
        self.assertTrue(efeat.shape[0] == 2*sum(len(c) for c in C))
        self.assertTrue(cfeat.shape[0] == len(C))

    def test_cp_features(self):
        pb = random_mis(50)
        C,nfeat,efeat,cfeat = get_features(pb, "CP")

        #self.assertTrue(C == pb.C)
        self.assertTrue(np.all([np.all(c1==c2) for c1,c2 in zip(C, pb.C)]))
        self.assertTrue(is_edge_cover(C, pb.g))
        self.assertTrue(nfeat.shape[0] == pb.nv())
        self.assertTrue(efeat.shape[0] == 2*sum(len(c) for c in C))
        self.assertTrue(cfeat.shape[0] == len(C))



if __name__=="__main__":
    unittest.main()
