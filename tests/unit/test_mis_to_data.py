import unittest
import time
import importlib.util

import numpy as np
import networkx as nx

from bpropy.co.mis import MIS, mis_equal
from bpropy.co.generators.mis import random_mis
from bpropy.clique_cover import clique_cover

class TestMIStoData(unittest.TestCase):
    def test_mis_to_data_and_back(self):
        pb = random_mis(100)
        data = MIS.to_data(pb)
        pb_1 = MIS.from_data(data)

        self.assertTrue(mis_equal(pb, pb_1))


if __name__=="__main__":
    unittest.main()

