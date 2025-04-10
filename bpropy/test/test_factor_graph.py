import unittest
import numpy as np

from bpropy.mrf.factor_graph import FactorGraph
from bpropy.mrf.sufficient_statistics import x2ss, ss2x

def simple_factor_graph():
    fg = FactorGraph()
    fg.add_variables_from([0,1,2])
    fg.add_factors_from([[0,1], [1,2], [0,2]])
    return fg


class TestFactorGraph(unittest.TestCase):
    def test_factor_graph_init(self):
        fg = simple_factor_graph()
        self.assertTrue(fg.nv() == 3)
        self.assertTrue(fg.nf() == 3)

        self.assertTrue(fg.number_sufficient_statistics() == 12)
    def test_factor_neighbors(self):
        fg = simple_factor_graph()

        for F in fg.iter_factors():
            self.assertTrue(fg.factor_size(F) == 2)
            F_vars = fg.factor_variables(F)
            self.assertTrue(F_vars == sorted(F_vars))
    def test_sufficient_statistics(self):
        fg = simple_factor_graph()

        for _ in range(5):
            x = np.random.choice([0,1], 3)
            x_ss = x2ss(x, fg)
            self.assertTrue(fg.number_sufficient_statistics() == x_ss.shape[0])
            self.assertTrue(np.all(ss2x(x_ss, fg) == x))


if __name__=="__main__":
    unittest.main()


