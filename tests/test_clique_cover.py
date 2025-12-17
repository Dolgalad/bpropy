import unittest
import networkx as nx

from bpropy.clique_cover import is_edge_cover, clique_cover

class TestCliqueCover(unittest.TestCase):
    def test_is_edge_cover(self):
        g = nx.Graph()
        g.add_nodes_from([0,1,2,3])
        g.add_edges_from([[0,1],[1,2],[2,3],[0,3],[0,2]])

        cc = [[0,1,2], [0,2,3]]
        cc_bad = [[0,1],[1,2],[2,3],[0,3]]
        self.assertTrue(is_edge_cover(cc, g))
        self.assertFalse(is_edge_cover(cc_bad, g))
    def test_greedy_clique_cover(self):
        g = nx.generators.random_graphs.erdos_renyi_graph(100, 0.15)
        cc = clique_cover(g)
        self.assertTrue(is_edge_cover(cc, g))


if __name__=="__main__":
    unittest.main()
