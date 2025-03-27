"""
FactorGraph class definition
"""
import numpy as np
import networkx as nx
from networkx import Graph

from bpropy.utils.set_sort import lenlex_sort

class FactorGraph(Graph):
    def __init__(self):
        super().__init__()
    def add_variable(self, n):
        """Add a variable node to the graph
        """
        self.add_node(n, bipartite=0)
    def add_variables_from(self, nl):
        """Add multiple variable nodes from list
        """
        self.add_nodes_from(nl, bipartite=0)
    def add_factor(self, nl):
        """Add a factor node connected to nodes in `nl`
        """
        f = self.number_of_nodes()
        self.add_node(f, bipartite=1)
        for n in nl:
            self.add_edge(n, f)
    def add_factors_from(self, nl):
        """Add multiple factors from list of adjacent nodes `nl`. `nl` should be a list of lists of nodes. Factor will first be sorted, first by increasing size and then by lexicographical order.
        """
        for n in lenlex_sort(nl):
            self.add_factor(n)
    def iter_variables(self):
        """Iterate over variable nodes, always iterates in increasing label value
        """
        for i in range(self.number_of_nodes()):
            if self.nodes[i]["bipartite"]==0:
                yield i
    def iter_factors(self):
        """Iterate over factor nodes, always iterates in increasing label value
        """
        for i in range(self.number_of_nodes()):
            if self.nodes[i]["bipartite"]==1:
                yield i
    def factor_variables(self, F):
        """Return list of variable nodes adjacent to factor node `F`. Always returns a sorted list.
        """
        return sorted(list(self.neighbors(F)))
    def factor_size(self, F):
        """Return number of variable nodes adjacent to factor node `F`
        """
        return len(self.factor_variables(F))
    def number_sufficient_statistics(self):
        """Number of sufficient statistics needed for the Over-Complete representation
        """
        return sum([2**self.factor_size(F) for F in self.iter_factors()])
    def number_of_variables(self):
        """Return number of variables
        """
        return len(list(self.iter_variables()))
    def nv(self):
        """Convenience function for getting number of variables
        """
        return self.number_of_variables()
    def number_of_factors(self):
        """Returns number of factor nodes
        """
        return len(list(self.iter_factors()))
    def nf(self):
        """Convenience function for getting number of factor nodes
        """
        return self.number_of_factors()
    def factor_sufficient_statistics(self, F, xF):
        """Return sufficient statistics corresponding to assignment xF
        """
        nF = self.factor_size(F)
        ss = np.zeros(2**nF, dtype=np.uint32)
        ss[np.packbits(xF, bitorder="little")] = 1
        return ss
    def factor_assignment(self, F, ssF):
        """Returns variable assignments corresponding to the values of sufficient statistics of F
        """
        nF = self.factor_size(F)
        return np.unpackbits(np.argwhere(ssF).astype(np.uint8), count=nF, bitorder="little")



    # Loopy Belief Propagation computation graph edges, will be moved
    def f2v_messages_dim(self):
        return sum([2*self.factor_size(F) for F in self.iter_factors()])
    def number_vm_edges(self):
        return sum([2*len(list(self.neighbors(n))) for n in self.iter_variables()])

