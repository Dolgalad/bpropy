"""
FactorGraph class definition
"""
import numpy as np
import networkx as nx
from networkx import Graph

from typing import Optional

from bpropy.utils.set_sort import lenlex_sort

try:
    import scipy.sparse as sp
except Exception:
    sp = None

class FactorGraph(Graph):
    def __init__(
            self,
            pb=None,
            *,
            problems=None,
            add_unaries: bool = True,
        ):
        super().__init__()
        self.num_vars = np.array([], dtype=int)
        self.num_factors = np.array([], dtype=int)
        self.num_graphs = 0

          # --- builder logic ---
        if pb is not None:
            if hasattr(pb, "get_problem") and hasattr(pb, "num_subproblems") and pb.num_subproblems > 1:
                for i in range(pb.num_subproblems):
                    self.append_instance(pb.get_problem(i), add_unaries=add_unaries)
            else:
                self.append_instance(pb, add_unaries=add_unaries)

        if problems is not None:
            for p in problems:
                if hasattr(p, "get_problem") and hasattr(p, "num_subproblems") and p.num_subproblems > 1:
                    for i in range(p.num_subproblems):
                        self.append_instance(p.get_problem(i), add_unaries=add_unaries)
                else:
                    self.append_instance(p, add_unaries=add_unaries)

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
    def factor_sizes(self):
        """Return all factor sizes
        """
        return np.array([self.factor_size(F) for F in self.iter_factors()], dtype=int)
    def containing_factors(self, i):
        return sorted(list(self.neighbors(i)))
    def factor_index_map(self):
        return {F:iF for iF,F in enumerate(self.iter_factors())} 
    def factor_variables_map(self):
        return {F:self.factor_variables(F) for F in self.iter_factors()}
    def containing_factors_map(self):
        return {v:self.containing_factors(v) for v in self.iter_variables()}
    def number_sufficient_statistics(self):
        """Number of sufficient statistics needed for the Over-Complete representation
        """
        return sum([2**self.factor_size(F) for F in self.iter_factors()])
    def nss(self):
        return self.number_sufficient_statistics()
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
        idx = int(sum((int(b) & 1) << k for k,b in enumerate(xF)))
        ss = np.zeros(1 << nF, dtype=np.uint32)
        ss[idx] = 1
        return ss

        ss = np.zeros(2**nF, dtype=np.uint32)
        ss[np.packbits(xF, bitorder="little")] = 1
        return ss
    def factor_assignment(self, F, ssF):
        """Returns variable assignments corresponding to the values of sufficient statistics of F
        """
        

        nF = self.factor_size(F)
        idx = int(np.argmax(ssF))
        bits = np.array([(idx >> k) & 1 for k in range(nF)], dtype=np.uint8)
        return bits

        return np.unpackbits(np.argwhere(ssF).astype(np.uint8), count=nF, bitorder="little")
    def ne(self):
        """Return number of edges
        """
        return self.number_of_edges()
    def num_f2v_edges(self):
        fs = self.factor_sizes()
        return (np.power(2,fs) * np.power(fs,2)).sum()
    def num_r_edges(self):
        fs = self.factor_sizes()
        return (np.power(2,fs) * fs).sum()
    def num_q_edges(self):
        res=0
        for F in self.iter_factors():
            for v in self.factor_variables(F):
                res += 2*(len(self.containing_factors(v))-1)
        return res
    def num_m_edges(self):
        return self.num_r_edges()
    def num_vm_edges(self):
        return (2 * np.array([len(self.containing_factors(v)) for v in self.iter_variables()], dtype=int)).sum()


    # Loopy Belief Propagation computation graph edges, will be moved
    def f2v_messages_dim(self):
        return sum([2*self.factor_size(F) for F in self.iter_factors()])
    def number_vm_edges(self):
        return sum([2*len(list(self.neighbors(n))) for n in self.iter_variables()])

    def append_instance(
        self,
        pb,  # LPProblem
        *,
        add_unaries: bool = True,
        include_empty_constraints: bool = False,
        unique_scopes: bool = False,
        A: Optional[object] = None,
    ) -> None:
        """
        Append one LPProblem instance to this (possibly batched) factor graph.

        Factors are derived from the support of the constraint matrix:
            scope(row i) = { j | A[i,j] != 0 }.

        Unary factors can be toggled with add_unaries.

        Parameters
        ----------
        pb : LPProblem
            Must expose constraint matrix and number of variables.
        add_unaries : bool
            If True, add one unary factor per variable.
        include_empty_constraints : bool
            If False, drop constraints with empty scope.
        unique_scopes : bool
            If True, deduplicate identical scopes (optional).
        A : optional
            Override constraint matrix (otherwise taken from pb).
        """
        # --- 1) get matrix A and n_vars ---
        if A is None:
            A,_,_ = pb.linear_constraints_matrix()

        # number of variables
        if hasattr(pb, "nv"):
            n_vars = int(pb.nv())
        elif hasattr(pb, "n"):
            n_vars = int(pb.n)
        else:
            # fallback from matrix shape
            n_vars = int(A.shape[1])

        # --- 2) extract scopes from nonzeros per row ---
        scopes: list[list[int]] = []
        if sp is not None and sp.issparse(A):
            A_csr = A.tocsr()
            for i in range(A_csr.shape[0]):
                start, end = A_csr.indptr[i], A_csr.indptr[i + 1]
                cols = A_csr.indices[start:end]
                if cols.size == 0 and not include_empty_constraints:
                    continue
                scopes.append([int(j) for j in cols])
        else:
            A_arr = np.asarray(A)
            for i in range(A_arr.shape[0]):
                cols = np.flatnonzero(A_arr[i] != 0)
                if cols.size == 0 and not include_empty_constraints:
                    continue
                scopes.append([int(j) for j in cols])

        if unique_scopes:
            seen = set()
            uniq = []
            for s in scopes:
                t = tuple(sorted(s))
                if t not in seen:
                    seen.add(t)
                    uniq.append(list(t))
            scopes = uniq

        # --- 3) add variables + factors with offsets; maintain ordering per instance ---
        # This assumes you have (or will add) a helper that appends using local scopes:
        self.append_instance_from_scopes(n_vars, scopes, add_unaries=add_unaries)

    def append_instance_from_scopes(self, n_vars: int, scopes: list[list[int]], *, add_unaries: bool = True) -> None:
        base = self.number_of_nodes()

        # variable nodes
        self.add_nodes_from(range(base, base + n_vars), bipartite=0)

        # build per-instance factor scopes (LOCAL), then sort per-instance
        local_scopes = []
        if add_unaries:
            local_scopes.extend([[i] for i in range(n_vars)])
        local_scopes.extend([sorted(map(int, s)) for s in scopes])

        # IMPORTANT: keep your invariant: size then lex, within the instance
        local_scopes.sort(key=lambda s: (len(s), s))

        # offset to GLOBAL ids
        global_scopes = [[base + v for v in s] for s in local_scopes]

        # add factors in given order (make sure add_factors_from doesn’t “global resort” old factors)
        self.add_factors_from(global_scopes)

        # bookkeeping if you track batching
        self.num_graphs += 1
        self.num_vars = np.append(self.num_vars, n_vars)
        self.num_factors = np.append(self.num_factors, len(local_scopes))

    def get_subgraph(self, idx: int) -> "FactorGraph":
        """
        Return the idx-th instance factor graph from a batched FactorGraph,
        relabeled so node ids start at 0.
        """
        if not hasattr(self, "num_graphs") or self.num_graphs == 0:
            raise ValueError("This FactorGraph does not look batched (num_graphs missing/0).")

        if idx < 0 or idx >= int(self.num_graphs):
            raise IndexError(f"idx={idx} out of range for num_graphs={self.num_graphs}")

        # each instance contributes (num_vars[idx] variable nodes + num_factors[idx] factor nodes)
        sizes = self.num_vars + self.num_factors
        offsets = np.concatenate(([0], np.cumsum(sizes[:-1])))

        start = int(offsets[idx])
        n_nodes = int(sizes[idx])
        nodes = list(range(start, start + n_nodes))

        g = nx.subgraph(self, nodes).copy()
        mapping = {n: n - start for n in nodes}
        g = nx.relabel_nodes(g, mapping)

        fg = FactorGraph()
        fg.add_nodes_from(g.nodes(data=True))
        fg.add_edges_from(g.edges())

        fg.num_graphs = 1
        fg.num_vars = np.array([int(self.num_vars[idx])], dtype=int)
        fg.num_factors = np.array([int(self.num_factors[idx])], dtype=int)
        return fg

