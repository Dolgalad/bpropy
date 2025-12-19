import networkx as nx
import numpy as np

import numpy as np
import networkx as nx

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None


def _as_max_objective(pb) -> np.ndarray:
    c = np.asarray(pb.objective_coeffs(), dtype=float).reshape(-1)
    return c if pb.objective_sense == "max" else -c


def variable_constraint_graph(pb, *, g=None):
    """
    Build VC graph using ONLY LPProblem public API:
      - pb.nv()
      - pb.objective_coeffs(), pb.objective_sense
      - pb.linear_constraints_matrix()
      - pb.variable_bounds() (optional)

    Returns: (g, features_dict)
      features_dict contains:
        - "node_x": np.ndarray [num_nodes, node_feat_dim]
        - "edge_e": dict[(u,v)] -> np.ndarray [edge_feat_dim]
    """
    if g is None:
        g = nx.Graph()

    n = int(pb.nv())
    A, lb, ub = pb.linear_constraints_matrix()
    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)
    m = int(lb.size)

    if ub.size != m:
        raise ValueError("lb/ub size mismatch in linear_constraints_matrix().")

    cmax = _as_max_objective(pb)
    if cmax.size != n:
        raise ValueError("objective_coeffs size mismatch with nv().")

    vb = pb.variable_bounds()
    if vb is None:
        lbx = np.full(n, -np.inf, dtype=float)
        ubx = np.full(n, np.inf, dtype=float)
    else:
        lbx, ubx = vb
        lbx = np.asarray(lbx, dtype=float).reshape(-1)
        ubx = np.asarray(ubx, dtype=float).reshape(-1)

    # ---- define feature dimensions ----
    # variable node: [cmax, lbx, ubx]
    var_feat_dim = 3
    # constraint node: [lb, ub, has_lb, has_ub, is_eq, nnz, row_l1, row_l2]
    con_feat_dim = 8
    node_x = np.zeros((n + m, max(var_feat_dim, con_feat_dim)), dtype=float)

    # ---- add variable nodes + features ----
    for j in range(n):
        g.add_node(j, bipartite=0)
        node_x[j, 0:3] = [cmax[j], lbx[j], ubx[j]]

    # ---- handle A (dense/sparse) ----
    A_is_sparse = (sp is not None) and sp.issparse(A)
    if A_is_sparse:
        A_csr = A.tocsr()
    else:
        A = np.asarray(A, dtype=float)
        if A.ndim == 1:
            A = A.reshape(1, -1)
        if A.shape[1] != n:
            raise ValueError("A has wrong number of columns.")
        if A.shape[0] != m:
            # if someone returns m from lb/ub but A rows differ, fail early
            raise ValueError("A has wrong number of rows versus lb/ub.")

    edge_e = {}

    # ---- add constraint nodes + features + edges ----
    for i in range(m):
        ci = n + i
        g.add_node(ci, bipartite=1)

        has_lb = float(np.isfinite(lb[i]))
        has_ub = float(np.isfinite(ub[i]))
        is_eq = float(has_lb and has_ub and abs(lb[i] - ub[i]) <= 1e-12)

        if A_is_sparse:
            start, end = A_csr.indptr[i], A_csr.indptr[i + 1]
            cols = A_csr.indices[start:end]
            vals = A_csr.data[start:end]
            nnz = float(end - start)
            row_l1 = float(np.sum(np.abs(vals))) if nnz else 0.0
            row_l2 = float(np.sqrt(np.sum(np.square(vals)))) if nnz else 0.0

            node_x[ci, 0:8] = [lb[i], ub[i], has_lb, has_ub, is_eq, nnz, row_l1, row_l2]

            for j, aij in zip(cols, vals):
                if aij == 0:
                    continue
                u, v = int(j), int(ci)
                g.add_edge(u, v)
                edge_e[(u, v)] = np.array([float(aij)], dtype=float)  # edge feature dim = 1

        else:
            row = A[i, :]
            nz = np.nonzero(row)[0]
            nnz = float(nz.size)
            vals = row[nz] if nz.size else np.array([], dtype=float)
            row_l1 = float(np.sum(np.abs(vals))) if nnz else 0.0
            row_l2 = float(np.sqrt(np.sum(np.square(vals)))) if nnz else 0.0

            node_x[ci, 0:8] = [lb[i], ub[i], has_lb, has_ub, is_eq, nnz, row_l1, row_l2]

            for j in nz:
                aij = float(row[j])
                u, v = int(j), int(ci)
                g.add_edge(u, v)
                edge_e[(u, v)] = np.array([aij], dtype=float)

    return g, {"node_x": node_x, "edge_e": edge_e}

class VCGraph(nx.Graph):
    def __init__(self, pb):
        super().__init__()
        self.graph["problem_type"] = pb.__class__.__name__  # "MIS" | "SCP" | ...
        _,self.features = variable_constraint_graph(pb, g=self)

    def iter_variables(self):
        for i in sorted(
            n for n, d in self.nodes(data=True)
            if d.get("bipartite") == 0
        ):
            yield i

    def iter_constraints(self):
        for i in sorted(
            n for n, d in self.nodes(data=True)
            if d.get("bipartite") == 1
        ):
            yield i


    #def iter_variables(self):
    #    for i, d in self.nodes(data=True):
    #        if d.get("bipartite") == 0:
    #            yield i
    #
    #def iter_constraints(self):
    #    for i, d in self.nodes(data=True):
    #        if d.get("bipartite") == 1:
    #            yield i
    #def iter_variables(self):
    #    """Iterate over variable nodes, always iterates in increasing label value
    #    """
    #    for i in range(self.number_of_nodes()):
    #        if self.nodes[i]["bipartite"]==0:
    #            yield i
    #def iter_constraints(self):
    #    """Iterate over constraint nodes, always iterates in increasing label value
    #    """
    #    for i in range(self.number_of_nodes()):
    #        if self.nodes[i]["bipartite"]==1:
    #            yield i
    def constraint_variables(self, C):
        """Return list of variable nodes adjacent to constraint node `C`. Always returns a sorted list.
        """
        return sorted(list(self.neighbors(C)))
    def constraint_size(self, C):
        """Return number of variable nodes adjacent to constraint node `C`
        """
        return len(self.constraint_variables(C))
    def number_of_variables(self):
        """Return number of variables
        """
        return len(list(self.iter_variables()))
    def nv(self):
        """Convenience function for getting number of variables
        """
        return self.number_of_variables()
    def number_of_constraints(self):
        """Returns number of constraint nodes
        """
        return len(list(self.iter_constraints()))
    def nc(self):
        """Convenience function for getting number of constraint nodes
        """
        return self.number_of_constraints()
