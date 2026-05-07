# bpropy/co/mis.py
from __future__ import annotations

import os
import tempfile
import itertools
from typing import Dict, Any, List, Literal, Optional, Tuple

import numpy as np
import networkx as nx
import mip
from mip import xsum, BINARY
from scipy.optimize import LinearConstraint, Bounds, milp

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

import importlib.util
docplex_spec = importlib.util.find_spec("docplex")
if docplex_spec is not None:
    from docplex.mp.model import Model

from bpropy.co.base_lp import LPProblem, LPSolveConfig
from bpropy.io.serde import SerializableLPMixin

from bpropy.utils.graph_utils import ordered_edges
from bpropy.utils.set_sort import lenlex_sort
from bpropy.clique_cover import is_edge_cover, clique_cover

#from bpropy.co.mis_solution import MISSolution
from bpropy.co.solution import Solution
from bpropy.co.kamis import write_metis_graph, kamis_solve, load_kamis_solution, solution_from_kamis



class MIS(LPProblem, SerializableLPMixin):
    """
    Maximum Independent Set (MIS) in clique-constraint form:
    $$
    \\max_{x\\in \\{0,1\\}^n}  \\sum_{i\\in V} x_i  \\quad \\text{s.t.} \\quad  \\sum_{i\\in c} x_i \\leq 1 \\quad \\forall c \\in \\mathcal{C}
    $$

    Parameters
    ----------
    g : nx.Graph
        Undirected graph
    cliques: List[List[int]], optional
        Set of cliques used for defining the LP constraints. Defaults to ``None`` in which case cliques defaults to set of edges.
    weights: ndarray, optional
        Vertex weights. If ``None`` use unit weights.
    num_subproblems: int
        Specifies if the problem is composed of multiple disjoint graphs. Defaults to 1. If greater than 1, ``num_subproblems`` should be set along with ``var_sizes``. Subproblem ``i`` is then defined on the subgraph of ``g`` containing vertices ``[var_sizes[:i-1].sum(), var_sizes[:i-1].sum()+var_sizes[i]]``.
    var_sizes: ndarray, optional
        Size of subproblems.

    """
    def __init__(
        self,
        g: nx.Graph,
        weights: Optional[np.ndarray] = None,
        cliques: Optional[List[List[int]]] = None,
        *,
        num_subproblems: int = 1,
        var_sizes: Optional[np.ndarray] = None,
    ):
        self.g = g

        n = g.number_of_nodes()
        self.w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float).reshape(-1)

        if self.w.size != n:
            raise ValueError(f"weights length {self.w.size} != number of nodes {n}")

        if cliques is None:
            C = [list(map(int, e)) for e in ordered_edges(self.g)]  # edges are 2-cliques
        else:
            C = [list(map(int, c)) for c in lenlex_sort(cliques)]
        self.C = [sorted(dict.fromkeys(c)) for c in C]

        if not is_edge_cover(self.C, self.g):
            raise ValueError("MIS: cliques do not cover all edges of graph")

        # batch metadata (optional)
        self._num_subproblems = int(num_subproblems)
        self._var_sizes = np.array([n], dtype=int) if var_sizes is None else np.asarray(var_sizes, dtype=int).reshape(-1)
        if self._var_sizes.sum() != n:
            # For a single graph MIS, this is usually just [n].
            # For disjoint-union batched MIS, this should sum to total nodes.
            raise ValueError("var_sizes must sum to number of nodes in g")

        self._offsets = np.concatenate([[0], np.cumsum(self._var_sizes)[:-1]])
        self.clique_indicator = self._compute_clique_indicator()

    # ---- LPProblem interface ----
    @property
    def objective_sense(self) -> Literal["min", "max"]:
        return "max"

    def ne(self):
        """
        Number of edges in ``pb.g``.

        Returns
        -------
        n : int
            Number of edges.
        """
        return nx.number_of_edges(self.g)

    def objective_coeffs(self) -> np.ndarray:
        """
        Get the LP's objective function coefficients (i.e. vertex weights)

        Returns
        -------
        w : ndarray
            Objective coefficients.
        """
        return self.w

    def nv(self) -> int:
        """
        Number of vertices in ``pb.g``.

        Returns
        -------
        n : int
            Number of vertices.
        """
        return self.g.number_of_nodes()

    def variable_bounds(self):
        """
        Get LP variable bounds.

        Returns
        -------
        lb, ub : Tuple[ndarray, ndarray]
            Variable lower and upper bound values.
        """
        n = self.nv()
        return np.zeros(n), np.ones(n)

    def linear_constraints_matrix(self, cliques=None):
        """
        Get LP constraint matrix, lower and upper bounds.

        Returns
        -------
        A, lb, ub : Tuple[ndarray, ndarray, ndarray]
            LP constraint matrix, lower and upper bound values.

        Examples
        --------
        >>> A,lb,ub = pb_house.linear_constraints_matrix()
        >>> A
        <Compressed Sparse Row sparse matrix of dtype 'float64'
        	with 12 stored elements and shape (6, 5)>
          Coords	Values
          (0, 0)	1.0
          (0, 1)	1.0
          (1, 0)	1.0
          (1, 3)	1.0
          (2, 1)	1.0
          (2, 2)	1.0
          (3, 1)	1.0
          (3, 4)	1.0
          (4, 2)	1.0
          (4, 3)	1.0
          (5, 2)	1.0
          (5, 4)	1.0
        >>> lb,ub
        [-inf -inf -inf -inf -inf -inf] [1. 1. 1. 1. 1. 1.]
        """
        n = self.nv()
        if cliques is None:
            cliques = self.C
            
        m = len(cliques)
        if m == 0:
            if sp is not None:
                A = sp.csr_matrix((0, n), dtype=float)
            else:
                A = np.zeros((0, n), dtype=float)
            lb = np.zeros(0, dtype=float) - np.inf
            ub = np.zeros(0, dtype=float)
            return A, lb, ub

        # Build sparse A where each row is a clique incidence
        if sp is not None:
            rows = []
            cols = []
            data = []
            for i, clq in enumerate(cliques):
                for u in clq:
                    rows.append(i)
                    cols.append(int(u))
                    data.append(1.0)
            A = sp.csr_matrix((data, (rows, cols)), shape=(m, n), dtype=float)
        else:
            A = np.zeros((m, n), dtype=float)
            for i, clq in enumerate(cliques):
                A[i, np.asarray(clq, dtype=int)] = 1.0

        lb = np.full(m, -np.inf, dtype=float)
        ub = np.ones(m, dtype=float)
        return A, lb, ub

    @property
    def num_subproblems(self) -> int:
        return self._num_subproblems

    @property
    def var_sizes(self) -> np.ndarray:
        return self._var_sizes.copy()

    # ---- utilities ----
    def edges(self):
        yield from ordered_edges(self.g)

    def _compute_clique_indicator(self) -> np.ndarray:
        """
        For batched MIS-as-disjoint-union graphs, map each clique to its component index
        based on the smallest vertex id in the clique and the stored offsets.
        """
        if self._num_subproblems <= 1:
            return np.zeros(len(self.C), dtype=int)

        # offsets are starts; ends are offsets+sizes
        ends = self._offsets + self._var_sizes
        ind = np.zeros(len(self.C), dtype=int)
        for i, clq in enumerate(self.C):
            v = min(clq)
            # find idx such that offsets[idx] <= v < ends[idx]
            idx = int(np.searchsorted(ends, v, side="right"))
            ind[i] = idx
        return ind

    def get_problem(self, idx: int) -> "MIS":
        """
        Get subproblem ``idx``.

        Parameters
        ----------
        idx : int
              Index of the subproblem.

        Returns
        -------
        pb : MIS
             Subproblem ``idx``.

        Examples
        --------
        >>> pb_batch = MIS.batch([pb_house,pb_tie])
        >>> pb_batch.get_problem(0).nv()
        5
        >>> pb_batch.get_problem(1).nv()
        5
        
        ![Batch subproblems](../../assets/mis_house_tie_batch_subproblems.svg)
        """
        if idx < 0 or idx >= self._num_subproblems:
            raise ValueError(f"idx={idx} out of range for num_subproblems={self._num_subproblems}")

        off = int(self._offsets[idx])
        nV = int(self._var_sizes[idx])
        nodes = list(range(off, off + nV))
        gsub = self.g.subgraph(nodes).copy()
        gsub = nx.relabel_nodes(gsub, {off + i: i for i in range(nV)})


        Csub = (np.array([np.array(c) for c in self.C], dtype=object)[self.clique_indicator == idx] - off)
        Csub = [list(map(int, c)) for c in Csub]


        wsub = self.w[off:off + nV]
        return MIS(gsub, weights=wsub, cliques=Csub)

    def solve(
        self,
        verbose: bool = False,
        cliques=None,
        solver: str = "python-mip",
        relax_integrality: bool = False,
        partial_solution: Optional[tuple] = None,
        timelimit=None,
    ) -> Solution:
    
        if cliques is None:
            cliques = self.C
    
        slv = (solver or "python-mip").lower()
    
        if timelimit is None:
            timelimit = np.inf
    
        # specialized solvers
        if slv == "greedy":
            if partial_solution is not None:
                raise ValueError("MIS greedy solver does not support partial_solution.")
            return self.greedy()
    
        if slv == "kamis":
            if partial_solution is not None:
                raise ValueError(
                    "MIS KaMIS solver does not support partial_solution directly. "
                    "Use a generic MILP solver such as --solver scip, gurobi, cbc, or python-mip "
                    "for fix-and-solve evaluation."
                )
    
            with tempfile.TemporaryDirectory() as d:
                graph_path = os.path.join(d, "graph.metis")
                solution_path = os.path.join(d, "solution")
                write_metis_graph(graph_path, self.g)
                kamis_solve(graph_path, solution_path, verbose=verbose)
                return solution_from_kamis(solution_path)
    
        if slv == "cplex":
            if partial_solution is not None:
                # Let the generic LPProblem path handle bound fixing.
                # The specialized docplex path below does not currently apply partial_solution.
                A, lb, ub = self.linear_constraints_matrix(cliques=cliques)
                return super().solve(
                    config=LPSolveConfig(
                        solver=slv,
                        relax_integrality=relax_integrality,
                        verbose=verbose,
                        timelimit=timelimit,
                    ),
                    override_linear_constraints=(A, lb, ub),
                    partial_solution=partial_solution,
                )
    
            if docplex_spec is None:
                raise RuntimeError("CPLEX solver not found. Missing package 'docplex'")
            md = cplex_model(self, continuous_relaxation=relax_integrality, cliques=cliques)
            sol = md.solve()
            return Solution.from_solver("cplex", sol=sol, md=md)
    
        # generic MILP solvers, with optional partial solution fixing
        A, lb, ub = self.linear_constraints_matrix(cliques=cliques)
    
        return super().solve(
            config=LPSolveConfig(
                solver=slv,
                relax_integrality=relax_integrality,
                verbose=verbose,
                timelimit=timelimit,
            ),
            override_linear_constraints=(A, lb, ub),
            partial_solution=partial_solution,
        )
    def constraints_for_scipy(self, cliques=None):
        # SciPy can take a single LinearConstraint with sparse A; better than list-of-rows.
        if cliques is None or cliques is self.C:
            A, lb, ub = self.linear_constraints_matrix()
            return [LinearConstraint(A, lb=lb, ub=ub)]

        # otherwise build from cliques explicitly
        n = self.nv()
        if sp is not None:
            rows, cols, data = [], [], []
            for i, clq in enumerate(cliques):
                for u in clq:
                    rows.append(i); cols.append(int(u)); data.append(1.0)
            A = sp.csr_matrix((data, (rows, cols)), shape=(len(cliques), n), dtype=float)
        else:
            A = np.zeros((len(cliques), n), dtype=float)
            for i, clq in enumerate(cliques):
                A[i, np.asarray(clq, dtype=int)] = 1.0
        lb = np.full(len(cliques), -np.inf, dtype=float)
        ub = np.ones(len(cliques), dtype=float)
        return [LinearConstraint(A, lb=lb, ub=ub)]

    # ---- batching (MIS override: disjoint union graph) ----
    @classmethod
    def batch(cls, problems: List["MIS"]) -> "MIS":
        """
        Efficient MIS batching: disjoint-union of graphs (components).
        This is equivalent to block-diagonal constraints but preserves native structure.

        Parameters
        ----------
        problems : List[MIS]
            List of ``MIS`` instances to batch.

        Returns
        -------
        pb : MIS
            Batched ``MIS`` instance.

        Examples
        --------
        >>> pb_batch = MIS.batch([pb_house,pb_tie])
        >>> pb_batch.nv()
        10

        ![Batched house and tie](../../assets/mis_house_tie_batch.svg)

        """
        g = nx.Graph()
        C = []
        w = []
        sizes = []
        offset = 0

        for pb in problems:
            n = pb.nv()
            sizes.append(n)
            w.append(np.asarray(pb.w, dtype=float).reshape(-1))

            # nodes + edges with offset
            g.add_nodes_from(range(offset, offset + n))
            g.add_edges_from([(u + offset, v + offset) for (u, v) in pb.g.edges()])

            # cliques with offset
            C.extend([[int(u) + offset for u in c] for c in pb.C])

            offset += n

        wcat = np.concatenate(w) if w else np.zeros(0, dtype=float)
        return cls(g, weights=wcat, cliques=C, num_subproblems=len(problems), var_sizes=np.array(sizes, dtype=int))

    # ---- serialization (keep your existing implementation; unchanged signature) ----
    def _to_serializable(self, use_npz: bool, arrays: Dict[str, np.ndarray], threshold_bytes: int = 200_000) -> Dict[str, Any]:
        edges = [[int(u), int(v)] for (u, v) in self.edges()]
        payload: Dict[str, Any] = {
            "nv": self.nv(),
            "weights": {"source": "inline", "key": "w"},
            "mis": {"edges": edges},
        }

        w = np.asarray(self.w, dtype=float).reshape(-1)
        if use_npz and w.nbytes > threshold_bytes:
            arrays["w"] = w
            payload["weights"] = {"source": "npz", "key": "w"}
        else:
            payload.setdefault("_arrays", {})["w"] = w.tolist()

        if hasattr(self, "C") and len(self.C) > 0:
            payload["mis"]["cliques"] = [sorted(dict.fromkeys(map(int, c))) for c in self.C]
        return payload

    @classmethod
    def _from_serializable(cls, man: Dict[str, Any], arrays: Dict[str, np.ndarray]):
        nv = int(man["nv"])

        wsrc = man["weights"]
        if wsrc["source"] == "npz":
            w = arrays[wsrc["key"]]
        else:
            w = np.array(man.get("_arrays", {}).get(wsrc["key"]), dtype=float)

        edges = man["mis"]["edges"]
        C = man["mis"].get("cliques", None)

        pb = cls.from_edges(nv=nv, edges=edges, weights=w)
        if C:
            pb.C = [list(map(int, c)) for c in C]
        return pb

    def greedy(self, 
               scores: np.ndarray | None = None,
               *,
               policy: Literal["argmax", "sample"] = "argmax",
               temperature: float = 1.0,
               rng: Optional[np.random.Generator] = None,
               ) -> Solution:
        """
        Greedy MIS heuristic with configurable vertex selection policy.
    
        At each step, pick a vertex from the remaining queue, set it to 1,
        then remove its neighbors from the queue (set them to 0).
    
        Parameters
        ----------
        pb: MIS
            MIS object with:
            - pb.nv() -> int
            - pb.g.neighbors(u) iterable
        scores : np.ndarray
            Score per vertex (higher is preferred).
        policy : {"argmax", "sample"}, default="argmax"
            Selection strategy:
            - "argmax": pick the remaining vertex with maximum score.
            - "sample": sample a vertex from remaining ones with probability
              proportional to softmax(scores/temperature).
        temperature : float, default=1.0
            Softmax temperature for policy="sample".
            Lower -> more greedy; higher -> more uniform.
        rng : numpy.random.Generator, optional
            RNG to use for sampling. If None, uses np.random.default_rng().
    
        Returns
        -------
        sol : Solution
            Binary solution vector of length pb.nv().

        Examples
        --------
        >>> sol = pb_tie.greedy()
        >>> sol.x
        [0. 0. 0. 1. 1.]
        """
        if scores is None:
            scores = np.array([-self.g.degree[n] for n in range(self.nv())], dtype=float)
        x = mis_greedy(self, scores, policy=policy, temperature=temperature, rng=rng)
        return Solution(x=x, solver="greedy")

    def set_cliques(
        self,
        cliques: Optional[Iterable[Sequence[int]]],
        *,
        sort: bool = False,
        unique: bool = False,
        drop_empty: bool = True,
        rebuild_indicator: bool = True,
    ) -> None:
        """
        Set / replace clique constraints for this MIS instance.

        Parameters
        ----------
        cliques:
            Iterable of cliques, each a sequence of vertex indices.
            Indices must be in the (global) vertex id space of this MIS object.
            If None, clears cliques.
        sort:
            If True, sort cliques by (size, lex).
        unique:
            If True, remove duplicate cliques after canonicalization.
        drop_empty:
            If True, drop empty cliques and cliques of size < 2.
        rebuild_indicator:
            If True and this is a batched MIS, recompute clique_indicator.

        Examples
        --------
        >>> pb_tie.C
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 4]]
        >>> pb_tie.set_cliques([[0,1,2],[1,2,4],[0,3]])
        >>> pb_tie.C
        [[0, 1, 2], [1, 2, 4], [0, 3]]

        ![Adding clique constraints](../../assets/mis_tie_constraint_matrix.svg)
        """
        if cliques is None:
            self.C = []
            if rebuild_indicator and hasattr(self, "_clique_indicator"):
                self._clique_indicator = None
            return

        # Canonicalize
        canon = []
        for c in cliques:
            if c is None:
                continue
            cc = [int(v) for v in c]
            if drop_empty and len(cc) < 2:
                continue
            canon.append(cc)

        if sort or unique:
            # canonicalize each clique to sorted unique vertices
            canon = [sorted(set(cc)) for cc in canon]
            #canon = lenlex_sort(canon)

            if drop_empty:
                canon = [cc for cc in canon if len(cc) >= 2]

        if unique:
            seen = set()
            uniq = []
            for cc in canon:
                t = tuple(cc)
                if t not in seen:
                    seen.add(t)
                    uniq.append(cc)
            canon = uniq

        if sort:
            #canon.sort(key=lambda cc: (len(cc), cc))
            canon = lenlex_sort(canon)

        self.C = canon

        # Invalidate / rebuild cached indicator for batched graphs
        if rebuild_indicator:
            # If you keep a cached clique indicator
            if hasattr(self, "_clique_indicator"):
                self._clique_indicator = None
            # If you have a method that recomputes it eagerly, call it here:
            if hasattr(self, "_compute_clique_indicator") and getattr(self, "_num_subproblems", 1) > 1:
                try:
                    self._clique_indicator = self._compute_clique_indicator()
                except Exception:
                    # keep it lazy if compute fails for any reason
                    self._clique_indicator = None


    @classmethod
    def from_edges(
        cls,
        nv: int,
        edges,
        weights=None,
        *,
        ensure_simple: bool = True,
        sort_edges: bool = True,
        validate: bool = True,
    ):
        if validate and (not isinstance(nv, int) or nv < 0):
            raise ValueError(f"nv must be nonnegative int, got {nv}")

        E = []
        for e in edges:
            if e is None or len(e) != 2:
                raise ValueError(f"Invalid edge {e!r}; expected a 2-tuple")
            u, v = int(e[0]), int(e[1])
            if ensure_simple and u == v:
                continue
            if sort_edges and u > v:
                u, v = v, u
            if validate and (u < 0 or v < 0 or u >= nv or v >= nv):
                raise ValueError(f"Edge {(u, v)} out of range for nv={nv}")
            E.append((u, v))

        if ensure_simple:
            E = list({(u, v) for (u, v) in E})

        g = nx.Graph()
        g.add_nodes_from(range(nv))
        g.add_edges_from(E)

        if weights is None:
            w = np.ones(nv, dtype=float)
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if validate and w.size != nv:
                raise ValueError(f"weights length {w.size} != nv {nv}")

        return cls(g, weights=w)

    @staticmethod
    def from_vcgraph(vcg: "VCGraph") -> "MIS":
        n = int(vcg.nv())
        node_x = vcg.features["node_x"]
        weights = np.asarray(node_x[:n, 0], dtype=float).reshape(-1)

        # each constraint node encodes a clique: neighbors among variable nodes
        cliques = [vcg.constraint_variables(C) for C in vcg.iter_constraints()]

        # reconstruct graph edges implied by cliques (complete graph per clique)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for clq in cliques:
            clq = list(map(int, clq))
            for i in range(len(clq)):
                for j in range(i + 1, len(clq)):
                    g.add_edge(clq[i], clq[j])

        return MIS(g, weights=weights, cliques=cliques)


def cplex_model(pb: MIS, continuous_relaxation: bool = False, cliques=None):
    model = Model()
    if continuous_relaxation:
        x = np.array([model.continuous_var(lb=0.0, ub=1.0, name=f"x{i}") for i in range(pb.nv())])
    else:
        x = np.array([model.binary_var(f"x{i}") for i in range(pb.nv())])

    if cliques is None:
        cliques = pb.C
    for clq in cliques:
        model.add_constraint(sum(x[u] for u in clq) <= 1)

    model.maximize(np.dot(x, pb.w))
    return model


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    z = (x - np.max(x)) / temperature
    e = np.exp(z)
    s = e.sum()
    if not np.isfinite(s) or s <= 0:
        # Fallback to uniform if numerical issues happen
        return np.full_like(e, 1.0 / e.size, dtype=float)
    return e / s


def mis_greedy(
    pb,
    scores: np.ndarray,
    *,
    policy: Literal["argmax", "sample"] = "argmax",
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    n = pb.nv()
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.size != n:
        raise ValueError(f"scores has size {scores.size}, expected {n}")

    if policy == "sample" and rng is None:
        rng = np.random.default_rng()

    queue = list(range(n))
    sol = np.zeros(n, dtype=float)

    while queue:
        q = np.array(queue, dtype=int)

        if policy == "argmax":
            u = int(q[np.argmax(scores[q])])

        elif policy == "sample":
            p = _softmax(scores[q], temperature=temperature)
            u = int(rng.choice(q, p=p))

        else:
            raise ValueError(f"Unknown policy={policy!r} (expected 'argmax' or 'sample')")

        sol[u] = 1.0
        queue.remove(u)

        # Remove neighbors (cannot be selected)
        for v in pb.g.neighbors(u):
            if v in queue:
                sol[v] = 0.0
                queue.remove(v)

    return sol

def mis_batch(pbs: list["MIS"]) -> "MIS":
    """Backwards-compatible alias for MIS.batch()."""
    return MIS.batch(pbs)

def mis_equal(pb1, pb2, *, check_weights: bool = True, atol: float = 0.0, rtol: float = 0.0) -> bool:
    """
    Check whether two MIS instances represent the same problem.

    This checks:
      - same number of vertices and edges
      - same undirected edge set
      - same clique family (as sets), ignoring ordering
      - optionally same weights (within tolerances)

    Parameters
    ----------
    pb1, pb2
        MIS instances.
    check_weights : bool, default=True
        Whether to also compare the weight vectors.
    atol, rtol : float
        Tolerances for weight comparison (np.allclose).

    Returns
    -------
    bool
        True if equivalent.
    """
    if pb1.nv() != pb2.nv():
        return False
    if pb1.g.number_of_edges() != pb2.g.number_of_edges():
        return False

    # Compare edge sets (undirected)
    e1 = {tuple(sorted(map(int, e))) for e in pb1.g.edges()}
    e2 = {tuple(sorted(map(int, e))) for e in pb2.g.edges()}
    if e1 != e2:
        return False

    # Compare cliques as a multiset of frozensets (ignore ordering inside each clique)
    C1 = sorted([frozenset(map(int, c)) for c in getattr(pb1, "C", [])], key=lambda s: (len(s), tuple(sorted(s))))
    C2 = sorted([frozenset(map(int, c)) for c in getattr(pb2, "C", [])], key=lambda s: (len(s), tuple(sorted(s))))
    if len(C1) != len(C2):
        return False
    if any(a != b for a, b in zip(C1, C2)):
        return False

    if check_weights:
        w1 = np.asarray(pb1.w, dtype=float).reshape(-1)
        w2 = np.asarray(pb2.w, dtype=float).reshape(-1)
        if w1.shape != w2.shape:
            return False
        if not np.allclose(w1, w2, atol=atol, rtol=rtol):
            return False

    return True

LPProblem.register_vcgraph_type("MIS", MIS)

