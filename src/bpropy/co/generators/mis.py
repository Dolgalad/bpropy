import networkx as nx

from bpropy.co.mis import MIS
from bpropy.clique_cover import clique_cover
from bpropy.graphs.random_graphs import random_bounded_degree

def random_mis(n=100, p=0.15, cliques=False):
    g = nx.generators.random_graphs.erdos_renyi_graph(n, p)
    if cliques:
        cc = clique_cover(g)
        pb = MIS(g, cliques=cc)
    else:
        pb = MIS(g)
    while pb.g.number_of_edges() == 0:
        g = nx.generators.random_graphs.erdos_renyi_graph(n, p)
        pb = MIS(g, cliques=clique_cover(g)) if cliques else MIS(g)
    return pb

def random_mis_bounded_degree(
    nv: int,
    max_deg: int,
    *,
    target_avg_deg: Optional[float] = None,
    p: Optional[float] = None,
    ensure_min_deg: int = 0,
    seed: Optional[int] = None,
    max_trials_factor: int = 20,
):
    nv_out, edges, w = random_mis_bounded_degree(
        nv=nv,
        max_deg=max_deg,
        target_avg_deg=target_avg_deg,
        p=p,
        ensure_min_deg=ensure_min_deg,
        seed=k,
    )
    # Build MIS problem
    return MIS.from_edges(nv_out, edges, weights=w)


