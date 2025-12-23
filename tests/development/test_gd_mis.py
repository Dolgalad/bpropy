from bpropy.gd.greedy_decimation import greedy_decimation
from bpropy.gd.mis_adapter import MISAdapter
#from bpropy.co.generators.mis import random_mis
from bpropy.gd.degree_backend import DegreeBackend
from bpropy.gd.lbp_backend import LBPBackend
from bpropy.canonical_parameters import *#mis_canonical_parameters
from bpropy.mrf.factor_graph import FactorGraph
from bpropy.gd.nmf_backend import NMFBackend
from bpropy.co.mis import MIS
from bpropy.graphs.random_graphs import random_mis_bounded_degree  # the bounded-degree generator

import sys
import time
def time_f(func, *args, **kwargs):
    t0 = time.time()
    r = func(*args, **kwargs)
    return time.time() - t0, r
def render_status(gaps, speedups):
    lines = []
    lines.append("=== Running MIS benchmark (live) ===")
    for t in ["BasicGreedy", "DegreeBackend", "LBPBackend", "NMFBackend"]:
        mg = np.mean(gaps[t]) if gaps[t] else float("nan")
        ms = np.mean(speedups[t]) if speedups[t] else float("nan")
        lines.append(f"{t:12s} | gap: {mg:8.3f}% | speedup: {ms:8.3f}% | n={len(gaps[t])}")
    return "\n".join(lines)

def clear_screen():
    # works on most terminals
    sys.stdout.write("\033[2J\033[H")  # clear + cursor home
    sys.stdout.flush()

titles = ["BasicGreedy", "DegreeBackend", "LBPBackend", "NMFBackend"]
gaps = {t:[] for t in titles}
speedups = {t:[] for t in titles}

for k in range(100):
    # random MIS
    #pb = random_mis(300)
    #pb = random_mis_bounded_degree(300, 15, p=0.15)
    nv_out, edges, w = random_mis_bounded_degree(
        nv=300,
        max_deg=15,
        #target_avg_deg=15,
        p=0.15,
        ensure_min_deg=1,
        seed=k,
    )
    # Build MIS problem
    pb = MIS.from_edges(nv_out, edges, weights=w)

    # solve baseline
    t_opt, sol_opt = time_f(pb.solve, solver="kamis")
    val_opt = sol_opt.x.sum()
    
    # solve with MIS.greedy (baseline solver)
    t_sol, sol = time_f(pb.greedy)
    val_sol = sol.x.sum()
    gap_sol = 100 * (val_opt - val_sol) / val_opt
    spd_sol = 100 * (t_opt - t_sol) / t_opt
    gaps["BasicGreedy"].append(gap_sol)
    speedups["BasicGreedy"].append(spd_sol)
   
    # solve with basic greedy
    backend = DegreeBackend()
    adapter = MISAdapter()
    t_deg, x_deg = time_f(greedy_decimation, pb, backend, adapter)
    val_deg = x_deg.sum()
    gap_deg = 100 * (val_opt - val_deg) / val_opt
    spd_deg = 100 * (t_opt - t_deg) / t_opt
    gaps["DegreeBackend"].append(gap_deg)
    speedups["DegreeBackend"].append(spd_deg)
    
    # solve with LBP inference
    def fg_builder(pb):
        return FactorGraph(pb)
    
    def theta_builder(pb, fg, device, dtype):
        return canonical_parameters(pb, unary_bias=1.0, M=10.0, device=device, dtype=dtype)
    
    backend = LBPBackend(fg_builder=fg_builder, theta_builder=theta_builder, damping=0.5, neg_inf=-1e9, max_iter=1000)
    t_lbp, x_lbp = time_f(greedy_decimation, pb, backend, adapter)
    val_lbp = x_lbp.sum()
    gap_lbp = 100 * (val_opt - val_lbp) / val_opt
    spd_lbp = 100 * (t_opt - t_lbp) / t_opt
    gaps["LBPBackend"].append(gap_lbp)
    speedups["LBPBackend"].append(spd_lbp)
    
    # solve with NMF inference
    def fg_builder(pb):
        return FactorGraph(pb)
    
    def theta_builder(pb, fg, device, dtype):
        return canonical_parameters(pb, unary_bias=1.0, M=10.0, device=device, dtype=dtype)
    
    backend = NMFBackend(fg_builder=fg_builder, theta_builder=theta_builder, damping=0.5, neg_inf=-100, max_iter=1000)
    t_nmf, x_nmf = time_f(greedy_decimation, pb, backend, adapter)
    val_nmf = x_nmf.sum()
    gap_nmf = 100 * (val_opt - val_nmf) / val_opt
    spd_nmf = 100 * (t_opt - t_nmf) / t_opt
    gaps["NMFBackend"].append(gap_nmf)
    speedups["NMFBackend"].append(spd_nmf)


    clear_screen()
    sys.stdout.write(render_status(gaps, speedups) + "\n")
    sys.stdout.flush()
