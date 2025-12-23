from bpropy.gd.greedy_decimation import greedy_decimation
from bpropy.gd.scp_adapter import SCPAdapter
from bpropy.gd.coverage_backend import CoverageBackend
from bpropy.gd.lbp_backend import LBPBackend
from bpropy.mrf.canonical_parameters import *#mis_canonical_parameters
from bpropy.mrf.factor_graph import FactorGraph
from bpropy.gd.nmf_backend import NMFBackend
from bpropy.co.scp import SCP
from bpropy.co.generators.scp import random_scp_items_k_balanced

import sys
import time
def time_f(func, *args, **kwargs):
    t0 = time.time()
    r = func(*args, **kwargs)
    return time.time() - t0, r
def render_status(gaps, speedups):
    lines = []
    lines.append("=== Running SCP benchmark (live) ===")
    for t in ["SCG3", "BasicGreedy", "CoverageBackend", "LBPBackend", "NMFBackend"]:
        mg = np.mean(gaps[t]) if gaps[t] else float("nan")
        ms = np.mean(speedups[t]) if speedups[t] else float("nan")
        lines.append(f"{t:12s} | gap: {mg:8.3f}% | speedup: {ms:8.3f}% | n={len(gaps[t])}")
    return "\n".join(lines)

def clear_screen():
    # works on most terminals
    sys.stdout.write("\033[2J\033[H")  # clear + cursor home
    sys.stdout.flush()

def optgap(opt_val, val, sense="min"):
    if sense=="min":
        return 100 * (val - opt_val) / opt_val
    return -opt_gap(opt_val, val, sense="min")
def speedup(opt_t, t):
    return 100 * t / opt_t

titles = ["SCG3", "BasicGreedy", "CoverageBackend", "LBPBackend", "NMFBackend"]
gaps = {t:[] for t in titles}
speedups = {t:[] for t in titles}

for k in range(100):
    # random SCP
    n_ = np.random.choice(range(200, 301))
    m_ = np.random.choice(range(150, 251))
    k_ = 5
    pb = random_scp_items_k_balanced(n=n_, m=m_, k=k_)


    # solve baseline
    #t_opt, sol_opt = time_f(pb.solve)
    #val_opt = np.dot(pb.w, sol_opt.x)

    # SCG3
    print("SCG3")
    t_opt, sol_opt= time_f(pb.solve, solver="scg3")
    val_opt = np.dot(pb.w, sol_opt.x)
    #gap_scg3 = optgap(val_opt, val_scg3)
    #spd_scg3 = speedup(t_opt, t_scg3)
    #gaps["SCG3"].append(gap_scg3)
    #speedups["SCG3"].append(spd_scg3)

    
    # solve with SCP.greedy (baseline solver)
    print("BasicGreedy")
    t_sol, sol = time_f(pb.greedy)
    val_sol = np.dot(pb.w, sol.x)
    gap_sol = optgap(val_opt, val_sol)
    spd_sol = speedup(t_opt, t_sol)
    gaps["BasicGreedy"].append(gap_sol)
    speedups["BasicGreedy"].append(spd_sol)
   
    # solve with basic greedy
    print("CoverageBackend")
    backend = CoverageBackend()
    adapter = SCPAdapter()
    t_deg, x_deg = time_f(greedy_decimation, pb, backend, adapter)
    val_deg = np.dot(pb.w, x_deg)
    gap_deg = optgap(val_opt, val_deg)
    spd_deg = speedup(t_opt, t_deg)
    gaps["CoverageBackend"].append(gap_deg)
    speedups["CoverageBackend"].append(spd_deg)
    
    # solve with LBP inference
    print("LBPBackend")
    def fg_builder(pb):
        return FactorGraph(pb)
    
    def theta_builder(pb, fg, device, dtype):
        return canonical_parameters(pb, unary_bias=1.0, M=10.0, device=device, dtype=dtype)
    
    backend = LBPBackend(fg_builder=fg_builder, theta_builder=theta_builder, damping=0.5, neg_inf=-100, max_iter=10)
    adapter = SCPAdapter()
    t_lbp, x_lbp = time_f(greedy_decimation, pb, backend, adapter)
    val_lbp = np.dot(pb.w, x_lbp)
    gap_lbp = optgap(val_opt, val_lbp)
    spd_lbp = speedup(t_opt, t_lbp)
    gaps["LBPBackend"].append(gap_lbp)
    speedups["LBPBackend"].append(spd_lbp)
    
    # solve with NMF inference
    print("NMFBackend")
    def fg_builder(pb):
        return FactorGraph(pb)
    
    def theta_builder(pb, fg, device, dtype):
        return canonical_parameters(pb, unary_bias=1.0, M=10.0, device=device, dtype=dtype)
    
    backend = NMFBackend(fg_builder=fg_builder, theta_builder=theta_builder, damping=0.5, neg_inf=-100, max_iter=10)
    adapter = SCPAdapter()
    t_nmf, x_nmf = time_f(greedy_decimation, pb, backend, adapter)
    val_nmf = np.dot(pb.w, x_nmf)
    gap_nmf = optgap(val_opt, val_nmf)
    spd_nmf = speedup(t_opt, t_nmf)
    gaps["NMFBackend"].append(gap_nmf)
    speedups["NMFBackend"].append(spd_nmf)


    clear_screen()
    sys.stdout.write(render_status(gaps, speedups) + "\n")
    sys.stdout.flush()
    print("LBP valid : ", pb.check_solution(x_lbp))
    print("NMF valid : ", pb.check_solution(x_nmf))

