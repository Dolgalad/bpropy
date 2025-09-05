import time
import sys
sys.path.append("../..")
import numpy as np
import torch
from torch.utils.data import DataLoader
from progress.bar import Bar
from typing import Dict, Optional, Tuple, Union, Mapping

from bpropy.models.mrf_models import Hetero_GCN_MRF_v4
from bpropy.datasets.mrf_dataset_2 import MRFDatasetConfig, MRFDataset
from bpropy.datasets.prefetcher import CUDAPrefetcher
from bpropy.datasets.collator import mrf_collate
from bpropy.adapters.vc_to_data import problem_to_data
from bpropy.mrf.lbp import loopy_belief_propagation
from bpropy.mrf.scp_factor_graph import SCPFactorGraph

ArrayLike = Union[np.ndarray, torch.Tensor]

def load_checkpoint(path, model, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])

def make_ds(root, split, cfg, hetero, feature_mode, with_lbp, with_nmf, processed_dir):
    return MRFDataset(
        root, split, cfg=cfg,
        preprocess=lambda pb: problem_to_data(
            pb, hetero=hetero, feature_mode=feature_mode,
            with_lbp=with_lbp, with_nmf=with_nmf
        ),
        processed_dir=processed_dir
    )

def make_ss_evidence_mask_dict(
    fg,
    evidence: Mapping[int, Optional[int]],
    *,
    as_torch: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bool,
    strict: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Build a mask over the over-complete sufficient-statistics vector that is True
    iff a factor configuration is consistent with the given (possibly partial) evidence.

    Args:
      fg: FactorGraph
      evidence: dict {var_id -> 0/1 or None}; missing keys are 'unknown'
      as_torch: return a torch.Tensor if True, else a np.ndarray
      device, dtype: only used when as_torch=True
      strict: if True, raise on bad var ids or values not in {0,1,None}

    Returns:
      mask of shape (sum_F 2**|F|,), in the same factor order and bit-endian
      convention as fg.factor_sufficient_statistics / your SS layout.
    """
    # Normalize evidence: keep only observed {0,1}
    obs = {}
    for k, v in evidence.items():
        if v is None:
            continue
        if v in (0, 1):
            obs[int(k)] = int(v)
        elif strict:
            raise ValueError(f"Evidence for var {k} must be 0/1/None, got {v!r}")

    blocks = []
    for F in fg.iter_factors():
        vars_F = fg.factor_variables(F)                   # sorted list of variable ids
        nF = len(vars_F)
        if nF == 0:
            # Degenerate factor: one configuration, always allowed
            blocks.append(np.array([True], dtype=bool))
            continue

        # Precompute all assignments for this factor as little-endian bits
        idx = np.arange(1 << nF, dtype=np.uint32)[:, None]     # [2^nF, 1]
        ks  = np.arange(nF, dtype=np.uint32)[None, :]          # [1, nF]
        bits = ((idx >> ks) & 1).astype(np.int64)              # [2^nF, nF], 0/1

        # Start with all allowed; constrain with observed vars in this factor
        allowed = np.ones((1 << nF), dtype=bool)
        for j, vj in enumerate(vars_F):
            if vj in obs:
                allowed &= (bits[:, j] == obs[vj])

        blocks.append(allowed)

    mask_np = np.concatenate(blocks, axis=0)  # length == fg.number_sufficient_statistics()
    if as_torch:
        out = torch.from_numpy(mask_np)
        if device is not None:
            out = out.to(device=device, dtype=dtype)
        else:
            out = out.to(dtype=dtype)
        return out
    return mask_np

@torch.no_grad()
def bp_guided_sampler(
    theta: torch.Tensor,
    edges,                              # LBPEdges (already built for this problem)
    pb, 
    evidence: Optional[Dict[int, int]] = None,  # {var_index: 0|1}
    steps: Optional[int] = None,
    max_iter: int = 10,
    tol: float = 1e-6,
    damping: float = 0.5,
    clamp_U: float = 12.0,              # strength for soft clamping
    state: Optional[dict] = None,       # warm-start: {"v2f": tensor}
) -> Tuple[torch.Tensor, float, dict]:
    """
    BP-guided sequential sampler (decimation).
    Returns:
      x          : sampled assignment over variables (0/1) as int64 tensor (shape [nv])
      log_q      : log proposal prob under product of marginals (Bethe-style)
      new_state  : {"v2f": tensor} for warm-starting the next call

    Notes:
      • We apply evidence by *shifting a copy* of theta (no in-place mutation).
      • We reuse and update v2f messages across steps for speed.
    """
    fg = SCPFactorGraph(pb)
    device = theta.device
    nv = int(edges.n_vars.sum().item())
    x  = torch.full((nv,), -1, dtype=torch.int64, device=device)
    log_q = 0.0

    # warm start messages
    v2f = state.get("v2f") if (state is not None and "v2f" in state) else torch.zeros(
        int(edges.n_f2v_msg.sum().item()), device=device, dtype=theta.dtype
    )

    # current (soft-)clamped parameters
    base_theta = theta
    cur_evid = dict(evidence or {})

    # helper: apply evidence to a *fresh* theta view
    def apply_evidence(th: torch.Tensor) -> torch.Tensor:
        if not cur_evid:
            return th
        th = th.clone()
        mask = make_ss_evidence_mask_dict(fg, cur_evid)
        NEG_INF = torch.finfo(th.dtype).min / 2  # or simply a large negative like -1e9 for float32
        th = torch.where(mask, th, torch.full_like(th, NEG_INF))
        #for i, yi in cur_evid.items():
        #    # Unary slots for var i live at [2*i : 2*i+2]
        #    # push probability mass to yi with a large positive shift and away from 1-yi
        #    if yi == 1:
        #        th[2*i + 0] /= clamp_U
        #        th[2*i + 1] *= clamp_U
        #    else:
        #        th[2*i + 0] *= clamp_U
        #        th[2*i + 1] /= clamp_U
        return th

    # how many sequential picks?
    T = steps if steps is not None else nv - (len(cur_evid) if cur_evid else 0)

    for _ in range(T):
        # 1) Run a few BP iters with current (soft-)evidence; warm-start with v2f
        theta_e = apply_evidence(base_theta)
        sol = loopy_belief_propagation(theta_e, v2f, edges,
                                       max_iter=max_iter, tol=tol, damping=damping)
        # sol.mu_v: size 2*nv, probs per state (normalized per variable)
        vm = sol.mu_v.view(nv, 2)  # [nv, {0,1}]  (your code already normalizes these) :contentReference[oaicite:1]{index=1}

        # 2) Exclude already fixed variables from sampling
        if cur_evid:
            idx = torch.tensor(list(cur_evid.keys()), device=device, dtype=torch.long)
            # lock their distributions to deltas (so they don't affect arg sampling)
            vm[idx, :] = 0.0
            vm[idx, torch.tensor([cur_evid[i] for i in cur_evid], device=device)] = 1.0

        # 3) Choose next variable to clamp (simple policy: most uncertain first → reduces contradictions)
        #    You can switch to "most confident first" by sorting on margin |p1 - 0.5|.
        undecided = (vm.sum(dim=1) > 0) & torch.tensor(
            [i not in (cur_evid or {}) for i in range(nv)], device=device, dtype=torch.bool
        )
        if not torch.any(undecided):
            break
        i_candidates = torch.nonzero(undecided, as_tuple=False).flatten()
        # pick the one with max entropy (closest to 0.5) to stabilize later steps
        p1 = vm[i_candidates, 1]
        ent = -(p1.clamp(1e-9, 1-1e-9).log()*p1 + (1-p1).clamp(1e-9,1-1e-9).log()*(1-p1))
        i_star = i_candidates[torch.argmax(ent)]

        # 4) Sample state from its unary marginal and accumulate proposal log-prob
        p = vm[i_star, :]
        yi = torch.distributions.Categorical(probs=p).sample().item()
        log_q += float(torch.log(p[yi] + 1e-12))

        # 5) Clamp and continue (update evidence & warm-start)
        cur_evid[int(i_star.item())] = int(yi)
        v2f = v2f.detach().clone()      # keep the latest messages as warm start for next iteration

        # (optional) early exit if everything is fixed
        if len(cur_evid) >= nv:
            break

    # finalize x from evidence; if some variables remain unset (possible with steps<T), fill by argmax
    for i in range(nv):
        if i in cur_evid:
            x[i] = cur_evid[i]
        else:
            # one last quick BP pass (cheap) to decide remaining vars by MAP of unary marginal
            theta_e = apply_evidence(base_theta)
            sol = loopy_belief_propagation(theta_e, v2f, edges,
                                           max_iter=max_iter, tol=tol, damping=damping)
            vm = sol.mu_v.view(nv, 2)
            x[i] = int(torch.argmax(vm[i]).item())
            log_q += float(torch.log(vm[i, x[i]] + 1e-12))

    new_state = {"v2f": v2f}
    return x.detach().cpu(), float(log_q), new_state, vm


def scp_repair(
    pb,
    x0: Optional[np.ndarray] = None,    # seed selection over sets, shape [pb.nv()]
    scores: Optional[np.ndarray] = None # guidance per set; higher = more desirable
) -> np.ndarray:
    """
    Feasibility repair for Set Cover + pruning, tailored to your SCP class.
    - Phase A (cover): greedily add sets until all items are covered.
      Priority ~ (effective_price / new_gain), where effective_price is cost
      or cost / score when scores are provided.
    - Phase B (prune): remove redundant picked sets while keeping coverage.

    Args:
      pb: SCP instance (with fields A, F, w and methods ne(), nv()).
      x0: optional 0/1 seed vector over sets (len = pb.nv()).
      scores: optional guidance (e.g., BP p(x=1) or model scores).

    Returns:
      y: feasible 0/1 vector over sets, shape [pb.nv()].
    """
    m = pb.nv()               # number of sets
    n = pb.ne()               # number of items
    cost = np.asarray(pb.w, dtype=float)  # per-set cost (already in SCP)  :contentReference[oaicite:1]{index=1}

    # Normalize/validate inputs
    if x0 is None:
        y = np.zeros(m, dtype=np.uint8)
    else:
        x0 = np.asarray(x0).reshape(-1)
        assert x0.size == m, f"x0 length {x0.size} != nv {m}"
        y = (x0 > 0).astype(np.uint8)

    if scores is not None:
        scores = np.asarray(scores, dtype=float).reshape(-1)
        assert scores.size == m, f"scores length {scores.size} != nv {m}"

    # Precompute coverage lists (items covered by each set) from pb.F  :contentReference[oaicite:2]{index=2}
    # pb.F[j] is a sorted list of item indices in pb.A covered by set j.
    sets_cover = [np.asarray(f, dtype=np.int32) for f in pb.F]

    # Build item -> sets map once to accelerate pruning & (optionally) cover phase
    item_sets = [[] for _ in range(n)]
    for j, cov in enumerate(sets_cover):
        for a in cov:
            item_sets[a].append(j)
    item_sets = [np.asarray(js, dtype=np.int32) for js in item_sets]

    # --- Coverage bookkeeping (from warm start) ---
    covered_cnt = np.zeros(n, dtype=np.int32)
    for j in np.flatnonzero(y):
        if sets_cover[j].size:
            covered_cnt[sets_cover[j]] += 1
    uncovered = np.flatnonzero(covered_cnt == 0)

    # Helper: “price” possibly biased by scores
    eps = 1e-12
    if scores is None:
        eff_price = cost
    else:
        # Prefer higher score at similar cost
        eff_price = cost / (scores + eps)

    # ----------------- Phase A: cover -----------------
    # Greedily add sets until all items are covered.
    # Gain(j) = number of still-uncovered items that set j would newly cover.
    # Pick argmin (eff_price / gain) among gain > 0.
    while uncovered.size > 0:
        # Mark uncovered items to compute gains quickly
        is_uncovered = np.zeros(n, dtype=np.uint8)
        is_uncovered[uncovered] = 1

        gains = np.zeros(m, dtype=np.int32)
        # Only compute gains for not-yet-picked sets
        for j in np.flatnonzero(1 - y):
            cov = sets_cover[j]
            if cov.size:
                gains[j] = int(is_uncovered[cov].sum())

        mask = gains > 0
        if not np.any(mask):
            # In typical SCP instances this shouldn't happen; bail with current y
            # (caller can decide what to do next)
            break

        cand = np.where(mask)[0]
        # ratio = price / gain
        ratio = eff_price[cand] / gains[cand].astype(np.float64)
        j_star = cand[int(np.argmin(ratio))]

        # Select j_star and update coverage
        y[j_star] = 1
        if sets_cover[j_star].size:
            covered_cnt[sets_cover[j_star]] += 1
        uncovered = np.flatnonzero(covered_cnt == 0)

    # ----------------- Phase B: prune -----------------
    # Remove any picked set whose covered items remain covered >= 1 without it.
    # Order removals to drop “least valuable” first:
    #   - If scores given: low (scores/cost) first.
    #   - Else: high cost first (drop expensive redundancies).
    picked = np.flatnonzero(y)
    if picked.size:
        if scores is None:
            order = picked[np.argsort(-cost[picked])]  # expensive first
        else:
            value = scores[picked] / (cost[picked] + eps)
            order = picked[np.argsort(value)]          # lower value first

        for j in order:
            cov = sets_cover[j]
            if cov.size == 0:
                y[j] = 0
                continue
            # If every item in cov would stay covered after removing j, drop it
            if np.all(covered_cnt[cov] - 1 >= 1):
                y[j] = 0
                covered_cnt[cov] -= 1

    return y

def greedy(pb, theta, data):
    fg = SCPFactorGraph(pb)
    covered = np.zeros(pb.ne(), dtype=bool)
    x = np.zeros(pb.nv(), dtype=bool)
    v2f = torch.zeros(
        int(data.lbp_edges.n_f2v_msg.sum().item()), device=device, dtype=theta.dtype
    )

    while not np.all(covered):
        gains = {i:(np.logical_not(covered) & pb.adj_vec(pb.F[i])).sum() for i in range(pb.nv())}
        cur_evid_1 = {i:1 for i in range(pb.nv()) if x[i]==1}
        cur_evid_0 = {i:0 for i in range(pb.nv()) if gains[i]==0}
        cur_evid = {**cur_evid_0, **cur_evid_1}

        th = theta.clone()
        mask = make_ss_evidence_mask_dict(fg, cur_evid)
        NEG_INF = torch.finfo(th.dtype).min / 2  # or simply a large negative like -1e9 for float32
        #theta = torch.where(mask, theta, torch.full_like(theta, NEG_INF))
        th = torch.where(mask, th, torch.full_like(th, NEG_INF))

        lbp_res = loopy_belief_propagation(th, v2f, data.lbp_edges,
                                       max_iter=10, tol=1e-6, damping=0.5)

        # NOTE: Gurobi will check feasibility; if infeasible, it's ignored.
        mu_evid = lbp_res.mu_v.reshape(-1,2)[:,1].numpy()
        #print(mu_evid)
        mu_evid[np.isnan(mu_evid)] = 0

        cand = [i for i in range(pb.nv()) if x[i] == 0]
        def ratio(i):
            gain = (np.logical_not(covered) & pb.adj_vec(pb.F[i])).sum()
            #return np.inf if gain == 0 else (-scores[i] / gain)
            return np.inf if gain == 0 else (1 - pb.sigmoid(mu_evid[i]))

        idx = min(cand, key=ratio)
        #print([gains[i] for i in range(pb.nv())])
        #print("Candidate ", idx, " score = ", mu_evid[idx], " gains = ", gains[idx])

        x[idx] = 1
        covered |= pb.adj_vec(pb.F[idx])
    return x



model_kwargs = {"in_variable_dim": 10, 
                "in_constraint_dim": 4, 
                "factor_sizes": [1,2,3,4,5],
                "inference": "loopy_belief_propagation"}

checkpoint_path = "../../models/scp/scp_n50_m50_k5/Hetero_GCN_MRF_v4_hetero_1756757098/checkpoints/best.pt"
dataset_root = "../../datasets/problems/scp/scp_n50_m50_k5"
processed_dir = "../../datasets/problems/scp/scp_n50_m50_k5/processed_lbp"
device = "cpu"

Model = Hetero_GCN_MRF_v4
model = Model(**model_kwargs)

load_checkpoint(checkpoint_path, model)

eval_cfg = MRFDatasetConfig(
    feature_mode="EB",
    yield_mode="both",
    refresh=False,
    return_assignment=True,
    assignment_solver="cbc",
)
ds_train_eval = make_ds(
    dataset_root, "test", eval_cfg,
    True, "EB", True, False,
    processed_dir
)
loader_eval = DataLoader(
    ds_train_eval, batch_size=1, shuffle=False,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    collate_fn=mrf_collate,
)

model.eval()

gaps_0, gaps_1, gaps_2 = [], [], []
time_0, time_1, time_2 = [], [], []
model.eval()
L,c = 502,1
with torch.no_grad():
    bar = Bar(f"[Test]          ", max=len(loader_eval))
    gaps_1, gaps_2 = [], []
    for pb, data, y in CUDAPrefetcher(loader_eval, device):
        t0 = time.time()
        lbp_res, theta= model(data)
        t_pred = time.time() - t0

        x = greedy(pb[0], theta, data)
        t_v0 = time.time() - t0
        gap_0 = 100 * (sum(x) - y.sum().item()) / y.sum().item()
        gaps_0.append(gap_0)
        time_0.append(t_v0)

        t_v1 = time.time()
        samp,_,_,mu_v= bp_guided_sampler(theta, data.lbp_edges, pb[0])
        p1 = mu_v.view(pb[0].nv(), 2)[:, 1].detach().cpu().numpy()
        t_guide = time.time() - t_v1
        sol_repaired = scp_repair(pb[0], samp.numpy(), p1)
        t_v1 = time.time() - t_v1 + t_pred
        #print(pb[0].check_solution(sol_repaired))

        gap_1 = 100 * (sol_repaired.sum().item() - y.sum().item()) / y.sum().item()
        gaps_1.append(gap_1)
        time_1.append(t_v1)

        #print()
        #print("GRD Gap 1 = ", gap, np.mean(gaps_1))

        t_v2 = time.time()
        sol_grd = pb[0].greedy(scores=p1)
        t_v2 = time.time() - t_v2 + t_guide
        gap_2 = 100 * (sol_grd.sum().item() - y.sum().item()) / y.sum().item()
        gaps_2.append(gap_2)
        time_2.append(t_v2)
        #print("GRD Gap 2 = ", gap, np.mean(gaps_2))

        print(f"{c} - V0: {gap_0:.3f}% ({np.mean(gaps_0):.3f}%, {np.mean(time_0):.3f}s), V1: {gap_1:.3f}% ({np.mean(gaps_1):.3f}%, {np.mean(time_1):.3f}s), V2: {gap_2:.3f}% ({np.mean(gaps_2):.3f}%, {np.mean(time_2):.3f}s)")

        #bar.next()
        c+= 1
        if c >= L:
            break
    bar.finish()

print(f"V0 gaps: {np.mean(gaps_0):.3f}% ({np.std(gaps_0):.3f}%), time: {np.mean(time_0):.3f}s ({np.std(time_0):.3f}s)")
print(f"V1 gaps: {np.mean(gaps_1):.3f}% ({np.std(gaps_1):.3f}%), time: {np.mean(time_1):.3f}s ({np.std(time_1):.3f}s)")
print(f"V1 gaps: {np.mean(gaps_2):.3f}% ({np.std(gaps_2):.3f}%), time: {np.mean(time_2):.3f}s ({np.std(time_2):.3f}s)")


