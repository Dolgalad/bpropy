#!/usr/bin/env python3
"""
Contrastive (energy-ranking) training script for bpropy using existing datasets,
ignoring ground-truth labels.

What it does (per batch):
  1) forward: theta = model(data)
  2) x_pos = greedy_decimation(pb, backend(theta.detach()))
  3) theta_neg = theta.detach() + sigma * N(0,1)
     x_neg = greedy_decimation(pb, backend(theta_neg))
  4) E_pos = E_theta(x_pos)   (computed with *clean* theta, with grad)
     E_neg = E_theta(x_neg)
     loss = softplus(E_pos - E_neg + margin)
  5) backward/step

Assumptions (match your current factor-graph pipeline):
  - model returns a flat theta tensor (factor-logits style), OR returns dict {"theta": ...}
  - data has:
      * data.factor_sizes: [nF]
      * data.fg_map: [sum_k, 2] with columns [factor_id, var_id]
    and fg_map rows are grouped by factor in the same order as factor_sizes
  - binary variables (assignment index is computed as sum_{j=0..k-1} x_j * 2^j
    where the variable order is the order inside fg_map for that factor).

If your within-factor bit ordering is different, adjust `assignment_index_from_bits()`.
"""

import argparse
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bpropy.cli.train.config import load_run_config
from bpropy.cli.train.method_registry import METHOD_REGISTRY
from bpropy.cli.train.loop import build_model, build_optimizer  # reuse your existing builder logic

from bpropy.cli.train.data import make_dataset  # yields MRFDataset configured from cfg
from bpropy.cli.train.metrics import gap_from_solution

from bpropy.datasets.collator import mrf_collate
from bpropy.gd.greedy_decimation import greedy_decimation


# ----------------------------
# Helpers
# ----------------------------

def as_theta(pred: Any) -> torch.Tensor:
    """Match your adapter helper behaviour: accept tensor or dict with 'theta'."""
    if isinstance(pred, dict):
        if "theta" in pred:
            return pred["theta"]
        # fallback if you ever return something else
        raise KeyError(f"pred dict keys={list(pred.keys())}, expected 'theta'")
    if torch.is_tensor(pred):
        return pred
    if isinstance(pred, tuple):
        return pred[1]
    raise TypeError(f"Unsupported pred type: {type(pred)}")


def assignment_index_from_bits(bits: torch.Tensor) -> torch.Tensor:
    """
    bits: [k] int/float in {0,1}
    returns scalar idx in [0, 2^k)
    Ordering: idx = sum bits[j] * 2^j.
    """
    k = bits.numel()
    if k == 0:
        return bits.new_zeros(())
    weights = (2 ** torch.arange(k, device=bits.device, dtype=torch.long))
    return (bits.long() * weights).sum()


def energy_from_theta_factorgraph(pb: Any, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute E_theta(x) = - sum_f theta_f(x_f)
    using (factor_sizes, fg_map) and flat theta layout with per-factor blocks of size 2^k.
    """
    from bpropy.mrf.factor_graph import FactorGraph
    from bpropy.mrf.sufficient_statistics import x2ss

    fg = FactorGraph(pb)
    x_ss = torch.from_numpy(x2ss(x.cpu().numpy(), fg)).float().to(theta.device)
    return torch.dot(theta, x_ss)

def perturb_theta(theta_detached: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return theta_detached
    return theta_detached + sigma * torch.randn_like(theta_detached)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to JSON config (same as bpropy-train).")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto).")
    ap.add_argument("--sigma", type=float, default=0.2, help="Stddev for theta perturbation (negative sample).")
    ap.add_argument("--alpha", type=float, default=0.0, help="Objective reweighting.")
    ap.add_argument("--margin", type=float, default=0.0, help="Contrastive margin.")
    ap.add_argument("--epochs", type=int, default=None, help="Override cfg.train.epochs")
    ap.add_argument("--batch-size", type=int, default=1, help="Batch size for this script (kept simple).")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    cfg = load_run_config(args.config)

    # Keep this script simple and correct first: greedy runs per instance.
    # (You can extend to true B>1 by slicing scores/assignments with ptr later.)
    cfg.setdefault("data_loader", {})
    cfg["data_loader"]["batch_size"] = int(args.batch_size)

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    # --- Build model + optimizer exactly like your CLI ---
    model = build_model(cfg, device=device)
    optimizer = build_optimizer(cfg, model)

    # --- Pick method adapter (we’ll reuse its decimation backend wiring) ---
    adapter = METHOD_REGISTRY["factor_marginals"]

    # --- Dataset/loader: force yield_mode='both' to get pb for greedy decimation ---
    split = cfg.get("data", {}).get("split", "train")
    loader_cfg = cfg.get("data", {}).get("loader", {})
    ds_train = make_dataset(cfg, split, yield_mode="both")
    train_loader = DataLoader(
        ds_train,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(loader_cfg.get("num_workers", 0)),
        pin_memory=bool(loader_cfg.get("pin_memory", False)),
        persistent_workers=bool(loader_cfg.get("persistent_workers", False)),
        prefetch_factor=int(loader_cfg.get("prefetch", 2)),
        collate_fn=mrf_collate,
    )

    epochs = int(args.epochs if args.epochs is not None else cfg.get("train", {}).get("epochs", 1))
    sigma = float(args.sigma)
    margin = float(args.margin)

    model.train()
    global_step = 0
    maximize = bool(cfg.get("problem", {}).get("maximize", True))

    for epoch in range(1, epochs + 1):
        running = 0.0
        n = 0

        for batch in train_loader:
            # batch is (pbs, data_batch, y) but we ignore y
            pbs, data, _y = batch
            # This script keeps it simple: batch_size=1 recommended.
            if isinstance(pbs, (list, tuple)):
                if len(pbs) != 1:
                    raise RuntimeError(
                        "This script currently expects batch_size=1 so pb.greedy can run per instance.\n"
                        "Set --batch-size 1 for now, or extend this script to loop over the batch elements."
                    )
                pb = pbs[0]
            else:
                pb = pbs

            data = data.to(device)

            optimizer.zero_grad(set_to_none=True)

            # 1) forward -> theta (with grad)
            pred = adapter.forward_fn(model, data)
            theta = as_theta(pred)

            # 2) x_pos using clean theta (detached for search)
            theta_det = theta.detach()
            pred_det = (None, theta_det)
            backend_pos = adapter.decimation_backend_fn(pb, data, pred_det, cfg, device=device)
            padapter = adapter.decimation_adapter_fn(pb, cfg)
            with torch.no_grad():
                x_pos = torch.as_tensor(greedy_decimation(pb, backend_pos, padapter), device=device).long()

            # 3) x_neg using perturbed theta (detached)
            theta_neg = perturb_theta(theta_det, sigma=sigma)
            pred_neg = (None, theta_neg)

            backend_neg = adapter.decimation_backend_fn(pb, data, pred_neg, cfg, device=device)
            with torch.no_grad():
                x_neg = torch.as_tensor(greedy_decimation(pb, backend_neg, padapter), device=device).long()

            # 4) energies under *clean* theta (grad flows into theta/model)
            E_pos = energy_from_theta_factorgraph(pb, theta, x_pos)
            E_neg = energy_from_theta_factorgraph(pb, theta, x_neg)

            R_pos = x_pos.sum()
            R_neg = x_neg.sum()
            dR = torch.tensor(R_neg - R_pos, device=device, dtype=torch.float32)
            w = 1.0 + args.alpha * torch.clamp(dR, min=0.0)
            w = w.detach()

            loss = w * F.softplus(E_neg - E_pos + margin)
            loss.backward()
            optimizer.step()

            running += float(loss.detach().item())
            n += 1
            global_step += 1

            gap = float(
                gap_from_solution(
                    y_true_sum=float(_y.sum().item()),
                    y_pred_sum=float(x_pos.sum()),
                    maximize=maximize,
                    eps=1e-8,  # match your timed eval path :contentReference[oaicite:11]{index=11}
                )
            )


            if global_step % int(args.log_every) == 0:
                print(
                    f"[epoch {epoch:03d} step {global_step:06d}] "
                    f"loss={running/max(n,1):.4f} "
                    f"Epos={float(E_pos.detach().item()):.3f} Eneg={float(E_neg.detach().item()):.3f} "
                    f"(Epos-Eneg)={float((E_pos-E_neg).detach().item()):.3f} sigma={sigma} "
                    f"(Epos gap)={gap}"

                )
                running = 0.0
                n = 0

    print("Done.")


if __name__ == "__main__":
    main()

