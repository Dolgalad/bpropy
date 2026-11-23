"""Minimal torch-only example (does not require PyG or torch-scatter).

Run from the repository root after installing the inference extra:
    python examples/inference_quickstart.py
"""
import torch

from bpropy.inference import FactorGraph, lbp, mean_field


def main():
    graph = FactorGraph.from_scopes(
        cardinalities=[2, 3, 2],
        scopes=[(0,), (0, 1), (1, 2)],
    )
    plan = graph.compile().to("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters can come from any torch.nn.Module.
    theta = torch.nn.Parameter(torch.randn(plan.num_configurations, device=plan.device))
    bp = lbp(plan, theta, iterations=15, damping=0.2)
    q = mean_field(plan, theta, iterations=15, damping=0.2)
    print("BP marginal P(x0):", bp.marginal(0).detach().cpu().tolist())
    print("NMF marginal P(x0):", q.marginal(0).detach().cpu().tolist())

    # Demonstrates end-to-end autograd through the unrolled GPU/CPU updates.
    objective = -bp.marginal(0)[1].clamp_min(1e-12).log()
    objective.backward()
    print("Gradient norm:", theta.grad.norm().item())


if __name__ == "__main__":
    main()
