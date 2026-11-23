"""CPU/CUDA numerical parity, gradients and hard-constraint smoke tests.

CUDA tests skip on CI runners without CUDA; run on an actual NVIDIA GPU with
``PYTHONPATH=src python -m unittest discover -s tests/inference -v``.
"""
from __future__ import annotations

import unittest

import torch

from bpropy.inference import FactorGraph, lbp, mean_field


def _loss(beliefs: torch.Tensor) -> torch.Tensor:
    weights = torch.linspace(0.1, 1.1, beliefs.numel(),
                             device=beliefs.device, dtype=beliefs.dtype)
    return (beliefs * weights).sum() / max(1, beliefs.numel())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device not available")
class CUDACorrectnessTests(unittest.TestCase):
    def _check_case(self, graph: FactorGraph, theta_cpu: torch.Tensor,
                    *, iterations: int = 9, damping: float = 0.15):
        cpu = graph.compile()
        gpu = cpu.to("cuda")
        self.assertEqual(gpu.device.type, "cuda")
        if theta_cpu.dtype == torch.float64:
            atol, rtol = 2e-7, 2e-6
        else:
            atol, rtol = 2e-5, 2e-4
        for algorithm, kwargs in (
            (lbp, {"return_factor_beliefs": True}),
            (mean_field, {"validate_finite": False}),
        ):
            cpu_theta = theta_cpu.detach().clone().requires_grad_(True)
            gpu_theta = theta_cpu.detach().to("cuda").requires_grad_(True)
            reference = algorithm(cpu, cpu_theta, iterations=iterations,
                                  damping=damping, **kwargs)
            actual = algorithm(gpu, gpu_theta, iterations=iterations,
                               damping=damping, **kwargs)
            torch.testing.assert_close(actual.variable_beliefs.cpu(),
                                       reference.variable_beliefs,
                                       atol=atol, rtol=rtol)
            self.assertTrue(torch.isfinite(actual.variable_beliefs).all().item())
            if algorithm is lbp:
                torch.testing.assert_close(actual.factor_beliefs.cpu(),
                                           reference.factor_beliefs,
                                           atol=atol, rtol=rtol)
            cpu_grad, = torch.autograd.grad(_loss(reference.variable_beliefs), cpu_theta)
            gpu_grad, = torch.autograd.grad(_loss(actual.variable_beliefs), gpu_theta)
            self.assertTrue(torch.isfinite(gpu_grad).all().item())
            torch.testing.assert_close(gpu_grad.cpu(), cpu_grad,
                                       atol=atol, rtol=rtol)

    def test_mixed_cardinalities_and_arity(self):
        torch.manual_seed(2026)
        graph = FactorGraph.from_scopes(
            [2, 3, 2, 4], [(0,), (0, 1), (0, 1, 2), (1, 3), (3,)])
        for dtype in (torch.float32, torch.float64):
            theta = torch.randn(graph.num_configurations, dtype=dtype) * 0.2
            with self.subTest(dtype=dtype):
                self._check_case(graph, theta)

    def test_batched_graphs_and_isolated_variables(self):
        a = FactorGraph.from_scopes([2, 3], [(0,), (0, 1)])
        b = FactorGraph.from_scopes([4, 2], [(0,), ()])
        c = FactorGraph.from_scopes([3], [])
        graph = FactorGraph.batch([a, b, c])
        theta = torch.linspace(-0.4, 0.3, graph.num_configurations,
                               dtype=torch.float64)
        self._check_case(graph, theta, iterations=12, damping=0.0)

    def test_cuda_lbp_hard_constraints(self):
        # No gradient comparison for -inf entries: hard-constraint
        # differentiation requires separate, explicit semantics.
        graph = FactorGraph.from_scopes([2, 2], [(0,), (0, 1)])
        theta = torch.tensor([.2, -.1, -torch.inf, 0., 0., -torch.inf],
                             dtype=torch.float64)
        cpu = lbp(graph.compile(), theta, iterations=8)
        gpu = lbp(graph.compile().to("cuda"), theta.cuda(), iterations=8)
        torch.testing.assert_close(gpu.variable_beliefs.cpu(),
                                   cpu.variable_beliefs, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(gpu.factor_beliefs.cpu(),
                                   cpu.factor_beliefs, atol=1e-9, rtol=1e-9)
        self.assertTrue(torch.isfinite(gpu.variable_beliefs).all().item())

    def test_cuda_repeated_deterministic_bounds(self):
        # Scatter reduction accumulation order can vary on GPUs. Require
        # numerically close results, not bitwise identity.
        graph = FactorGraph.from_scopes([2, 2, 2],
                                        [(0, 1), (1, 2), (0, 2), (0,)])
        plan = graph.compile().to("cuda")
        theta = torch.linspace(-.2, .3, graph.num_configurations,
                               dtype=torch.float32, device="cuda")
        for method in (lbp, mean_field):
            kwargs = {"validate_finite": False} if method is mean_field else {}
            a = method(plan, theta, iterations=10, **kwargs).variable_beliefs
            b = method(plan, theta, iterations=10, **kwargs).variable_beliefs
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)


if __name__ == "__main__":
    unittest.main()
