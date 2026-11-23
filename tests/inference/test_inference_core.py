"""Self-contained correctness tests: no external datasets, optimizers or PyG needed."""
from __future__ import annotations

import itertools
import math
import random
import unittest

import torch

from bpropy.inference import FactorGraph, LBP, NaiveMeanField, lbp, mean_field


def exact_variable_marginals(graph: FactorGraph, theta: torch.Tensor):
    """Independent brute-force reference implementation (small graphs only)."""
    cards = graph.cardinalities.cpu().tolist()
    offsets = graph.potential_ptr.cpu().tolist()
    probs = torch.zeros(sum(cards), dtype=theta.dtype)
    logits = []
    assignments = list(itertools.product(*(range(c) for c in cards)))
    for x in assignments:
        score = theta.new_zeros(())
        for f in range(graph.num_factors):
            local = graph.scope(f)
            index = 0
            for i in local:
                index = index * cards[i] + x[i]
            score = score + theta[offsets[f] + index]
        logits.append(score)
    weights = torch.stack(logits).softmax(0)
    p = 0
    for v, size in enumerate(cards):
        for state in range(size):
            probs[p + state] = weights[[x[v] == state for x in assignments]].sum()
        p += size
    return probs


class TopologyTests(unittest.TestCase):
    def test_scope_offsets_and_row_major_order(self):
        g = FactorGraph.from_scopes([2, 3, 2], [(0,), (0, 2), (1, 0), (), (1, 2)])
        self.assertEqual(g.num_variables, 3)
        self.assertEqual(g.num_factors, 5)
        self.assertEqual(g.factor_ptr.tolist(), [0, 1, 3, 5, 5, 7])
        self.assertEqual(g.potential_sizes.tolist(), [2, 4, 6, 1, 6])
        self.assertEqual(g.potential_ptr.tolist(), [0, 2, 6, 12, 13, 19])
        self.assertEqual(g.scope(2), (1, 0))
        p = g.compile()
        # (x_1, x_0) has 3x2 entries, with x_0 varying fastest.
        rows = p.entry_message.tolist()
        cfg_start, cfg_end = p.config_ptr[2:4].tolist()
        self.assertEqual(cfg_end - cfg_start, 6)
        self.assertEqual(len(rows), sum(g.factor_sizes.mul(g.potential_sizes).tolist()))
        self.assertEqual(p.num_message_states, sum(g.cardinalities[g.factor_vars].tolist()))

    def test_invalid_scope_and_batch_detection(self):
        for scopes in ([(0, 0)], [(2,)], [(-1,) ]):
            with self.assertRaises(ValueError):
                FactorGraph.from_scopes([2, 2], scopes)
        with self.assertRaises(ValueError):
            FactorGraph.from_scopes([2, 0], [(0,)])
        with self.assertRaises(ValueError):
            FactorGraph.batch([])

    def test_batch_disconnected_graphs(self):
        a = FactorGraph.from_scopes([2, 2], [(0,), (0, 1)])
        b = FactorGraph.from_scopes([3], [(0,)])
        c = FactorGraph.batch([a, b])
        self.assertEqual(c.num_graphs, 2)
        self.assertEqual(c.variable_graph_ptr.tolist(), [0, 2, 3])
        self.assertEqual(c.factor_graph_ptr.tolist(), [0, 2, 3])
        self.assertEqual(c.scope(2), (2,))
        self.assertEqual(c.num_configurations, a.num_configurations + b.num_configurations)
        self.assertEqual(FactorGraph.batch([a, c]).num_graphs, 3)

    def test_large_dense_factor_is_capped(self):
        g = FactorGraph.from_scopes([2] * 16, [range(16)])
        with self.assertRaisesRegex(ValueError, "exceeding max_factor_states"):
            g.compile(max_factor_states=1000)
        with self.assertRaisesRegex(ValueError, "max_entries"):
            g.compile(max_entries=1000)


class LoopyBeliefPropagationTests(unittest.TestCase):
    def setUp(self):
        self.g = FactorGraph.from_scopes([2, 2, 3], [(0,), (0, 1), (1, 2), (2,)])
        self.p = self.g.compile()
        self.theta = torch.tensor([.2, -.3, .1, .4, -.2, .7,
                                   .3, -.1, .5, .2, -.1, .8,
                                   -.5, .1, .6], dtype=torch.float64)
        self.assertEqual(len(self.theta), self.p.num_configurations)

    def test_tree_equals_exact_enumeration(self):
        ref = exact_variable_marginals(self.g, self.theta)
        result = lbp(self.p, self.theta, iterations=12)
        torch.testing.assert_close(result.variable_beliefs, ref, rtol=1e-9, atol=1e-9)
        self.assertEqual(result.iterations, 12)
        self.assertEqual(result.algorithm, "lbp")
        for f in range(self.g.num_factors):
            torch.testing.assert_close(result.factor_marginal(f).sum(), torch.tensor(1., dtype=self.theta.dtype))
        self.assertEqual(result.marginal(2).numel(), 3)

    def test_damping_and_early_stop(self):
        for damping in (0., .3, .65):
            out = lbp(self.p, self.theta, iterations=120, damping=damping, tol=1e-11)
            torch.testing.assert_close(out.variable_beliefs, exact_variable_marginals(self.g, self.theta),
                                       atol=1e-8, rtol=1e-8)
            self.assertTrue(out.converged)
            self.assertLess(out.iterations, 120)

    def test_hard_constraints_and_unaries(self):
        g = FactorGraph.from_scopes([2, 2], [(0, 1)])
        theta = torch.tensor([-torch.inf, 0., 0., -torch.inf], dtype=torch.float64)
        out = lbp(g.compile(), theta, iterations=5)
        self.assertTrue(torch.isfinite(out.variable_beliefs).all())
        torch.testing.assert_close(out.variable_beliefs, torch.full((4,), .5, dtype=theta.dtype))
        torch.testing.assert_close(out.factor_marginal(0),
                                   torch.tensor([0., .5, .5, 0.], dtype=theta.dtype))
        unary = FactorGraph.from_scopes([2], [(0,)])
        constrained = lbp(unary.compile(), torch.tensor([0., -torch.inf]), iterations=2)
        torch.testing.assert_close(constrained.marginal(0), torch.tensor([1., 0.]))

    def test_isolated_variables_and_empty_factors(self):
        graph = FactorGraph.from_scopes([2, 3], [()])
        out = lbp(graph.compile(), torch.tensor([42.]), iterations=2)
        torch.testing.assert_close(out.variable_beliefs, torch.tensor([.5, .5, 1/3, 1/3, 1/3]))
        torch.testing.assert_close(out.factor_marginal(0), torch.tensor([1.]))

    def test_batched_results_match_separate_results(self):
        other = FactorGraph.from_scopes([3], [(0,)])
        theta_other = torch.tensor([.1, -.7, .8], dtype=self.theta.dtype)
        union = FactorGraph.batch([self.g, other])
        actual = lbp(union.compile(), torch.cat((self.theta, theta_other)), iterations=14)
        a = lbp(self.p, self.theta, iterations=14)
        b = lbp(other.compile(), theta_other, iterations=14)
        torch.testing.assert_close(actual.variable_beliefs,
                                   torch.cat((a.variable_beliefs, b.variable_beliefs)))

    def test_gradcheck(self):
        theta = self.theta.detach().clone().requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(lambda t: lbp(self.p, t, iterations=5,
                                         damping=.2).variable_beliefs, (theta,), eps=1e-6, atol=1e-5))
        result = LBP(iterations=5)(self.p, theta)
        result.marginal(1)[0].backward()
        self.assertIsNotNone(theta.grad)
        self.assertTrue(torch.isfinite(theta.grad).all())

    def test_shape_device_and_parameter_validation(self):
        with self.assertRaises(ValueError):
            lbp(self.p, self.theta[:-1])
        with self.assertRaises(ValueError):
            lbp(self.p, self.theta, damping=-.1)
        with self.assertRaises(ValueError):
            lbp(self.p, self.theta, iterations=0)
        with self.assertRaises(ValueError):
            lbp(self.p, self.theta, initial_messages=torch.zeros(3))
        with self.assertRaises(IndexError):
            lbp(self.p, self.theta).marginal(99)


class MeanFieldTests(unittest.TestCase):
    def test_unary_factors_are_exact(self):
        graph = FactorGraph.from_scopes([2, 3], [(0,), (1,)])
        theta = torch.tensor([-.2, .9, .3, -.4, .7], dtype=torch.float64)
        out = mean_field(graph.compile(), theta, iterations=4)
        torch.testing.assert_close(out.variable_beliefs, torch.cat((theta[:2].softmax(0), theta[2:].softmax(0))))
        self.assertIsNone(out.factor_beliefs)
        with self.assertRaisesRegex(ValueError, "not provided"):
            out.factor_marginal(0)

    def test_pairwise_first_iteration_against_reference(self):
        graph = FactorGraph.from_scopes([2, 2], [(0, 1)])
        theta = torch.tensor([-.3, .2, 1.1, -.1], dtype=torch.float64)
        expected0 = torch.stack((theta[:2].mean(), theta[2:].mean())).softmax(0)
        expected1 = torch.stack((theta[[0, 2]].mean(), theta[[1, 3]].mean())).softmax(0)
        actual = mean_field(graph.compile(), theta, iterations=1)
        torch.testing.assert_close(actual.variable_beliefs,
                                   torch.cat((expected0, expected1)), atol=1e-13, rtol=1e-13)

    def test_batching_damping_early_stopping(self):
        a = FactorGraph.from_scopes([2], [(0,)])
        b = FactorGraph.from_scopes([3], [(0,)])
        theta = torch.tensor([0., 0., 1., 0., -.8], dtype=torch.float64)
        plan = FactorGraph.batch([a,b]).compile()
        out = mean_field(plan, theta, iterations=120, damping=.3, tol=1e-11)
        self.assertTrue(out.converged)
        torch.testing.assert_close(out.variable_beliefs,
                                   torch.cat((theta[:2].softmax(0), theta[2:].softmax(0))), atol=1e-10, rtol=1e-10)

    def test_autograd_and_gradcheck(self):
        graph = FactorGraph.from_scopes([2, 3], [(0, 1), (0,)])
        theta = torch.tensor([-.1, .5, .8, -.4, .2, .1, .4, -.3],
                             dtype=torch.float64, requires_grad=True)
        plan = graph.compile()
        self.assertTrue(torch.autograd.gradcheck(lambda t: mean_field(plan, t, iterations=6,
                                    damping=.1).variable_beliefs, (theta,), eps=1e-6, atol=1e-5))
        result = NaiveMeanField(iterations=8)(plan, theta)
        result.marginal(1)[1].backward()
        self.assertTrue(torch.isfinite(theta.grad).all())

    def test_hard_constraints_rejected_and_invalid_params(self):
        p = FactorGraph.from_scopes([2], [(0,)]).compile()
        with self.assertRaisesRegex(ValueError, "finite log potentials"):
            mean_field(p, torch.tensor([0., -torch.inf]))
        with self.assertRaises(ValueError):
            mean_field(p, torch.tensor([0., 1.]), eps=0.)


class PropertyAndCompileTests(unittest.TestCase):
    def test_random_categorical_trees_against_exact(self):
        rng = random.Random(2026)
        torch.manual_seed(2026)
        for _ in range(20):
            n = rng.randint(1, 5)
            cards = [rng.randint(2, 4) for _ in range(n)]
            scopes = [(i,) for i in range(n)] + [
                (i, rng.randrange(i)) for i in range(1, n)
            ]
            rng.shuffle(scopes)
            g = FactorGraph.from_scopes(cards, scopes)
            theta = torch.randn(g.num_configurations, dtype=torch.float64)
            actual = lbp(g.compile(), theta, iterations=12)
            exact = exact_variable_marginals(g, theta)
            torch.testing.assert_close(actual.variable_beliefs, exact, rtol=1e-9, atol=1e-9)

    @unittest.skipUnless(hasattr(torch, "compile"), "torch.compile unavailable")
    def test_fullgraph_torch_compile_fixed_iterations(self):
        g = FactorGraph.from_scopes([2, 3], [(0, 1), (1,)])
        plan = g.compile()
        theta = torch.randn(plan.num_configurations, dtype=torch.float64,
                            requires_grad=True)
        # The optional eager backend validates fullgraph traceability without
        # requiring inductor/Triton or a CUDA machine in the CI runner.
        for module in (LBP(iterations=3),
                       NaiveMeanField(iterations=3, validate_finite=False)):
            compiled = torch.compile(module, backend="eager", fullgraph=True)
            actual = compiled(plan, theta).variable_beliefs
            reference = module(plan, theta).variable_beliefs
            torch.testing.assert_close(actual, reference)



@unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable in this environment")
class CUDAInferenceTests(unittest.TestCase):
    def test_cuda_parity_and_autograd(self):
        g = FactorGraph.from_scopes([2, 3], [(0,), (0, 1)])
        cpu = g.compile()
        theta = torch.arange(g.num_configurations, dtype=torch.float64) / 10
        gpu = cpu.to("cuda")
        actual = lbp(gpu, theta.cuda().requires_grad_(True), iterations=8)
        reference = lbp(cpu, theta, iterations=8)
        torch.testing.assert_close(actual.variable_beliefs.cpu(), reference.variable_beliefs,
                                   rtol=1e-8, atol=1e-8)
        actual.variable_beliefs[0].backward()
        self.assertTrue(torch.isfinite(actual.variable_beliefs).all())
        mf = mean_field(gpu, theta.cuda(), iterations=8)
        torch.testing.assert_close(mf.variable_beliefs.cpu(),
                                   mean_field(cpu, theta, iterations=8).variable_beliefs)


try:
    import torch_geometric  # noqa: F401
except ImportError:
    HAS_PYG = False
else:
    HAS_PYG = True


@unittest.skipUnless(HAS_PYG, "torch_geometric not installed")
class PyGTests(unittest.TestCase):
    def test_heterodata_and_loader_round_trip(self):
        from torch_geometric.loader import DataLoader
        a = FactorGraph.from_scopes([2, 3], [(0, 1), (0,), (0, 1)])
        b = FactorGraph.from_scopes([2], [(0,)])
        pa = (torch.arange(a.num_configurations, dtype=torch.float32) * .1).requires_grad_()
        pb = (torch.arange(b.num_configurations, dtype=torch.float32) * .1).requires_grad_()
        data = [a.to_pyg(pa), b.to_pyg(pb)]
        for g, d in zip((a, b), data):
            restored = FactorGraph.from_pyg(d)
            self.assertEqual(restored.factor_vars.tolist(), g.factor_vars.tolist())
            self.assertEqual(restored.potential_ptr.tolist(), g.potential_ptr.tolist())
        batch = next(iter(DataLoader(data, batch_size=2)))
        reconstructed = FactorGraph.from_pyg(batch)
        expected = FactorGraph.batch([a, b])
        self.assertEqual(reconstructed.num_graphs, 2)
        self.assertEqual(reconstructed.factor_vars.tolist(), expected.factor_vars.tolist())
        tensor = FactorGraph.potentials_from_pyg(batch)
        self.assertEqual(tensor.numel(), expected.num_configurations)
        out = lbp(reconstructed.compile(), tensor, iterations=5)
        loss = out.marginal(0)[1] + out.marginal(2)[1]
        loss.backward()
        self.assertIsNotNone(pa.grad)
        self.assertIsNotNone(pb.grad)
        self.assertGreater(pa.grad.abs().sum().item(), 0.)
        self.assertGreater(pb.grad.abs().sum().item(), 0.)


if __name__ == "__main__":
    unittest.main()
