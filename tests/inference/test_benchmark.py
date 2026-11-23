"""Benchmark CLI correctness and output schema tests (CPU-only and fast)."""
from __future__ import annotations

import contextlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch


BENCHMARK_FILE = Path(__file__).resolve().parents[2] / "benchmarks" / "inference_core.py"
MODULE_SPEC = importlib.util.spec_from_file_location("inference_benchmark", BENCHMARK_FILE)
benchmark = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def test_cli_report_backward_and_exports(self):
        with tempfile.TemporaryDirectory() as root:
            json_out = Path(root) / "nested" / "report.json"
            csv_out = Path(root) / "nested" / "report.csv"
            report = benchmark.run([
                "--device", "cpu", "--variables", "3", "--factors", "4",
                "--graphs", "2", "--cardinality", "3", "--arity", "2",
                "--mixed-cardinalities", "--dtype", "float64",
                "--iterations", "2", "--repeats", "2", "--warmup", "1",
                "--json-out", str(json_out), "--csv-out", str(csv_out),
            ])
            self.assertEqual(report["schema_version"], 2)
            self.assertEqual(report["graph"]["variables"], 6)
            self.assertEqual(report["graph"]["factors"], 8)
            self.assertGreater(report["graph"]["configuration_entries"], 0)
            self.assertGreater(report["graph"]["compiled_index_mib"], 0)
            self.assertTrue(json_out.exists() and csv_out.exists())
            self.assertEqual(json.loads(json_out.read_text())["parameters"]["graphs"], 2)
            self.assertEqual(len(csv_out.read_text().splitlines()), 7)  # 2 algorithms x 3 phases
            for algorithm in ("lbp", "mean_field"):
                for phase in ("forward", "backward_only", "forward_backward"):
                    self.assertGreater(report["benchmarks"][algorithm][phase]["wall_ms_median"], 0)

    def test_forward_only_and_compiled_eager(self):
        report = benchmark.run([
            "--device", "cpu", "--variables", "2", "--factors", "2",
            "--iterations", "2", "--warmup", "1", "--repeats", "1",
            "--compile", "eager", "--no-backward",
        ])
        for name in ("lbp", "mean_field"):
            self.assertEqual(list(report["benchmarks"][name]), ["forward"])

    def test_graph_generation_is_seeded(self):
        kw = dict(variables=4, factors=5, cardinality=3, arity=2,
                  graphs=2, seed=2026, mixed_cardinalities=True)
        a = benchmark.build_graph(**kw)
        b = benchmark.build_graph(**kw)
        torch.testing.assert_close(a.cardinalities, b.cardinalities)
        torch.testing.assert_close(a.factor_vars, b.factor_vars)
        self.assertEqual(a.variable_graph_ptr.tolist(), [0, 4, 8])

    def test_legacy_compatible_scope_sorting(self):
        graph = benchmark.build_graph(variables=5, factors=12, cardinality=2,
                                     arity=2, graphs=1, seed=2026,
                                     mixed_cardinalities=False, legacy_compatible=True)
        scopes = [graph.scope(f) for f in range(graph.num_factors)]
        self.assertEqual(scopes, sorted(scopes))
        self.assertTrue(all(x == tuple(sorted(x)) for x in scopes))

    def test_cpu_reference_requires_cuda(self):
        with contextlib.redirect_stderr(__import__("io").StringIO()):
            with self.assertRaises(SystemExit):
                benchmark.run(["--device", "cpu", "--verify-cpu"])


if __name__ == "__main__":
    unittest.main()
