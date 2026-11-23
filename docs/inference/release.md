# Publishing readiness and compatibility checklist

This work is a **development preview**, not a published release.

## Before a PyPI release

- [ ] Choose an explicit, appropriate copyright/license policy for **original**
      inference code; add a repository license only after review. Do not assume
      rights to relicense bundled third-party code.
- [ ] Audit `src/bpropy/vendor/scg3/` licensing and provenance; decouple its
      C compilation from a pure-PyTorch install, or separately package it.
- [ ] Restructure the legacy extras so `pip install bpropy` installs only a
      documented minimal dependency set. The current top-level distribution
      still pulls in combinatorial optimization / plotting dependencies.
- [x] Repair the pre-existing legacy LBP batch-test convergence mismatch;
      162 tests pass (4 skipped) on Python 3.10/3.11/3.12. Keep both the
      legacy test matrix and new inference-core tests as separate CI gates.
- [ ] Test Windows/Linux/macOS packaging, Python version compatibility and
      PyTorch CPU-only installation from a fresh virtual environment.
- [ ] Run the new CUDA tests on supported NVIDIA hardware and record numerical
      parity (beliefs and gradients), peak memory and runtime. The CUDA suite
      and reproducible benchmarks are implemented, but GPU measurements are
      not yet available.
- [ ] Benchmark varying graph sizes, factor degrees, cardinalities, batch sizes,
      inference iteration counts and changing versus cached topologies.
- [x] Validate PyG `HeteroData` / `DataLoader` interoperability and gradient
      propagation in a dedicated **CPU** PyG CI job.
- [ ] Exercise `torch.compile`, mixed precision and (if relevant) CUDA Graphs;
      clearly document supported compilation and autograd modes.
- [ ] Handle contradictory hard factors (`-inf`) with informative diagnostics;
      NMF currently requires all finite energies.
- [ ] Decide whether `bpropy` (which contains many research experiments) or a
      distinct PyPI distribution is the right long-term public package name.
- [ ] Configure reproducible versioning, provenance and release artifacts;
      build wheel/sdist, inspect contents, upload to **TestPyPI** first.

No branch merge, repo visibility change, artifact upload or publishing should
happen without an explicit approval and passing tests.
