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
- [ ] Reproduce and repair the **existing** `bpropy` full unit-test CI failures
      separately from the new standalone inference suite.
- [ ] Test Windows/Linux/macOS packaging, Python version compatibility and
      PyTorch CPU-only installation from a fresh virtual environment.
- [ ] Run CUDA tests on supported NVIDIA hardware and record numerical parity,
      peak memory and real runtime against the old scatter implementation.
- [ ] Benchmark varying graph sizes, factor degrees, cardinalities, batch sizes,
      inference iteration counts and changing versus cached topologies.
- [ ] Validate PyG `HeteroData` / `DataLoader` interoperability and gradient
      propagation in a dedicated PyG CI job.
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
