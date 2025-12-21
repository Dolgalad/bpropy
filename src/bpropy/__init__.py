"""
`bpropy` defines a differentiable and GPU-friendly MRF inference functions that can be integrated in a `PyTorch` model. The package is split into two main submodules : 

- the `bpropy.co` module defines a base `LPProblem` class that provides functions usefull for representing LP problems, exposes functions for retrieving the decision variable, constraints and is used as input for computing the bipartite representation of the problem
- the `bpropy.mrf` module implements the Loopy Belief Propagation and Naive Mean Field routines for solving inference problems on Graphical Models defined by a Factor Graph.

Submodules:

- [`bpropy.co`](co.md): combinatorial optimization problems (MIS, SCP, ...)
- [`bpropy.mrf`](mrf.md): factor graphs and inference (LBP, TRW, ...)
- [`bpropy.datasets`](api/datasets.md): dataset utilities
- [`bpropy.graphs`](api/graphs.md): graph generators / representations
- [`bpropy.models`](api/models.md): model components
- [`bpropy.utils`](api/utils.md): helpers

"""
