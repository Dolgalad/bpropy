# SCG3
SCG3 is a Lagrangian-relaxation-based solver for the Set Covering Problem that combines:

- Exact-preserving preprocessing (reductions)
- Dual ascent and subgradient optimization to compute strong lower bounds
- Primal recovery heuristics to construct feasible solutions
- Penalty and column-fixing techniques to interatively reduce the problem

It is a heuristic solver, but can prove optimality when the dual and primal bounds meet.
