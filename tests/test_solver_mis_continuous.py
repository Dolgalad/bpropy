from dopy.maxcut import *

import networkx as nx
import numpy as np
import time

g = nx.generators.random_graphs.erdos_renyi_graph(1000, 0.15)

pb = MaxCut(g)

md = cplex_model(pb, continuous_relaxation=True)

t0 = time.time()
sol = md.solve()
t0 = time.time() - t0

print("solved in ", t0)
print(sol)
