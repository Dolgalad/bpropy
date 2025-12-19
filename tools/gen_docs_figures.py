"""
Generate images for documentation
"""

import matplotlib.pyplot as plt
import networkx as nx

# MIS examples
from bpropy.co.mis import MIS
g_house = nx.Graph()
g_house.add_edges_from([(0,1),(1,2),(2,3),(3,0),(1,4),(2,4)])
pb_house = MIS(g_house)
pos_house = nx.layout.kamada_kawai_layout(g_house)

g_tie = nx.Graph()
g_tie.add_edges_from([(0,1),(1,2),(2,0),(3,0),(1,4),(2,4)])
pb_tie = MIS(g_tie)
pos_tie = nx.layout.kamada_kawai_layout(g_tie)

pb_batch = MIS.batch([pb_house,pb_tie])
sol_batch = pb_batch.solve()

def translate_pos(pos, n_offset=0, dx=0.0, dy=0.0):
    return {n+n_offset: (x + dx, y + dy) for n, (x, y) in pos.items()}

pos_tie_shifted = translate_pos(pos_tie, n_offset=5, dx=3.0)
pos_batch = {}
pos_batch.update(pos_house)
pos_batch.update(pos_tie_shifted)


fig,ax = plt.subplots(1,2)
nx.draw(pb_house.g, ax=ax[0], pos=pos_house)
nx.draw_networkx_labels(pb_house.g, pos_house, ax=ax[0])
sol = pb_house.solve()
nx.draw(pb_house.g, ax=ax[1], pos=pos_house, node_color=sol.x, cmap="PiYG")
nx.draw_networkx_labels(pb_house.g, pos_house, ax=ax[1])
plt.tight_layout()
plt.savefig("docs/assets/mis_solution_house_example.svg")
plt.close(fig)

fig,ax = plt.subplots(1,2)
nx.draw(pb_tie.g, ax=ax[0], pos=pos_tie)
nx.draw_networkx_labels(pb_tie.g, pos_tie, ax=ax[0])
sol = pb_tie.solve()
nx.draw(pb_tie.g, ax=ax[1], pos=pos_tie, node_color=sol.x, cmap="PiYG")
nx.draw_networkx_labels(pb_tie.g, pos_tie, ax=ax[1])
plt.tight_layout()
plt.savefig("docs/assets/mis_solution_tie_example.svg")
plt.close(fig)

fig,ax = plt.subplots()
nx.draw(pb_batch.g, pos=pos_batch, ax=ax)
nx.draw_networkx_labels(pb_batch.g, pos_batch, ax=ax)
plt.tight_layout()
plt.savefig("docs/assets/mis_house_tie_batch.svg")
plt.close()

fig,ax = plt.subplots(1,3)
nx.draw(pb_batch.g, pos=pos_batch, ax=ax[0])
nx.draw_networkx_labels(pb_batch.g, pos_batch, ax=ax[0])
ax[0].set_title("Batch")

pb1 = pb_batch.get_problem(0)
pb2 = pb_batch.get_problem(1)

nx.draw(pb1.g, pos=pos_house, ax=ax[1])
nx.draw_networkx_labels(pb1.g, pos_house, ax=ax[1])
ax[1].set_title("Subproblem 0")

nx.draw(pb2.g, pos=pos_tie, ax=ax[2])
nx.draw_networkx_labels(pb2.g, pos_tie, ax=ax[2])
ax[2].set_title("Subproblem 1")

plt.tight_layout()
plt.savefig("docs/assets/mis_house_tie_batch_subproblems.svg")
plt.close()

print("MIS.solve output")
print(">>> sol = pb_house.solve()")
print(">>> sol.x")
sol = pb_house.solve()
print(sol.x)

print("MIS.greedy output")
print(">>> sol = pb_tie.greedy()")
print(">>> sol.x")
sol = pb_tie.greedy()
print(sol.x)

print("MIS.set_cliques output")
print(">>> pb_tie.C")
print(pb_tie.C)

fig,ax = plt.subplots(1,2)
A,_,_ = pb_tie.linear_constraints_matrix()
ax[0].imshow(A.toarray(), cmap="viridis")
ax[0].set_title("Edge constraints")

print(">>> pb_tie.set_cliques([[0,1,2],[1,2,4],[0,3]])")
pb_tie.set_cliques([[0,1,2],[1,2,4],[0,3]])
print(">>> pb_tie.C")
print(pb_tie.C)

A,_,_ = pb_tie.linear_constraints_matrix()
ax[1].imshow(A.toarray(), cmap="viridis")
ax[1].set_title("Clique constraints")
plt.tight_layout()
plt.savefig("docs/assets/mis_tie_constraint_matrix.svg")
plt.close(fig)

print("MIS.linear_constraints_matrix output")
print(">>> A,lb,ub = pb_house.linear_constraints_matrix()")
A,lb,ub = pb_house.linear_constraints_matrix()
print(">>> A")
print(A)
print(">>> lb,ub")
print(lb,ub)
