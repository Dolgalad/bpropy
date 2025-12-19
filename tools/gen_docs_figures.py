"""
Generate images for documentation
"""

import matplotlib.pyplot as plt

from bpropy.co.generators.scp import random_scp_items_k

# SCP matrix
pb = random_scp_items_k(n=15, m=10, k=3)
A,_,_ = pb.linear_constraints_matrix()

fig = plt.figure()
#ax.spy(A, markersize=5)
plt.imshow(A, aspect="auto", interpolation="nearest")
plt.colorbar(label="value")
plt.title("SCP constraint matrix")
plt.savefig("assets/scp_matrix.png")
plt.close(fig)
