from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor

import numpy as np

nodes = MarkovNetwork([(0,1),(0,2),(0,3),(0,4),(1,3),(2,3),(3,4)])
for n in range(5):
    nodes.add_factors(
                DiscreteFactor([n], cardinality=[2], values=np.array([0,0.1]))

            )
factor01 = DiscreteFactor([0,1], cardinality=[2,2], values=np.array([0,.5,.5,0]))
factor02 = DiscreteFactor([0,2], cardinality=[2,2], values=np.array([0,.5,.5,0]))
factor03 = DiscreteFactor([0,3], cardinality=[2,2], values=np.array([0,.5,.5,0]))
factor04 = DiscreteFactor([0,4], cardinality=[2,2], values=np.array([0,.5,.5,0]))
factor13 = DiscreteFactor([1,3], cardinality=[2,2], values=np.array([0,.5,.5,0]))
factor23 = DiscreteFactor([2,3], cardinality=[2,2], values=np.array([0,.5,.5,0]))
factor34 = DiscreteFactor([3,4], cardinality=[2,2], values=np.array([0,.5,.5,0]))

nodes.add_factors(factor01)
nodes.add_factors(factor02)
nodes.add_factors(factor03)
nodes.add_factors(factor04)
nodes.add_factors(factor13)
nodes.add_factors(factor23)
nodes.add_factors(factor34)

print(nodes)

from pgmpy.inference import BeliefPropagation

bp = BeliefPropagation(nodes)
bp.calibrate()

print(bp.get_clique_beliefs())
print(bp.query(variables=[0,1,2,3,4]))

print(bp.query(variables=[1,2,3,4], evidence={0:1}))
