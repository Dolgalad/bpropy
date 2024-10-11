from docplex.mp.model import Model

from dopy.maxcut import *

import time

# number of nodes
n = 150

# Randomly generate a graph
g = nx.generators.erdos_renyi_graph(n, 0.15)
print("Graph")
for e in g.edges:
    print("\t", e)

# Solve ILP
w = np.random.random_sample(n)
#w = np.ones(n)
pb = MaxCut(g, weights=w)

print("Solving problem...")
t_sol = time.time()
ilp_sol = pb.solve()
t_sol = time.time() - t_sol
ilp_val = np.dot(w, ilp_sol.x > 0.5)
print(f"Done in {t_sol}s, val = {ilp_val}")

def get_cplex_model(pb):
    mdl = Model("maxcut")
    # add variables
    variables = []
    for i in range(pb.nv()):
        variables.append(mdl.binary_var(name=f"x{i}"))
    variables = np.array(variables, dtype=object)
    ci = 0
    for c in pb.constraints():
        for i in range(c.A.shape[0]):
            if c.ub[i] == c.lb[i]:
                mdl.add_constraint( np.dot(c.A[i,:], variables) == c.ub[i])
                ci += 1
            else:
                if c.ub[i] != np.inf:
                    mdl.add_constraint( np.dot(c.A[i,:], variables ) <= c.ub[i], f"c{ci}" )
                    ci += 1
                if c.lb[i] != -np.inf:
                    mdl.add_constraint( np.dot(c.A[i,:], variables ) >= c.lb[i], f"c{ci+1}" )
                    ci += 1

    mdl.maximize( np.dot( pb.w , variables))
    return mdl

model = get_cplex_model(pb)
model.parameters.mip.pool.intensity=4
model.parameters.mip.pool.relgap = 0.1
print(model)
model.solve()
print(model.get_solve_status())

opt_val = model.objective_value
print(opt_val)

cpx = model.get_cplex()
try:
    cpx.populate_solution_pool()
except CplexSolverError:
    print("Error")

print("Solution pool size : ", cpx.solution.pool.get_num())
for i in range(cpx.solution.pool.get_num()):
    objval_i = cpx.solution.pool.get_objective_value(i)
    sol = np.array(cpx.solution.pool.get_values(i), dtype=int)
    
    print(objval_i, opt_val - objval_i,  np.where(sol > 0)[0])
