import numpy as np
import mip
from mip import xsum, minimize, BINARY

from bpropy.co.scp_solution import SCPSolution

def number_containing_sets(a, F):
    return sum([a in f for f in F])

def sort_covering(A, F):
    return sorted(A, key=lambda a: number_containing_sets(a, F))

class SCP:
    def __init__(self, A, F, weights=None):
        self.A = sort_covering(A, F)
        self.F = F
        if weights is None:
            self.w = np.ones(len(F))
        else:
            self.w = weights

    def ne(self):
        """Get number of elements
        """
        return len(self.A)
    def nv(self):
        """Get number of variables
        """
        return len(self.F)
    def constraints(self):
        """Get list of scipy.LinearConstraint objects.
        """
        c = []
        for a in A: 
            row = np.array([a in f for f in F])
            c.append(LinearConstraint(row, lb=1))
        return c
    def solve(self, disp=False, solver="cbc", relax_integrality=False):
        """Solve SCP problem
        """
        if solver=="cplex":
            if docplex_spec is None:
                raise Exception("CPLEX solver not found. Missing package 'docplex'")
            md = cplex_model(self, continuous_relaxation=relax_integrality)
            sol = md.solve()
            return SCPSolution.from_cplex(sol, md)
        if solver=="scipy":
            if relax_integrality:
                integrality = np.zeros(self.nv())
            else:
                integrality = np.ones(self.nv())
            return SCPSolution.from_scipy_milp(milp(
                    self.w,
                    integrality=integrality,
                    constraints=self.constraints(cliques=cliques),
                    bounds=Bounds(np.zeros(self.nv()), np.ones(self.nv())),
                    options={"disp":disp}
                    ))
        # default is using CBC
        md = mip.Model("scp")
        x = [md.add_var(lb=0.0, ub=1.0, var_type=BINARY) for i in range(self.nv())]
        md.objective = mip.minimize(xsum(x))
        for a in self.A:
            md += mip.xsum(x[i] for i,f in enumerate(self.F) if a in f) >= 1
        md.verbose = 0
        status = md.optimize(relax=relax_integrality)
        return SCPSolution.from_python_mip(x, md)
    def adj_vec(self, f):
        return np.array([a in f for a in self.A])
    def greedy(self, scores=None):
        # initialize list of covered elements
        covered = np.zeros(self.ne(), dtype=bool)
        x = np.zeros(self.nv(), dtype=bool)
        while not np.all(covered):
            # list candidate sets
            candidate_sets = [i for i in range(self.nv()) if x[i]==0]
            # compute improvement ratio for each set
            improvement_ratios = [self.w[i] / (np.logical_not(covered) & self.adj_vec(self.F[i])).sum() for i in candidate_sets]
            idx = np.argmin(improvement_ratios)
            x[candidate_sets[idx]] = 1
            covered = covered | self.adj_vec(self.F[candidate_sets[idx]])
        return x
    def check_solution(self, x):
        """Check if solution is feasible
        """
        r = np.stack([self.adj_vec(self.F[i]) for i in range(self.nv()) if x[i]==1]).sum(axis=0)
        print(r)
        return np.all(r >= 1)



def adjacency_matrix_to_sets(adj):
    F = []
    for col in adj.T:
        F.append(np.argwhere(col).flatten().tolist())
    return F

def random_scp(n, m):
    adj = np.random.rand(n,m) > 0.5
    # check that every element is covered by at least one set
    for i in range(n):
        if adj[i,:].sum() == 0:
            adj[i, np.random.choice(range(m))] = 1
    return SCP(list(range(n)), adjacency_matrix_to_sets(adj))



if __name__=="__main__":
    import time
    import matplotlib.pyplot as plt
    # generate a random SCP problem
    pb = random_scp(10, 5)
    print("A = ", pb.A)
    print("F = ", pb.F)

    sol = pb.solve()
    print("Opt    = ", sol.x)
    print("\tvalid = ", pb.check_solution(sol.x))

    sol_grd = pb.greedy()
    print("Greedy = ", sol_grd.astype(float))
    print("\tvalid = ", pb.check_solution(sol_grd))

