"""MIS solution class
"""
import numpy as np

class MISSolution:
    def __init__(self, x, duals=None, slacks=None, varbasis_onehot=None, reduced_costs=None, solve_time=None):
        self.x = x
        self.duals = duals
        self.slacks = slacks
        self.varbasis_onehot = varbasis_onehot
        self.reduced_costs = reduced_costs
        self.solve_time = solve_time

    @staticmethod
    def from_scipy_milp(r):
        return MISSolution(r.x)
    @staticmethod
    def from_cplex(sol, md):
        n = md.number_of_variables
        varlist = [md.get_var_by_name(f"x{i}") for i in range(n)]
        #n = len(varlist)

        varbasis = np.array([bstt.value for bstt in md.var_basis_statuses(varlist)])
        varbasis_onehot = np.zeros((varbasis.size, 4))
        varbasis_onehot[np.arange(varbasis.size), varbasis] = 1

        x = np.array([sol.get_value(var) for var in varlist])
        reduced_costs = np.array(sol.get_reduced_costs(varlist))

        # constraints
        constraints = list(md.iter_constraints())
    
        slacks = np.array(md.slack_values(constraints))
        duals = np.array(md.dual_values(constraints))


        return MISSolution(
                x,
                duals,
                slacks,
                varbasis_onehot,
                reduced_costs,
                )
    @staticmethod
    def from_python_mip(x, md):
        n = len(md.vars)
        m = len(md.constrs)
        varbasis = np.zeros(n, dtype=int)
        x = np.zeros(n)
        rc = np.zeros(n)
        duals = np.zeros(m)
        slacks = np.zeros(m)
        for v in md.vars:
            if v.rc == 0:
                varbasis[v.idx] = 1
            else:
                if v.x == v.lb:
                    varbasis[v.idx] = 0
                elif v.x == v.ub:
                    varbasis[v.idx] = 2
                else:
                    varbasis[v.idx] = 3
            x[v.idx] = v.x
            rc[v.idx] = v.rc
        for c in md.constrs:
            duals[c.idx] = c.pi
            slacks[c.idx] = c.slack

        varbasis_onehot = np.zeros((varbasis.size, 4))
        varbasis_onehot[np.arange(varbasis.size), varbasis] = 1
        return MISSolution(
                x,#np.array([x[i].x for i in range(len(x))]),
                duals,#np.array([c.pi for c in md.constrs]),
                slacks,#np.array([c.slack for c in md.constrs]),
                varbasis_onehot,
                rc,#np.array([x[i].rc for i in range(len(x))]),
                )



