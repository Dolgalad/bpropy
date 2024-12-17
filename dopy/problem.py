"""
Base class defining an optimization problem
"""

class Problem:
    def __init__(self):
        pass
    def iter_constraints(self):
        """
        iterate over constraint, yields tuples (lb, A, ub, type)
        """
        yield None
