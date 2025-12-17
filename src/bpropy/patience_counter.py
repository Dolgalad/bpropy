import numpy as np

class PatienceCounter:
    def __init__(self, limit, sense="min", dim=1):
        self.limit = np.ones(dim, dtype=int) * limit
        self.best_value = np.array([1e30 for _ in range(dim)])
        if sense == "max":
            self.best_value = np.array([-1e30 for _ in range(dim)])
        self.counter = np.zeros(dim)
        self.dim = dim
    def limit_reached(self):
        return np.all(self.counter >= self.limit)
    def update(self, val):
        idx = val < self.best_value
        self.best_value[idx] = val[idx]
        self.counter[idx] = 0
        self.counter[np.logical_not(idx)] += 1


        #for i in range(self.dim):
        #    if val[i] < self.best_value[i]:
        #        self.counter[i] = 0
        #        print(self.best_value[i])
        #        self.best_value[i] = val[i]
        #    else:
        #        self.counter[i] += 1

