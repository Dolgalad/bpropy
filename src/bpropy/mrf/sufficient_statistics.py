"""Sufficient statistics manipulation
"""
import numpy as np

def x2ss(x, fg):
    factor_ss = []
    for F in fg.iter_factors():
        factor_ss.append(fg.factor_sufficient_statistics(F, x[fg.factor_variables(F)]))
    return np.concatenate(factor_ss, dtype=np.uint32)

def ss2x(ss, fg):
    x = np.zeros(fg.nv(), dtype=np.uint32)
    offset = 0
    for F in fg.iter_factors():
        nF = fg.factor_size(F)
        x[fg.factor_variables(F)] = fg.factor_assignment(F, ss[offset:offset+2**nF])
        offset += 2**nF
    return x



