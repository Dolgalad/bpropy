from functools import cmp_to_key

"""Comparison function for two lists
"""
def lenlex_cmp(c1, c2):
    if len(c1)==len(c2):
        if list(c1) < list(c2):
            return -1
        else:
            return 1
    else:
        return len(c1) - len(c2)

"""Sort list of list by increasing size and lexicographical order
"""
def lenlex_sort(l):
    return sorted(l, key=cmp_to_key(lenlex_cmp))
