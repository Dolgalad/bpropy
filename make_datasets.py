"""
Create labeled datasets for supervised training
"""

import sys
import argparse
import networkx as nx

from dopy.maxcut import *

parser = argparse.ArgumentParser(prog="make_datasets",
                                 description="dataset generation",
                                 )
dataset_dir = "."

# Maximum Independent Set Erdos-Renyi graph
def generate_instance(n, p):
    g = nx.generators.random_graphs.erdos_renyi_graph(n, p)
    pb = MaxCut(g)

    # get pool of solutions
