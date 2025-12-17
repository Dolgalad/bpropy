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
dataset_dir = sys.argv[1]

# Maximum Independent Set Erdos-Renyi graph
def generate_instance(n, p):
    g = nx.generators.random_graphs.erdos_renyi_graph(n, p)
    # solve with kamis
