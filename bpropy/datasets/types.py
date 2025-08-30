from dataclasses import dataclass
import numpy as np

@dataclass
class FeatureExtractionResult:
    node_features: np.ndarray        # [num_vars, d_v]
    constraint_features: np.ndarray  # [num_constraints, d_c]
    edge_features: np.ndarray        # [num_edges, d_e]
    constraints: list                # raw constraint objects (or cliques for MIS)
    feature_mode: str = None         # record how they were extracted

