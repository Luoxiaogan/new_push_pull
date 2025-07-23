import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from scripts_pushpull_differ_lr.experiment_utils import generate_topology_matrices, compute_learning_rates, compute_c_value
from utils.algebra_utils import (
    show_col, 
    show_row, 
    get_left_perron, 
    get_right_perron,
    compute_kappa_col,
    compute_kappa_row
    )

from scripts_pushpull_differ_lr.experiment_utils import generate_topology_matrices, compute_possible_c

# 生成通信拓扑
n = 16
# A, B = generate_topology_matrices("neighbor", n=n, matrix_seed=51583, k=3)
index = 0
max_kappa = 1

for i in range(60000):
    A, B = generate_topology_matrices("ring", n=n, matrix_seed=i, k=2)
    kappa = compute_kappa_row(A)
    if kappa > max_kappa and kappa <100000 and nx.is_strongly_connected(nx.DiGraph(A)):
        max_kappa = kappa
        index = i
        kappa_b = compute_kappa_col(B)
        print(f"New maximum kappa_A: {max_kappa} and kappa_B: {kappa_b} at seed {index}")
