import os
import sys
import numpy as np
import pandas as pd
# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from scripts_pushpull_differ_lr.experiment_utils import generate_topology_matrices, compute_learning_rates, compute_c_value
from utils.algebra_utils import show_col, show_row, get_left_perron, get_right_perron

from scripts_pushpull_differ_lr.experiment_utils import generate_topology_matrices, compute_possible_c

# 生成通信拓扑
n = 16
A, B = generate_topology_matrices("neighbor", n=16, matrix_seed=51583, k=3)

# 计算所有可能的c值
lr_basic = 2e-3
results = compute_possible_c(
    A=A, 
    B=B, 
    lr_basic=lr_basic, 
    n=n,
    num_samples=20,
    sample_seed=42
    )

print("\n")
print(results[19][1])
