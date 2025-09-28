import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加父目录到路径以导入函数
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from useful_functions_with_batch import (
    get_xinmeng_matrix,
    get_xinmeng_like_matrix,
    compute_kappa_col,
    compute_beta_col,
    compute_S_B_col,
    get_right_perron,
    compute_2st_eig_value
)
# Test with multiple seeds
for seed in range(10, 100, 1):
    A = get_xinmeng_like_matrix(20, seed)
    pi_A = get_right_perron(A)
    max_index = np.argmax(pi_A)
    print(f"Seed {seed}: Max element at index {max_index}")