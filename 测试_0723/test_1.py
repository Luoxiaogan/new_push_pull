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

n=16
lr_basic = 2e-3
A, B = generate_topology_matrices("neighbor", n=n, matrix_seed=51583, k=3)
show_row(A)
show_col(B)

pi_A = get_left_perron(A)
pi_B = get_right_perron(B)
# 计算 hadamard 积
pi_A_hadamard_pi_B = pi_A * pi_B
print("Hadamard Product:", pi_A_hadamard_pi_B )
# 计算hadamard 积 pi_A_hadamard_pi_B 的按照从大到小排序的索引
sorted_indices = np.argsort(pi_A_hadamard_pi_B)[::-1]
print("Sorted Indices:", sorted_indices)

print("\n pi_a_inverse:")
lr_list = compute_learning_rates(
    strategy="pi_a_inverse",
    A=A,
    B=B,
    lr_basic=lr_basic,
    n=n,
)
c = compute_c_value(
    A=A,
    B=B,
    lr_list=lr_list,
    lr_basic=lr_basic,
)
# print("Learning Rates:", lr_list)
print("C Value:", c)

print("\n pi_b_inverse:")
lr_list = compute_learning_rates(
    strategy="pi_b_inverse",
    A=A,
    B=B,
    lr_basic=lr_basic,
    n=n,
)
c = compute_c_value(
    A=A,
    B=B,
    lr_list=lr_list,
    lr_basic=lr_basic,
)
# print("Learning Rates:", lr_list)
print("C Value:", c)


print("\n uniform:")
lr_list = compute_learning_rates(
    strategy="uniform",
    A=A,
    B=B,
    lr_basic=lr_basic,
    n=n,
)
c = compute_c_value(
    A=A,
    B=B,
    lr_list=lr_list,
    lr_basic=lr_basic,
)
# print("Learning Rates:", lr_list)
print("C Value:", c)


from utils.d_matrix_utils import generate_d_matrices_with_increasing_c  
tuple_list = generate_d_matrices_with_increasing_c(
    A=A,
    B=B,
    num_c_values=10,
    num_samples=1000,  # Not used in theoretical approach
    distribution="vertices"
)
for d_diagonal, c_value in tuple_list:
    print(f"C Value: {c_value}")
