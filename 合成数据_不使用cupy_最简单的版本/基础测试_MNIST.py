from useful_functions_with_batch import *
from opt_function_with_batch import *
import os
import sys
import json
import time
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from scripts_pushpull_differ_lr.experiment_utils import generate_topology_matrices, compute_possible_c
from opt_function_with_batch import PushPull_with_batch_different_lr
from training import train_track_grad_norm_different_learning_rate, train_track_grad_norm_with_hetero_different_learning_rate

time_integer = int(time.time())
random_integer = np.random.randint(1, 1000000000, size=2)[0]

max_it = 10000

alpha = 0.1  # 高值接近均匀分布，低值（如0.1）高度异质

d = 10
L_total = 1440000
n = 4 # num_nodes, 6,12,18,24

topology = "exp"
matrix_seed = 51583
lr_basic = 8e-1
sample_seed = 42
num_samples = 20 

# 使用标准的全局数据生成函数（数据本身是uniform的）
h_global_cpu, y_global_cpu, x_opt_cpu = init_global_data(d=d, L_total=L_total, seed=42)
print("h:",h_global_cpu.shape)
print("y:",y_global_cpu.shape)
print("Number of positive samples:", np.sum(y_global_cpu == 1))
print("Number of negative samples:", np.sum(y_global_cpu == -1))

init_x_cpu_single = init_x_func(n=n, d=d, seed=42)

# 使用异质数据分配函数
# h_tilde_cpu, y_tilde_cpu = distribute_data(h=h_global_cpu, y=y_global_cpu, n=n)
h_tilde_cpu, y_tilde_cpu = distribute_data_hetero(h=h_global_cpu, y=y_global_cpu, n=n, alpha=alpha, seed=42)
print("h_tilde:",h_tilde_cpu.shape)
print("y_tilde:",y_tilde_cpu.shape)

# 打印每个节点的数据分布信息
print("\nData distribution across nodes:")
for i in range(n):
    n_pos = np.sum(y_tilde_cpu[i] == 1)
    n_neg = np.sum(y_tilde_cpu[i] == -1)
    pos_ratio = n_pos / len(y_tilde_cpu[i])
    print(f"Node {i}: {n_pos} positive ({pos_ratio:.2%}), {n_neg} negative ({(1-pos_ratio):.2%})")

A_cpu, B_cpu = generate_topology_matrices(topology=topology, n=n, matrix_seed=matrix_seed, k=3)
print("\nCPU数据准备完毕")

lr_list = [lr_basic] * n

results = compute_possible_c(
    A=A_cpu, 
    B=B_cpu, 
    lr_basic=lr_basic, 
    n=n,
    num_samples=num_samples,
    sample_seed=sample_seed
    )

# Get user input for i and strategy
i = int(input("Enter an integer value for i: "))
strategy = input("Enter a value for strategy (string): ")

# Use the input values
c = results[i][0]
d_list = results[i][1]

print(sum(d_list))
lr_list = [lr_basic* i for i in d_list]

config = {
    "n": n,
    "strategy": strategy,
    "num_samples": num_samples,
    "topology": topology,
    "matrix_seed": matrix_seed,
    "c": c,
    "lr_basic": lr_basic,
    "d_list": d_list,
    "lr_list": lr_list,
    "heterogeneous": True,
    "alpha": alpha,
}

# save config to a file
config_file_path = f"/Users/luogan/Code/new_push_pull/合成数据_不使用cupy_最简单的版本/result_differ_lr_and_hetero/{strategy}_{topology}_hetero_alpha{alpha}_config.json"
# Create directory if it doesn't exist
print(f"\nSaving config to {config_file_path}")
os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
with open(config_file_path, 'w') as config_file:
    json.dump(config, config_file, indent=4)
# 保存这个配置文件
print("配置文件保存成功.")

rho=0.1
df = train_track_grad_norm_with_hetero_different_learning_rate(
    algorithm="PushPull",
    lr_list=lr_list,
    A=A_cpu,
    B=B_cpu,
    dataset_name="MNIST",
    batch_size=128,
    num_epochs=10,
    remark="test",
    alpha=alpha,
    root=None,
    use_hetero=True,
    device="cuda:0",
    seed=42,
)

df.to_csv(f"/Users/luogan/Code/new_push_pull/合成数据_不使用cupy_最简单的版本/result_differ_lr_and_hetero/rho_{rho}_{strategy}_{topology}_hetero_alpha{alpha}_lr_basic={lr_basic}.csv")