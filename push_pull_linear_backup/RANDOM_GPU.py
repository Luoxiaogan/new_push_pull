import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from useful_functions_with_batch import *
from opt_function_with_batch import *
from network_utils import *
import cupy as cp
from cupy_fuc import grad_with_batch_batched_gpu, PushPull_with_batch_batched_gpu

d = 10
L_total = 1440000
n = 6 # num_nodes
num_runs = 20 # 并行运行的次数
device_id = "cuda:0" # 指定GPU

# --- 选择GPU设备 ---
gpu_id_int = int(device_id.split(':')[1])
cp.cuda.Device(gpu_id_int).use()
print(f"Using GPU: {cp.cuda.Device(gpu_id_int).pci_bus_id}")

h_global_cpu, y_global_cpu, x_opt_cpu = init_global_data(d=d, L_total=L_total, seed=42)
print("h:",h_global_cpu.shape)
print("y:",y_global_cpu.shape)

h_tilde_cpu, y_tilde_cpu = distribute_data(h=h_global_cpu, y=y_global_cpu, n=n)
print("h_tilde:",h_tilde_cpu.shape)
print("y_tilde:",y_tilde_cpu.shape)

init_x_cpu_single = init_x_func(n=n, d=d, seed=42)
A_cpu, B_cpu = generate_random_graph_matrices(n=n, seed=42)
print("CPU数据准备完毕")

# --- 将数据迁移到GPU ---
A_gpu = cp.asarray(A_cpu)
B_gpu = cp.asarray(B_cpu)
h_tilde_gpu_nodes = cp.asarray(h_tilde_cpu) # Shape: (n, L, d)
y_tilde_gpu_nodes = cp.asarray(y_tilde_cpu) # Shape: (n, L)

# 为 num_runs 扩展 init_x
init_x_gpu_batched = cp.repeat(cp.asarray(init_x_cpu_single)[cp.newaxis, ...], num_runs, axis=0)
# init_x_gpu_batched shape: (num_runs, n, d)

print("Data moved to GPU.")
print("A_gpu shape:", A_gpu.shape)
print("h_tilde_gpu_nodes shape:", h_tilde_gpu_nodes.shape)
print("init_x_gpu_batched shape:", init_x_gpu_batched.shape)

print(f"\nStarting batched experiment with n={n}, num_runs={num_runs} on GPU {device_id}")

L1_avg_df = PushPull_with_batch_batched_gpu(
    A_gpu=A_gpu,
    B_gpu=B_gpu,
    init_x_gpu_batched=init_x_gpu_batched,
    h_data_nodes_gpu=h_tilde_gpu_nodes,
    y_data_nodes_gpu=y_tilde_gpu_nodes,
    grad_func_batched_gpu=grad_with_batch_batched_gpu,
    rho=1e-2,
    lr=5e-2,
    sigma_n=0, # 或你的设定值
    max_it=4000,
    batch_size=200,
    num_runs=num_runs
)
print("\nL1_avg_df (from GPU batched execution):")
print(L1_avg_df.head())

#保存到 CSV
output_path = f"/home/lg/ICML2025_project/push_pull_linear/RANDOM_out/RANDOM_avg_n={n}_gpu_batched.csv"
L1_avg_df.to_csv(output_path, index_label="iteration")
print(f"Average results saved to {output_path}")

#清理GPU内存 (可选, 如果需要立即释放)
# cp.get_default_memory_pool().free_all_blocks()
# cp.get_default_pinned_memory_pool().free_all_blocks()