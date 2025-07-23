from useful_functions_with_batch import *
from opt_function_with_batch import *
import cupy as cp
from cupy_fuc import grad_with_batch_batched_gpu, PushPull_with_batch_batched_gpu_differ_lr, loss_with_batch_batched_gpu
import os
import sys
import json
import time
import numpy as np
# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from scripts_pushpull_differ_lr.experiment_utils import generate_topology_matrices, compute_possible_c

time_integer = int(time.time())
random_integer = np.random.randint(1, 1000000000, size=2)[0]

max_it = 10000

d = 10
L_total = 1440000
n = 16 # num_nodes, 6,12,18,24
num_runs = 2 # 并行运行的次数

# 144000/ 16 = 9000 
# 每个节点的数据量

# 异质性参数
alpha = 0.1  # 高值接近均匀分布，低值（如0.1）高度异质

topology = "neighbor"
matrix_seed = 51583
lr_basic = 1e-2
sample_seed = 42
num_samples = 20 

device_id = input("Enter the GPU device ID (e.g., 'cuda:0', 'cuda:1'): ")

# --- 选择GPU设备 ---
gpu_id_int = int(device_id.split(':')[1])
cp.cuda.Device(gpu_id_int).use()
print(f"Using GPU: {cp.cuda.Device(gpu_id_int).pci_bus_id}")

# 使用标准的全局数据生成函数（数据本身是uniform的）
h_global_cpu, y_global_cpu, x_opt_cpu = init_global_data(d=d, L_total=L_total, seed=42)
print("h:",h_global_cpu.shape)
print("y:",y_global_cpu.shape)
print("Number of positive samples:", np.sum(y_global_cpu == 1))
print("Number of negative samples:", np.sum(y_global_cpu == -1))

# 使用异质数据分配函数
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

init_x_cpu_single = init_x_func(n=n, d=d, seed=42)
A_cpu, B_cpu = generate_topology_matrices(topology=topology, n=n, matrix_seed=matrix_seed, k=3)
print("\nCPU数据准备完毕")

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

# --- 将数据迁移到GPU ---
A_gpu = cp.asarray(A_cpu)
B_gpu = cp.asarray(B_cpu)
h_tilde_gpu_nodes = cp.asarray(h_tilde_cpu) # Shape: (n, L, d)
y_tilde_gpu_nodes = cp.asarray(y_tilde_cpu) # Shape: (n, L)

# 为 num_runs 扩展 init_x
init_x_gpu_batched = cp.repeat(cp.asarray(init_x_cpu_single)[cp.newaxis, ...], num_runs, axis=0)

print("Data moved to GPU.")
print("A_gpu shape:", A_gpu.shape)
print("h_tilde_gpu_nodes shape:", h_tilde_gpu_nodes.shape)
print("init_x_gpu_batched shape:", init_x_gpu_batched.shape)

print(f"\nStarting batched experiment with n={n}, num_runs={num_runs} on GPU {device_id}")
print(f"Using heterogeneous data distribution with alpha={alpha}")

print(sum(d_list))
lr_list = [lr_basic* i for i in d_list]

config = {
    "n": n,
    "num_runs": num_runs,
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
config_file_path = f"/home/lg/new_push_pull/合成数据/hetero_full/{strategy}_{topology}_hetero_alpha{alpha}_config.json"
# Create directory if it doesn't exist
print(f"\nSaving config to {config_file_path}")
os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
with open(config_file_path, 'w') as config_file:
    json.dump(config, config_file, indent=4)
# 保存这个配置文件
print("配置文件保存成功.")


# 

L1_avg_df = PushPull_with_batch_batched_gpu_differ_lr(
    A_gpu=A_gpu,
    B_gpu=B_gpu,
    init_x_gpu_batched=init_x_gpu_batched,
    h_data_nodes_gpu=h_tilde_gpu_nodes,
    y_data_nodes_gpu=y_tilde_gpu_nodes,
    loss_func_batched_gpu = loss_with_batch_batched_gpu,
    grad_func_batched_gpu=grad_with_batch_batched_gpu,
    rho=1e-2,
    lr_list=lr_list,
    sigma_n=0, # 或你的设定值
    max_it=max_it,
    batch_size=9000 , # 每个节点9000个样本, 完全批处理
    num_runs=num_runs
)
print("\nL1_avg_df (from GPU batched execution):")
print(L1_avg_df.head())

output_path = f"/home/lg/new_push_pull/合成数据/hetero_full/{strategy}_{topology}_hetero_alpha{alpha}_c={c}_avg_n={n}_gpu_batched.csv"
L1_avg_df.to_csv(output_path, index_label="iteration")
print(f"Average results saved to {output_path}")