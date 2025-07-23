import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from useful_functions_with_batch import *
from opt_function_with_batch import *
from network_utils import *
import matplotlib.pyplot as plt

d=10
L_total=8000
h_global, y_global, x_opt = init_global_data(d=d, L_total=L_total, seed=42)

# 设置参数
n = 16
lr=1.22e-4


h_tilde, y_tilde = distribute_data(h=h_global, y=y_global, n=n)
x_star = generate_x_star(n=n, d=d, x_opt=x_opt, sigma_h=10)
init_x = init_x_func(n=n, d=d, seed=42)
A,B= ring1(n)
print("h_tilde:", h_tilde.shape, "\n")

# 初始化列表，用于存储每次的结果
gradient_norms_list = []

# 循环 20 次
for i in range(20):
    print(f"Running iteration {i+1}/20")
    
    # 运行 PullDiag_GD_with_batch
    L1 = PullDiag_GT_with_batch(
        A=A,
        init_x=init_x,
        h_data=h_tilde,
        y_data=y_tilde,
        grad_func=grad_with_batch,
        rho=1e-2,
        lr=lr,
        sigma_n=0,
        max_it=100000,
        batch_size=200
    )
    
    # 将 gradient_norm_on_full_trainset 添加到列表中
    gradient_norms_list.append(L1["gradient_norm_on_full_trainset"])

# 将列表转换为 numpy 数组以便计算平均值
gradient_norms_array = np.array(gradient_norms_list)

# 计算逐元素平均值
average_gradient_norm = np.mean(gradient_norms_array, axis=0)

# 将 average_gradient_norm 保存为 CSV 文件
# 创建一个 DataFrame
df = pd.DataFrame(average_gradient_norm, columns=["average_gradient_norm"])

# 保存为 CSV 文件
df.to_csv(f"/root/GanLuo/ICML2025_project/模拟数据/linear_speedup实验/output/GT_环状图,重复20次,lr={lr},n={n}.csv", index=False)

print("average_gradient_norm 已保存为 average_gradient_norm.csv")