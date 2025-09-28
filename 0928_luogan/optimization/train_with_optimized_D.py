#!/usr/bin/env python3
"""
使用优化的D矩阵进行PushPull训练
从best_configs.npz读取优化配置
"""

from re import T
import numpy as np
import os
import sys
import time
from useful_functions_with_batch import *
from opt_function_with_batch import *

# 硬编码参数（与basic_test.py保持一致）
n = 5  # 节点数，可修改为4-29之间的任意值
max_it = 9000  # 最大迭代次数
alpha = 0.1  # 异质性参数（高度异质）
d = 10  # 特征维度
L_total = 1440000  # 总样本数
lr_basic = 1e-1  # 基础学习率
sample_seed = 42  # 随机种子

# 加载优化的配置
print(f"加载n={n}的优化配置...")
data = np.load('/Users/luogan/Code/new_push_pull/0928_luogan/profile/best_configs.npz')

# 检查配置是否存在
if f'A_{n}' not in data:
    print(f"错误: 没有n={n}的配置(可用范围: 4-29)")
    sys.exit(1)

A = data[f'A_{n}']  # 列随机矩阵
D = data[f'D_{n}']  # 优化的对角元素
seed = int(data[f'seed_{n}'])
norm = float(data[f'norm_{n}'])
efficiency = float(data[f'efficiency_{n}'])

print(f"使用seed={seed}的最佳配置")
print(f"||D*π_A||_2 = {norm:.4f}")
print(f"效率 = {efficiency*100:.2f}%")
print(f"D向量: {D}")

# 生成B矩阵（A的转置，从列随机变为行随机）
B = A.T
print("\nA是列随机矩阵, B=A.T是行随机矩阵")

# 计算学习率列表
lr_list = [lr_basic * d_i for d_i in D]
print(f"\n学习率列表: {lr_list}")
print(f"学习率总和: {sum(lr_list):.4f} (应该约等于 {lr_basic * n:.4f})")

# 生成全局数据（与basic_test.py一致）
print("\n生成全局数据...")
h_global, y_global, x_opt = init_global_data(d=d, L_total=L_total, seed=42)
print(f"全局数据形状:")
print(f"h: {h_global.shape}")
print(f"y: {y_global.shape}")
print(f"正样本数: {np.sum(y_global == 1)}")
print(f"负样本数: {np.sum(y_global == -1)}")

# 初始化x
init_x_single = init_x_func(n=n, d=d, seed=42)

# 异质数据分配
print(f"\n使用alpha={alpha}进行异质数据分配...")
h_tilde, y_tilde = distribute_data_hetero(h=h_global, y=y_global, n=n, alpha=alpha, seed=42)
print(f"分布式数据形状:")
print(f"h_tilde: {h_tilde.shape}")
print(f"y_tilde: {y_tilde.shape}")

# 打印每个节点的数据分布
print("\n各节点数据分布:")
for i in range(n):
    n_pos = np.sum(y_tilde[i] == 1)
    n_neg = np.sum(y_tilde[i] == -1)
    pos_ratio = n_pos / len(y_tilde[i]) if len(y_tilde[i]) > 0 else 0
    print(f"节点 {i}: {n_pos:6d} 正样本 ({pos_ratio:6.2%}), {n_neg:6d} 负样本 ({(1-pos_ratio):6.2%})")

# 运行PushPull训练
print("\n" + "="*60)
print("开始PushPull训练...")
print("="*60)

start_time = time.time()
L1_df = PushPull_with_batch_different_lr(
    A=A,
    B=B,
    init_x=init_x_single,
    h_data=h_tilde,
    y_data=y_tilde,
    lr_list=lr_list,
    rho=0,
    sigma_n=0,
    max_it=max_it,
    batch_size=None,  # 使用全批次
)
end_time = time.time()

print(f"\n训练完成!")
print(f"训练时间: {end_time - start_time:.2f} 秒")
print(f"最终损失: {L1_df.iloc[-1]['loss_on_full_trainset']:.6f}")
print(f"最终梯度范数: {L1_df.iloc[-1]['gradient_norm_on_full_trainset']:.6f}")

# 显示损失下降情况
print(f"\n损失下降情况:")
checkpoints = [0, 100, 500, 1000, 2000, 4000, 6000, 8000, max_it-1]
print(f"{'迭代':>6} | {'损失':>12} | {'梯度范数':>12}")
print("-" * 40)
for i in checkpoints:
    if i < len(L1_df):
        print(f"{i:6d} | {L1_df.iloc[i]['loss_on_full_trainset']:12.6f} | {L1_df.iloc[i]['gradient_norm_on_full_trainset']:12.6f}")

# 保存结果
time_str = time.strftime("%Y%m%d_%H%M%S")
output_file = f"/Users/luogan/Code/new_push_pull/0928_luogan/output/results_n{n}_optimized_D_{time_str}.csv"
L1_df.to_csv(output_file)
print(f"\n结果已保存到: {output_file}")