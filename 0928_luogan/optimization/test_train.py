#!/usr/bin/env python3
"""
测试版本：使用优化的D矩阵进行PushPull训练
只运行100次迭代以快速测试
"""

import numpy as np
import os
import sys
import time
from useful_functions_with_batch import *
from opt_function_with_batch import *

# 硬编码参数（与basic_test.py保持一致）
n = 4  # 节点数，可修改为4-29之间的任意值
max_it = 100  # 减少到100次迭代用于测试
alpha = 0.1  # 异质性参数（高度异质）
d = 10  # 特征维度
L_total = 1440000  # 总样本数
lr_basic = 1e-1  # 基础学习率
sample_seed = 42  # 随机种子

print("测试版本：只运行100次迭代")
print("="*60)

# 加载优化的配置
print(f"加载n={n}的优化配置...")
data = np.load('/Users/luogan/Code/new_push_pull/0928_luogan/profile/best_configs.npz')

# 检查配置是否存在
if f'A_{n}' not in data:
    print(f"错误: 没有n={n}的配置（可用范围: 4-29）")
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
print("开始PushPull训练（测试版本，100次迭代）...")
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

print(f"\n测试完成!")
print(f"训练时间: {end_time - start_time:.2f} 秒")
print(f"最终损失: {L1_df.iloc[-1]['loss_on_full_trainset']:.6f}")

# 显示损失下降情况
print(f"\n损失下降情况:")
checkpoints = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, max_it-1]
print(f"{'迭代':>6} | {'损失':>12}")
print("-" * 20)
for i in checkpoints:
    if i < len(L1_df):
        print(f"{i:6d} | {L1_df.iloc[i]['loss_on_full_trainset']:12.6f}")

print("\n测试成功！可以运行完整版本的train_with_optimized_D.py")