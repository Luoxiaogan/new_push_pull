#!/usr/bin/env python3
"""
展示优化配置的矩阵性质
"""

import numpy as np
import sys
import os

# 添加profile目录到path以使用其中的函数
sys.path.append('/Users/luogan/Code/new_push_pull/0928_luogan/profile')
from useful_functions_with_batch import (
    show_col, show_row,
    get_right_perron, get_left_perron,
    compute_kappa_col, compute_kappa_row,
    compute_beta_col, compute_beta_row
)

def show_optimized_config(n):
    """展示指定n的优化配置和矩阵性质"""

    # 加载配置
    data = np.load('/Users/luogan/Code/new_push_pull/0928_luogan/profile/best_configs.npz')

    # 检查n是否存在
    if f'A_{n}' not in data:
        print(f"错误: 没有n={n}的配置（可用范围: 4-29）")
        available_ns = sorted([int(k.split('_')[1]) for k in data.keys() if k.startswith('A_')])
        print(f"可用的n值: {available_ns}")
        return False

    # 读取配置
    A = data[f'A_{n}']  # 列随机矩阵
    D = data[f'D_{n}']  # 优化的对角元素
    seed = int(data[f'seed_{n}'])
    norm = float(data[f'norm_{n}'])
    efficiency = float(data[f'efficiency_{n}'])

    print(f"\n{'='*80}")
    print(f"n = {n} 的优化配置分析")
    print(f"{'='*80}")

    # 基本信息
    print(f"\n【基本信息】")
    print(f"生成seed: {seed}")
    print(f"||D*π_A||_2: {norm:.4f}")
    print(f"效率: {efficiency*100:.2f}%")

    # D向量信息
    print(f"\n【D向量（对角元素）】")
    print(f"D = {D}")
    print(f"sum(D) = {np.sum(D):.4f} (应该等于n={n})")
    print(f"max(D) = {np.max(D):.4f}")
    print(f"min(D) = {np.min(D):.4f}")
    print(f"κ(D) = max/min = {np.max(D)/np.min(D):.2f}")

    # 识别D的分配策略
    k = np.sum(D > np.min(D) * 1.1)  # 计算有多少个节点获得了较大权重
    print(f"主要权重分配给 {k} 个节点")

    # A矩阵结构
    print(f"\n【A矩阵（列随机）】")
    print("A矩阵:")
    with np.printoptions(precision=4, suppress=True):
        print(A)
    print(f"\n每列和: {A.sum(axis=0)} (应该都是1)")

    # A矩阵性质（列随机用show_col）
    print(f"\n【A矩阵性质（列随机）】")
    show_col(A)

    # Perron向量
    pi_A = get_right_perron(A)
    print(f"\nπ_A (右Perron向量): ")
    with np.printoptions(precision=6, suppress=True):
        print(pi_A)
    print(f"max(π_A) = {np.max(pi_A):.6f}")
    print(f"min(π_A) = {np.min(pi_A):.6f}")
    print(f"κ(π_A) = {np.max(pi_A)/np.min(pi_A):.2f}")

    # 找出最大的k个π_A元素
    top_k_indices = np.argsort(pi_A)[-k:][::-1]
    print(f"\nπ_A最大的{k}个元素:")
    for idx in top_k_indices:
        print(f"  节点{idx}: π_A[{idx}] = {pi_A[idx]:.6f}, D[{idx}] = {D[idx]:.4f}")

    # B矩阵（转置）
    print(f"\n{'='*60}")
    print(f"【B矩阵（行随机，B=A.T）】")
    print("B矩阵:")
    B = A.T
    with np.printoptions(precision=4, suppress=True):
        print(B)
    print(f"\n每行和: {B.sum(axis=1)} (应该都是1)")

    # B矩阵性质（行随机用show_row）
    print(f"\n【B矩阵性质（行随机）】")
    show_row(B)

    # D*π_A的计算验证
    print(f"\n{'='*60}")
    print(f"【优化结果验证】")
    D_pi = D * pi_A
    computed_norm = np.linalg.norm(D_pi)
    print(f"D * π_A = ")
    with np.printoptions(precision=6, suppress=True):
        print(D_pi)
    print(f"\n||D*π_A||_2 = {computed_norm:.6f} (存储值: {norm:.6f})")

    # 理论上界
    theoretical_upper = n * np.max(pi_A)
    print(f"\n理论上界 = n × max(π_A) = {n} × {np.max(pi_A):.6f} = {theoretical_upper:.6f}")
    print(f"实际效率 = {computed_norm/theoretical_upper*100:.2f}%")

    # 对比均匀分配
    uniform_D = np.ones(n)
    uniform_norm = np.linalg.norm(uniform_D * pi_A)
    print(f"\n【对比均匀分配】")
    print(f"均匀D (都是1): ||D*π_A||_2 = {uniform_norm:.6f}")
    print(f"优化提升: {computed_norm/uniform_norm:.2f}倍")

    return True

def main():
    """主函数"""
    print("="*80)
    print("矩阵性质展示工具")
    print("="*80)

    # 交互式输入或命令行参数
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = int(input("请输入n值 (4-29): "))

    # 展示配置
    success = show_optimized_config(n)

    if success:
        print(f"\n{'='*80}")
        print("分析完成")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()