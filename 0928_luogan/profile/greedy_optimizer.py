#!/usr/bin/env python3
"""
贪心优化算法核心库
提供D矩阵优化的贪心算法实现
"""

import numpy as np


def optimize_D_matrix(pi_A, n, kappa_bound=100):
    """
    贪心算法优化对角矩阵D

    目标: maximize ||D*pi_A||_2
    约束:
        - sum(diag(D)) = n
        - κ(D) ≤ kappa_bound
        - D_ii > 0

    参数:
        pi_A: Perron向量 (numpy array)
        n: 节点数/矩阵维度
        kappa_bound: 条件数上界 (默认100)

    返回:
        dict: 包含以下键值
            - 'D': 优化后的对角矩阵值 (numpy array)
            - 'norm': ||D*pi_A||_2
            - 'k': 最优的k值（分配权重给top-k个元素）
            - 'kappa': 实际条件数
            - 'efficiency': 相对理论上界的效率
            - 'theoretical_upper': 理论上界值
    """
    # 按Perron向量值降序排列的索引
    sorted_indices = np.argsort(pi_A)[::-1]

    best_result = None
    best_norm = -1

    # 尝试不同的k值
    for k in range(1, n):  # 不包括k=n（所有元素相等）
        # 选择top-k个元素
        top_k_indices = sorted_indices[:k]

        # 通过求解方程组计算d_max和d_min：
        # k * d_max + (n-k) * d_min = n
        # d_max / d_min = kappa_bound
        # 解得：
        d_min = n / (k * kappa_bound + (n - k))
        d_max = kappa_bound * d_min

        # 构建D矩阵
        D = np.ones(n) * d_min
        D[top_k_indices] = d_max

        # 计算条件数（应该恰好等于kappa_bound）
        kappa = d_max / d_min

        # 使用小的容差处理浮点数比较
        if kappa <= kappa_bound * (1 + 1e-10):
            # 计算目标函数值
            norm = np.linalg.norm(D * pi_A)

            if norm > best_norm:
                best_norm = norm
                best_result = {
                    'D': D.copy(),
                    'norm': norm,
                    'k': k,
                    'kappa': kappa
                }
        else:
            # 条件数超过约束，后续k值都会超过
            break

    # 如果没有找到满足约束的解，返回均匀分配作为默认解
    if best_result is None:
        D = np.ones(n)  # 均匀分配
        best_result = {
            'D': D,
            'norm': np.linalg.norm(D * pi_A),
            'k': 0,  # 表示使用了默认解
            'kappa': 1.0
        }

    # 计算理论上界和效率
    # 理论上界：如果没有条件数约束，最优是把所有权重都给pi_A最大的元素
    theoretical_upper = n * np.max(pi_A)
    best_result['theoretical_upper'] = theoretical_upper
    best_result['efficiency'] = best_result['norm'] / theoretical_upper

    return best_result


def compute_statistics(pi_A):
    """
    计算Perron向量的统计信息

    参数:
        pi_A: Perron向量

    返回:
        dict: 统计信息
    """
    return {
        'min': float(np.min(pi_A)),
        'max': float(np.max(pi_A)),
        'mean': float(np.mean(pi_A)),
        'std': float(np.std(pi_A)),
        'ratio': float(np.max(pi_A) / np.min(pi_A)) if np.min(pi_A) > 0 else float('inf'),
        'sum': float(np.sum(pi_A))
    }


def analyze_result(result, n):
    """
    分析优化结果

    参数:
        result: optimize_D_matrix的返回值
        n: 节点数

    返回:
        dict: 分析结果
    """
    return {
        'norm': result['norm'],
        'norm_per_n': result['norm'] / n,
        'efficiency_percent': result['efficiency'] * 100,
        'k_ratio': result['k'] / n,
        'kappa': result['kappa'],
        'theoretical_upper': result['theoretical_upper']
    }