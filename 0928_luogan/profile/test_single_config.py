#!/usr/bin/env python3
"""
单配置贪心优化测试
对单个(n, seed)配置进行详细分析
"""

import numpy as np
import matplotlib.pyplot as plt
from useful_functions_with_batch import get_xinmeng_like_matrix, get_right_perron
from greedy_optimizer import optimize_D_matrix, compute_statistics, analyze_result


def test_single_configuration(n=16, seed=49, kappa_bound=100, save_plot=True):
    """
    测试单个配置的贪心优化

    参数:
        n: 节点数
        seed: 随机种子
        kappa_bound: 条件数约束
        save_plot: 是否保存图片

    返回:
        dict: 包含A矩阵、D矩阵、优化结果等
    """
    # 生成矩阵
    A = get_xinmeng_like_matrix(n, seed)
    pi_A = get_right_perron(A)

    # 贪心优化
    result = optimize_D_matrix(pi_A, n, kappa_bound)

    # 计算统计信息
    pi_stats = compute_statistics(pi_A)
    analysis = analyze_result(result, n)

    # 打印结果
    print(f"配置: n={n}, seed={seed}, κ≤{kappa_bound}")
    print(f"\nPerron向量统计:")
    print(f"  范围: [{pi_stats['min']:.6f}, {pi_stats['max']:.6f}]")
    print(f"  比值: {pi_stats['ratio']:.1f}")

    print(f"\n贪心优化结果:")
    print(f"  ||D*π_A||₂ = {result['norm']:.4f}")
    print(f"  理论上界 = {result['theoretical_upper']:.4f}")
    print(f"  效率 = {analysis['efficiency_percent']:.2f}%")
    print(f"  最优 k = {result['k']} (k/n = {analysis['k_ratio']:.2f})")
    print(f"  条件数 κ(D) = {result['kappa']:.2f}")

    # 可视化
    if save_plot:
        visualize_configuration(pi_A, result['D'], n, analysis, seed)

    return {
        'A': A,
        'pi_A': pi_A,
        'D': result['D'],
        'result': result,
        'statistics': pi_stats,
        'analysis': analysis
    }


def visualize_configuration(pi_A, D, n, analysis, seed):
    """生成单配置的可视化"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    indices = np.arange(n)

    # 子图1: Perron向量分布
    ax = axes[0]
    ax.bar(indices, pi_A, color='steelblue', alpha=0.7)
    ax.set_xlabel('Node Index')
    ax.set_ylabel('Value')
    ax.set_title('Perron Vector Distribution')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 子图2: D矩阵对角元素
    ax = axes[1]
    ax.bar(indices, D, color='coral', alpha=0.7)
    ax.set_xlabel('Node Index')
    ax.set_ylabel('Diagonal Value')
    ax.set_title('Optimized D Matrix')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Uniform (D=I)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Single Config Analysis (n={n}, seed={seed}, efficiency={analysis["efficiency_percent"]:.1f}%)')
    plt.tight_layout()
    plt.savefig(f'single_config_n{n}_seed{seed}.png', dpi=100, bbox_inches='tight')
    print(f"\n图已保存: single_config_n{n}_seed{seed}.png")


def main():
    """主函数 - 运行默认配置测试"""
    # 默认参数
    n = 16
    seed = 49
    kappa_bound = 100

    # 运行测试
    result = test_single_configuration(n, seed, kappa_bound)

    return result


if __name__ == "__main__":
    main()