#!/usr/bin/env python3
"""
批量n-seed分析
寻找每个n的最佳配置并保存
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from useful_functions_with_batch import get_xinmeng_like_matrix, get_right_perron
from greedy_optimizer import optimize_D_matrix


def batch_optimize(n_range, n_samples_per_n=100, kappa_bound=100, random_seed=42):
    """
    批量优化：对每个n测试多个随机seed

    参数:
        n_range: n值范围
        n_samples_per_n: 每个n的采样数
        kappa_bound: 条件数约束
        random_seed: 随机种子（确保可重现）

    返回:
        results: 完整结果字典
        best_configs: 每个n的最佳配置
        all_points: 所有(n, norm)数据点
    """
    results = {}
    best_configs = {}
    all_points = []

    np.random.seed(random_seed)

    for n in tqdm(n_range, desc='Optimizing'):
        seeds = np.random.randint(0, 10000, n_samples_per_n)
        n_results = []
        best_for_n = {'norm': -1}

        for seed in seeds:
            # 生成矩阵
            A = get_xinmeng_like_matrix(n, seed)
            pi_A = get_right_perron(A)

            # 贪心优化
            opt_result = optimize_D_matrix(pi_A, n, kappa_bound)

            # 记录结果
            result = {
                'seed': int(seed),
                'norm': opt_result['norm'],
                'efficiency': opt_result['efficiency'],
                'k': opt_result['k'],
                'D': opt_result['D']
            }
            n_results.append(result)
            all_points.append((n, opt_result['norm']))

            # 更新最佳配置
            if opt_result['norm'] > best_for_n['norm']:
                best_for_n = result.copy()
                best_for_n['A'] = A
                best_for_n['pi_A'] = pi_A

        # 保存该n的结果
        norms = [r['norm'] for r in n_results]
        results[n] = {
            'best': best_for_n,
            'worst': min(n_results, key=lambda x: x['norm']),
            'mean': np.mean(norms),
            'std': np.std(norms),
            'median': np.median(norms),
            'all_norms': norms
        }

        # 保存最佳配置
        best_configs[n] = {
            'seed': best_for_n['seed'],
            'A': best_for_n['A'],
            'D': best_for_n['D'],
            'norm': best_for_n['norm'],
            'efficiency': best_for_n['efficiency']
        }

    return results, best_configs, all_points


def save_best_configs(best_configs, filename='/Users/luogan/Code/new_push_pull/0928_luogan/profile/best_configs.npz'):
    """保存每个n的最佳配置"""
    save_dict = {}

    for n, config in best_configs.items():
        save_dict[f'A_{n}'] = config['A']
        save_dict[f'D_{n}'] = config['D']
        save_dict[f'seed_{n}'] = np.array(config['seed'])
        save_dict[f'norm_{n}'] = np.array(config['norm'])
        save_dict[f'efficiency_{n}'] = np.array(config['efficiency'])

    np.savez(filename, **save_dict)
    print(f"\n最佳配置已保存: {filename}")

    # 打印读取说明
    print("\n读取示例:")
    print("```python")
    print(f"data = np.load('{filename}')")
    print("n = 10  # 要读取的n值")
    print("A = data[f'A_{n}']")
    print("D = data[f'D_{n}']")
    print("seed = int(data[f'seed_{n}'])")
    print("norm = float(data[f'norm_{n}'])")
    print("```")


def visualize_batch_results(results, all_points, save_filename='/Users/luogan/Code/new_push_pull/0928_luogan/profile/batch_analysis.png'):
    """生成批量分析的4子图可视化"""
    n_values = sorted(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ========== 子图1：散点图 + 最佳值 ==========
    ax = axes[0, 0]
    all_n = [p[0] for p in all_points]
    all_norms = [p[1] for p in all_points]
    ax.scatter(all_n, all_norms, alpha=0.3, s=10, color='lightblue',
               label=f'All samples ({len(all_points)} points)')

    best_n = [n for n in n_values]
    best_norms = [results[n]['best']['norm'] for n in n_values]
    ax.scatter(best_n, best_norms, color='red', s=100, marker='*',
               zorder=5, label='Best')

    mean_norms = [results[n]['mean'] for n in n_values]
    ax.plot(n_values, mean_norms, 'g-', linewidth=2, alpha=0.7, label='Mean')

    ax.set_xlabel('n (number of nodes)')
    ax.set_ylabel('||D*π_A||₂')
    ax.set_title('All Data Points with Best Values Highlighted')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ========== 子图2：性能统计 ==========
    ax = axes[0, 1]
    worst_norms = [results[n]['worst']['norm'] for n in n_values]
    median_norms = [results[n]['median'] for n in n_values]

    # 均值±标准差
    mean_array = np.array(mean_norms)
    std_array = np.array([results[n]['std'] for n in n_values])
    ax.fill_between(n_values, mean_array - std_array, mean_array + std_array,
                     alpha=0.3, color='green', label='±1 std')

    ax.plot(n_values, best_norms, 'r-o', label='Best', linewidth=2)
    ax.plot(n_values, mean_norms, 'g-s', label='Mean', linewidth=2)
    ax.plot(n_values, median_norms, 'm-^', label='Median', linewidth=2)
    ax.plot(n_values, worst_norms, 'b-v', label='Worst', linewidth=2)

    ax.set_xlabel('n (number of nodes)')
    ax.set_ylabel('||D*π_A||₂')
    ax.set_title('Performance Statistics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ========== 子图3：效率分布 ==========
    ax = axes[1, 0]
    best_effs = [results[n]['best']['efficiency'] * 100 for n in n_values]
    ax.plot(n_values, best_effs, 'r-o', linewidth=2)

    ax.set_xlabel('n (number of nodes)')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Best Efficiency vs Network Size')
    ax.grid(True, alpha=0.3)

    # 添加趋势线
    z = np.polyfit(n_values, best_effs, 1)
    p = np.poly1d(z)
    ax.plot(n_values, p(n_values), 'k--', alpha=0.5,
            label=f'Trend: {z[0]:.2f}n + {z[1]:.2f}')
    ax.legend()

    # ========== 子图4：归一化性能 ==========
    ax = axes[1, 1]
    best_ratios = [results[n]['best']['norm'] / n for n in n_values]
    mean_ratios = [results[n]['mean'] / n for n in n_values]
    worst_ratios = [results[n]['worst']['norm'] / n for n in n_values]

    ax.plot(n_values, best_ratios, 'r-o', label='Best', linewidth=2)
    ax.plot(n_values, mean_ratios, 'g-s', label='Mean', linewidth=2)
    ax.plot(n_values, worst_ratios, 'b-^', label='Worst', linewidth=2)

    ax.set_xlabel('n (number of nodes)')
    ax.set_ylabel('Norm/n Ratio')
    ax.set_title('Normalized Performance (||D*π_A||₂ / n)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Batch Greedy Optimization Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_filename, dpi=150, bbox_inches='tight')
    print(f"图已保存: {save_filename}")


def print_summary(results):
    """打印结果汇总表格"""
    n_values = sorted(results.keys())

    print("\n" + "=" * 80)
    print("批量优化结果汇总")
    print("=" * 80)
    print(f"{'n':^5} {'Best Norm':^12} {'Mean±Std':^20} {'Efficiency(%)':^12} {'Best Seed':^10}")
    print("-" * 80)

    for n in n_values:
        r = results[n]
        print(f"{n:^5} {r['best']['norm']:^12.4f} "
              f"{r['mean']:>8.4f}±{r['std']:<7.4f} "
              f"{r['best']['efficiency']*100:^12.2f} "
              f"{r['best']['seed']:^10}")

    # 找出全局最佳
    best_n = max(n_values, key=lambda n: results[n]['best']['norm'])
    print("-" * 80)
    print(f"全局最佳: n={best_n}, norm={results[best_n]['best']['norm']:.4f}, "
          f"seed={results[best_n]['best']['seed']}")


def main():
    """主函数"""
    # 参数设置
    n_range = range(4, 30)
    n_samples = 2000
    kappa_bound = 200

    print("批量贪心优化分析")
    print(f"参数: n∈{list(n_range)}, 每个n测试{n_samples}个seed, κ≤{kappa_bound}")

    # 运行批量优化
    results, best_configs, all_points = batch_optimize(
        n_range, n_samples, kappa_bound
    )

    # 保存最佳配置
    save_best_configs(best_configs)

    # 打印汇总
    print_summary(results)

    # 生成可视化
    visualize_batch_results(results, all_points)

    return results, best_configs


if __name__ == "__main__":
    main()