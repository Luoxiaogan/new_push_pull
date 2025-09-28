#!/usr/bin/env python3
"""
贪心优化算法的n-seed系统性分析
分析网络规模(n)和随机种子(seed)对贪心优化性能的影响
"""

import re
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pickle
import time

# 添加父目录到路径以导入自定义函数
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# 导入必要的函数
from useful_functions_with_batch import (
    get_xinmeng_like_matrix,
    get_right_perron,
)

# ============================================================================
# 基础计算函数（从test_pi_A_D.py复制）
# ============================================================================

def compute_D_pi_norm(D_diag, pi_A):
    """计算||D*pi_A||_2"""
    return np.linalg.norm(D_diag * pi_A)

def compute_kappa_D(D_diag):
    """计算对角矩阵D的条件数"""
    if np.min(D_diag) <= 0:
        return np.inf
    return np.max(D_diag) / np.min(D_diag)

# ============================================================================
# 贪心优化核心函数
# ============================================================================

def detect_top_cluster(pi_A, threshold_ratio=0.8):
    """
    检测pi_A中的顶层集群（值相近的大元素）
    
    参数:
        pi_A: Perron向量
        threshold_ratio: 相对于最大值的阈值比例
    
    返回:
        list: 属于顶层集群的索引
    """
    # 按值降序排序
    sorted_indices = np.argsort(pi_A)[::-1]
    sorted_values = pi_A[sorted_indices]
    
    # 最大值
    max_val = sorted_values[0]
    
    # 找出值相近的元素（相对于最大值的比例）
    cluster_indices = []
    for i, idx in enumerate(sorted_indices):
        if sorted_values[i] >= max_val * threshold_ratio:
            cluster_indices.append(idx)
        else:
            break
    
    # 如果只有一个元素，尝试使用更宽松的阈值
    if len(cluster_indices) == 1:
        # 计算前几个元素的相对差异
        if len(sorted_values) > 1:
            # 如果第二大值与最大值的比例超过0.5，也加入集群
            if sorted_values[1] / max_val >= 0.5:
                cluster_indices.append(sorted_indices[1])
            # 如果第三大值与第二大值很接近，也考虑加入
            if len(sorted_values) > 2 and len(cluster_indices) > 1:
                if sorted_values[2] / sorted_values[1] >= 0.9:
                    cluster_indices.append(sorted_indices[2])
    
    return cluster_indices

def greedy_allocation_core(pi_A, n, kappa_bound=100):
    """
    智能贪心算法：自适应地给pi_A大的元素分配权重
    
    参数:
        pi_A: Perron向量
        n: 节点数
        kappa_bound: 条件数上界
    
    返回:
        dict: 包含最优D、范数、kappa等信息
    """
    # 按pi_A值降序排序索引
    sorted_indices = np.argsort(pi_A)[::-1]
    
    # 检测顶层集群
    top_cluster = detect_top_cluster(pi_A, threshold_ratio=0.7)
    
    # 存储候选解
    candidates = []
    
    # 策略1：原始贪心（逐个尝试k值）
    for k in range(1, min(n, 6)):  # 限制k最大为5，避免过度分散
        # 计算d_max和d_min
        # k * d_max + (n-k) * d_min = n
        # d_max / d_min = kappa_bound
        d_min = n / (k * kappa_bound + (n - k))
        d_max = kappa_bound * d_min
        
        # 构建D
        D_trial = np.ones(n) * d_min
        D_trial[sorted_indices[:k]] = d_max
        
        # 计算范数
        norm_val = compute_D_pi_norm(D_trial, pi_A)
        
        candidates.append({
            'D': D_trial.copy(),
            'norm': norm_val,
            'kappa': compute_kappa_D(D_trial),
            'k': k,
            'strategy': 'standard'
        })
    
    # 策略2：集群分配（给顶层集群相同的权重）
    if len(top_cluster) > 1:
        k_cluster = len(top_cluster)
        d_min = n / (k_cluster * kappa_bound + (n - k_cluster))
        d_max = kappa_bound * d_min
        
        D_cluster = np.ones(n) * d_min
        for idx in top_cluster:
            D_cluster[idx] = d_max
        
        norm_val = compute_D_pi_norm(D_cluster, pi_A)
        
        candidates.append({
            'D': D_cluster.copy(),
            'norm': norm_val,
            'kappa': compute_kappa_D(D_cluster),
            'k': k_cluster,
            'strategy': 'cluster'
        })
    
    # 策略3：加权分配（根据pi_A值的比例分配权重）
    # 给前k个元素分配与其pi_A值成比例的权重
    for k in [2, 3, 4]:
        if k >= n:
            break
            
        # 获取前k个元素的pi_A值
        top_k_indices = sorted_indices[:k]
        top_k_values = pi_A[top_k_indices]
        
        # 归一化权重
        weights = top_k_values / np.sum(top_k_values)
        
        # 分配总权重
        total_top_weight = n * kappa_bound / (kappa_bound + 1)  # 近似值
        total_bottom_weight = n - total_top_weight
        
        D_weighted = np.ones(n) * (total_bottom_weight / (n - k) if n > k else 1)
        for i, idx in enumerate(top_k_indices):
            D_weighted[idx] = weights[i] * total_top_weight
        
        # 调整以满足和约束
        D_weighted = D_weighted * n / np.sum(D_weighted)
        
        # 检查kappa约束
        if compute_kappa_D(D_weighted) <= kappa_bound:
            norm_val = compute_D_pi_norm(D_weighted, pi_A)
            candidates.append({
                'D': D_weighted.copy(),
                'norm': norm_val,
                'kappa': compute_kappa_D(D_weighted),
                'k': k,
                'strategy': 'weighted'
            })
    
    # 策略4：自适应分配（基于pi_A的分布特征）
    # 计算pi_A的统计特征
    pi_sorted = pi_A[sorted_indices]
    ratios = pi_sorted[1:] / pi_sorted[:-1]  # 相邻元素的比例
    
    # 找到显著下降点（比例小于0.5的位置）
    k_adaptive = 1
    for i, ratio in enumerate(ratios):
        if ratio < 0.5:
            k_adaptive = i + 1
            break
    else:
        k_adaptive = min(len(ratios), 3)
    
    if k_adaptive > 1:
        d_min = n / (k_adaptive * kappa_bound + (n - k_adaptive))
        d_max = kappa_bound * d_min
        
        D_adaptive = np.ones(n) * d_min
        D_adaptive[sorted_indices[:k_adaptive]] = d_max
        
        norm_val = compute_D_pi_norm(D_adaptive, pi_A)
        
        candidates.append({
            'D': D_adaptive.copy(),
            'norm': norm_val,
            'kappa': compute_kappa_D(D_adaptive),
            'k': k_adaptive,
            'strategy': 'adaptive'
        })
    
    # 选择最佳候选
    if candidates:
        best_candidate = max(candidates, key=lambda x: x['norm'])
        return {
            'D': best_candidate['D'],
            'norm': best_candidate['norm'],
            'kappa': best_candidate['kappa'],
            'k': best_candidate['k'],
            'strategy': best_candidate['strategy']
        }
    else:
        # 如果没有候选，返回均匀分布
        D_uniform = np.ones(n)
        return {
            'D': D_uniform,
            'norm': compute_D_pi_norm(D_uniform, pi_A),
            'kappa': 1.0,
            'k': n,
            'strategy': 'uniform'
        }

# ============================================================================
# 单次优化函数
# ============================================================================

def greedy_optimization_single(n, seed, kappa_bound=100):
    """
    对单个(n, seed)组合运行贪心优化
    
    参数:
        n: 节点数
        seed: 随机种子
        kappa_bound: 条件数上界
    
    返回:
        dict: 包含最优D、范数、kappa、pi_A统计等
    """
    try:
        # 1. 生成Xinmeng-like矩阵
        A = get_xinmeng_like_matrix(n, seed)
        
        # 2. 计算右Perron向量
        pi_A = get_right_perron(A)
        
        # 3. 运行贪心优化
        result = greedy_allocation_core(pi_A, n, kappa_bound)
        
        # 4. 计算理论上界
        theoretical_upper = n * np.max(pi_A)
        
        # 5. 返回完整结果
        return {
            'n': n,
            'seed': seed,
            'D': result['D'],
            'norm': result['norm'],
            'kappa': result['kappa'],
            'k': result['k'],
            'strategy': result.get('strategy', 'unknown'),
            'pi_A': pi_A,
            'pi_A_max': float(np.max(pi_A)),
            'pi_A_min': float(np.min(pi_A)),
            'pi_A_ratio': float(np.max(pi_A) / np.min(pi_A)),
            'theoretical_upper': theoretical_upper,
            'efficiency': result['norm'] / theoretical_upper,
            'compare_to_n': result['norm'] / n
        }
    except Exception as e:
        print(f"错误: n={n}, seed={seed}, 错误信息: {e}")
        return None

# ============================================================================
# 批量分析函数
# ============================================================================

def analyze_n_seed_dependency(n_range, n_samples_per_n=10, kappa_bound=100):
    """
    分析贪心优化性能对n和seed的依赖性
    
    参数:
        n_range: n值的范围，如range(5, 21)
        n_samples_per_n: 每个n值的随机种子采样数
        kappa_bound: 条件数上界
    
    返回:
        results_dict: 完整的结果字典
    """
    results = {}
    
    # 设置随机种子以确保可重现性
    np.random.seed(42)
    
    for n in tqdm(n_range, desc='Testing n values'):
        n_results = []
        
        # 对每个n，随机采样seed
        seeds = np.random.randint(0, 10000, n_samples_per_n)
        # print("n_samples_per_n=", n_samples_per_n)
        
        for seed in seeds:
            result = greedy_optimization_single(n, int(seed), kappa_bound)
            if result is not None:
                n_results.append(result)
        
        # 找出最佳和最差seed
        if n_results:
            norms = [r['norm'] for r in n_results]
            best_idx = np.argmax(norms)
            worst_idx = np.argmin(norms)
            
            results[n] = {
                'all_results': n_results,
                'best': n_results[best_idx],
                'worst': n_results[worst_idx],
                'mean_norm': float(np.mean(norms)),
                'std_norm': float(np.std(norms)),
                'median_norm': float(np.median(norms)),
                'seeds_tested': seeds.tolist()
            }
    
    return results

# ============================================================================
# 数据存储函数
# ============================================================================

def save_results(results, base_filename='greedy_n_seed_results'):
    """
    保存分析结果到多种格式
    
    参数:
        results: 分析结果字典
        base_filename: 基础文件名
    """
    # 1. JSON格式（元数据和统计）
    json_data = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_values': list(results.keys()),
            'samples_per_n': len(results[list(results.keys())[0]]['all_results']) if results else 0
        },
        'summary': {}
    }
    
    for n, n_data in results.items():
        json_data['summary'][str(n)] = {
            'best_norm': n_data['best']['norm'],
            'best_seed': n_data['best']['seed'],
            'best_k': n_data['best']['k'],
            'best_efficiency': n_data['best']['efficiency'],
            'worst_norm': n_data['worst']['norm'],
            'worst_seed': n_data['worst']['seed'],
            'mean_norm': n_data['mean_norm'],
            'std_norm': n_data['std_norm'],
            'median_norm': n_data['median_norm'],
            'all_norms': [r['norm'] for r in n_data['all_results']],
            'all_seeds': [r['seed'] for r in n_data['all_results']],
            'all_efficiencies': [r['efficiency'] for r in n_data['all_results']]
        }
    
    with open(f'{base_filename}.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # 2. NPZ格式（包含D矩阵）
    npz_data = {}
    for n, n_data in results.items():
        npz_data[f'n{n}_best_D'] = n_data['best']['D']
        npz_data[f'n{n}_best_norm'] = n_data['best']['norm']
        npz_data[f'n{n}_best_seed'] = n_data['best']['seed']
        npz_data[f'n{n}_best_pi_A'] = n_data['best']['pi_A']
        npz_data[f'n{n}_all_norms'] = np.array([r['norm'] for r in n_data['all_results']])
        npz_data[f'n{n}_all_seeds'] = np.array([r['seed'] for r in n_data['all_results']])
    
    np.savez_compressed(f'{base_filename}.npz', **npz_data)
    
    # 3. CSV格式（便于分析）
    rows = []
    for n, n_data in results.items():
        for result in n_data['all_results']:
            rows.append({
                'n': n,
                'seed': result['seed'],
                'norm': result['norm'],
                'kappa': result['kappa'],
                'k': result['k'],
                'strategy': result.get('strategy', 'unknown'),
                'pi_A_max': result['pi_A_max'],
                'pi_A_min': result['pi_A_min'],
                'pi_A_ratio': result['pi_A_ratio'],
                'theoretical_upper': result['theoretical_upper'],
                'efficiency': result['efficiency']
            })
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(f'{base_filename}.csv', index=False)
    
    print(f"结果已保存到 {base_filename}.[json/npz/csv]")

# ============================================================================
# 可视化函数
# ============================================================================

def visualize_results(results, save_filename, kappa_bound):
    """
    创建综合可视化图表
    
    参数:
        results: 分析结果字典
        save_filename: 保存的图片文件名
    """
    # 准备数据
    n_values = sorted(results.keys())
    all_data_points = []
    best_points = []
    worst_points = []
    mean_points = []
    median_points = []
    
    for n in n_values:
        n_data = results[n]
        # 所有数据点
        for r in n_data['all_results']:
            all_data_points.append((n, r['norm'], r['seed']))
        # 统计点
        best_points.append((n, n_data['best']['norm']))
        worst_points.append((n, n_data['worst']['norm']))
        mean_points.append((n, n_data['mean_norm']))
        median_points.append((n, n_data['median_norm']))
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ========== 子图1：所有数据点 + 最佳值标记 ==========
    ax = axes[0, 0]
    
    # 绘制所有点（使用向量化方式更高效）
    print(f"  正在绘制{len(all_data_points)}个数据点...")
    if all_data_points:
        # 提取所有n和norm值
        all_n = [point[0] for point in all_data_points]
        all_norms = [point[1] for point in all_data_points]
        # 一次性绘制所有点，s=10使点更小以减少重叠，alpha=0.2增加透明度
        ax.scatter(all_n, all_norms, alpha=0.2, s=10, color='steelblue', 
                  edgecolors='none', label=f'All samples ({len(all_data_points)} points)')
    
    # 标记最佳值（大红星）
    best_n, best_norms = zip(*best_points)
    ax.scatter(best_n, best_norms, color='red', s=150, 
               marker='*', label='Best', zorder=5, edgecolors='darkred', linewidth=1)
    
    # 连接最佳值
    ax.plot(best_n, best_norms, 'r--', alpha=0.5, linewidth=1.5)
    
    # 添加平均值线
    mean_n, mean_norms = zip(*mean_points)
    ax.plot(mean_n, mean_norms, 'g-', alpha=0.7, 
            linewidth=2, label='Mean')
    
    ax.set_xlabel('n (number of nodes)', fontsize=12)
    ax.set_ylabel('||D*pi_A||_2', fontsize=12)
    ax.set_title('All Data Points with Best Values Highlighted', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # ========== 子图2：最佳、最差、平均值对比 ==========
    ax = axes[0, 1]
    worst_n, worst_norms = zip(*worst_points)
    median_n, median_norms = zip(*median_points)
    
    ax.plot(best_n, best_norms, 'r-o', label='Best', linewidth=2, markersize=8)
    ax.plot(mean_n, mean_norms, 'g-s', label='Mean', linewidth=2, markersize=7)
    ax.plot(median_n, median_norms, 'm-d', label='Median', linewidth=2, markersize=7)
    ax.plot(worst_n, worst_norms, 'b-^', label='Worst', linewidth=2, markersize=7)
    
    # 添加误差带（标准差）
    std_values = [results[n]['std_norm'] for n in n_values]
    mean_minus_std = [m - s for m, s in zip(mean_norms, std_values)]
    mean_plus_std = [m + s for m, s in zip(mean_norms, std_values)]
    ax.fill_between(n_values, mean_minus_std, mean_plus_std,
                     alpha=0.2, color='green', label='±1 std')
    
    ax.set_xlabel('n (number of nodes)', fontsize=12)
    ax.set_ylabel('||D*pi_A||_2', fontsize=12)
    ax.set_title('Performance Statistics', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # ========== 子图3：效率分布（箱线图） ==========
    ax = axes[1, 0]
    
    # 收集所有效率数据用于箱线图
    efficiency_data = []
    positions = []
    
    for n in n_values:
        n_data = results[n]
        # 收集该n值的所有效率
        all_eff = [r['efficiency'] * 100 for r in n_data['all_results']]
        efficiency_data.append(all_eff)
        positions.append(n)
    
    # 创建箱线图
    bp = ax.boxplot(efficiency_data, positions=positions, widths=0.6,
                     patch_artist=True, 
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='blue', linewidth=1),
                     capprops=dict(color='blue', linewidth=1),
                     flierprops=dict(marker='o', markersize=3, alpha=0.5, color='gray'))
    
    # 添加平均值线
    mean_efficiencies = [np.mean(eff_list) for eff_list in efficiency_data]
    ax.plot(n_values, mean_efficiencies, 'g--', linewidth=1.5, label='Mean', alpha=0.7)
    
    ax.set_xlabel('n (number of nodes)', fontsize=12)
    ax.set_ylabel('Efficiency (%)', fontsize=12)
    ax.set_title('Efficiency Distribution (Relative to Theoretical Upper Bound)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='lower left')
    
    # 设置 y 轴范围以更好地显示变化
    all_eff_flat = [e for eff_list in efficiency_data for e in eff_list]
    y_min = min(all_eff_flat) - 0.5
    y_max = max(all_eff_flat) + 0.5
    ax.set_ylim(y_min, y_max)
    
    # ========== 子图4：compare_to_n 比例 (norm/n) ==========
    ax = axes[1, 1]
    
    # 计算 norm/n 比例的统计
    best_ratios = []
    mean_ratios = []
    worst_ratios = []
    
    for n in n_values:
        n_data = results[n]
        # 最佳比例
        best_ratios.append(n_data['best']['norm'] / n)
        # 所有比例
        all_ratios = [r['norm'] / n for r in n_data['all_results']]
        mean_ratios.append(np.mean(all_ratios))
        # 最差比例
        worst_ratios.append(n_data['worst']['norm'] / n)
    
    # 绘制三条线
    ax.plot(n_values, best_ratios, 'r-o', label='Best', linewidth=2)
    ax.plot(n_values, mean_ratios, 'g-s', label='Mean', linewidth=2)
    ax.plot(n_values, worst_ratios, 'b-^', label='Worst', linewidth=2)
    
    ax.set_xlabel('n (number of nodes)', fontsize=12)
    ax.set_ylabel('Norm/n Ratio', fontsize=12)
    ax.set_title('Normalized Performance (||D*pi_A||₂ / n)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # 添加趋势线（对平均值）
    if len(n_values) > 1:
        z = np.polyfit(n_values, mean_ratios, 1)
        p = np.poly1d(z)
        ax.plot(n_values, p(n_values), "k--", alpha=0.5, linewidth=1)
        # 在图上添加趋势方程
        trend_text = f'Mean trend: {z[0]:.4f}n + {z[1]:.4f}'
        ax.text(0.05, 0.95, trend_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 总标题
    plt.suptitle(f'Greedy Optimization: n-seed Dependency Analysis. kappa_D={kappa_bound}', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_filename, dpi=150, bbox_inches='tight')
    print(f"可视化已保存到 {save_filename}")
    
    return fig

# ============================================================================
# 主执行函数
# ============================================================================

def main():
    """
    主执行函数
    """
    # 设置参数
    n_range = range(5, 40)  # n从5到20
    n_samples = 1000  # 每个n采样10个seed
    kappa_bound = 1000  # 条件数上界
    
    print("="*60)
    print("贪心优化算法 n-seed 依赖性分析")
    print("="*60)
    print(f"n范围: {list(n_range)}")
    print(f"每个n的采样数: {n_samples}")
    print(f"条件数上界: {kappa_bound}")
    print()
    
    # 运行分析
    results = analyze_n_seed_dependency(n_range, n_samples, kappa_bound)
    
    # 保存结果
    save_results(results)
    
    # 可视化
    fig = visualize_results(results=results, save_filename=f"greedy_n_seed_analysis_kappa-D={kappa_bound}.png", kappa_bound=kappa_bound)

    # 打印总结
    print("\n" + "="*60)
    print("分析总结")
    print("="*60)
    
    for n in sorted(results.keys()):
        n_data = results[n]
        print(f"\nn = {n}:")
        print(f"  最佳: norm={n_data['best']['norm']:.4f}, seed={n_data['best']['seed']}, k={n_data['best']['k']}, 策略={n_data['best'].get('strategy', 'N/A')}")
        print(f"  最差: norm={n_data['worst']['norm']:.4f}, seed={n_data['worst']['seed']}")
        print(f"  平均: {n_data['mean_norm']:.4f} ± {n_data['std_norm']:.4f}")
        print(f"  中位数: {n_data['median_norm']:.4f}")
        print(f"  变异系数: {n_data['std_norm']/n_data['mean_norm']*100:.1f}%")
        print(f"  最佳效率: {n_data['best']['efficiency']*100:.1f}%")
        
        # 统计不同策略的使用情况
        strategies = [r.get('strategy', 'unknown') for r in n_data['all_results']]
        strategy_counts = {}
        for s in strategies:
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
        if len(strategy_counts) > 1:
            print(f"  策略使用: {strategy_counts}")
    
    # 找出全局最佳
    global_best_n = None
    global_best_norm = 0
    global_best_seed = None
    
    for n, n_data in results.items():
        if n_data['best']['norm'] > global_best_norm:
            global_best_norm = n_data['best']['norm']
            global_best_n = n
            global_best_seed = n_data['best']['seed']
    
    print(f"\n" + "="*60)
    print(f"全局最佳配置:")
    print(f"  n = {global_best_n}")
    print(f"  seed = {global_best_seed}")
    print(f"  norm = {global_best_norm:.4f}")
    if global_best_n:
        print(f"  效率 = {results[global_best_n]['best']['efficiency']*100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()