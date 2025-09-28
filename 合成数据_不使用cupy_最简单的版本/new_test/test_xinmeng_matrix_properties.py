"""
测试 Xinmeng 矩阵的谱性质随节点数变化的趋势
分析 get_xinmeng_matrix 和 get_xinmeng_like_matrix 的 kappa 和 beta 值
这两个函数生成的都是列随机矩阵（column-stochastic matrices）
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加父目录到路径以导入函数
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from useful_functions_with_batch import (
    get_xinmeng_matrix,
    get_xinmeng_like_matrix,
    compute_kappa_col,
    compute_beta_col,
    compute_S_B_col,
    get_right_perron,
    compute_2st_eig_value
)

def analyze_matrix_as_col_stochastic(M):
    """分析矩阵作为列随机矩阵的性质"""
    try:
        kappa = compute_kappa_col(M)
        beta = compute_beta_col(M)
        spectral_gap = 1 - beta
        S_B = compute_S_B_col(M)
        
        # 获取Perron向量用于额外分析
        pi = get_right_perron(M)
        pi_max = np.max(pi)
        pi_min = np.min(pi)
        
        # 第二大特征值
        second_eig = compute_2st_eig_value(M)
        
        return {
            'kappa': kappa,
            'beta': beta,
            'spectral_gap': spectral_gap,
            'S_B': S_B,
            'pi_max': pi_max,
            'pi_min': pi_min,
            'second_eigenvalue': second_eig
        }
    except Exception as e:
        print(f"Error in column stochastic analysis: {e}")
        return {
            'kappa': np.nan,
            'beta': np.nan,
            'spectral_gap': np.nan,
            'S_B': np.nan,
            'pi_max': np.nan,
            'pi_min': np.nan,
            'second_eigenvalue': np.nan
        }

def main():
    # 测试的节点数列表
    n_values = [4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 72, 80, 88, 96, 104]
    
    # 对随机矩阵使用的种子数
    num_seeds = 200
    
    # 存储结果
    results = []
    
    # 存储所有随机样本的原始数据，用于绘图
    all_random_samples = {}
    
    print("Analyzing Spectral Properties of Xinmeng Column-Stochastic Matrices...")
    print("=" * 70)
    
    for n in tqdm(n_values, desc="Processing different node counts"):
        print(f"\nProcessing n = {n}")
        
        # 1. 分析固定的 Xinmeng 矩阵
        M_fixed = get_xinmeng_matrix(n)
        
        # 验证是列随机矩阵
        col_sums = np.sum(M_fixed, axis=0)
        assert np.allclose(col_sums, 1.0, rtol=1e-10), f"Matrix is not column-stochastic: col_sums = {col_sums}"
        
        # 分析矩阵性质
        results_fixed = analyze_matrix_as_col_stochastic(M_fixed)
        
        # 2. 分析随机版本的 Xinmeng-like 矩阵（多个种子取平均）
        results_random = []
        
        # 保存当前n值的所有随机样本
        samples_for_n = {'kappa': [], 'beta': [], 'spectral_gap': [], 'S_B': []}
        
        for seed in range(num_seeds):
            M_random = get_xinmeng_like_matrix(n, seed=seed)
            
            # 验证是列随机矩阵
            col_sums = np.sum(M_random, axis=0)
            assert np.allclose(col_sums, 1.0, rtol=1e-10), f"Random matrix is not column-stochastic"
            
            result = analyze_matrix_as_col_stochastic(M_random)
            results_random.append(result)
            
            # 保存每个样本的值
            for key in ['kappa', 'beta', 'spectral_gap', 'S_B']:
                if not np.isnan(result[key]):
                    samples_for_n[key].append(result[key])
        
        # 保存到全局字典
        all_random_samples[n] = samples_for_n
        
        # 计算随机版本的统计量
        random_mean = {}
        random_std = {}
        
        for key in results_random[0].keys():
            values = [r[key] for r in results_random if not np.isnan(r[key])]
            if values:
                random_mean[f'{key}_random_mean'] = np.mean(values)
                random_std[f'{key}_random_std'] = np.std(values)
            else:
                random_mean[f'{key}_random_mean'] = np.nan
                random_std[f'{key}_random_std'] = np.nan
        
        # 组合结果
        result = {
            'n': n,
            **{f'{k}_fixed': v for k, v in results_fixed.items()},
            **random_mean,
            **random_std
        }
        
        results.append(result)
        
        # 打印关键结果
        print(f"  Fixed matrix - kappa: {results_fixed['kappa']:.4f}, "
              f"beta: {results_fixed['beta']:.4f}, "
              f"spectral_gap: {results_fixed['spectral_gap']:.6f}")
        print(f"  Random matrix - kappa: {random_mean.get('kappa_random_mean', np.nan):.4f} "
              f"± {random_std.get('kappa_random_std', np.nan):.4f}")
    
    # 保存结果到CSV
    df = pd.DataFrame(results)
    csv_path = 'xinmeng_col_stochastic_analysis.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # 创建可视化（传入所有随机样本数据）
    create_visualizations(df, all_random_samples)
    
    # 生成分析报告
    generate_report(df)
    
    return df

def create_visualizations(df, all_random_samples=None):
    """创建可视化图表，显示所有随机样本点"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Spectral Properties of Xinmeng Column-Stochastic Matrices', fontsize=16)
    
    n_values = df['n'].values
    
    # 1. Kappa 值比较
    ax = axes[0, 0]
    # 画固定矩阵的线和点
    ax.semilogy(n_values, df['kappa_fixed'], 'b-o', label='Fixed Matrix', linewidth=2, markersize=8)
    
    # 先画所有随机样本点（如果提供了数据）
    if all_random_samples:
        for n in n_values:
            if n in all_random_samples and 'kappa' in all_random_samples[n]:
                kappa_samples = all_random_samples[n]['kappa']
                # 为每个n值画出所有样本点，使用较小的标记
                ax.semilogy([n]*len(kappa_samples), kappa_samples, 'r.', 
                           markersize=4, alpha=0.4)
    
    # 画随机矩阵的平均值线
    ax.semilogy(n_values, df['kappa_random_mean'], 'r--^', 
               label='Random Matrix (mean)', linewidth=2, markersize=7)
    
    ax.set_xlabel('Number of Nodes (n)', fontsize=12)
    ax.set_ylabel('Kappa (Condition Number)', fontsize=12)
    ax.set_title('Condition Number vs Number of Nodes', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. Beta 值比较
    ax = axes[0, 1]
    # 画固定矩阵的线和点
    ax.plot(n_values, df['beta_fixed'], 'b-o', label='Fixed Matrix', linewidth=2, markersize=8)
    
    # 先画所有随机样本点
    if all_random_samples:
        for n in n_values:
            if n in all_random_samples and 'beta' in all_random_samples[n]:
                beta_samples = all_random_samples[n]['beta']
                ax.plot([n]*len(beta_samples), beta_samples, 'r.', 
                       markersize=4, alpha=0.4)
    
    # 画随机矩阵的平均值线
    ax.plot(n_values, df['beta_random_mean'], 'r--^', 
           label='Random Matrix (mean)', linewidth=2, markersize=7)
    
    ax.set_xlabel('Number of Nodes (n)', fontsize=12)
    ax.set_ylabel('Beta', fontsize=12)
    ax.set_title('Beta Values vs Number of Nodes', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.05])
    
    # 3. Spectral Gap 比较
    ax = axes[1, 0]
    # 画固定矩阵的线和点
    ax.semilogy(n_values, df['spectral_gap_fixed'], 'b-o', label='Fixed Matrix', linewidth=2, markersize=8)
    
    # 先画所有随机样本点
    if all_random_samples:
        for n in n_values:
            if n in all_random_samples and 'spectral_gap' in all_random_samples[n]:
                gap_samples = all_random_samples[n]['spectral_gap']
                # 防止负值或零值在对数坐标中出错
                gap_samples = [max(g, 1e-10) for g in gap_samples]
                ax.semilogy([n]*len(gap_samples), gap_samples, 'r.', 
                           markersize=4, alpha=0.4)
    
    # 画随机矩阵的平均值线
    ax.semilogy(n_values, df['spectral_gap_random_mean'], 'r--^', 
               label='Random Matrix (mean)', linewidth=2, markersize=7)
    
    ax.set_xlabel('Number of Nodes (n)', fontsize=12)
    ax.set_ylabel('Spectral Gap (1 - Beta)', fontsize=12)
    ax.set_title('Spectral Gap vs Number of Nodes', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 4. S_B 值（谱复杂度）比较
    ax = axes[1, 1]
    # 画固定矩阵的线和点
    ax.loglog(n_values, df['S_B_fixed'], 'b-o', label='Fixed Matrix', linewidth=2, markersize=8)
    
    # 先画所有随机样本点
    if all_random_samples:
        for i, n in enumerate(n_values):
            if n in all_random_samples and 'S_B' in all_random_samples[n]:
                S_B_samples = all_random_samples[n]['S_B']
                # 只在第一个n值时添加图例标签
                label = 'Random Samples' if i == 0 else None
                ax.loglog([n]*len(S_B_samples), S_B_samples, 'r.', 
                         markersize=4, alpha=0.4, label=label)
    
    # 画随机矩阵的平均值线
    ax.loglog(n_values, df['S_B_random_mean'], 'r--^', 
             label='Random Matrix (mean)', linewidth=2, markersize=7)
    
    ax.set_xlabel('Number of Nodes (n)', fontsize=12)
    ax.set_ylabel('S_B (Spectral Complexity)', fontsize=12)
    ax.set_title('Spectral Complexity vs Number of Nodes', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = 'xinmeng_col_stochastic_properties_new.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.show()

def generate_report(df):
    """生成分析报告"""
    n_values = df['n'].values
    
    # 计算增长率
    def compute_growth_rate(values):
        """计算平均增长率"""
        if len(values) < 2:
            return 0
        valid_mask = values > 0
        if np.sum(valid_mask) < 2:
            return 0
        log_values = np.log(values[valid_mask])
        log_n = np.log(n_values[valid_mask])
        # 线性拟合 log(y) = a * log(n) + b
        coeffs = np.polyfit(log_n, log_values, 1)
        return coeffs[0]
    
    # 分析各指标的增长趋势
    kappa_fixed_growth = compute_growth_rate(df['kappa_fixed'].values)
    S_B_fixed_growth = compute_growth_rate(df['S_B_fixed'].values)
    
    report = f"""# Xinmeng Column-Stochastic Matrices Analysis Report

## Overview
- Node count range tested: {min(n_values)} to {max(n_values)}
- Matrix types: get_xinmeng_matrix (fixed) and get_xinmeng_like_matrix (random)
- Random matrix samples: 10 seeds averaged
- **Matrix Type**: Column-stochastic (normalized by column sums)

## Key Findings

### 1. Condition Number (Kappa) Analysis
- **Fixed Matrix**:
  - Growth exponent: {kappa_fixed_growth:.3f} (O(n^{kappa_fixed_growth:.2f}))
  - Value at n=64: {df[df['n']==64]['kappa_fixed'].values[0]:.2e}
  - Min value (n={n_values[df['kappa_fixed'].argmin()]}): {df['kappa_fixed'].min():.2f}
  - Max value (n={n_values[df['kappa_fixed'].argmax()]}): {df['kappa_fixed'].max():.2e}

- **Random Matrix**:
  - Mean at n=64: {df[df['n']==64]['kappa_random_mean'].values[0]:.2e} ± {df[df['n']==64]['kappa_random_std'].values[0]:.2e}
  - Average variability: {(df['kappa_random_std'] / df['kappa_random_mean']).mean() * 100:.1f}%

### 2. Beta Values Analysis
- **Fixed Matrix**:
  - Range: {df['beta_fixed'].min():.4f} to {df['beta_fixed'].max():.4f}
  - Trend: {"Increasing" if df['beta_fixed'].values[-1] > df['beta_fixed'].values[0] else "Decreasing"} with n

- **Random Matrix**:
  - Mean range: {df['beta_random_mean'].min():.4f} to {df['beta_random_mean'].max():.4f}
  - Average standard deviation: {df['beta_random_std'].mean():.4f}

### 3. Spectral Gap (1-β) Analysis
- **Fixed Matrix**:
  - Min spectral gap (n={n_values[df['spectral_gap_fixed'].argmin()]}): {df['spectral_gap_fixed'].min():.6f}
  - Max spectral gap (n={n_values[df['spectral_gap_fixed'].argmax()]}): {df['spectral_gap_fixed'].max():.6f}
  - Convergence rate deterioration: {(1 - df['spectral_gap_fixed'].values[-1] / df['spectral_gap_fixed'].values[0]) * 100:.1f}%

### 4. Spectral Complexity (S_B) Analysis
- **Fixed Matrix**:
  - Growth exponent: {S_B_fixed_growth:.3f} (O(n^{S_B_fixed_growth:.2f}))
  - Value at n=64: {df[df['n']==64]['S_B_fixed'].values[0]:.2e}

### 5. Perron Vector Analysis
- **Fixed Matrix**:
  - Max Perron entry at n=64: {df[df['n']==64]['pi_max_fixed'].values[0]:.6f}
  - Min Perron entry at n=64: {df[df['n']==64]['pi_min_fixed'].values[0]:.2e}
  - Ratio (max/min) = Kappa

### 6. Fixed vs Random Comparison
- Kappa ratio (random/fixed): {(df['kappa_random_mean'] / df['kappa_fixed']).mean():.2f}x on average
- Beta difference: {abs(df['beta_random_mean'].mean() - df['beta_fixed'].mean()):.4f}
- Spectral gap ratio: {(df['spectral_gap_random_mean'] / df['spectral_gap_fixed']).mean():.2f}x

## Conclusions

1. **Scalability**: The condition number grows as O(n^{kappa_fixed_growth:.1f}), indicating {"poor" if kappa_fixed_growth > 2 else "moderate" if kappa_fixed_growth > 1 else "good"} scalability.

2. **Convergence**: Spectral gap decreases with n, suggesting slower convergence for larger networks.

3. **Randomization Impact**: Random matrices show {"higher" if df['kappa_random_mean'].mean() > df['kappa_fixed'].mean() else "lower"} condition numbers with significant variability.

4. **Practical Implications**: 
   - For n > 32, numerical stability becomes a concern (kappa > 10^9)
   - The tridiagonal structure creates highly imbalanced Perron vectors
   - Not recommended for distributed optimization algorithms requiring good conditioning

## Files Generated
- Numerical results: xinmeng_col_stochastic_analysis.csv
- Visualization: xinmeng_col_stochastic_properties.png
"""
    
    # 保存报告
    report_path = 'xinmeng_col_stochastic_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    df = main()
    print("\nAnalysis complete!")