
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加父目录到路径以导入自定义函数
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# 导入Xinmeng矩阵相关函数和谱性质计算函数
from useful_functions_with_batch import (
    get_xinmeng_matrix,
    get_xinmeng_like_matrix,
    compute_kappa_col,
    compute_beta_col,
    compute_S_B_col,
    get_right_perron,
    compute_2st_eig_value
)

# 设置参数：节点数和随机种子
n = 16
seed = 49

# 生成随机Xinmeng-like矩阵（列随机矩阵）
A = get_xinmeng_like_matrix(n, seed)

# 计算右Perron向量（列随机矩阵的稳态分布）
pi_A = get_right_perron(A)
# print("colunm stochastic matrix", A)
# print("Perron vector values:")
# print(pi_A)

# 创建可视化图形
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# 使用对数尺度绘制Perron向量值
indices = np.arange(len(pi_A))
ax.semilogy(indices, pi_A, 'b-o', linewidth=2, markersize=8, label='Right Perron Vector')

# 添加网格线以提高可读性
ax.grid(True, which="both", ls="-", alpha=0.2)  # 主网格线
ax.grid(True, which="minor", ls=":", alpha=0.1)  # 次网格线

# 设置坐标轴标签和标题
ax.set_xlabel('Index', fontsize=12)
ax.set_ylabel('Value (log scale)', fontsize=12)
ax.set_title(f'Right Perron Vector of Xinmeng-like Matrix (n={n}, seed={seed})', fontsize=14)

# 设置x轴刻度，显示所有索引
ax.set_xticks(indices)
ax.set_xticklabels(indices)

# 添加图例
ax.legend(loc='upper right', fontsize=10)

# 添加参考线：均匀分布时的值（1/n）
uniform_value = 1.0 / n
ax.axhline(y=uniform_value, color='r', linestyle='--', alpha=0.5, 
           label=f'Uniform distribution (1/n = {uniform_value:.3f})')
ax.legend(loc='upper right', fontsize=10)

# 标注最大值和最小值
max_idx = np.argmax(pi_A)
min_idx = np.argmin(pi_A)

# 为最大值添加标注
ax.annotate(f'Max: {pi_A[max_idx]:.3e}', 
            xy=(max_idx, pi_A[max_idx]), 
            xytext=(max_idx+0.5, pi_A[max_idx]*2),
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
            fontsize=9)

# 为最小值添加标注
ax.annotate(f'Min: {pi_A[min_idx]:.3e}', 
            xy=(min_idx, pi_A[min_idx]), 
            xytext=(min_idx-1, pi_A[min_idx]/2),
            arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
            fontsize=9)

# 调整布局并保存图像
plt.tight_layout()
plt.savefig('perron_vector_visualization.png', dpi=150)
print(f"\nVisualization saved to: perron_vector_visualization.png")

# 显示图形
# plt.show()

# ============================================================================
# 优化部分：找到最优的对角矩阵D以最大化||D*pi_A||_2
# ============================================================================

from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import time

# -----------------------------------------------------------------------------
# 基础计算函数
# -----------------------------------------------------------------------------

def compute_D_pi_norm(D_diag, pi_A):
    """计算||D*pi_A||_2"""
    return np.linalg.norm(D_diag * pi_A)

def compute_kappa_D(D_diag):
    """计算对角矩阵D的条件数"""
    if np.min(D_diag) <= 0:
        return np.inf
    return np.max(D_diag) / np.min(D_diag)

def check_constraints(D_diag, n, kappa_bound):
    """检查D是否满足所有约束"""
    # 检查正定性
    if np.any(D_diag <= 0):
        return False
    # 检查和约束
    if not np.isclose(np.sum(D_diag), n, rtol=1e-6):
        return False
    # 检查条件数约束
    if compute_kappa_D(D_diag) > kappa_bound:
        return False
    return True

# -----------------------------------------------------------------------------
# 优化策略1：贪心分配法
# -----------------------------------------------------------------------------

def greedy_allocation(pi_A, n, kappa_bound, return_top_k=5):
    """
    贪心算法：优先给pi_A大的元素分配权重
    返回前top_k个最优解
    """
    # 按pi_A值降序排序索引
    sorted_indices = np.argsort(pi_A)[::-1]
    
    # 存储所有候选解
    candidates = []
    
    for k in range(1, n):
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
        kappa_val = compute_kappa_D(D_trial)
        
        candidates.append({
            'D': D_trial.copy(),
            'norm': norm_val,
            'kappa': kappa_val,
            'k': k  # 记录使用了前k个元素
        })
    
    # 按范数排序，返回前top_k个
    candidates.sort(key=lambda x: x['norm'], reverse=True)
    
    if return_top_k == 1:
        # 保持向后兼容
        return candidates[0]['D'], candidates[0]['norm']
    else:
        return candidates[:return_top_k]

# -----------------------------------------------------------------------------
# 优化策略2：幂律缩放法
# -----------------------------------------------------------------------------

def power_scaling(pi_A, n, kappa_bound, alpha=1.0):
    """
    按pi_A的幂次分配权重
    """
    # 避免数值问题，加小量epsilon
    eps = 1e-10
    pi_A_safe = np.maximum(pi_A, eps)
    
    # 设置D_ii ∝ pi_A[i]^alpha
    D_diag = pi_A_safe ** alpha
    
    # 归一化使sum(D_ii) = n
    D_diag = D_diag * n / np.sum(D_diag)
    
    # 裁剪以满足kappa约束
    kappa_current = compute_kappa_D(D_diag)
    if kappa_current > kappa_bound:
        # 压缩范围
        d_min = np.min(D_diag)
        d_max = np.max(D_diag)
        d_target_min = n / (1 + kappa_bound * (n - 1))
        d_target_max = kappa_bound * d_target_min
        
        # 线性映射
        D_diag = d_target_min + (D_diag - d_min) * (d_target_max - d_target_min) / (d_max - d_min)
        # 重新归一化
        D_diag = D_diag * n / np.sum(D_diag)
    
    norm_val = compute_D_pi_norm(D_diag, pi_A)
    return D_diag, norm_val

# -----------------------------------------------------------------------------
# 优化策略3：分层优化法
# -----------------------------------------------------------------------------

def layer_based_optimization(pi_A, n, kappa_bound):
    """
    将索引按pi_A值分层
    """
    sorted_indices = np.argsort(pi_A)
    
    # 定义层（底层20%，中层60%，顶层20%）
    n_bottom = int(0.2 * n)
    n_top = int(0.2 * n)
    n_middle = n - n_bottom - n_top
    
    best_D = None
    best_norm = 0
    
    # 网格搜索权重组合
    for w_ratio_top in np.linspace(0.5, 0.9, 10):
        for w_ratio_middle in np.linspace(0.05, 0.45, 10):
            w_ratio_bottom = 1 - w_ratio_top - w_ratio_middle
            if w_ratio_bottom <= 0:
                continue
            
            # 计算实际权重
            w_bottom = w_ratio_bottom * n / n_bottom if n_bottom > 0 else 0
            w_middle = w_ratio_middle * n / n_middle if n_middle > 0 else 0
            w_top = w_ratio_top * n / n_top if n_top > 0 else 0
            
            # 检查kappa约束
            if w_top / w_bottom > kappa_bound:
                continue
            
            # 构建D
            D_trial = np.zeros(n)
            D_trial[sorted_indices[:n_bottom]] = w_bottom
            D_trial[sorted_indices[n_bottom:n-n_top]] = w_middle
            D_trial[sorted_indices[-n_top:]] = w_top
            
            # 确保和约束（由于舍入误差）
            D_trial = D_trial * n / np.sum(D_trial)
            
            # 计算范数
            norm_val = compute_D_pi_norm(D_trial, pi_A)
            if norm_val > best_norm:
                best_norm = norm_val
                best_D = D_trial.copy()
    
    return best_D, best_norm

# -----------------------------------------------------------------------------
# 优化策略4：scipy数值优化
# -----------------------------------------------------------------------------

def scipy_optimization(pi_A, n, kappa_bound):
    """
    使用scipy.optimize.minimize求解
    改进版：更好的初始点和约束处理
    """
    # 目标函数（负值用于最小化，因为我们要最大化）
    def objective(D_diag):
        return -compute_D_pi_norm(D_diag, pi_A)
    
    # 梯度（修正版本）
    def gradient(D_diag):
        D_pi = D_diag * pi_A
        norm = np.linalg.norm(D_pi)
        if norm > 0:
            # 正确的梯度：∂||D*pi||_2/∂D_i = (D_i * pi_i^2) / ||D*pi||_2
            # 由于我们最小化负值，所以返回负梯度
            grad = (D_pi * pi_A) / norm
            return -grad
        return np.zeros(n)
    
    # 和约束
    linear_constraint = LinearConstraint(np.ones(n), n, n)
    
    # kappa约束（改进的非线性约束）
    def kappa_constraint(D_diag):
        min_d = np.min(D_diag)
        if min_d <= 1e-10:  # 避免除零
            return -1
        max_d = np.max(D_diag)
        return kappa_bound - (max_d / min_d)
    
    # 更智能的初始点：基于pi_A的分布
    # 给大的pi_A分配更多权重作为初始猜测
    sorted_indices = np.argsort(pi_A)[::-1]
    x0 = np.ones(n)
    # 前20%分配更多权重
    top_k = max(1, int(0.2 * n))
    x0[sorted_indices[:top_k]] = n * 0.6 / top_k  # 60%的总权重
    x0[sorted_indices[top_k:]] = n * 0.4 / (n - top_k) if n > top_k else 1
    
    # 边界（所有元素必须为正，且满足和约束的可能范围）
    min_val = n / (kappa_bound * n)  # 理论最小值
    max_val = kappa_bound * min_val  # 理论最大值
    bounds = [(min_val, max_val) for _ in range(n)]
    
    # 使用trust-constr方法，因为它对非线性约束处理更好
    try:
        # 定义非线性约束
        nonlinear_constraint = NonlinearConstraint(
            kappa_constraint, 
            0, 
            np.inf,
            keep_feasible=False
        )
        
        # 执行优化
        result = minimize(
            objective,
            x0,
            method='trust-constr',
            jac=gradient,
            bounds=bounds,
            constraints=[linear_constraint, nonlinear_constraint],
            options={
                'maxiter': 3000,
                'verbose': 0,
                'gtol': 1e-8,
                'xtol': 1e-8
            }
        )
        
        # 检查结果
        if result.x is not None:
            # 强制满足和约束
            D_opt = result.x * n / np.sum(result.x)
            
            # 检查kappa约束
            kappa_actual = compute_kappa_D(D_opt)
            if kappa_actual <= kappa_bound * 1.01:  # 允许1%的误差
                norm = compute_D_pi_norm(D_opt, pi_A)
                return D_opt, norm
    except Exception as e:
        pass
    
    # 如果优化失败，尝试不带梯度的SLSQP
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[linear_constraint],
            options={'maxiter': 2000}
        )
        
        if result.success and result.x is not None:
            D_opt = result.x
            # 手动调整以满足kappa约束
            kappa_actual = compute_kappa_D(D_opt)
            if kappa_actual > kappa_bound:
                # 压缩范围
                d_min = np.min(D_opt)
                d_max = np.max(D_opt)
                scale = (kappa_bound - 1) / (kappa_actual - 1)
                D_opt = d_min + (D_opt - d_min) * scale
                D_opt = D_opt * n / np.sum(D_opt)
            
            norm = compute_D_pi_norm(D_opt, pi_A)
            return D_opt, norm
    except:
        pass
    
    # 如果所有方法都失败，返回贪心解作为备选
    print("  警告：Scipy优化未找到有效解，使用贪心备选方案")
    return greedy_allocation(pi_A, n, kappa_bound)

# -----------------------------------------------------------------------------
# 优化策略5：随机采样搜索
# -----------------------------------------------------------------------------

def random_sampling_search(pi_A, n, kappa_bound, num_samples=1000, return_top_k=5):
    """
    随机采样搜索最优D
    返回前top_k个最优解
    """
    # 存储所有满足约束的候选解
    candidates = []
    
    # 获取pi_A的排序索引
    sorted_indices = np.argsort(pi_A)[::-1]
    
    for i in range(num_samples):
        # 使用Dirichlet分布生成满足和约束的正数
        alpha = np.random.uniform(0.1, 2.0, n)
        D_trial = np.random.dirichlet(alpha) * n
        
        # 检查并调整kappa约束
        kappa_current = compute_kappa_D(D_trial)
        if kappa_current > kappa_bound:
            # 压缩到满足约束
            d_min = np.min(D_trial)
            d_max = np.max(D_trial)
            
            # 二分搜索找到合适的压缩系数
            low, high = 0, 1
            for _ in range(20):  # 二分迭代
                mid = (low + high) / 2
                D_compressed = d_min + (D_trial - d_min) * mid
                if compute_kappa_D(D_compressed) <= kappa_bound:
                    low = mid
                else:
                    high = mid
            
            D_trial = d_min + (D_trial - d_min) * low
            D_trial = D_trial * n / np.sum(D_trial)  # 重新归一化
        
        # 计算范数
        norm_val = compute_D_pi_norm(D_trial, pi_A)
        kappa_val = compute_kappa_D(D_trial)
        
        candidates.append({
            'D': D_trial.copy(),
            'norm': norm_val,
            'kappa': kappa_val,
            'sample_id': i
        })
    
    # 按范数排序，返回前top_k个
    candidates.sort(key=lambda x: x['norm'], reverse=True)
    
    if return_top_k == 1:
        # 保持向后兼容
        if candidates:
            return candidates[0]['D'], candidates[0]['norm']
        else:
            return np.ones(n), compute_D_pi_norm(np.ones(n), pi_A)
    else:
        return candidates[:return_top_k]

# ============================================================================
# 执行优化并比较结果
# ============================================================================

# 设置优化参数
kappa_bound = 100
num_samples = 5000

print("\n" + "="*80)
print("开始优化对角矩阵D以最大化||D*pi_A||_2")
print("="*80)

# 理论上界
theoretical_upper = n * np.max(pi_A)
print(f"\n参数设置:")
print(f"  节点数 n = {n}")
print(f"  条件数上界 kappa_bound = {kappa_bound}")
print(f"  随机采样数 = {num_samples}")
print(f"  理论上界 = {theoretical_upper:.6f}")
print(f"  均匀分布基准 = {compute_D_pi_norm(np.ones(n), pi_A):.6f}")

# 存储结果
results = {}

# 运行策略1：贪心分配
print("\n运行策略1: 贪心分配法...")
start_time = time.time()
# 获取前5个最优解
greedy_top5 = greedy_allocation(pi_A, n, kappa_bound, return_top_k=5)
time_greedy = time.time() - start_time

# 最佳解用于后续分析
D_greedy = greedy_top5[0]['D']
norm_greedy = greedy_top5[0]['norm']

results['Greedy'] = {
    'D': D_greedy,
    'norm': norm_greedy,
    'kappa': compute_kappa_D(D_greedy),
    'time': time_greedy,
    'efficiency': norm_greedy / theoretical_upper,
    'top_5': greedy_top5  # 存储前5个解
}
print(f"  范数: {norm_greedy:.6f}, kappa: {compute_kappa_D(D_greedy):.2f}, 时间: {time_greedy:.3f}s")
print(f"  前5个解的范数: {['{:.4f}'.format(sol['norm']) for sol in greedy_top5]}")

# 运行策略2：幂律缩放（尝试不同的alpha值）
print("\n运行策略2: 幂律缩放法...")
best_power = {'D': None, 'norm': 0, 'alpha': 0}
for alpha in [0.5, 1.0, 1.5, 2.0]:
    start_time = time.time()
    D_power, norm_power = power_scaling(pi_A, n, kappa_bound, alpha)
    time_power = time.time() - start_time
    if norm_power > best_power['norm']:
        best_power = {
            'D': D_power,
            'norm': norm_power,
            'alpha': alpha,
            'kappa': compute_kappa_D(D_power),
            'time': time_power
        }
results['Power'] = {
    'D': best_power['D'],
    'norm': best_power['norm'],
    'kappa': best_power['kappa'],
    'time': best_power['time'],
    'efficiency': best_power['norm'] / theoretical_upper,
    'alpha': best_power['alpha']
}
print(f"  最佳alpha: {best_power['alpha']}, 范数: {best_power['norm']:.6f}, kappa: {best_power['kappa']:.2f}")

# 运行策略3：分层优化
print("\n运行策略3: 分层优化法...")
start_time = time.time()
D_layer, norm_layer = layer_based_optimization(pi_A, n, kappa_bound)
time_layer = time.time() - start_time
results['Layer'] = {
    'D': D_layer,
    'norm': norm_layer,
    'kappa': compute_kappa_D(D_layer),
    'time': time_layer,
    'efficiency': norm_layer / theoretical_upper
}
print(f"  范数: {norm_layer:.6f}, kappa: {compute_kappa_D(D_layer):.2f}, 时间: {time_layer:.3f}s")

# 运行策略4：scipy优化
print("\n运行策略4: Scipy数值优化...")
start_time = time.time()
D_scipy, norm_scipy = scipy_optimization(pi_A, n, kappa_bound)
time_scipy = time.time() - start_time
results['Scipy'] = {
    'D': D_scipy,
    'norm': norm_scipy,
    'kappa': compute_kappa_D(D_scipy),
    'time': time_scipy,
    'efficiency': norm_scipy / theoretical_upper
}
print(f"  范数: {norm_scipy:.6f}, kappa: {compute_kappa_D(D_scipy):.2f}, 时间: {time_scipy:.3f}s")

# 运行策略5：随机采样
print(f"\n运行策略5: 随机采样搜索 ({num_samples}个样本)...")
start_time = time.time()
# 获取前5个最优解
random_top5 = random_sampling_search(pi_A, n, kappa_bound, num_samples, return_top_k=5)
time_random = time.time() - start_time

# 最佳解用于后续分析
D_random = random_top5[0]['D']
norm_random = random_top5[0]['norm']

results['Random'] = {
    'D': D_random,
    'norm': norm_random,
    'kappa': compute_kappa_D(D_random),
    'time': time_random,
    'efficiency': norm_random / theoretical_upper,
    'top_5': random_top5  # 存储前5个解
}
print(f"  范数: {norm_random:.6f}, kappa: {compute_kappa_D(D_random):.2f}, 时间: {time_random:.3f}s")
print(f"  前5个解的范数: {['{:.4f}'.format(sol['norm']) for sol in random_top5]}")

# 找出最佳策略
best_strategy = max(results.keys(), key=lambda k: results[k]['norm'])

print("\n" + "="*80)
print("优化结果总结")
print("="*80)
print(f"\n最佳策略: {best_strategy}")
print(f"最大范数: {results[best_strategy]['norm']:.6f}")
print(f"效率(相对理论上界): {results[best_strategy]['efficiency']*100:.2f}%")
print(f"对应kappa: {results[best_strategy]['kappa']:.2f}")

# 保存结果到CSV
df_results = pd.DataFrame({
    'Strategy': list(results.keys()),
    'Norm': [results[k]['norm'] for k in results],
    'Kappa': [results[k]['kappa'] for k in results],
    'Time': [results[k]['time'] for k in results],
    'Efficiency': [results[k]['efficiency'] for k in results]
})
df_results.to_csv('D_optimization_results.csv', index=False)
print(f"\n结果已保存到: D_optimization_results.csv")

# ============================================================================
# 改进的存储方式：保存所有最优解
# ============================================================================

import json
import pickle

# 1. JSON格式（人类可读）
json_data = {
    'metadata': {
        'n': int(n),
        'seed': int(seed),
        'kappa_bound': float(kappa_bound),
        'theoretical_upper': float(theoretical_upper),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'pi_A_stats': {
            'max': float(np.max(pi_A)),
            'min': float(np.min(pi_A)),
            'mean': float(np.mean(pi_A)),
            'std': float(np.std(pi_A))
        }
    },
    'strategies': {}
}

for strategy_name, strategy_results in results.items():
    strategy_data = {
        'best': {
            'norm': float(strategy_results['norm']),
            'kappa': float(strategy_results['kappa']),
            'efficiency': float(strategy_results['efficiency']),
            'time': float(strategy_results['time'])
        }
    }
    
    # 如果有top_5数据
    if 'top_5' in strategy_results and strategy_results['top_5']:
        strategy_data['top_5'] = []
        for i, solution in enumerate(strategy_results['top_5'][:5]):
            sol_data = {
                'rank': i + 1,
                'norm': float(solution['norm']),
                'kappa': float(solution['kappa']),
                'D_stats': {
                    'max': float(np.max(solution['D'])),
                    'min': float(np.min(solution['D'])),
                    'mean': float(np.mean(solution['D'])),
                    'std': float(np.std(solution['D']))
                }
            }
            # 添加额外的元数据
            if 'k' in solution:
                sol_data['k'] = int(solution['k'])
            if 'sample_id' in solution:
                sol_data['sample_id'] = int(solution['sample_id'])
            strategy_data['top_5'].append(sol_data)
    
    json_data['strategies'][strategy_name] = strategy_data

# 保存JSON文件
with open('D_optimization_results_detailed.json', 'w') as f:
    json.dump(json_data, f, indent=2)
print(f"详细结果已保存到: D_optimization_results_detailed.json")

# 2. 改进的NPZ格式（包含所有top_5解）
npz_data = {
    'pi_A': pi_A,
    'n': n,
    'seed': seed,
    'kappa_bound': kappa_bound
}

for strategy_name, strategy_results in results.items():
    # 保存最佳解
    npz_data[f'{strategy_name}_best_D'] = strategy_results['D']
    npz_data[f'{strategy_name}_best_norm'] = strategy_results['norm']
    
    # 保存top_5解
    if 'top_5' in strategy_results and strategy_results['top_5']:
        for i, solution in enumerate(strategy_results['top_5'][:5]):
            npz_data[f'{strategy_name}_top{i+1}_D'] = solution['D']
            npz_data[f'{strategy_name}_top{i+1}_norm'] = solution['norm']
            npz_data[f'{strategy_name}_top{i+1}_kappa'] = solution['kappa']

np.savez_compressed('D_optimization_all_solutions.npz', **npz_data)
print(f"所有D矩阵已保存到: D_optimization_all_solutions.npz")

# 3. Pickle格式（保存完整的Python对象）
with open('D_optimization_complete_results.pkl', 'wb') as f:
    pickle.dump({
        'results': results,
        'pi_A': pi_A,
        'n': n,
        'seed': seed,
        'kappa_bound': kappa_bound,
        'theoretical_upper': theoretical_upper,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }, f)
print(f"完整结果对象已保存到: D_optimization_complete_results.pkl")

# ============================================================================
# 可视化分析
# ============================================================================

# 创建综合可视化图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 策略性能对比（柱状图）
ax = axes[0, 0]
strategies = list(results.keys())
norms = [results[s]['norm'] for s in strategies]
colors = ['green' if s == best_strategy else 'steelblue' for s in strategies]
bars = ax.bar(strategies, norms, color=colors)
ax.axhline(y=theoretical_upper, color='r', linestyle='--', alpha=0.5, label=f'Theoretical Upper Bound')
ax.axhline(y=compute_D_pi_norm(np.ones(n), pi_A), color='gray', linestyle=':', alpha=0.5, label='Uniform Baseline')
ax.set_ylabel('||D*pi_A||_2', fontsize=12)
ax.set_title('Strategy Performance Comparison', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 在柱状图上添加数值标签
for bar, norm in zip(bars, norms):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{norm:.2f}', ha='center', va='bottom')

# 2. 最佳D矩阵的对角元素分布
ax = axes[0, 1]
best_D = results[best_strategy]['D']
indices = np.arange(n)
sorted_indices = np.argsort(pi_A)[::-1]
ax.bar(indices, best_D[sorted_indices], color='darkblue', alpha=0.7)
ax.set_xlabel('Index (sorted by pi_A)', fontsize=12)
ax.set_ylabel('D diagonal value', fontsize=12)
ax.set_title(f'Best D Distribution ({best_strategy})', fontsize=14)
ax.grid(True, alpha=0.3)

# 3. D*pi_A vs pi_A 关系图（散点图）
ax = axes[0, 2]
D_pi_product = best_D * pi_A
ax.scatter(pi_A, D_pi_product, s=50, alpha=0.6, c=range(n), cmap='viridis')
ax.set_xlabel('pi_A value', fontsize=12)
ax.set_ylabel('D_ii * pi_A[i]', fontsize=12)
ax.set_title('Component-wise Contribution to Norm', fontsize=14)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')

# 添加颜色条
cbar = plt.colorbar(ax.scatter(pi_A, D_pi_product, s=50, alpha=0.6, c=range(n), cmap='viridis'), ax=ax)
cbar.set_label('Index', fontsize=10)

# 4. Kappa vs Norm 权衡（所有策略）
ax = axes[1, 0]
kappas = [results[s]['kappa'] for s in strategies]
for s, kappa, norm in zip(strategies, kappas, norms):
    ax.scatter(kappa, norm, s=100, label=s, alpha=0.7)
ax.set_xlabel('Kappa (Condition Number)', fontsize=12)
ax.set_ylabel('||D*pi_A||_2', fontsize=12)
ax.set_title('Kappa vs Norm Trade-off', fontsize=14)
ax.set_xscale('log')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# 5. 效率对比（相对理论上界的百分比）
ax = axes[1, 1]
efficiencies = [results[s]['efficiency'] * 100 for s in strategies]
bars = ax.bar(strategies, efficiencies, color=['green' if s == best_strategy else 'coral' for s in strategies])
ax.set_ylabel('Efficiency (%)', fontsize=12)
ax.set_title('Efficiency Relative to Theoretical Upper Bound', fontsize=14)
ax.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100% (Theoretical Max)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 添加数值标签
for bar, eff in zip(bars, efficiencies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{eff:.1f}%', ha='center', va='bottom')

# 6. 各策略的D分布对比（热图）
ax = axes[1, 2]
D_matrix = np.zeros((len(strategies), n))
for i, s in enumerate(strategies):
    D_matrix[i, :] = results[s]['D'][sorted_indices]  # 按pi_A排序显示

im = ax.imshow(D_matrix, aspect='auto', cmap='hot', interpolation='nearest')
ax.set_yticks(range(len(strategies)))
ax.set_yticklabels(strategies)
ax.set_xlabel('Index (sorted by pi_A)', fontsize=12)
ax.set_ylabel('Strategy', fontsize=12)
ax.set_title('D Values Heatmap (sorted by pi_A)', fontsize=14)

# 添加颜色条
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('D diagonal value', fontsize=10)

# 调整布局
plt.suptitle(f'D Matrix Optimization Results (n={n}, kappa_bound={kappa_bound})', fontsize=16, y=1.02)
plt.tight_layout()

# 保存可视化
plt.savefig('D_optimization_visualization.png', dpi=150, bbox_inches='tight')
print(f"\n可视化已保存到: D_optimization_visualization.png")
# plt.show()

# 打印详细的D矩阵分析
print("\n" + "="*80)
print("最佳D矩阵详细分析")
print("="*80)
print(f"\n策略: {best_strategy}")
print(f"D矩阵统计:")
print(f"  最大值: {np.max(best_D):.6f}")
print(f"  最小值: {np.min(best_D):.6f}")
print(f"  均值: {np.mean(best_D):.6f}")
print(f"  标准差: {np.std(best_D):.6f}")
print(f"  条件数: {compute_kappa_D(best_D):.2f}")

# 显示前5个最大贡献的元素
contributions = best_D * pi_A
top_indices = np.argsort(contributions)[::-1][:5]
print(f"\n前5个最大贡献元素:")
for i, idx in enumerate(top_indices):
    print(f"  {i+1}. Index {idx}: pi_A={pi_A[idx]:.6f}, D_ii={best_D[idx]:.6f}, 贡献={contributions[idx]:.6f}")

print(f"\n总范数贡献分解:")
print(f"  前20%元素贡献: {np.sum(contributions[top_indices[:int(0.2*n)]]**2)**0.5:.6f}")
print(f"  前50%元素贡献: {np.sum(contributions[top_indices[:int(0.5*n)]]**2)**0.5:.6f}")
print(f"  全部元素贡献: {np.linalg.norm(contributions):.6f}")