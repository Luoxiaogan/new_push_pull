# 贪心优化算法的n-seed系统性分析

## 任务信息
- **创建时间**: 2025-09-02  
- **任务类型**: 系统性分析与可视化
- **基础代码**: test_pi_A_D.py中的贪心优化算法

## 研究目标

分析网络规模(n)和随机种子(seed)对贪心优化算法性能的影响，特别是：
1. 性能如何随n缩放
2. 不同seed导致的性能变异性
3. 识别产生最优/最差性能的seed

## 实施方案

### 第一步：创建新脚本文件
**文件名**: `greedy_n_seed_analysis.py`
**位置**: `/Users/luogan/Code/new_push_pull/合成数据_不使用cupy_最简单的版本/new_test/`

### 第二步：封装贪心优化函数

```python
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
    # 1. 生成Xinmeng-like矩阵
    A = get_xinmeng_like_matrix(n, seed)
    
    # 2. 计算Perron向量
    pi_A = get_right_perron(A)
    
    # 3. 运行贪心优化
    result = greedy_allocation(pi_A, n, kappa_bound, return_top_k=1)
    
    # 4. 返回结果
    return {
        'n': n,
        'seed': seed,
        'D': result[0] if isinstance(result, tuple) else result['D'],
        'norm': result[1] if isinstance(result, tuple) else result['norm'],
        'kappa': compute_kappa_D(D),
        'pi_A': pi_A,
        'pi_A_max': np.max(pi_A),
        'pi_A_min': np.min(pi_A),
        'pi_A_ratio': np.max(pi_A) / np.min(pi_A)
    }
```

### 第三步：主分析函数

```python
def analyze_n_seed_dependency(n_range, n_samples_per_n=10):
    """
    分析贪心优化性能对n和seed的依赖性
    
    参数:
        n_range: n值的范围，如range(5, 21)
        n_samples_per_n: 每个n值的随机种子采样数
    
    返回:
        results_dict: 完整的结果字典
    """
    results = {}
    
    for n in tqdm(n_range, desc='Testing n values'):
        n_results = []
        
        # 对每个n，随机采样seed
        seeds = np.random.randint(0, 10000, n_samples_per_n)
        
        for seed in seeds:
            try:
                result = greedy_optimization_single(n, seed)
                n_results.append(result)
            except Exception as e:
                print(f"Error for n={n}, seed={seed}: {e}")
                continue
        
        # 找出最佳seed
        if n_results:
            best_idx = np.argmax([r['norm'] for r in n_results])
            worst_idx = np.argmin([r['norm'] for r in n_results])
            
            results[n] = {
                'all_results': n_results,
                'best': n_results[best_idx],
                'worst': n_results[worst_idx],
                'mean_norm': np.mean([r['norm'] for r in n_results]),
                'std_norm': np.std([r['norm'] for r in n_results]),
                'seeds_tested': seeds
            }
    
    return results
```

### 第四步：数据存储方案

使用多种格式存储结果：

```python
def save_results(results, base_filename='greedy_n_seed_results'):
    """
    保存分析结果到多种格式
    """
    # 1. JSON格式（元数据和统计）
    json_data = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_values': list(results.keys()),
            'samples_per_n': len(results[list(results.keys())[0]]['all_results'])
        },
        'summary': {}
    }
    
    for n, n_data in results.items():
        json_data['summary'][str(n)] = {
            'best_norm': n_data['best']['norm'],
            'best_seed': n_data['best']['seed'],
            'worst_norm': n_data['worst']['norm'],
            'worst_seed': n_data['worst']['seed'],
            'mean_norm': n_data['mean_norm'],
            'std_norm': n_data['std_norm'],
            'all_norms': [r['norm'] for r in n_data['all_results']],
            'all_seeds': [r['seed'] for r in n_data['all_results']]
        }
    
    with open(f'{base_filename}.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # 2. NPZ格式（包含D矩阵）
    npz_data = {}
    for n, n_data in results.items():
        npz_data[f'n{n}_best_D'] = n_data['best']['D']
        npz_data[f'n{n}_best_norm'] = n_data['best']['norm']
        npz_data[f'n{n}_best_seed'] = n_data['best']['seed']
        npz_data[f'n{n}_all_norms'] = [r['norm'] for r in n_data['all_results']]
    
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
                'pi_A_max': result['pi_A_max'],
                'pi_A_min': result['pi_A_min'],
                'pi_A_ratio': result['pi_A_ratio']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(f'{base_filename}.csv', index=False)
    
    print(f"Results saved to {base_filename}.[json/npz/csv]")
```

### 第五步：可视化函数

```python
def visualize_results(results):
    """
    创建综合可视化图表
    """
    # 准备数据
    n_values = sorted(results.keys())
    all_data_points = []
    best_points = []
    worst_points = []
    mean_points = []
    
    for n in n_values:
        n_data = results[n]
        # 所有数据点
        for r in n_data['all_results']:
            all_data_points.append((n, r['norm'], r['seed']))
        # 最佳点
        best_points.append((n, n_data['best']['norm']))
        # 最差点
        worst_points.append((n, n_data['worst']['norm']))
        # 平均值
        mean_points.append((n, n_data['mean_norm']))
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 子图1：所有数据点 + 最佳值标记
    ax = axes[0, 0]
    # 绘制所有点（小点，半透明）
    for n, norm, seed in all_data_points:
        ax.scatter(n, norm, alpha=0.3, s=30, color='blue')
    
    # 标记最佳值（大红点）
    best_n, best_norms = zip(*best_points)
    ax.scatter(best_n, best_norms, color='red', s=100, 
               marker='*', label='Best', zorder=5)
    
    # 连接最佳值
    ax.plot(best_n, best_norms, 'r--', alpha=0.5, linewidth=2)
    
    # 添加平均值线
    mean_n, mean_norms = zip(*mean_points)
    ax.plot(mean_n, mean_norms, 'g-', alpha=0.7, 
            linewidth=2, label='Mean')
    
    ax.set_xlabel('n (number of nodes)', fontsize=12)
    ax.set_ylabel('||D*pi_A||_2', fontsize=12)
    ax.set_title('Greedy Optimization Performance vs Network Size', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 子图2：最佳、最差、平均值对比
    ax = axes[0, 1]
    worst_n, worst_norms = zip(*worst_points)
    
    ax.plot(best_n, best_norms, 'r-o', label='Best', linewidth=2)
    ax.plot(mean_n, mean_norms, 'g-s', label='Mean', linewidth=2)
    ax.plot(worst_n, worst_norms, 'b-^', label='Worst', linewidth=2)
    
    # 添加误差带（标准差）
    std_values = [results[n]['std_norm'] for n in n_values]
    ax.fill_between(n_values, 
                     [m - s for m, s in zip(mean_norms, std_values)],
                     [m + s for m, s in zip(mean_norms, std_values)],
                     alpha=0.2, color='green', label='±1 std')
    
    ax.set_xlabel('n (number of nodes)', fontsize=12)
    ax.set_ylabel('||D*pi_A||_2', fontsize=12)
    ax.set_title('Performance Statistics', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 子图3：变异系数（CV = std/mean）
    ax = axes[1, 0]
    cv_values = [results[n]['std_norm'] / results[n]['mean_norm'] 
                 for n in n_values]
    ax.plot(n_values, cv_values, 'b-o', linewidth=2)
    ax.set_xlabel('n (number of nodes)', fontsize=12)
    ax.set_ylabel('Coefficient of Variation', fontsize=12)
    ax.set_title('Performance Variability', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 子图4：最佳种子分布
    ax = axes[1, 1]
    best_seeds = [results[n]['best']['seed'] for n in n_values]
    ax.scatter(n_values, best_seeds, s=50, alpha=0.7)
    ax.set_xlabel('n (number of nodes)', fontsize=12)
    ax.set_ylabel('Best Seed Value', fontsize=12)
    ax.set_title('Optimal Seeds Distribution', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 为最佳点添加标注
    for n, seed, norm in zip(n_values, best_seeds, best_norms):
        ax.annotate(f'{norm:.2f}', 
                   xy=(n, seed), 
                   xytext=(3, 3),
                   textcoords='offset points',
                   fontsize=8,
                   alpha=0.7)
    
    plt.suptitle('Greedy Optimization: n-seed Dependency Analysis', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('greedy_n_seed_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig
```

### 第六步：主执行流程

```python
def main():
    """
    主执行函数
    """
    # 设置参数
    n_range = range(5, 21)  # n从5到20
    n_samples = 10  # 每个n采样10个seed
    
    print("="*60)
    print("贪心优化算法 n-seed 依赖性分析")
    print("="*60)
    print(f"n范围: {list(n_range)}")
    print(f"每个n的采样数: {n_samples}")
    
    # 运行分析
    results = analyze_n_seed_dependency(n_range, n_samples)
    
    # 保存结果
    save_results(results)
    
    # 可视化
    fig = visualize_results(results)
    
    # 打印总结
    print("\n" + "="*60)
    print("分析总结")
    print("="*60)
    
    for n in sorted(results.keys()):
        n_data = results[n]
        print(f"\nn = {n}:")
        print(f"  最佳: norm={n_data['best']['norm']:.4f}, seed={n_data['best']['seed']}")
        print(f"  最差: norm={n_data['worst']['norm']:.4f}, seed={n_data['worst']['seed']}")
        print(f"  平均: {n_data['mean_norm']:.4f} ± {n_data['std_norm']:.4f}")
        print(f"  变异系数: {n_data['std_norm']/n_data['mean_norm']:.3f}")
    
    # 找出全局最佳
    global_best_n = None
    global_best_norm = 0
    for n, n_data in results.items():
        if n_data['best']['norm'] > global_best_norm:
            global_best_norm = n_data['best']['norm']
            global_best_n = n
    
    print(f"\n全局最佳: n={global_best_n}, norm={global_best_norm:.4f}")
    print(f"对应seed: {results[global_best_n]['best']['seed']}")

if __name__ == "__main__":
    main()
```

## 预期输出

### 文件输出
1. `greedy_n_seed_results.json` - 完整的统计数据
2. `greedy_n_seed_results.npz` - 包含最优D矩阵
3. `greedy_n_seed_results.csv` - 表格数据，便于进一步分析
4. `greedy_n_seed_analysis.png` - 4子图综合可视化

### 可视化内容
1. **主图**：所有数据点散点图，标记最佳值
2. **统计图**：最佳/平均/最差值对比，含标准差带
3. **变异性图**：变异系数随n的变化
4. **种子分布图**：最优种子的分布模式

## 关键洞察期望

1. **性能缩放**：||D*pi_A||_2如何随n增长？是线性、亚线性还是超线性？
2. **变异性**：随着n增大，性能的变异性是增加还是减少？
3. **种子敏感性**：某些种子是否系统性地产生更好的结果？
4. **理论界限**：实际性能与理论上界n*max(pi_A)的比例如何变化？

## 实施顺序

1. [ ] 创建greedy_n_seed_analysis.py文件
2. [ ] 导入必要的库和函数
3. [ ] 实现greedy_optimization_single函数
4. [ ] 实现analyze_n_seed_dependency函数
5. [ ] 实现save_results函数
6. [ ] 实现visualize_results函数
7. [ ] 实现main函数
8. [ ] 运行测试（先用小范围如n=5-8测试）
9. [ ] 运行完整分析（n=5-20）
10. [ ] 分析结果并生成报告

## 成功标准

- 成功运行所有n值和seed组合
- 生成清晰的可视化显示趋势
- 识别出性能的缩放规律
- 量化性能的变异性
- 保存可重现的结果数据