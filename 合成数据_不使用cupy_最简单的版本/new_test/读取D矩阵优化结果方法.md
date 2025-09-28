# D矩阵优化结果读取方法

## 存储格式说明

优化程序生成了三种格式的存储文件，每种都有其特定用途：

### 1. JSON格式 (`D_optimization_results_detailed.json`)
**特点**：人类可读，易于跨语言使用
**用途**：快速查看结果，与其他工具集成

### 2. NPZ格式 (`D_optimization_all_solutions.npz`)
**特点**：NumPy原生格式，紧凑高效
**用途**：数值计算和进一步分析

### 3. Pickle格式 (`D_optimization_complete_results.pkl`)
**特点**：保存完整Python对象
**用途**：完整恢复所有数据结构

## 读取方法示例

### 1. 读取JSON文件

```python
import json
import numpy as np

# 读取JSON文件
with open('D_optimization_results_detailed.json', 'r') as f:
    data = json.load(f)

# 查看元数据
print("节点数:", data['metadata']['n'])
print("条件数上界:", data['metadata']['kappa_bound'])
print("理论上界:", data['metadata']['theoretical_upper'])

# 查看每个策略的最佳结果
for strategy_name, strategy_data in data['strategies'].items():
    best = strategy_data['best']
    print(f"\n策略: {strategy_name}")
    print(f"  最佳范数: {best['norm']:.6f}")
    print(f"  效率: {best['efficiency']*100:.2f}%")
    print(f"  条件数: {best['kappa']:.2f}")
    
    # 如果有top_5数据，显示前5个解
    if 'top_5' in strategy_data:
        print(f"  前5个解的范数:")
        for sol in strategy_data['top_5']:
            print(f"    第{sol['rank']}名: {sol['norm']:.6f} (kappa={sol['kappa']:.2f})")
```

### 2. 读取NPZ文件

```python
import numpy as np

# 加载NPZ文件
data = np.load('D_optimization_all_solutions.npz')

# 查看所有保存的数组名称
print("保存的数组:", list(data.keys()))

# 读取基本信息
n = data['n']
seed = data['seed']
kappa_bound = data['kappa_bound']
pi_A = data['pi_A']

print(f"n={n}, seed={seed}, kappa_bound={kappa_bound}")

# 读取各策略的最佳D矩阵
strategies = ['Greedy', 'Power', 'Layer', 'Scipy', 'Random']
for strategy in strategies:
    if f'{strategy}_best_D' in data:
        D_best = data[f'{strategy}_best_D']
        norm_best = data[f'{strategy}_best_norm']
        print(f"\n{strategy}策略:")
        print(f"  最佳范数: {norm_best:.6f}")
        print(f"  D矩阵统计: max={np.max(D_best):.4f}, min={np.min(D_best):.4f}")
        
        # 读取前5个解（如果存在）
        for i in range(1, 6):
            if f'{strategy}_top{i}_D' in data:
                D_i = data[f'{strategy}_top{i}_D']
                norm_i = data[f'{strategy}_top{i}_norm']
                kappa_i = data[f'{strategy}_top{i}_kappa']
                print(f"  Top{i}: norm={norm_i:.4f}, kappa={kappa_i:.2f}")
```

### 3. 读取Pickle文件

```python
import pickle
import numpy as np

# 加载完整的结果对象
with open('D_optimization_complete_results.pkl', 'rb') as f:
    complete_data = pickle.load(f)

# 访问所有数据
results = complete_data['results']
pi_A = complete_data['pi_A']
n = complete_data['n']
theoretical_upper = complete_data['theoretical_upper']
timestamp = complete_data['timestamp']

print(f"优化运行时间: {timestamp}")
print(f"理论上界: {theoretical_upper:.6f}")

# 遍历所有策略
for strategy_name, strategy_results in results.items():
    print(f"\n策略: {strategy_name}")
    print(f"  范数: {strategy_results['norm']:.6f}")
    print(f"  效率: {strategy_results['efficiency']*100:.2f}%")
    print(f"  计算时间: {strategy_results['time']:.3f}秒")
    
    # 如果有top_5数据
    if 'top_5' in strategy_results:
        top5 = strategy_results['top_5']
        print(f"  找到{len(top5)}个解")
        for i, sol in enumerate(top5[:3]):  # 显示前3个
            print(f"    解{i+1}: norm={sol['norm']:.4f}, kappa={sol['kappa']:.2f}")
            # 贪心策略有额外的k参数
            if 'k' in sol:
                print(f"      使用前{sol['k']}个元素")
            # 随机策略有sample_id
            if 'sample_id' in sol:
                print(f"      来自第{sol['sample_id']}个样本")
```

## 高级分析示例

### 比较不同策略的前5个解

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = np.load('D_optimization_all_solutions.npz')

# 收集所有策略的前5个解的范数
strategies = ['Greedy', 'Random']
strategy_norms = {}

for strategy in strategies:
    norms = []
    for i in range(1, 6):
        if f'{strategy}_top{i}_norm' in data:
            norms.append(float(data[f'{strategy}_top{i}_norm']))
    if norms:
        strategy_norms[strategy] = norms

# 绘制对比图
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(1, 6)

for strategy, norms in strategy_norms.items():
    ax.plot(x[:len(norms)], norms, marker='o', label=strategy, linewidth=2)

ax.set_xlabel('Rank')
ax.set_ylabel('||D*pi_A||_2')
ax.set_title('Top 5 Solutions Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### 分析D矩阵的分布模式

```python
import numpy as np
import pandas as pd

# 加载完整数据
with open('D_optimization_complete_results.pkl', 'rb') as f:
    data = pickle.load(f)

results = data['results']
pi_A = data['pi_A']

# 创建DataFrame分析
analysis_data = []

for strategy_name, strategy_results in results.items():
    if 'top_5' not in strategy_results:
        continue
    
    for i, sol in enumerate(strategy_results['top_5'][:3]):
        D = sol['D']
        
        # 计算D和pi_A的相关性
        correlation = np.corrcoef(D, pi_A)[0, 1]
        
        # 找出最大权重的位置
        max_idx = np.argmax(D)
        max_pi_A_idx = np.argmax(pi_A)
        
        analysis_data.append({
            'Strategy': strategy_name,
            'Rank': i + 1,
            'Norm': sol['norm'],
            'Kappa': sol['kappa'],
            'D_pi_correlation': correlation,
            'Max_D_at_max_pi': max_idx == max_pi_A_idx,
            'D_max': np.max(D),
            'D_min': np.min(D),
            'D_std': np.std(D)
        })

df_analysis = pd.DataFrame(analysis_data)
print(df_analysis.to_string())

# 保存分析结果
df_analysis.to_csv('D_optimization_analysis.csv', index=False)
```

## 重新运行优化的示例

```python
import numpy as np

# 从保存的文件中恢复参数
data = np.load('D_optimization_all_solutions.npz')
pi_A = data['pi_A']
n = int(data['n'])
seed = int(data['seed'])
kappa_bound = float(data['kappa_bound'])

print(f"恢复的参数: n={n}, seed={seed}, kappa_bound={kappa_bound}")

# 现在可以用这些参数重新运行优化或进行新的实验
# 例如，测试不同的kappa_bound值
for new_kappa in [50, 100, 200, 500]:
    # 运行优化...
    pass
```

## 文件大小和性能比较

| 格式 | 文件大小 | 读取速度 | 跨语言兼容性 | 人类可读性 |
|------|---------|----------|-------------|-----------|
| JSON | 较大 | 慢 | 优秀 | 优秀 |
| NPZ | 中等 | 快 | 仅NumPy | 差 |
| Pickle | 较小 | 最快 | 仅Python | 差 |

## 建议使用场景

1. **快速查看结果**：使用JSON文件
2. **数值计算和可视化**：使用NPZ文件
3. **完整恢复和继续实验**：使用Pickle文件
4. **与MATLAB/R集成**：使用JSON或导出为CSV
5. **长期归档**：保留所有三种格式

## 注意事项

1. Pickle文件可能存在版本兼容性问题，建议记录Python和NumPy版本
2. JSON文件中的数值都转换为float，可能有精度损失
3. NPZ文件使用压缩格式，读取时会自动解压
4. 对于大规模问题（n>1000），考虑使用HDF5格式