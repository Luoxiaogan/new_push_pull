# 贪心算法 D 矩阵优化工具

## 项目结构

```
0928_luogan/
│
├── greedy_optimizer.py           # 核心算法库（贪心优化算法）
├── test_single_config.py         # 单配置测试工具
├── batch_analysis.py             # 批量n-seed分析工具
├── useful_functions_with_batch.py # 基础函数库（矩阵生成等）
└── README.md                      # 本文档
```

## 核心问题

优化对角矩阵 D 以最大化 ||D·π_A||₂，约束条件：
- sum(diag(D)) = n
- κ(D) ≤ 100 (条件数约束)
- D_ii > 0

理论上界：n × max(π_A)

## 快速开始

### 1. 单配置测试

测试单个 (n, seed) 配置的优化结果：

```bash
python test_single_config.py
```

默认参数：n=16, seed=49, κ≤100

### 2. 批量分析

对多个n值，每个n测试多个随机seed，寻找最佳配置：

```bash
python batch_analysis.py
```

默认参数：n∈[5,29], 每个n测试100个seed

## 详细使用

### 核心算法库 (greedy_optimizer.py)

```python
from greedy_optimizer import optimize_D_matrix
from useful_functions_with_batch import get_xinmeng_like_matrix, get_right_perron

# 生成测试矩阵
n = 10
A = get_xinmeng_like_matrix(n, seed=42)
pi_A = get_right_perron(A)

# 贪心优化
result = optimize_D_matrix(pi_A, n, kappa_bound=100)

print(f"优化结果: {result['norm']:.4f}")
print(f"效率: {result['efficiency']*100:.2f}%")
print(f"最优k: {result['k']}")
```

### 单配置测试 (test_single_config.py)

```python
from test_single_config import test_single_configuration

# 自定义参数测试
result = test_single_configuration(n=20, seed=123, kappa_bound=100)

# 结果包含：
# - A矩阵
# - D矩阵
# - 优化结果
# - 统计信息
```

### 批量分析 (batch_analysis.py)

```python
from batch_analysis import batch_optimize, save_best_configs

# 批量优化
n_range = range(5, 21)
n_samples = 50
results, best_configs, all_points = batch_optimize(n_range, n_samples, 100)

# 保存最佳配置
save_best_configs(best_configs, 'my_best_configs.npz')

# 读取已保存的配置
import numpy as np
data = np.load('my_best_configs.npz')
n = 10
A = data[f'A_{n}']
D = data[f'D_{n}']
seed = int(data[f'seed_{n}'])
```

## 输出文件

### 单配置测试
- `single_config_n{n}_seed{seed}.png` - 可视化图片

### 批量分析
- `best_configs.npz` - 每个n的最佳A和D矩阵
- `batch_analysis.png` - 4子图综合分析

## 算法原理

### 贪心策略

1. 将Perron向量元素按值降序排列
2. 对于k=1到n，尝试将权重分配给top-k个元素
3. 在满足条件数约束的前提下，选择使||D·π_A||₂最大的k值

### 效率定义

```
效率 = ||D·π_A||₂ / (n × max(π_A))
```

其中 n × max(π_A) 是理论上界（无条件数约束时的最优值）

## 关键发现

1. **k=1策略占优**：对于Xinmeng-like矩阵，通常k=1（只给最大元素分配权重）是最优的
2. **效率随n递减**：网络规模越大，相对效率越低（约80-95%）
3. **种子依赖性**：不同随机种子产生的矩阵，优化性能差异可达3-4倍

## 注意事项

- 条件数约束κ≤100确保数值稳定性
- Xinmeng-like矩阵会产生极不均匀的Perron向量分布
- 效率值应在0-100%之间，超过100%表示计算错误

## 依赖包

```
numpy
matplotlib
tqdm
```

## 更新日志

- 2024-09-28: 重构代码结构，分离核心算法和应用
- 2024-09-28: 修正理论上界计算错误
- 2024-09-28: 添加批量分析和最佳配置保存功能