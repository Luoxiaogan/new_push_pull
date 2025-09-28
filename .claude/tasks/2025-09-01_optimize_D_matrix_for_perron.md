# 优化对角矩阵D以最大化||D*pi_A||_2

## 任务信息
- **创建时间**: 2025-09-01
- **任务类型**: 数值优化与分析
- **基础文件**: /Users/luogan/Code/new_push_pull/合成数据_不使用cupy_最简单的版本/new_test/test_pi_A_D.py

## 问题定义

### 已知条件
- **pi_A**: Xinmeng-like矩阵的右Perron向量（列随机矩阵的稳态分布）
- **n**: 节点数（当前设置为16）
- **特点**: pi_A元素差异极大，最大值/最小值比值通常>1000

### 优化目标
找到n×n对角矩阵D，使得||D*pi_A||_2最大

### 约束条件
1. **正定性**: D_ii > 0 for all i
2. **和约束**: sum(D_ii) = n
3. **条件数约束**: kappa_D = max(D_ii)/min(D_ii) ≤ kappa_D_upper_bound（例如100）

### 数学表述
- **目标函数**: maximize ||D*pi_A||_2 = sqrt(sum((D_ii * pi_A[i])^2))
- **理论上界**: ||D*pi_A||_2 ≤ n * max(pi_A)（当所有权重集中在最大元素时）

## 实施步骤

### 第一阶段：扩展基础代码
在`test_pi_A_D.py`中添加以下功能模块：

```python
# 1. 基础计算函数
def compute_D_pi_norm(D_diag, pi_A):
    """计算||D*pi_A||_2"""
    return np.linalg.norm(D_diag * pi_A)

def compute_kappa_D(D_diag):
    """计算对角矩阵D的条件数"""
    return np.max(D_diag) / np.min(D_diag)

def check_constraints(D_diag, n, kappa_bound):
    """检查D是否满足所有约束"""
    # 检查和约束、正定性、条件数
    pass
```

### 第二阶段：实现多种优化策略

#### 策略1：贪心分配法
```python
def greedy_allocation(pi_A, n, kappa_bound):
    """
    贪心算法：优先给pi_A大的元素分配权重
    1. 按pi_A值降序排序索引
    2. 给前k个分配最大权重d_max
    3. 给剩余分配最小权重d_min
    4. 满足：k*d_max + (n-k)*d_min = n且d_max/d_min ≤ kappa_bound
    """
```

#### 策略2：幂律缩放法
```python
def power_scaling(pi_A, n, kappa_bound, alpha):
    """
    按pi_A的幂次分配权重
    1. 设置D_ii ∝ pi_A[i]^alpha
    2. 归一化使sum(D_ii) = n
    3. 裁剪以满足kappa约束
    """
```

#### 策略3：分层优化法
```python
def layer_based_optimization(pi_A, n, kappa_bound):
    """
    将索引按pi_A值分层
    - 顶层20%（最大值）: 权重w1
    - 中层60%: 权重w2
    - 底层20%（最小值）: 权重w3
    优化w1, w2, w3满足约束
    """
```

#### 策略4：数值优化法
```python
def scipy_optimization(pi_A, n, kappa_bound):
    """
    使用scipy.optimize.minimize
    - 目标：-||D*pi_A||_2（负值用于最小化）
    - 约束：线性约束（和=n）+ 非线性约束（kappa≤bound）
    - 方法：SLSQP或trust-constr
    """
```

### 第三阶段：随机采样与评估

```python
def random_sampling_search(pi_A, n, kappa_bound, num_samples=1000):
    """
    随机采样搜索
    1. 使用Dirichlet分布生成满足和约束的D
    2. 调整以满足kappa约束
    3. 评估||D*pi_A||_2
    4. 保留最佳的top-k个解
    """
```

特殊采样技巧：
- 基于pi_A排序的偏向采样
- 使用(1-ε)*n分配给大元素，ε*n分配给小元素
- ε取值范围：[0.01, 0.1]

### 第四阶段：可视化与分析

创建4个子图展示：
1. **策略对比图**: 不同策略的||D*pi_A||_2值对比
2. **D对角元素分布**: 最优解的D_ii值分布（柱状图）
3. **权重分配热图**: 显示D_ii与pi_A[i]的关系
4. **帕累托前沿**: ||D*pi_A||_2 vs kappa_D的权衡曲线

### 第五阶段：参数敏感性分析

测试不同参数配置：
- **kappa_bound取值**: [10, 50, 100, 500, 1000, inf]
- **节点数n**: [8, 16, 32, 64]
- **不同seed**: 测试10个不同的随机种子

分析内容：
1. kappa约束如何影响可达到的最大范数
2. 最优解的稳定性（对pi_A扰动的敏感度）
3. 不同n值下的缩放规律

### 第六阶段：结果输出

#### 数据文件
- `D_optimization_results.csv`: 包含所有策略的结果
  - 列：strategy, norm_value, kappa_D, D_diagonal_values, computation_time
- `best_D_matrices.npz`: 保存最优的D矩阵
- `parameter_study_results.csv`: 参数研究结果

#### 可视化文件
- `D_optimization_comparison.png`: 策略对比图
- `D_distribution_best.png`: 最优D的分布
- `pareto_front_analysis.png`: 帕累托前沿分析

#### 分析报告
- `D_optimization_report.md`: 详细分析报告，包括：
  - 各策略性能对比
  - 最优解特征分析
  - 参数敏感性结论
  - 实际应用建议

## 实验配置

```python
# 默认参数
DEFAULT_CONFIG = {
    'n': 16,
    'seed': 49,
    'kappa_bound': 100,
    'num_samples': 1000,
    'epsilon_range': [0.01, 0.05, 0.1],
    'alpha_range': [0.5, 1.0, 1.5, 2.0],
    'layer_percentiles': [20, 80]  # 顶层20%，底层20%
}
```

## 预期成果

1. **找到最优D矩阵**，在kappa约束下最大化||D*pi_A||_2
2. **理解权重分配模式**：哪些索引应该获得更多权重
3. **量化约束影响**：kappa_bound如何限制可达到的范数
4. **提供实用指南**：在实际应用中如何选择D

## 技术要点

1. **数值稳定性**: 处理极小的pi_A值时避免下溢
2. **优化效率**: 对大规模n使用向量化操作
3. **约束处理**: 确保所有解严格满足约束条件
4. **结果验证**: 独立验证最优解的正确性

## 执行顺序

1. [ ] 扩展test_pi_A_D.py添加基础函数
2. [ ] 实现四种优化策略
3. [ ] 运行随机采样搜索
4. [ ] 生成可视化图表
5. [ ] 进行参数敏感性分析
6. [ ] 编写分析报告
7. [ ] 整理并保存所有结果

## 成功标准

- 找到的D使||D*pi_A||_2 ≥ 0.8 * n * max(pi_A)（理论上界的80%）
- 所有策略都能在1秒内完成计算（n≤100）
- 可视化清晰展示优化权衡
- 报告提供可操作的结论