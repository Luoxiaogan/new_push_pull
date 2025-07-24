# 不同学习率策略的系统化测试

## 概述

本文档规定了实现一个简洁接口的要求，用于系统化测试分布式优化实验中的不同学习率策略。目标是在现有训练函数的基础上创建一个更友好的封装器，同时保持不同策略之间的公平比较。

## 设计原则

1. **公平比较**：比较不同学习率策略时，保持总学习率和 `lr_basic * n` 在所有策略中恒定
2. **模块化设计**：将拓扑生成、学习率策略和实验执行分离
3. **可重现性**：为网络拓扑和随机策略提供清晰的种子管理
4. **清晰输出**：使用清晰的命名约定系统化组织结果

## API 规范

### 主函数

```python
def run_distributed_optimization_experiment(
    # 拓扑参数
    topology: str,          # 选项: "exp", "grid", "ring", "random", "geometric", "neighbor"
    n: int,                 # 节点数量
    matrix_seed: int,       # 拓扑生成种子
    
    # 学习率参数
    lr_basic: float,        # 基础学习率（总和为 lr_basic * n）
    strategy: str,          # 选项: "uniform", "pi_a_inverse", "pi_b_inverse", "random"
    random_seed: int = None,  # 仅在 strategy="random" 时使用
    
    # 训练参数
    dataset_name: str,      # "MNIST" 或 "CIFAR10"
    batch_size: int,
    num_epochs: int,
    alpha: float,           # 异质性参数（越高越均匀）
    use_hetero: bool,       # 启用异质数据分布
    
    # 实验参数
    repetitions: int = 1,   # 重复次数用于求平均
    remark: str = "",       # 实验标识符
    device: str = "cuda:0", # GPU 设备
    
    # 输出参数
    output_dir: str,        # 保存结果的基础目录
) -> pd.DataFrame:
    """
    使用指定配置运行分布式优化实验。
    
    返回：
        包含跨重复平均结果的 DataFrame
    """
```

## 参数详情

### 拓扑选项

`topology` 参数支持 6 种预定义的网络结构：
- `"exp"`：指数图（使用 `get_matrixs_from_exp_graph`）
- `"grid"`：网格拓扑（使用 `generate_grid_matrices`）
- `"ring"`：带快捷方式的环形拓扑（使用 `generate_ring_matrices`）
- `"random"`：边概率为 p=1/3 的随机图（使用 `generate_random_graph_matrices`）
- `"geometric"`：随机几何图（使用 `generate_stochastic_geometric_matrices`）
- `"neighbor"`：k-最近邻图（使用 `generate_nearest_neighbor_matrices`）

### 学习率策略选项

`strategy` 参数决定了总学习率如何在节点间分配：

1. **`"uniform"`**：所有节点使用相同的学习率
   - `lr_list = [lr_basic] * n`

2. **`"pi_a_inverse"`**：学习率与 A 的左 Perron 向量的倒数成比例
   - 计算 `pi_a = get_left_perron(A)`
   - 设置 `D = diag(1/pi_a)`，标准化使得 trace(D) = n
   - `lr_list[i] = lr_basic * D[i,i]`

3. **`"pi_b_inverse"`**：学习率与 B 的右 Perron 向量的倒数成比例
   - 计算 `pi_b = get_right_perron(B)`
   - 设置 `D = diag(1/pi_b)`，标准化使得 trace(D) = n
   - `lr_list[i] = lr_basic * D[i,i]`

4. **`"random"`**：保持总和不变的随机分布
   - 使用 `random_seed` 确保可重现性
   - 生成随机正值，标准化使其和为 n
   - `lr_list[i] = lr_basic * random_value[i]`

### 重复处理

处理多次重复以进行统计平均：
- 每次重复应该使用不同的训练函数种子
- 建议方法：`training_seed = base_seed + repetition_index`
- 对梯度范数 CSV 结果进行跨重复平均
- 保存单独和平均结果

## 输出规范

### 文件命名约定

单次运行输出（来自 `train_track_grad_norm_with_hetero_different_learning_rate`）：
- 损失 CSV：`{remark}_hetero={use_hetero}, alpha={alpha}, {algorithm}, lr[0]={lr_list[0]}, n_nodes={n}, batch_size={batch_size}, {date}.csv`
- 梯度范数 CSV：`{remark}_grad_norm,hetero={use_hetero},s alpha={alpha}, {algorithm}, lr[0]={lr_list[0]}, n_nodes={n}, batch_size={batch_size}, {date}.csv`

平均结果（多次重复后）：
```
{output_dir}/averaged_results/topology={topology}_n={n}_strategy={strategy}_lr_total={lr_basic*n}_seed={matrix_seed}_reps={repetitions}.csv
```

### 计算 c 值

理论收敛因子 c 定义为：
```
c = n * pi_A^T * D * pi_B
```

其中：
- `pi_A`：矩阵 A 的左 Perron 向量
- `pi_B`：矩阵 B 的右 Perron 向量
- `D`：表示学习率倍数的对角矩阵
- `n`：节点数量

实现：
```python
def compute_c_value(A, B, lr_list, lr_basic):
    n = A.shape[0]
    pi_a = get_left_perron(A)
    pi_b = get_right_perron(B)
    
    # 从学习率构建 D 矩阵
    D = np.diag([lr / lr_basic for lr in lr_list])
    
    # 计算 c = n * pi_A^T * D * pi_B
    c = n * pi_a.T @ D @ pi_b
    return c
```

## 使用示例

```python
# 示例 1：在最近邻拓扑上使用均匀学习率
df = run_distributed_optimization_experiment(
    topology="neighbor",
    n=16,
    matrix_seed=42,
    lr_basic=0.007,
    strategy="uniform",
    dataset_name="MNIST",
    batch_size=128,
    num_epochs=100,
    alpha=1000,
    use_hetero=True,
    repetitions=5,
    remark="uniform_lr_test",
    device="cuda:0",
    output_dir="./experiments"
)

# 示例 2：在网格拓扑上使用 Pi_b 逆策略
df = run_distributed_optimization_experiment(
    topology="grid",
    n=16,
    matrix_seed=123,
    lr_basic=0.007,
    strategy="pi_b_inverse",
    dataset_name="CIFAR10",
    batch_size=64,
    num_epochs=200,
    alpha=100,
    use_hetero=True,
    repetitions=3,
    remark="pi_b_inverse_test",
    device="cuda:1",
    output_dir="./experiments"
)
```

## 实现注意事项

1. **矩阵验证**：确保生成的矩阵 A 和 B 满足：
   - A 是行随机的（行和为 1）
   - B 是列随机的（列和为 1）
   - 两者都表示强连通图

2. **错误处理**：包括以下验证：
   - 有效的拓扑名称
   - 有效的策略名称
   - 网格拓扑的适当 n 值（完全平方数）
   - 当 strategy="random" 时提供 random_seed

3. **日志记录**：考虑添加以下日志：
   - 选定的拓扑及其属性（kappa、beta 值）
   - 学习率分布
   - 计算的 c 值
   - 重复进度

4. **内存管理**：对于大 n 或多次重复，考虑：
   - 将中间结果保存到磁盘
   - 在重复之间清除 GPU 内存
   - 批处理结果