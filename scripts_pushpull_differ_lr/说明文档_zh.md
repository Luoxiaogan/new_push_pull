# 分布式优化实验运行器使用说明

## 概述

本文档介绍如何使用实验运行器系统化地测试不同的学习率策略在分布式优化中的表现。该系统支持多种网络拓扑、学习率分配策略，并提供了完整的实验管理功能。

## 文件结构

```
scripts_pushpull_differ_lr/
├── experiment_utils.py      # 实验工具函数（拓扑生成、学习率计算等）
├── run_experiment.py        # 主运行脚本
├── run_experiments.sh       # 批量实验Shell脚本
├── network_utils.py         # 网络拓扑生成函数（原有文件）
└── 说明文档_zh.md          # 本文档
```

## 主要功能

1. **支持6种网络拓扑**：exp（指数图）、grid（网格）、ring（环形）、random（随机）、geometric（几何）、neighbor（k近邻）
2. **支持4种学习率策略**：uniform（均匀）、pi_a_inverse（A矩阵Perron向量倒数）、pi_b_inverse（B矩阵Perron向量倒数）、random（随机）
3. **自动实验管理**：时间戳命名、配置文件保存、结果平均
4. **灵活的参数配置**：支持命令行参数和Shell脚本批量运行

## 参数说明

### 必需参数（无默认值）

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `--topology` | str | 网络拓扑类型 | exp, grid, ring, random, geometric, neighbor |
| `--n` | int | 节点数量 | 16 |
| `--matrix_seed` | int | 拓扑生成种子 | 42 |
| `--lr_basic` | float | 基础学习率 | 0.007 |
| `--strategy` | str | 学习率策略 | uniform, pi_a_inverse, pi_b_inverse, random |
| `--dataset_name` | str | 数据集名称 | MNIST, CIFAR10 |
| `--batch_size` | int | 批次大小 | 128 |
| `--num_epochs` | int | 训练轮数 | 100 |
| `--alpha` | float | 异质性参数（越大越均匀） | 1000 |
| `--use_hetero` | flag | 启用异质数据分布 | --use_hetero |

### 可选参数（有默认值）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--random_seed` | int | None | 随机策略的种子 |
| `--repetitions` | int | 1 | 重复次数（用于求平均） |
| `--remark` | str | "" | 实验备注标识 |
| `--device` | str | cuda:0 | GPU设备 |
| `--output_dir` | str | ./experiments | 输出目录 |
| `--k` | int | 3 | k近邻拓扑的邻居数 |

## 使用方法

### 1. 运行单个实验

```bash
python run_experiment.py \
    --topology neighbor \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --strategy uniform \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 100 \
    --alpha 1000 \
    --use_hetero \
    --repetitions 3 \
    --remark "test_uniform" \
    --device cuda:0 \
    --output_dir "./my_experiments"
```

### 2. 使用Shell脚本批量运行

#### 运行预定义的批量实验：
```bash
# 首先赋予执行权限
chmod +x run_experiments.sh

# 运行批量实验
./run_experiments.sh
```

#### 创建自定义批量实验脚本：
```bash
#!/bin/bash

# 设置Python环境
PYTHON=python3

# 设置输出目录
OUTPUT_BASE="./experiments/my_batch_$(date +%Y%m%d_%H%M%S)"

# 实验1：比较不同的学习率策略
for strategy in uniform pi_a_inverse pi_b_inverse; do
    $PYTHON run_experiment.py \
        --topology neighbor \
        --n 16 \
        --matrix_seed 42 \
        --lr_basic 0.007 \
        --strategy $strategy \
        --dataset_name MNIST \
        --batch_size 128 \
        --num_epochs 100 \
        --alpha 1000 \
        --use_hetero \
        --repetitions 5 \
        --remark "strategy_${strategy}" \
        --device cuda:0 \
        --output_dir "$OUTPUT_BASE"
done
```

### 3. 在Python脚本中调用

```python
from run_experiment import run_distributed_optimization_experiment

# 运行实验
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
    repetitions=3,
    remark="python_test",
    device="cuda:0",
    output_dir="./experiments"
)
```

## 输出文件结构

每次实验会创建一个独立的目录，包含所有相关文件：

```
experiments/
└── experiment_20250122_143022_neighbor_uniform_16nodes/
    ├── config.yaml                                    # 实验配置文件
    ├── test_uniform_hetero=True, alpha=1000, PushPull, lr[0]=0.007, n_nodes=16, batch_size=128, 2025-01-22.csv
    ├── test_uniform_grad_norm,hetero=True,s alpha=1000, PushPull, lr[0]=0.007, n_nodes=16, batch_size=128, 2025-01-22.csv
    └── averaged_grad_norm.csv                         # 多次重复的平均结果（仅当repetitions>1时）
```

### config.yaml 文件内容

配置文件包含以下信息：
- **experiment_info**: 时间戳、实验名称、输出目录
- **topology_parameters**: 拓扑类型、节点数、种子、κ和β值
- **learning_rate_parameters**: 学习率策略、分布统计、c值、各节点学习率
- **training_parameters**: 数据集、批次大小、训练轮数等
- **experiment_parameters**: 重复次数、备注、设备等
- **generated_files**: 生成的文件列表

## 学习率策略说明

1. **uniform（均匀）**: 所有节点使用相同的学习率
   - `lr_i = lr_basic`

2. **pi_a_inverse（π_A倒数）**: 学习率与A矩阵的左Perron向量成反比
   - `lr_i = lr_basic * (1/π_A[i]) * (n/sum(1/π_A))`

3. **pi_b_inverse（π_B倒数）**: 学习率与B矩阵的右Perron向量成反比
   - `lr_i = lr_basic * (1/π_B[i]) * (n/sum(1/π_B))`

4. **random（随机）**: 随机分配学习率（保持总和不变）
   - 需要提供 `--random_seed` 参数

## 注意事项

1. **网格拓扑要求**: 使用grid拓扑时，节点数n必须是完全平方数（如4, 9, 16, 25等）

2. **GPU内存管理**: 程序会在每次重复之间自动清理GPU内存

3. **公平比较**: 所有策略都保持总学习率和 `lr_basic * n` 相等，确保公平比较

4. **随机策略**: 使用random策略时必须提供 `--random_seed` 参数

5. **异质性参数**: `alpha` 越大，数据分布越均匀；越小，异质性越强

## 常见用例

### 1. 比较不同拓扑的性能
```bash
for topo in ring grid neighbor random; do
    python run_experiment.py \
        --topology $topo \
        --n 16 \
        --matrix_seed 42 \
        --lr_basic 0.007 \
        --strategy uniform \
        --dataset_name MNIST \
        --batch_size 128 \
        --num_epochs 100 \
        --alpha 1000 \
        --use_hetero \
        --device cuda:0 \
        --output_dir "./topology_comparison"
done
```

### 2. 测试不同的异质性水平
```bash
for alpha in 100 1000 10000; do
    python run_experiment.py \
        --topology neighbor \
        --n 16 \
        --matrix_seed 42 \
        --lr_basic 0.007 \
        --strategy uniform \
        --dataset_name MNIST \
        --batch_size 128 \
        --num_epochs 50 \
        --alpha $alpha \
        --use_hetero \
        --remark "alpha_${alpha}" \
        --device cuda:0 \
        --output_dir "./heterogeneity_test"
done
```

### 3. 大规模参数扫描
```bash
#!/bin/bash
# 参数扫描实验

for n in 8 16 32; do
    for lr in 0.001 0.005 0.01; do
        for strategy in uniform pi_a_inverse pi_b_inverse; do
            python run_experiment.py \
                --topology neighbor \
                --n $n \
                --matrix_seed 42 \
                --lr_basic $lr \
                --strategy $strategy \
                --dataset_name MNIST \
                --batch_size 128 \
                --num_epochs 50 \
                --alpha 1000 \
                --use_hetero \
                --remark "n${n}_lr${lr}_${strategy}" \
                --device cuda:0 \
                --output_dir "./parameter_sweep"
        done
    done
done
```

## 故障排除

1. **CUDA内存不足**: 减小batch_size或节点数n
2. **找不到模块**: 确保在正确的目录下运行，或检查PYTHONPATH
3. **权限错误**: 使用 `chmod +x` 给Shell脚本添加执行权限
4. **拓扑生成失败**: 检查参数是否合理（如grid需要完全平方数）