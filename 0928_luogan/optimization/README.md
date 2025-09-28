# 优化D矩阵的PushPull训练工具

## 📁 文件说明

### 核心脚本
- **`train_with_optimized_D.py`** - 使用优化的D矩阵进行PushPull训练
- **`show_matrix_properties.py`** - 展示矩阵性质和优化配置
- **`test_train.py`** - 测试版本（只运行100次迭代）

### 依赖文件
- **`useful_functions_with_batch.py`** - 工具函数库（从合成数据目录复制）
- **`opt_function_with_batch.py`** - PushPull优化算法（从合成数据目录复制）
- **`../profile/best_configs.npz`** - 预优化的最佳配置（n=4到n=29）

## 🚀 快速开始

### 1. 查看优化配置
```bash
# 交互式输入
python show_matrix_properties.py
# 输入: 4  (或任意4-29之间的n值)

# 命令行参数
python show_matrix_properties.py 16
```

输出内容：
- A矩阵（列随机）和B矩阵（行随机）的性质
- 优化的D向量分配策略
- Perron向量π_A的分布
- ||D*π_A||₂的值和效率

### 2. 运行训练
```bash
# 编辑文件中的n值（第14行）
vim train_with_optimized_D.py
# 修改: n = 4  # 可以改为4-29之间的任意值

# 运行训练
python train_with_optimized_D.py
```

输出内容：
- 数据分布情况
- 训练过程（每次迭代的损失和梯度范数）
- 最终结果保存为CSV文件

### 3. 快速测试
```bash
# 运行测试版本（只跑100次迭代）
python test_train.py
```

## ⚙️ 参数说明

### 硬编码参数（与basic_test.py保持一致）
```python
n = 4              # 节点数（可修改为4-29）
max_it = 9000      # 最大迭代次数
alpha = 0.1        # 异质性参数（0.1=高度异质）
d = 10             # 特征维度
L_total = 1440000  # 总样本数
lr_basic = 1e-1    # 基础学习率
```

### D矩阵优化策略
- 从`best_configs.npz`读取预优化的D向量
- 学习率设置：`lr_i = lr_basic * D_i`
- D向量满足：`sum(D) = n`，`κ(D) ≤ 200`

## 📊 实验对比

### 优化D vs 均匀D
- **均匀分配**：所有节点使用相同学习率
- **优化分配**：根据Perron向量优化学习率分配
- **性能提升**：通常可达3-4倍

### 示例结果（n=4）
```
优化配置：
- ||D*π_A||₂ = 3.9381
- 效率 = 98.52%
- 提升倍数 = 3.94x

训练收敛：
- 初始损失: 0.683
- 最终损失: 0.541 (100次迭代)
```

## 📝 注意事项

1. **n值范围**：必须在4-29之间（best_configs.npz中的可用范围）
2. **内存占用**：n越大，内存占用越高（建议n≤20）
3. **训练时间**：完整9000次迭代约需几分钟
4. **B矩阵生成**：简单使用`B = A.T`（A的转置）

## 🔗 相关文件

- 优化算法：`../profile/greedy_optimizer.py`
- 批量分析：`../profile/batch_analysis.py`
- 原始实验：`../../合成数据_不使用cupy_最简单的版本/basic_test.py`