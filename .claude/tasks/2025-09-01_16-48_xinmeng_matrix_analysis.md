# 研究Xinmeng矩阵的谱性质随节点数变化趋势

## 任务信息
- **创建时间**: 2025-09-01 16:48
- **任务类型**: 数值实验与分析
- **涉及文件**: 合成数据_不使用cupy_最简单的版本/new_test/

## 任务目标
研究 `get_xinmeng_matrix` 和 `get_xinmeng_like_matrix` 两种特殊**列随机矩阵**的谱性质（kappa值和beta值）随节点数n变化的趋势。

**重要说明**: 这两个函数生成的都是列随机矩阵（column-stochastic matrices），因为归一化是按列进行的（`M /= np.sum(M,axis=0)`）。

## 实施步骤

### 1. 创建测试目录结构
- [x] 在 `/Users/luogan/Code/new_push_pull/合成数据_不使用cupy_最简单的版本/` 下创建 `new_test/` 目录
- [x] 保存计划文档到 `.claude/tasks/` 目录

### 2. 编写测试脚本
创建文件：`合成数据_不使用cupy_最简单的版本/new_test/test_xinmeng_matrix_properties.py`

脚本功能：
- 测试不同节点数 n = [4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64]
- 对每个n值：
  - 生成 `get_xinmeng_matrix(n)` （确定性矩阵）
  - 生成 `get_xinmeng_like_matrix(n, seed)` （随机矩阵，多个seed取平均）
  - 计算两种矩阵作为行随机和列随机时的性质
  
### 3. 计算和记录的指标
对每种列随机矩阵计算：
- kappa_col: 条件数  
- beta_col: 第二大特征值相关
- spectral_gap: 1 - beta_col
- S_B: 谱复杂度
- 右Perron向量的最大值和最小值

### 4. 数据可视化
创建4个子图展示：
1. kappa值随n的变化（行随机 vs 列随机）
2. beta值随n的变化（行随机 vs 列随机）
3. spectral gap随n的变化
4. S值（谱复杂度）随n的变化

对比固定矩阵和随机矩阵的差异。

### 5. 结果保存
- 将数值结果保存为CSV文件：`new_test/xinmeng_matrix_analysis_results.csv`
- 将图表保存为：`new_test/xinmeng_matrix_properties.png`
- 生成分析报告：`new_test/xinmeng_matrix_analysis_report.md`

## 预期输出
1. CSV数据文件包含所有计算结果
2. 可视化图表展示趋势
3. 分析报告总结关键发现：
   - 两种矩阵的扩展性特征
   - kappa和beta的增长趋势（线性/对数/指数）
   - 作为行随机vs列随机的性能差异
   - 随机化对谱性质的影响

## 技术要点
- 使用对数坐标展示可能的指数增长
- 对随机矩阵运行多个seed（如10个）取平均值和标准差
- 检查矩阵的强连通性
- 处理可能的数值稳定性问题

## 执行状态
- [x] 计划制定完成
- [x] 目录创建
- [x] 脚本编写
- [x] 实验运行
- [x] 结果分析
- [x] 报告生成

## 任务完成总结
- **完成时间**: 2025-09-01 16:49
- **生成文件**:
  - 测试脚本: `test_xinmeng_matrix_properties.py`
  - 数据结果: `xinmeng_matrix_analysis_results.csv`
  - 可视化图表: `xinmeng_matrix_properties.png`
  - 分析报告: `xinmeng_matrix_analysis_report.md`

## 关键发现
1. **行随机 vs 列随机**: 固定Xinmeng矩阵作为行随机时条件数恒为1（完美条件），作为列随机时呈指数级增长（O(n^13.12)）
2. **收敛性恶化**: 当n≥12时，行随机矩阵的beta值达到1，spectral gap变为0，导致算法无法收敛
3. **随机化影响**: 随机版本的矩阵在大规模时表现出极大的不稳定性，条件数方差极大
4. **实际应用建议**: 对于分布式优化，应避免使用此类矩阵结构作为行随机混合矩阵，特别是在节点数较多时