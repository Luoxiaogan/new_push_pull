focus on “不同节点使用不同学习率”
+  `training/training_track_grad_norm_different_learning_rate.py`已经实现了这一点
   +  传入一个lr_list
+  接下来我是希望能够系统性地测试.
   +  `scripts_pushpull_differ_lr/Nearest_neighbor_MNIST.py`显得太过于复杂了，可能还需要包装一下
+  你首先阅读`scripts_pushpull_differ_lr/Nearest_neighbor_MNIST.py`
   +  `lr_basic`表示的是, 在不同的lr_list对比的时候, 要保持总和`lr_basic*n`是一样的, 这样对比更公平
```python
n=16
A, B = generate_nearest_neighbor_matrices(n = n, k=3, seed=seed)

pi_a = get_left_perron(A)
pi_b = get_right_perron(B)
```
这部分我需要自己调, 因为seed的调整本质上是为了调kappa, beta. see `utils/algebra_utils.py`.
+ 这部分你不需要操心，你需要做的是，把这个写成一个我更容易用的接口.

首先
+ `scripts_pushpull_differ_lr/network_utils.py`里面定义了我要用的拓扑选择:
  + get_matrixs_from_exp_graph, generate_grid_matrices, generate_ring_matrices, generate_random_graph_matrices, generate_stochastic_geometric_matrices, generate_nearest_neighbor_matrices
  + 总的接口提供6个拓扑选项: exp, grid, ring, random, geometric, neighbor
+ `n`和`matrix_seed`也作为输入参数. 
+ `lr_basic`也是传入的参数

对于lr切分的方法。这里的讨论是:
用一个对角矩阵D来代表不同节点学习率的倍数。identity, 代表都是相同的学习率。为了公平，D的对角线元素之和最好保持为n(这样和我们之前`lr_basic*n`的总和不变一个意思)
加入D之后，我们c的定义应该从 n\pi_A^\top \pi_B变成   n\pi_A^\top  D \pi_B。因此下列D的取值比较值得研究：
【1】identity
【2】diag(pi_A)^{-1}  （还需要标准化）
【3】diag(pi_B)^{-1}
【4】random,多采样几组

因此, 还有一个选项参数
+ strategy: identity, pia-1, pib-1, random
+ 可能我给的参数不太好看, 你可以给个好看的参数选项
+ random使用的seed, 和`matrix_seed`不同， 是新的参数`random_seed`. 只有 strategy = "random" 的时候采用

然后跑实验的时候, 实际上主要的部分train_track_grad_norm_with_hetero_different_learning_rate没有动.
+ `use_hetero`和`alpha`, `bs`, `num_epochs`这些参数当然有

+ 以及有一个repitition的选项, 本质上是为了兼容多次跑来取平均(但是这里没想到一个好的方法, 因为train_track_grad_norm_with_hetero_different_learning_rate的seed需要每次都不一样)


最后主要是文件保存地址和文件名称这些
+ train_track_grad_norm_with_hetero_different_learning_rate程序里面其实root的设置, 主要是保存的的基础位置
+ 但是多次求平均之后的保存(只对于grad_norm这个csv, train_track_grad_norm_with_hetero_different_learning_rate本身会产出两个csv), scripts_pushpull_differ_lr/Nearest_neighbor_MNIST.py里面定义了
df_output.to_csv(f"/home/lg/ICML2025_project/NEW_PROJECT_20250717/init_test_mnist_0717/neighbor_repeate/n={n}_lr_total={lr_total}_seed={seed}.csv")

以及最后的，我希望计算c.
\pi_A,  \pi_B 你可以看utils/algebra_utils.py