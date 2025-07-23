1. read `测试_0723/test_1.py`, 需要写一个compute_possible_c的函数
2. 阅读'claude_code_prompt/demand_4_strategy.md' and 'claude_code_prompt/response_gemini.md', 你可以知道我实现了根据单纯形法得到c_max and c_min的方法
3. 但是 in practice, D 对角矩阵的对角元素, 对应了每个节点的学习率: lr_list[i]=lr_basic*diag(D)[i]
   1. 因此不能是0
4. compute_possible_c, 输入矩阵A,B,lr_basic,n
   1. 首先show_row(A), show_col(B), 然后展示 pi_A_hadamard_pi_B and hadamard 积 pi_A_hadamard_pi_B 的按照从大到小排序的索引
   2. 然后根据compute_learning_rates, 根据已经有的一些选项，uniform, pi_a_inverse, pi_b_inverse.
      1. 得到相应的c和D
      2. 保存tuple_list and print(策略和c)
   3. 用utils/d_matrix_utils.py里面的generate_d_matrices_theoretical, vertices选项, 得到单纯形算出来的c,D的tuple_list。
      1. 和前面的tuple_list合并到一起
      2. print(策略(都是vertices, 但是可以说是在pi_A_hadamard_pi_B 的按照从大到小排序的索引)和c)
   4. 最后返回最后的tuple_list
      1. 这个需要先整合一下，就是说tuple_list里面，对于D要进行分析，检查对角元素，检查是不是包含0，最后输出的是一个三元组list
         1. (c, d_list(也就是对角元素序列), remark也就是对角元素是不是包含0)