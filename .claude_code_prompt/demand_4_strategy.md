我在思考一种构造li_list的逻辑
1. 就是这样的:
   1. 通过某种构造方式，对于确定的pi_a和pi_b(固定了A和B之后当然是确定的了)
   2. 使得c从小到大的变化
2. 不过这个和/Users/luogan/Code/new_push_pull/scripts_pushpull_differ_lr/run_experiment.py的逻辑不太match
3. 因为它是在确定参数下重复
4. 因此，你可以给strategy加一个custom的属性
5. 这样可以传入一个list(不是lr_list, 而是对角矩阵D的对角线)
6. 然后在untils里面写一个新的函数，接收的参数是使得c从小到大的变化，这样的c有的个数
   1. 问题是，A和B的选择已经在run_experiment.py里面了
   2. 所以在外部好像不太好操作
7. 希望这个方法是独立于scripts_pushpull_differ_lr/run_experiment.py，也就是说不要影响现在
8. 其他都保留相同
9. 你可以仔细阅读`scripts_pushpull_differ_lr/说明文档_zh.md`