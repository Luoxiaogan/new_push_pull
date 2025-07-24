1. 着眼于`合成数据`
2. 不要破坏目前的逻辑.
3. 对于生成的数据，目前是uniform的.
   1. see `init_global_data` and `distribute_data` in `合成数据/useful_functions_with_batch.py`
4. 现在我想要一个异质性数据分布的版本.
5. 你可以考虑用dirichlet分布
6. 异质性有2点
   1. 一点是，distribution上的区别
   2. 第二点是，每个节点的sample的数目的区别
7. 考虑适配`PushPull_with_batch_batched_gpu_differ_lr`, `loss_with_batch_batched_gpu`, `grad_with_batch_batched_gpu`
   1. 看起来需要每个节点的sample数目一样
   2. 所以首先着眼于distribution上的区别
8. 得到的新函数, 可以平行替代`合成数据/MAIN_TEST_GPU.py`里面的
```python 
h_global_cpu, y_global_cpu, x_opt_cpu = init_global_data(d=d, L_total=L_total, seed=42)
print("h:",h_global_cpu.shape)
print("y:",y_global_cpu.shape)
h_tilde_cpu, y_tilde_cpu = distribute_data(h=h_global_cpu, y=y_global_cpu, n=n)
print("h_tilde:",h_tilde_cpu.shape)
print("y_tilde:",y_tilde_cpu.shape)
```
这部分. alpha是异质性参数