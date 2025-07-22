1. 这个项目的目的是在单张GPU上模拟分布式优化
2. 这个是本地项目，实际上训练的数据和路径都是按照服务器的，因此你可以忽略这方面的问题
   1. `data`存放训练数据, 本地是空文件
      1. MNIST and CIFAR10
   2. `datasets`存放处理数据的脚本
      1. 作用是为分布式优化做准备，分割数据
      2. 可以支持生成 hiterogeneous的 数据分布
   3. `models`保存模型结构代码
      1. fully connected
      2. cnn
   4. `training`
      1. 保存了训练的函数的代码
   5. `utils`
      1. 保存了各种可能用到的小函数
3. `NEW_PROJECT_20250717`一般用来存放training函数输出的csv文件, 以及我的画图分析.
   1. 你可以不管这个
4. `scripts_pushpull_differ_lr`一般是我调用函数来跑实验的地方
   1. `scripts_pushpull_differ_lr/network_utils.py`, 我在这里专用的, 你不用改动
   2. `scripts_pushpull_differ_lr/Nearest_neighbor_MNIST.py`你可以阅读这个, 首先熟悉基本的api方法.