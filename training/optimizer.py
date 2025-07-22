# training/optimizer.py

import torch
from torch.optim import Optimizer

class PullDiag_GT(Optimizer):
    def __init__(self, model_list, lr=1e-2, A=None, closure=None):
        self.model_list = model_list
        self.lr = lr
        self.A = A.to(next(model_list[0].parameters()).device)

        # Compute initial gradients
        closure()

        # Store previous parameters and gradients as vectors
        self.prev_params = [
            torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
            for model in self.model_list
        ]
        self.prev_grads = [
            torch.nn.utils.parameters_to_vector([p.grad for p in model.parameters()])
            .detach()
            .clone()
            for model in self.model_list
        ]

        # Initialize v_list
        self.v_list = [prev_grad.clone() for prev_grad in self.prev_grads]

        # Initialize w_vector and prev_w_vector
        self.w_vector = torch.ones(
            self.A.shape[0], device=next(model_list[0].parameters()).device
        )
        self.prev_w_vector = self.w_vector.clone()

        defaults = dict(lr=lr)
        super(PullDiag_GT, self).__init__(model_list[0].parameters(), defaults)

    def step(self, closure, lr):
        self.lr = lr  # Update lr if provided

        # -------------------------- #
        #     Step 2 + Step 1       #
        #     x = A(x - lr * v)     #
        # -------------------------- #
        with torch.no_grad():
            # 先更新 w_vector
            self.w_vector = torch.matmul(self.A, self.w_vector)

            # 把所有模型的参数堆叠起来
            prev_params_tensor = torch.stack(self.prev_params)    # shape: (n_models, param_size)
            v_tensor = torch.stack(self.v_list)                   # shape: (n_models, param_size)

            # 先 (x - lr*v)，再乘 A，一步完成
            new_params_tensor = torch.matmul(
                self.A, 
                prev_params_tensor - self.lr * v_tensor
            )

            # 写回到每个 model
            for i, model in enumerate(self.model_list):
                torch.nn.utils.vector_to_parameters(
                    new_params_tensor[i], model.parameters()
                )

        # -------------------------- #
        #        Step 3: 计算梯度   #
        # -------------------------- #
        for model in self.model_list:
            model.zero_grad()  # 或者 for p in model.parameters(): p.grad = None
        loss = closure()       # 前向 + 反向，计算新的梯度

        # -------------------------- #
        #           Step 4           #
        #   v = A[v + 1/w*g - 1/w'*g']
        # -------------------------- #
        with torch.no_grad():
            new_grads = [
                torch.nn.utils.parameters_to_vector(
                    [p.grad for p in model.parameters()]
                ).detach().clone()
                for model in self.model_list
            ]
            new_grads_tensor = torch.stack(new_grads)    # shape: (n_models, param_size)

            # 计算需要的中间量
            v_tensor = torch.stack(self.v_list)
            prev_grads_tensor = torch.stack(self.prev_grads)

            w_vector_inv = 1.0 / self.w_vector.unsqueeze(1)            # shape: (n_models, 1)
            prev_w_vector_inv = 1.0 / self.prev_w_vector.unsqueeze(1)  # shape: (n_models, 1)

            # (1/w)*g
            W_g = w_vector_inv * new_grads_tensor
            # (1/w_prev)*prev_g
            prev_W_prev_g = prev_w_vector_inv * prev_grads_tensor

            # v + (1/w)*g - (1/w_prev)*prev_g
            tmp = v_tensor + W_g - prev_W_prev_g
            # 再乘 A，一步完成
            new_v_tensor = torch.matmul(self.A, tmp)

            # 写回 v_list
            self.v_list = [new_v_tensor[i].clone() for i in range(len(self.model_list))]

            # -------------------------- #
            #         Step 5 更新缓存    #
            # -------------------------- #
            self.prev_params = [
                new_params_tensor[i].clone() for i in range(len(self.model_list))
            ]
            self.prev_grads = [
                new_grads_tensor[i].clone() for i in range(len(self.model_list))
            ]
            self.prev_w_vector = self.w_vector.clone()

        return loss

class PullDiag_GD(Optimizer):
    """
    PullDiag_GD 优化器:
    1. 不需要 gradient tracking 向量 v
    2. 直接使用每次前向+反向计算得到的梯度 g
    3. 更新公式: x = A [ x - lr * (1 / w) * g ], 同时 w = A w。
    """
    def __init__(self, model_list, lr=1e-2, A=None, closure=None):
        """
        参数:
            model_list: 由 n 个模型组成的列表, 每个模型对应一个节点
            lr: 学习率
            A: 拉普拉斯混合矩阵或邻接矩阵, 用于聚合
            closure: 用于在初始化时计算一次梯度（可选）
        """
        if A is None:
            raise ValueError("Matrix A must be provided for PullDiag_GD.")
        self.model_list = model_list
        self.lr = lr
        self.A = A.to(next(model_list[0].parameters()).device)

        # 如果需要，可以先做一次 forward + backward，用于初始化(可选)
        # 若不需要，可以删掉下面这几行
        if closure is not None:
            closure()

        # 记录当前模型的参数 (上一轮的参数)
        # 注: 我们这里还是存一下，以便在每次 step() 时做矩阵运算
        self.prev_params = [
            torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
            for model in self.model_list
        ]

        # 初始化 w 向量, 形状与 A 的行数相匹配 (n_nodes)
        self.w_vector = torch.ones(
            self.A.shape[0], device=next(model_list[0].parameters()).device
        )

        defaults = dict(lr=lr)
        super(PullDiag_GD, self).__init__(model_list[0].parameters(), defaults)

    def step(self, closure=None, lr=None):
        """
        执行一次 GD 更新:
        1) 先对各模型 zero_grad() 并调用 closure 计算新的梯度
        2) 更新 w 向量: w = A w
        3) 按照公式 x = A [ x - lr * (1 / w) * g ] 更新各模型参数
        """
        if lr is not None:
            self.lr = lr  # 动态更新学习率

        # ---------- 1) 计算新的梯度 ---------- #
        if closure is None:
            raise RuntimeError("closure must be provided to compute new gradients.")

        # 清空梯度
        for model in self.model_list:
            # 更推荐: for p in model.parameters(): p.grad = None
            model.zero_grad()

        # 前向 + 反向传播，得到新的梯度 g
        loss = closure()

        # 将梯度打包到一个张量中
        new_grads = [
            torch.nn.utils.parameters_to_vector(
                [p.grad for p in model.parameters()]
            ).detach().clone()
            for model in self.model_list
        ]
        new_grads_tensor = torch.stack(new_grads)  # 形状: (n_models, param_size)

        with torch.no_grad():
            # ---------- 2) 更新 w 向量 ---------- #
            # w = A w
            self.w_vector = torch.matmul(self.A, self.w_vector)

            # ---------- 3) 更新模型参数 ---------- #
            # x = A [ x - lr * (1 / w) * g ]
            prev_params_tensor = torch.stack(self.prev_params)  # (n_models, param_size)

            # 1 / w, shape: (n_models, 1)
            w_vector_inv = 1.0 / self.w_vector.unsqueeze(1)

            # x - lr*(1/w)*g
            temp_params_tensor = prev_params_tensor - self.lr * (w_vector_inv * new_grads_tensor)

            # 再做 Ax
            new_params_tensor = torch.matmul(self.A, temp_params_tensor)

            # 将 new_params_tensor 写回各个模型
            for i, model in enumerate(self.model_list):
                torch.nn.utils.vector_to_parameters(
                    new_params_tensor[i], model.parameters()
                )

            # 记录新的参数到 prev_params，以便下次更新使用
            self.prev_params = [
                new_params_tensor[i].clone() for i in range(len(self.model_list))
            ]

        return loss

class PushPull(Optimizer):
    def __init__(
            self, 
            model_list, 
            lr=1e-2, 
            A=None, 
            B=None, 
            closure=None):
        self.model_list = model_list
        self.lr = lr
        self.A = A.to(next(model_list[0].parameters()).device)
        self.B = B.to(next(model_list[0].parameters()).device)

        # Compute initial gradients
        closure()

        # Store previous parameters and gradients as vectors
        self.prev_params = [
            torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
            for model in self.model_list
        ]
        self.prev_grads = [
            torch.nn.utils.parameters_to_vector([p.grad for p in model.parameters()])
            .detach()
            .clone()
            for model in self.model_list
        ]

        # Initialize v_list
        self.v_list = [prev_grad.clone() for prev_grad in self.prev_grads]
        # 梯度跟踪向量 v

        defaults = dict(lr=lr)
        super(PushPull, self).__init__(model_list[0].parameters(), defaults)

    def step(self, closure, lr):
        self.lr = lr  # Update lr if provided

        # -------------------------- #
        #     Step 2 + Step 1       #
        #     x = Ax - lr * v     #
        # -------------------------- #
        with torch.no_grad():

            # 把所有模型的参数堆叠起来
            prev_params_tensor = torch.stack(self.prev_params)    # shape: (n_models, param_size)
            v_tensor = torch.stack(self.v_list)                   # shape: (n_models, param_size)

            # 先进行聚合 Ax，再减去 lr*v
            new_params_tensor = torch.matmul(self.A, prev_params_tensor) - self.lr * v_tensor

            # 写回到每个 model
            for i, model in enumerate(self.model_list):
                torch.nn.utils.vector_to_parameters(
                    new_params_tensor[i], model.parameters()
                )

        # -------------------------- #
        #        Step 3: 计算梯度   #
        # -------------------------- #
        for model in self.model_list:
            model.zero_grad()  # 或者 for p in model.parameters(): p.grad = None
        loss = closure()       # 前向 + 反向，计算新的梯度

        # -------------------------- #
        #           Step 4           #
        #   v = Bv + g - g_prev
        # -------------------------- #
        with torch.no_grad():
            new_grads = [
                torch.nn.utils.parameters_to_vector(
                    [p.grad for p in model.parameters()]
                ).detach().clone()
                for model in self.model_list
            ]
            new_grads_tensor = torch.stack(new_grads)    # shape: (n_models, param_size)

            # 计算需要的中间量
            v_tensor = torch.stack(self.v_list)
            prev_grads_tensor = torch.stack(self.prev_grads)

            # 使用更新公式：v = Bv + g - g_prev
            # 先计算 Bv
            Bv = torch.matmul(self.B, v_tensor)

            # 然后计算 Bv + g - g_prev
            new_v_tensor = Bv + new_grads_tensor - prev_grads_tensor

            # 写回 v_list
            self.v_list = [new_v_tensor[i].clone() for i in range(len(self.model_list))]

            # -------------------------- #
            #         Step 5 更新缓存    #
            # -------------------------- #
            self.prev_params = [
                new_params_tensor[i].clone() for i in range(len(self.model_list))
            ]
            self.prev_grads = [
                new_grads_tensor[i].clone() for i in range(len(self.model_list))
            ]

        return loss