# training/optimizer.py

import torch
from torch.optim import Optimizer

class PushPull_grad_norm_track(Optimizer):
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
        super(PushPull_grad_norm_track, self).__init__(model_list[0].parameters(), defaults)

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
        loss, grad_norm, avg_grad_norm = closure()       # 前向 + 反向，计算新的梯度

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

        return loss, grad_norm, avg_grad_norm