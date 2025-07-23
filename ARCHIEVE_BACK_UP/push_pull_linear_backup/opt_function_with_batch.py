import numpy as np
import pandas as pd
from useful_functions_with_batch import *
import copy

def PushPull_with_batch(
    A,
    B,
    init_x,
    h_data,
    y_data,
    grad_func,
    rho=0.1,
    lr=0.1,
    sigma_n=0.1,
    max_it=200,
    batch_size=None,
):
    """
    Push-Pull算法
    支持批次训练的分布式梯度下降
    """
    h_data, y_data = copy.deepcopy(h_data), copy.deepcopy(y_data)
    x = copy.deepcopy(init_x)
    n, d = x.shape

    # 初始化梯度（使用批次）
    g = grad_func(x, y_data, h_data, rho=rho, batch_size=batch_size) + sigma_n * np.random.normal(
        size=(n, d)
    )
    y = copy.deepcopy(g) # gradient tracking

    # 记录训练过程
    gradient_history_onfull = []

    for _ in range(max_it):
        x = A @ x - lr * y
        g_new = grad_func(
            x, y_data, h_data, rho=rho, batch_size=batch_size
        ) + sigma_n * np.random.normal(size=(n, d))
        y = B @ y + g_new - g
        g = g_new

        # 记录平均梯度范数和参数范数
        x_mean = np.mean(x, axis=0, keepdims=True)
        x_mean_expand = np.broadcast_to(x_mean, (n, d))
        _grad = grad_func(x_mean_expand, y_data, h_data, rho=rho, batch_size=None).reshape(
            x.shape
        )
        # 在整个训练数据集上计算梯度
        mean_grad = np.mean(_grad, axis=0, keepdims=True)
        gradient_history_onfull.append(np.linalg.norm(mean_grad))

    return pd.DataFrame(
        {
            "gradient_norm_on_full_trainset": gradient_history_onfull,
        }
    )