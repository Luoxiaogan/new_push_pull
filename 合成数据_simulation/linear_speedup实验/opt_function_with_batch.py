import numpy as np
import pandas as pd
from useful_functions_with_batch import *
import copy


def PullDiag_GD(
    A,
    init_x,
    h_data,
    y_data,
    grad_func,
    rho=0.1,
    lr=0.1,
    sigma_n=0.1,
    max_it=200,
):
    """PullDiag_GD"""
    h, y = copy.deepcopy(h_data), copy.deepcopy(y_data)
    x = copy.deepcopy(init_x)
    n, d = x.shape
    W = np.eye(n)
    g = grad_func(x, y, h, rho=rho).reshape(x.shape) + sigma_n * np.random.normal(
        size=(n, d)
    )
    x_mean = np.mean(x, axis=0, keepdims=True)  # 1*d
    x_mean_expand = np.broadcast_to(x_mean, (n, d))  # n*d
    gradient_x_mean = grad_func(x_mean_expand, y, h, rho=rho).reshape(x.shape)
    mean_gradient_x_mean = np.mean(gradient_x_mean, axis=0, keepdims=True)  # 1*d
    gradient_history = [np.linalg.norm(mean_gradient_x_mean)]
    x_history = [np.linalg.norm(x_mean)]

    for _ in range(max_it):
        W = A @ W
        Diag_W_inv = np.diag(1 / np.diag(W))
        x_new = x - lr * Diag_W_inv @ g
        x = A @ x_new
        g = grad_func(x, y, h, rho=rho).reshape(x.shape) + sigma_n * np.random.normal(
            size=(n, d)
        )
        x_mean = np.mean(x, axis=0, keepdims=True)  # 1*d
        x_mean_expand = np.broadcast_to(x_mean, (n, d))  # n*d
        gradient_x_mean = grad_func(x_mean_expand, y, h, rho=rho).reshape(x.shape)
        mean_gradient_x_mean = np.mean(gradient_x_mean, axis=0, keepdims=True)  # 1*d
        gradient_history.append(np.linalg.norm(mean_gradient_x_mean))
        x_history.append(np.linalg.norm(x_mean))

    result_df = pd.DataFrame(
        {"gradient_norm": gradient_history, "x_mean_norm": x_history}
    )
    return result_df


def PullDiag_GT(
    A,
    init_x,
    h_data,
    y_data,
    grad_func,
    rho=0.1,
    lr=0.1,
    sigma_n=0.1,
    max_it=200,
):
    """PullDiag_GT"""
    h, y = copy.deepcopy(h_data), copy.deepcopy(y_data)
    x = copy.deepcopy(init_x)
    n, d = x.shape
    W = np.eye(n)
    g = grad_func(x, y, h, rho=rho).reshape(x.shape) + sigma_n * np.random.normal(
        size=(n, d)
    )
    w = np.diag(1 / np.diag(W)) @ g
    v = copy.deepcopy(g)

    v_history = [np.linalg.norm(v)]
    x_mean = np.mean(x, axis=0, keepdims=True)  # 1*d
    x_mean_expand = np.broadcast_to(x_mean, (n, d))  # n*d
    gradient_x_mean = grad_func(x_mean_expand, y, h, rho=rho).reshape(x.shape)
    mean_gradient_x_mean = np.mean(gradient_x_mean, axis=0, keepdims=True)  # 1*d
    gradient_history = [np.linalg.norm(mean_gradient_x_mean)]
    x_history = [np.linalg.norm(x_mean)]

    for _ in range(max_it):
        W = A @ W
        x = A @ x - lr * v
        g = grad_func(x, y, h, rho=rho).reshape(x.shape) + sigma_n * np.random.normal(
            size=(n, d)
        )
        v = A @ v + np.linalg.inv(np.diag(np.diag(W))) @ g - w
        w = (
            np.linalg.inv(np.diag(np.diag(W))) @ g
        )  # 这一步计算的w是下一步用到的w，因此程序没有问题
        v_history.append(np.linalg.norm(v))

        x_mean = np.mean(x, axis=0, keepdims=True)  # 1*d
        x_mean_expand = np.broadcast_to(x_mean, (n, d))
        gradient_x_mean = grad_func(x_mean_expand, y, h, rho=rho).reshape(x.shape)
        mean_gradient_x_mean = np.mean(gradient_x_mean, axis=0, keepdims=True)
        gradient_history.append(np.linalg.norm(mean_gradient_x_mean))
        x_history.append(np.linalg.norm(x_mean))

    result_df = pd.DataFrame(
        {
            "gradient_norm": gradient_history,
            "gradient_tracking_norm": v_history,
            "x_mean": x_history,
        }
    )
    return result_df


def PullDiag_GD_with_batch(
    A,
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
    """支持批次训练的分布式梯度下降"""
    h, y = copy.deepcopy(h_data), copy.deepcopy(y_data)
    x = copy.deepcopy(init_x)
    n, d = x.shape
    W = np.eye(n)

    # 初始化梯度（使用批次）
    g = grad_func(x, y, h, rho=rho, batch_size=batch_size) + sigma_n * np.random.normal(
        size=(n, d)
    )

    # 记录训练过程
    gradient_history_onfull = []

    for _ in range(max_it):
        W = A @ W
        Diag_W_inv = np.diag(1 / np.diag(W))
        x_new = x - lr * Diag_W_inv @ g
        x = A @ x_new  # 注意A在外面

        # 计算梯度（使用批次）
        g = grad_func(
            x, y, h, rho=rho, batch_size=batch_size
        ) + sigma_n * np.random.normal(size=(n, d))

        # 记录平均梯度范数和参数范数
        x_mean = np.mean(x, axis=0, keepdims=True)
        x_mean_expand = np.broadcast_to(x_mean, (n, d))
        _grad = grad_func(x_mean_expand, y, h, rho=rho, batch_size=None).reshape(
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


def PullDiag_GT_with_batch(
    A,
    init_x,
    h_data,
    y_data,
    grad_func,
    rho=0.1,
    lr=0.1,
    sigma_n=0.1,
    max_it=200,
    batch_size=None,  # 新增批次参数
):
    """支持批次训练的梯度跟踪算法 (Gradient Tracking with Batch)"""
    h, y = copy.deepcopy(h_data), copy.deepcopy(y_data)
    x = copy.deepcopy(init_x)
    n, d = x.shape
    W = np.eye(n)

    # 初始化梯度（使用批次）
    g = grad_func(x, y, h, rho=rho, batch_size=batch_size).reshape(
        x.shape
    ) + sigma_n * np.random.normal(size=(n, d))
    w = np.diag(1 / np.diag(W)) @ g
    v = copy.deepcopy(g)

    # 记录历史数据
    # v_history = [np.linalg.norm(v)]
    x_mean = np.mean(x, axis=0, keepdims=True)  # 1*d
    x_mean_expand = np.broadcast_to(x_mean, (n, d))  # n*d
    gradient_x_mean = grad_func(x_mean_expand, y, h, rho=rho, batch_size=None).reshape(
        x.shape
    )
    mean_gradient_x_mean = np.mean(gradient_x_mean, axis=0, keepdims=True)  # 1*d
    gradient_history_onfull = [np.linalg.norm(mean_gradient_x_mean)]
    # x_history = [np.linalg.norm(x_mean)]
    # gradient_history = [np.linalg.norm(np.mean(g, axis=0))]

    for _ in range(max_it):
        # 更新权重矩阵和参数
        W = A @ W
        x_tmp = x - lr * v
        x = A @ x_tmp  # 注意A在外面
        #x = A @ x - lr * v

        # 计算梯度（使用批次）
        g = grad_func(x, y, h, rho=rho, batch_size=batch_size).reshape(
            x.shape
        ) + sigma_n * np.random.normal(size=(n, d))

        # 更新梯度跟踪变量
        Diag_W_inv = np.diag(1 / np.diag(W))
        tmp = v + Diag_W_inv @ g - w
        v = A @ tmp  # 注意A在外面
        #v = A @ v + Diag_W_inv @ g - w
        w = Diag_W_inv @ g  # 下一步的 w

        # 记录状态
        # v_history.append(np.linalg.norm(v))

        # 计算平均参数和梯度
        x_mean = np.mean(x, axis=0, keepdims=True)
        x_mean_expand = np.broadcast_to(x_mean, (n, d))
        _grad = grad_func(x_mean_expand, y, h, rho=rho, batch_size=None).reshape(
            x.shape
        )
        # 在整个训练数据集上计算梯度
        mean_grad = np.mean(_grad, axis=0, keepdims=True)
        gradient_history_onfull.append(np.linalg.norm(mean_grad))
        # x_history.append(np.linalg.norm(x_mean))
        # gradient_history.append(np.linalg.norm(np.mean(g, axis=0)))

    # 返回结果
    result_df = pd.DataFrame(
        {
            "gradient_norm_on_full_trainset": gradient_history_onfull,
            # "gradient_tracking_norm": v_history,
            # "gradient_norm_on_batch": gradient_history,
            # "x_mean": x_history,
        }
    )
    return result_df
