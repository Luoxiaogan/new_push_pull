import numpy as np
import pandas as pd
# from useful_functions_with_batch import *
import copy

def stable_log_exp(x):
    """
    Stable computation of log(1 + exp(x)) to avoid overflow.
    
    Parameters:
        x (np.ndarray): Input array.
    
    Returns:
        np.ndarray: log(1 + exp(x)) computed in a numerically stable way.
    """
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def loss_with_batch(x, y, h, rho=0.001, batch_size=None):
    """
    计算损失函数，支持批次采样
    """
    n, L, d = h.shape
    if batch_size is None or batch_size >= L:
        batch_size = L
        h_batch = h
        y_batch = y
    else:
        # 每个节点独立采样批次
        batch_indices = np.random.choice(L, batch_size, replace=False)
        h_batch = h[:, batch_indices, :]
        y_batch = y[:, batch_indices]
    
    # x should be shape (n, d), compute h_dot_x for each node
    h_dot_x = np.einsum('ijk,ik->ij', h_batch, x, optimize=True)

    # 计算 logistic loss: log(1 + exp(-y * h_dot_x))
    exp_val = np.exp(-y_batch * h_dot_x)
    np.clip(exp_val, None, 1e300, out=exp_val)
    
    # Calculate loss per sample, then average over batch dimension
    loss_per_sample = np.log1p(exp_val)  # log(1 + exp(-y * h_dot_x))
    loss_per_node = np.mean(loss_per_sample, axis=1)  # Average over batch samples, shape (n,)
    
    # L2 regularization term
    x_squared = x**2
    reg_term = rho * np.log(1 + x_squared)  # Shape (n, d)
    reg_per_node = np.mean(reg_term, axis=1)  # Average over features, shape (n,)
    
    # Total loss: logistic loss + regularization
    total_loss_per_node = loss_per_node + reg_per_node  # Shape (n,)
    
    # Average loss across all nodes
    total_loss = np.mean(total_loss_per_node)
    
    return total_loss

def grad_with_batch(x, y, h, rho=0.001, batch_size=None):
    """
    计算每个节点的局部梯度，支持批次采样
    """
    n, L, d = h.shape
    if batch_size is None or batch_size >= L:
        batch_size = L
        h_batch = h
        y_batch = y
    else:
        # 每个节点独立采样批次
        batch_indices = np.random.choice(L, batch_size, replace=False)
        h_batch = h[:, batch_indices, :]
        y_batch = y[:, batch_indices]
    
    # 计算梯度
    h_dot_x = np.einsum('ijk,ik->ij', h_batch, x)
    exp_val = np.exp(y_batch * h_dot_x)
    np.clip(exp_val, None, 1e300, out=exp_val)
    
    g1 = -np.einsum('ijk,ij->ik', h_batch, y_batch / (1 + exp_val)) / batch_size
    x_squared = x**2
    g2 = 2 * x / (1 + x_squared)**2
    
    return (g1 + rho * g2).reshape(n, d)


def PushPull_with_batch_different_lr(
    A,
    B,
    init_x,
    h_data,
    y_data,
    lr_list,
    rho=0.1,
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

    grad_func = grad_with_batch  # 使用定义的梯度函数
    loss_func = loss_with_batch  # 使用定义的损失函数

    # 初始化梯度（使用批次）
    g = grad_func(x, y_data, h_data, rho=rho, batch_size=batch_size) + sigma_n * np.random.normal(
        size=(n, d)
    )
    y = copy.deepcopy(g) # gradient tracking

    # 记录训练过程
    gradient_history_onfull = []
    loss_history_onfull = []

    for _ in range(max_it):
        # print("A:", A.shape, "B:", B.shape, "x:", x.shape, "y:", y.shape)
        x = A @ x - np.array(lr_list).reshape(-1, 1) * y
        g_new = grad_func(
            x, y_data, h_data, rho=rho, batch_size=batch_size
        ) + sigma_n * np.random.normal(size=(n, d))
        y = B @ y + g_new - g
        g = g_new

        # 记录平均梯度范数和参数范数
        x_mean = np.mean(x, axis=0, keepdims=True)
        x_mean_expand = np.broadcast_to(x_mean, (n, d))
        _grad = grad_func(x_mean_expand, y_data, h_data, rho=rho, batch_size=batch_size).reshape(
            x.shape
        )
        _loss = loss_func(x_mean_expand, y_data, h_data, rho=rho, batch_size=batch_size)
        loss_history_onfull.append(_loss)
        # 在整个训练数据集上计算梯度
        mean_grad = np.mean(_grad, axis=0, keepdims=True)
        _grad_norm = np.linalg.norm(mean_grad)
        gradient_history_onfull.append(_grad_norm)
        print("Iteration: ", _, "loss: ", _loss, " gradient norm: ", _grad_norm)

    return pd.DataFrame(
        {
            "gradient_norm_on_full_trainset": gradient_history_onfull,
            "loss_on_full_trainset": loss_history_onfull,
        }
    )


def PushPull_with_batch(
    A,
    B,
    init_x,
    h_data,
    y_data,
    grad_func,
    lr_list,
    rho=0.1,
    sigma_n=0.1,
    max_it=200,
    batch_size=None,
):
    """
    Push-Pull算法
    允许不同节点使用不同的学习率
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
        # .reshape(-1, 1) 将形状从 [n] 变为 [n, 1]
        # 这样就可以与形状为 [n, d] 的 y 进行广播乘法，每一行使用对应的学习率
        x = A @ x - np.array(lr_list).reshape(-1, 1) * y
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