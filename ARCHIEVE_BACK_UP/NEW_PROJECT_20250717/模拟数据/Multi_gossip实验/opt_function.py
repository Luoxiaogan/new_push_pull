import numpy as np
import pandas as pd
from useful_functions import *
import copy


def PullDiag_GD(
    A,
    init_x,
    h_data,
    y_data,
    rho=0.1,
    lr=0.1,
    sigma_n=0.1,
    max_it=200,
):
    """PullDiag_GD"""
    def grad(x, y, h, rho=0.001):
        """
        计算每个节点的局部梯度 ∇f_i(x_i)，结果是一个 (n, d) 的矩阵，每一行是 ∇f_i(x_i)。
        
        Parameters:
            x (np.ndarray): 输入参数，形状为 (n, d)。
            y (np.ndarray): 标签数据，形状为 (n, L)。
            h (np.ndarray): 模型参数，形状为 (n, L, d)。
            rho (float): 正则化系数，默认为 0.001。
        
        Returns:
            np.ndarray: 梯度矩阵，形状为 (n, d)。
        """
        n, L, d = h.shape
        
        # 计算 h_dot_x: h 和 x 的点积，形状为 (n, L)
        h_dot_x = np.einsum('ijk,ik->ij', h, x, optimize=True)  # 使用 optimize=True 加速
        
        # 计算 exp_val: exp(y * h_dot_x)，形状为 (n, L)
        exp_val = np.exp(y * h_dot_x)
        np.clip(exp_val, None, 1e300, out=exp_val)  # 防止数值溢出
        
        # 计算 g1: 损失函数的梯度部分，形状为 (n, d)
        g1 = -np.einsum('ijk,ij->ik', h, y / (1 + exp_val), optimize=True) / L
        
        # 计算 g2: 正则化项的梯度部分，形状为 (n, d)
        x_squared = x**2
        g2 = 2 * x / (1 + x_squared)**2
        
        # 返回最终的梯度: g1 + rho * g2，形状为 (n, d)
        return (g1 + g2 * rho).reshape(n, d)
    
    grad_func = grad

    h, y = copy.deepcopy(h_data), copy.deepcopy(y_data)
    x = copy.deepcopy(init_x)
    n, d = x.shape
    W = np.eye(n)
    g = grad_func(x, y, h, rho=rho).reshape(x.shape) + sigma_n * np.random.normal(
        size=(n, d)
    )
    x_mean = np.mean(x, axis=0, keepdims=True)# 1*d
    x_mean_expand = np.broadcast_to(x_mean, (n, d))#n*d
    gradient_x_mean = grad_func(x_mean_expand, y, h, rho=rho).reshape(x.shape)+ sigma_n * np.random.normal(
        size=(n, d)
    )
    mean_gradient_x_mean = np.mean(gradient_x_mean, axis=0, keepdims=True)#1*d
    gradient_history = [np.linalg.norm(mean_gradient_x_mean)]

    for _ in range(max_it):
        W = A @ W
        Diag_W_inv = np.diag(1 / np.diag(W))
        x_new = x - lr * Diag_W_inv @ g
        x = A @ x_new
        g = grad_func(x, y, h, rho=rho).reshape(x.shape) + sigma_n * np.random.normal(
            size=(n, d)
        )
        x_mean = np.mean(x, axis=0, keepdims=True)# 1*d
        x_mean_expand = np.broadcast_to(x_mean, (n, d))#n*d
        gradient_x_mean = grad_func(x_mean_expand, y, h, rho=rho).reshape(x.shape)+ sigma_n * np.random.normal(
        size=(n, d)
    )
        mean_gradient_x_mean = np.mean(gradient_x_mean, axis=0, keepdims=True)#1*d
        gradient_history.append(np.linalg.norm(mean_gradient_x_mean))

    result_df = pd.DataFrame({"gradient_norm": gradient_history})
    return result_df


def PullDiag_GT(
    A,
    init_x,
    h_data,
    y_data,
    rho=0.1,
    lr=0.1,
    sigma_n=0.1,
    max_it=200,
):
    """PullDiag_GT"""
    def grad(x, y, h, rho=0.001):
        """
        计算每个节点的局部梯度 ∇f_i(x_i)，结果是一个 (n, d) 的矩阵，每一行是 ∇f_i(x_i)。
        
        Parameters:
            x (np.ndarray): 输入参数，形状为 (n, d)。
            y (np.ndarray): 标签数据，形状为 (n, L)。
            h (np.ndarray): 模型参数，形状为 (n, L, d)。
            rho (float): 正则化系数，默认为 0.001。
        
        Returns:
            np.ndarray: 梯度矩阵，形状为 (n, d)。
        """
        n, L, d = h.shape
        
        # 计算 h_dot_x: h 和 x 的点积，形状为 (n, L)
        h_dot_x = np.einsum('ijk,ik->ij', h, x, optimize=True)  # 使用 optimize=True 加速
        
        # 计算 exp_val: exp(y * h_dot_x)，形状为 (n, L)
        exp_val = np.exp(y * h_dot_x)
        np.clip(exp_val, None, 1e300, out=exp_val)  # 防止数值溢出
        
        # 计算 g1: 损失函数的梯度部分，形状为 (n, d)
        g1 = -np.einsum('ijk,ij->ik', h, y / (1 + exp_val), optimize=True) / L
        
        # 计算 g2: 正则化项的梯度部分，形状为 (n, d)
        x_squared = x**2
        g2 = 2 * x / (1 + x_squared)**2
        
        # 返回最终的梯度: g1 + rho * g2，形状为 (n, d)
        return (g1 + g2 * rho).reshape(n, d)
    
    grad_func = grad
    h, y = copy.deepcopy(h_data), copy.deepcopy(y_data)
    x = copy.deepcopy(init_x)
    n, d = x.shape
    W = np.eye(n)
    g = grad_func(x, y, h, rho=rho).reshape(x.shape) + sigma_n * np.random.normal(
        size=(n, d)
    )
    w = np.diag(1 / np.diag(W)) @ g
    v = copy.deepcopy(g)

    x_mean = np.mean(x, axis=0, keepdims=True)# 1*d
    x_mean_expand = np.broadcast_to(x_mean, (n, d))#n*d
    gradient_x_mean = grad_func(x_mean_expand, y, h, rho=rho).reshape(x.shape)+ sigma_n * np.random.normal(
        size=(n, d)
    )
    mean_gradient_x_mean = np.mean(gradient_x_mean, axis=0, keepdims=True)#1*d
    gradient_history = [np.linalg.norm(mean_gradient_x_mean)]

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

        x_mean = np.mean(x, axis=0, keepdims=True)# 1*d
        x_mean_expand = np.broadcast_to(x_mean, (n, d))
        gradient_x_mean = grad_func(x_mean_expand, y, h, rho=rho).reshape(x.shape)+ sigma_n * np.random.normal(
        size=(n, d)
    )
        mean_gradient_x_mean = np.mean(gradient_x_mean, axis=0, keepdims=True)
        gradient_history.append(np.linalg.norm(mean_gradient_x_mean))

    result_df = pd.DataFrame(
        {
            "gradient_norm": gradient_history,
        }
    )
    return result_df
