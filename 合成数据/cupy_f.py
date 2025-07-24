import cupy as cp

def grad_with_batch_cupy(x, y, h, rho=0.001, batch_size=None):
    """
    支持 CuPy 的批次梯度计算函数（输入需为 CuPy 数组）
    """
    n, L, d = h.shape
    if batch_size is None or batch_size >= L:
        batch_size = L
        h_batch = h
        y_batch = y
    else:
        # 使用 CuPy 的随机采样（每个节点独立采样）
        batch_indices = cp.random.choice(L, batch_size, replace=False)
        h_batch = h[:, batch_indices, :]
        y_batch = y[:, batch_indices]
    
    # 计算梯度（全部使用 CuPy 操作）
    h_dot_x = cp.einsum('ijk,ik->ij', h_batch, x)
    exp_val = cp.exp(y_batch * h_dot_x)
    cp.clip(exp_val, None, 1e300, out=exp_val)
    
    g1 = -cp.einsum('ijk,ij->ik', h_batch, y_batch / (1 + exp_val)) / batch_size
    x_squared = x**2
    g2 = 2 * x / (1 + x_squared)**2
    
    return (g1 + rho * g2).reshape(n, d)