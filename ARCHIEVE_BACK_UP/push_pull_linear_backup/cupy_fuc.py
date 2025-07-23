import cupy as cp
import numpy as np
import pandas as pd

def get_right_perron_gpu(W):
    """
    For a column-stochastic matrix, gets the right Perron vector using CuPy.
    The right Perron vector is the eigenvector corresponding to the eigenvalue 1.
    """
    eigenvalues, eigenvectors = cp.linalg.eigh(W)
    # Find the index of the eigenvalue closest to 1
    max_eigen_idx = cp.argmin(cp.abs(eigenvalues - 1.0))
    # Extract the corresponding eigenvector
    vector = eigenvectors[:, max_eigen_idx]
    # The Perron vector is real and non-negative. We take the real part to discard
    # numerical noise and the absolute value to handle potential sign flips.
    perron_vector = cp.real(vector)
    # Normalize to make it a probability distribution (sum to 1)
    return perron_vector / cp.sum(perron_vector)

def get_left_perron_gpu(W):
    """
    For a row-stochastic matrix, gets the left Perron vector using CuPy.
    This is equivalent to finding the right Perron vector of the transposed matrix.
    """
    return get_right_perron_gpu(W.T)

def grad_with_batch_batched_gpu(
    x_batched,  # Shape: (num_runs, n, d)
    y_nodes_gpu,  # Shape: (n, L) - global y_tilde on GPU
    h_nodes_gpu,  # Shape: (n, L, d) - global h_tilde on GPU
    rho,
    batch_size,
    num_runs # 传入 num_runs
):
    n_nodes, L_samples, d_dims = h_nodes_gpu.shape # n, L, d from h_tilde

    if batch_size is None or batch_size >= L_samples:
        batch_size_eff = L_samples
        # Expand h_nodes_gpu and y_nodes_gpu for num_runs using broadcasting
        # h_batch_gpu shape: (num_runs, n, L, d)
        # y_batch_gpu shape: (num_runs, n, L)
        h_batch_gpu = cp.broadcast_to(h_nodes_gpu[cp.newaxis, ...], (num_runs, n_nodes, L_samples, d_dims))
        y_batch_gpu = cp.broadcast_to(y_nodes_gpu[cp.newaxis, ...], (num_runs, n_nodes, L_samples))
    else:
        batch_size_eff = batch_size
        # Sample indices for each run and each node independently
        # batch_indices shape: (num_runs, n, batch_size_eff)
        # CuPy's random.choice doesn't directly support this multi-axis independent sampling easily.
        # We can generate for one run-node and then tile, or loop (less efficient but clear).
        # A more efficient way is to generate a large pool of random numbers.
        # For simplicity here, let's assume we can get appropriately shaped indices.
        # A practical way for (num_runs, n, batch_size_eff):
        all_indices = cp.random.rand(num_runs, n_nodes, L_samples).argsort(axis=-1)[:, :, :batch_size_eff]
        batch_indices = all_indices.astype(cp.int32) # Ensure integer indices

        # Gather h_batch and y_batch using these indices
        # h_batch_gpu shape: (num_runs, n, batch_size_eff, d)
        # y_batch_gpu shape: (num_runs, n, batch_size_eff)
        
        # Create indices for gathering
        run_idx = cp.arange(num_runs)[:, cp.newaxis, cp.newaxis] # (num_runs, 1, 1)
        node_idx = cp.arange(n_nodes)[cp.newaxis, :, cp.newaxis]   # (1, n, 1)
        
        h_batch_gpu = h_nodes_gpu[node_idx, batch_indices, :] 
        y_batch_gpu = y_nodes_gpu[node_idx, batch_indices]

    # x_batched is (num_runs, n, d)
    # h_batch_gpu is (num_runs, n, batch_size_eff, d)
    # y_batch_gpu is (num_runs, n, batch_size_eff)

    # einsum for h_dot_x: result shape (num_runs, n, batch_size_eff)
    h_dot_x = cp.einsum('rnbd,rnd->rnb', h_batch_gpu, x_batched)
    
    exp_val = cp.exp(y_batch_gpu * h_dot_x)
    cp.clip(exp_val, a_min=None, a_max=1e300, out=exp_val)

    # einsum for g1: result shape (num_runs, n, d)
    g1 = -cp.einsum('rnbd,rnb->rnd', h_batch_gpu, y_batch_gpu / (1 + exp_val)) / batch_size_eff
    
    x_squared = x_batched**2
    g2 = 2 * x_batched / (1 + x_squared)**2
    
    grad_val = (g1 + rho * g2) # Shape (num_runs, n, d)
    return grad_val # No reshape needed if einsum is correct

def PushPull_with_batch_batched_gpu(
    A_gpu, B_gpu, init_x_gpu_batched, # init_x shape: (num_runs, n, d)
    h_data_nodes_gpu, y_data_nodes_gpu, # Original h_tilde, y_tilde on GPU
    grad_func_batched_gpu, # The new batched gradient function for GPU
    rho, lr, sigma_n,
    max_it, batch_size, num_runs
):
    x = cp.copy(init_x_gpu_batched) # Shape: (num_runs, n, d)
    num_n, num_d = x.shape[1], x.shape[2] # n, d

    # Initial gradient calculation
    g = grad_func_batched_gpu(x, y_data_nodes_gpu, h_data_nodes_gpu, rho, batch_size, num_runs)
    if sigma_n > 0:
        g += sigma_n * cp.random.normal(size=(num_runs, num_n, num_d))
    
    y = cp.copy(g) # Shape: (num_runs, n, d)

    # Store average gradient norm over runs
    avg_gradient_norm_history = []

    for iter_num in range(max_it):
        # x_update: x = A @ x - lr * y
        # A_gpu is (n,n). x is (num_runs, n, d).
        # einsum: 'jk,rkl->rjl' where j=n_out, k=n_in, r=num_runs, l=d
        term_Ax = cp.einsum('jk,rkl->rjl', A_gpu, x)
        x = term_Ax - lr * y
        
        # New gradient
        g_new = grad_func_batched_gpu(x, y_data_nodes_gpu, h_data_nodes_gpu, rho, batch_size, num_runs)
        if sigma_n > 0:
            g_new += sigma_n * cp.random.normal(size=(num_runs, num_n, num_d))

        # y_update: y = B @ y + g_new - g
        term_By = cp.einsum('jk,rkl->rjl', B_gpu, y)
        y = term_By + g_new - g
        g = g_new # Update old gradient

        # --- Record history (averaged over runs) ---
        # 1. Calculate mean_x for each run: x_mean_per_run shape (num_runs, 1, d)
        x_mean_per_run = cp.mean(x, axis=1, keepdims=True)
        
        # 2. Expand x_mean_per_run for grad_func: shape (num_runs, n, d)
        x_mean_expand_per_run = cp.broadcast_to(x_mean_per_run, (num_runs, num_n, num_d))
        
        # 3. Calculate full batch gradient for each run's x_mean: _grad_on_full_per_run shape (num_runs, n, d)
        #    Use batch_size=None for full dataset, no sigma_n noise for this evaluation
        _grad_on_full_per_run = grad_func_batched_gpu(
            x_mean_expand_per_run, y_data_nodes_gpu, h_data_nodes_gpu,
            rho=rho, batch_size=None, num_runs=num_runs
        )
        
        # 4. Calculate mean gradient over nodes for each run: mean_grad_per_run shape (num_runs, 1, d)
        mean_grad_per_run = cp.mean(_grad_on_full_per_run, axis=1, keepdims=True)
        
        # 5. Calculate norm of mean_grad for each run: norm_per_run shape (num_runs,)
        norm_per_run = cp.linalg.norm(mean_grad_per_run, axis=2).squeeze() # Squeeze out d-dim (norm result) and then 1-dim
        
        # 6. Average these norms over all runs: avg_norm_over_runs (scalar)
        avg_norm_over_runs_scalar = cp.mean(norm_per_run)
        avg_gradient_norm_history.append(cp.asnumpy(avg_norm_over_runs_scalar)) # Store as numpy float for pandas

        if (iter_num + 1) % 10 == 0: # Print progress
            print(f"Iteration {iter_num+1}/{max_it}, Avg Grad Norm: {avg_norm_over_runs_scalar:.6f}")


    return pd.DataFrame({
        "gradient_norm_on_full_trainset_avg": avg_gradient_norm_history,
    })


import cupy as cp
import numpy as np
import pandas as pd

# Assume grad_with_batch_batched_gpu and the Perron vector functions are defined as above

def PushPull_with_batch_batched_gpu_new(
    A_gpu, B_gpu, init_x_gpu_batched, # init_x shape: (num_runs, n, d)
    h_data_nodes_gpu, y_data_nodes_gpu, # Original h_tilde, y_tilde on GPU
    grad_func_batched_gpu, # The new batched gradient function for GPU
    rho, lr, sigma_n,
    max_it, batch_size, num_runs
):
    x = cp.copy(init_x_gpu_batched) # Shape: (num_runs, n, d)
    num_n, num_d = x.shape[1], x.shape[2] # n, d

    # --- New Feature: Pre-computation for new metrics ---
    # Get Perron vectors
    pi_A_gpu = get_left_perron_gpu(A_gpu).reshape(1, num_n)   # Shape (1, n)
    pi_B_gpu = get_right_perron_gpu(B_gpu).reshape(num_n, 1) # Shape (n, 1)
    
    # Calculate B_infinity
    B_inf_gpu = pi_B_gpu @ cp.ones((1, num_n)) # Shape (n, n)
    
    # Pre-calculate the contraction term for efficiency in the loop
    I_gpu = cp.eye(num_n)
    term_for_y_norm = pi_A_gpu @ (I_gpu - B_inf_gpu) # Shape (1, n)

    # Initial gradient calculation
    g = grad_func_batched_gpu(x, y_data_nodes_gpu, h_data_nodes_gpu, rho, batch_size, num_runs)
    if sigma_n > 0:
        g += sigma_n * cp.random.normal(size=(num_runs, num_n, num_d))
    
    y = cp.copy(g) # Shape: (num_runs, n, d)

    # --- History Tracking ---
    avg_gradient_norm_history = []
    piA_y_norm_history = []
    g_bar_norm_history = []


    for iter_num in range(max_it):
        # x_update: x = A @ x - lr * y
        term_Ax = cp.einsum('jk,rkl->rjl', A_gpu, x)
        x = term_Ax - lr * y
        
        # New gradient
        g_new = grad_func_batched_gpu(x, y_data_nodes_gpu, h_data_nodes_gpu, rho, batch_size, num_runs)
        if sigma_n > 0:
            g_new += sigma_n * cp.random.normal(size=(num_runs, num_n, num_d))

        # y_update: y = B @ y + g_new - g
        term_By = cp.einsum('jk,rkl->rjl', B_gpu, y)
        y = term_By + g_new - g
        
        # Update old gradient
        g = g_new 

        # --- Record History (averaged over runs) ---
        
        # (1) Original metric: Norm of the full gradient at the average iterate
        x_mean_per_run = cp.mean(x, axis=1, keepdims=True)
        x_mean_expand_per_run = cp.broadcast_to(x_mean_per_run, (num_runs, num_n, num_d))
        _grad_on_full_per_run = grad_func_batched_gpu(
            x_mean_expand_per_run, y_data_nodes_gpu, h_data_nodes_gpu,
            rho=rho, batch_size=None, num_runs=num_runs
        )
        mean_grad_per_run = cp.mean(_grad_on_full_per_run, axis=1, keepdims=True)
        norm_per_run = cp.linalg.norm(mean_grad_per_run, axis=2).squeeze()
        avg_norm_over_runs_scalar = cp.mean(norm_per_run)
        avg_gradient_norm_history.append(cp.asnumpy(avg_norm_over_runs_scalar))

        # (2) New metric: 2-norm of pi_A^T @ (I - B_inf) @ y
        # We use einsum to efficiently contract y with our pre-computed term
        # 'kn,rnd->rd' contracts (1, n) with (runs, n, d) -> (runs, d)
        y_contracted = cp.einsum('kn,rnd->rd', term_for_y_norm, y)
        # Calculate norm for each run, then average
        norm_per_run_y = cp.linalg.norm(y_contracted, axis=1)
        avg_norm_y = cp.mean(norm_per_run_y)
        piA_y_norm_history.append(cp.asnumpy(avg_norm_y))

        # # (3) New metric: 2-norm of the average gradient g_bar
        # # Average g over nodes (axis=1) for each run
        # g_bar_per_run = cp.mean(g, axis=1) # Shape (num_runs, d)
        # # Calculate norm for each run, then average
        # norm_per_run_g_bar = cp.linalg.norm(g_bar_per_run, axis=1)
        # avg_norm_g_bar = cp.mean(norm_per_run_g_bar)
        # g_bar_norm_history.append(cp.asnumpy(avg_norm_g_bar))


        if (iter_num + 1) % 10 == 0: # Print progress
            print(f"Iteration {iter_num+1}/{max_it}, "
                  f"Avg Grad Norm: {avg_norm_over_runs_scalar:.6f}, "
                  f"piA(I-Binf)y Norm: {avg_norm_y:.6f}")
                  #f"g_bar Norm: {avg_norm_g_bar:.6f}")

    # Return results in a pandas DataFrame
    return pd.DataFrame({
        "gradient_norm_on_full_trainset_avg": avg_gradient_norm_history,
        "piA_I_minus_Binf_y_norm": piA_y_norm_history,
        #"g_bar_norm": g_bar_norm_history,
    })