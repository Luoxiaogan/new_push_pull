"""
Utility functions for distributed optimization experiments.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from typing import Optional, Tuple, List

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.algebra_utils import get_left_perron, get_right_perron
from network_utils import (
    get_matrixs_from_exp_graph,
    generate_grid_matrices,
    generate_ring_matrices,
    generate_random_graph_matrices,
    generate_stochastic_geometric_matrices,
    generate_nearest_neighbor_matrices
)


def generate_topology_matrices(
    topology: str, 
    n: int, 
    matrix_seed: int,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate A and B matrices based on topology type.
    
    Args:
        topology: Type of network topology
        n: Number of nodes
        matrix_seed: Seed for topology generation
        **kwargs: Additional parameters for specific topologies
        
    Returns:
        Tuple of (A, B) matrices where A is row-stochastic and B is column-stochastic
    """
    topology_generators = {
        "exp": lambda: get_matrixs_from_exp_graph(n, seed=matrix_seed),
        "grid": lambda: generate_grid_matrices(n, seed=matrix_seed),
        "ring": lambda: generate_ring_matrices(n, seed=matrix_seed),
        "random": lambda: generate_random_graph_matrices(n, seed=matrix_seed),
        "geometric": lambda: generate_stochastic_geometric_matrices(n, seed=matrix_seed),
        "neighbor": lambda: generate_nearest_neighbor_matrices(n, k=kwargs.get('k', 3), seed=matrix_seed)
    }
    
    if topology not in topology_generators:
        raise ValueError(f"Invalid topology '{topology}'. Valid options: {list(topology_generators.keys())}")
    
    # Special validation for grid topology
    if topology == "grid":
        grid_size = int(np.sqrt(n))
        if grid_size * grid_size != n:
            raise ValueError(f"For grid topology, n must be a perfect square. Got n={n}")
    
    A, B = topology_generators[topology]()
    
    # Convert to numpy arrays if they're not already
    if isinstance(A, torch.Tensor):
        A = A.numpy()
    if isinstance(B, torch.Tensor):
        B = B.numpy()
    
    # Validate matrices
    if not np.allclose(A.sum(axis=1), 1.0, rtol=1e-6):
        raise ValueError("Matrix A is not row-stochastic")
    if not np.allclose(B.sum(axis=0), 1.0, rtol=1e-6):
        raise ValueError("Matrix B is not column-stochastic")
    
    return A, B


def compute_learning_rates(
    strategy: str,
    A: np.ndarray,
    B: np.ndarray,
    lr_basic: float,
    n: int,
    random_seed: Optional[int] = None,
    d_diagonal: Optional[np.ndarray] = None
) -> List[float]:
    """
    Compute learning rate list based on strategy.
    
    Args:
        strategy: Learning rate distribution strategy
        A: Row-stochastic matrix
        B: Column-stochastic matrix
        lr_basic: Base learning rate (total will be lr_basic * n)
        n: Number of nodes
        random_seed: Random seed for "random" strategy
        d_diagonal: Diagonal values for "custom" strategy
        
    Returns:
        List of learning rates for each node
    """
    if strategy == "uniform":
        return [lr_basic] * n
    
    elif strategy == "pi_a_inverse":
        pi_a = get_left_perron(A)
        # Create diagonal matrix D with inverse of pi_a
        d_values = 1.0 / pi_a
        # Normalize so sum equals n
        d_values = d_values * n / np.sum(d_values)
        return [lr_basic * d for d in d_values]
    
    elif strategy == "pi_b_inverse":
        pi_b = get_right_perron(B)
        # Create diagonal matrix D with inverse of pi_b
        d_values = 1.0 / pi_b
        # Normalize so sum equals n
        d_values = d_values * n / np.sum(d_values)
        return [lr_basic * d for d in d_values]
    
    elif strategy == "random":
        if random_seed is None:
            raise ValueError("random_seed must be provided when strategy='random'")
        
        np.random.seed(random_seed)
        # Generate random positive values
        random_values = np.random.rand(n) + 0.1  # Add 0.1 to avoid values too close to 0
        # Normalize to sum to n
        random_values = random_values * n / np.sum(random_values)
        return [lr_basic * r for r in random_values]
    
    elif strategy == "custom":
        if d_diagonal is None:
            raise ValueError("d_diagonal must be provided when strategy='custom'")
        
        if len(d_diagonal) != n:
            raise ValueError(f"d_diagonal length ({len(d_diagonal)}) must match number of nodes ({n})")
        
        # d_diagonal should already be normalized to sum to n
        # Convert to learning rates
        return [lr_basic * d for d in d_diagonal]
    
    else:
        raise ValueError(f"Invalid strategy '{strategy}'. Valid options: uniform, pi_a_inverse, pi_b_inverse, random, custom")


def compute_c_value(
    A: np.ndarray,
    B: np.ndarray,
    lr_list: List[float],
    lr_basic: float
) -> float:
    """
    Compute theoretical convergence factor c = n * pi_A^T * D * pi_B.
    
    Args:
        A: Row-stochastic matrix
        B: Column-stochastic matrix
        lr_list: List of learning rates for each node
        lr_basic: Base learning rate
        
    Returns:
        Convergence factor c
    """
    n = A.shape[0]
    pi_a = get_left_perron(A)
    pi_b = get_right_perron(B)
    
    # Construct D matrix from learning rates
    D = np.diag([lr / lr_basic for lr in lr_list])
    
    # Compute c = n * pi_A^T * D * pi_B
    c = n * pi_a.T @ D @ pi_b
    return float(c)


def average_gradient_norm_results(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Average gradient norm DataFrames across repetitions.
    
    Args:
        df_list: List of DataFrames from multiple runs
        
    Returns:
        Averaged DataFrame
    """
    if not df_list:
        raise ValueError("df_list is empty")
    
    # Sum all DataFrames
    df_sum = df_list[0].copy()
    for df in df_list[1:]:
        df_sum = df_sum.add(df, fill_value=0)
    
    # Average
    df_avg = df_sum / len(df_list)
    return df_avg