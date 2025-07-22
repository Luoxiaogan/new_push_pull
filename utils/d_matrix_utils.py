"""
Utility functions for generating diagonal matrices D with varying convergence factors c.
"""

import numpy as np
from typing import List, Tuple
from scipy.optimize import minimize_scalar
from utils.algebra_utils import get_left_perron, get_right_perron


def compute_c_from_d_diagonal(
    d_diagonal: np.ndarray,
    pi_a: np.ndarray,
    pi_b: np.ndarray,
    n: int
) -> float:
    """
    Compute convergence factor c = n * pi_A^T * D * pi_B.
    
    Args:
        d_diagonal: Diagonal values of matrix D
        pi_a: Left Perron vector of matrix A
        pi_b: Right Perron vector of matrix B
        n: Number of nodes
        
    Returns:
        Convergence factor c
    """
    # Ensure d_diagonal sums to n (normalization constraint)
    d_diagonal_normalized = d_diagonal * n / np.sum(d_diagonal)
    
    # Compute c = n * pi_A^T * D * pi_B
    c = n * np.sum(pi_a * d_diagonal_normalized * pi_b)
    return c


def generate_random_d_diagonal(n: int, seed: int = None) -> np.ndarray:
    """
    Generate a random diagonal for D matrix that satisfies constraints.
    
    Args:
        n: Number of nodes
        seed: Random seed
        
    Returns:
        Diagonal values that sum to n
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random positive values
    d_diagonal = np.random.rand(n) + 0.1  # Add 0.1 to avoid values too close to 0
    
    # Normalize to sum to n
    d_diagonal = d_diagonal * n / np.sum(d_diagonal)
    
    return d_diagonal


def generate_d_matrices_with_increasing_c(
    A: np.ndarray,
    B: np.ndarray,
    num_c_values: int,
    c_min: float = None,
    c_max: float = None,
    num_samples: int = 1000,
    seed: int = 42
) -> List[Tuple[np.ndarray, float]]:
    """
    Generate D matrices (as diagonal arrays) that produce increasing c values.
    
    Args:
        A: Row-stochastic matrix
        B: Column-stochastic matrix
        num_c_values: Number of different c values to generate
        c_min: Minimum c value (if None, will be determined automatically)
        c_max: Maximum c value (if None, will be determined automatically)
        num_samples: Number of random samples to generate for finding c range
        seed: Random seed
        
    Returns:
        List of tuples (d_diagonal, c_value) sorted by increasing c
    """
    n = A.shape[0]
    pi_a = get_left_perron(A)
    pi_b = get_right_perron(B)
    
    # If c_min or c_max not specified, find the range by sampling
    if c_min is None or c_max is None:
        np.random.seed(seed)
        c_values_sampled = []
        
        # Sample random D matrices to find c range
        for i in range(num_samples):
            d_diagonal = generate_random_d_diagonal(n, seed=seed + i)
            c = compute_c_from_d_diagonal(d_diagonal, pi_a, pi_b, n)
            c_values_sampled.append(c)
        
        if c_min is None:
            c_min = min(c_values_sampled)
        if c_max is None:
            c_max = max(c_values_sampled)
    
    # Generate specific c values
    if num_c_values == 1:
        target_c_values = [(c_min + c_max) / 2]
    else:
        target_c_values = np.linspace(c_min, c_max, num_c_values)
    
    # For each target c value, find a D matrix that produces it
    results = []
    
    for target_c in target_c_values:
        # Use optimization to find D diagonal that produces target c
        best_d_diagonal = None
        best_c_diff = float('inf')
        
        # Try multiple random initializations
        for attempt in range(100):
            # Start with random D diagonal
            d_diagonal_init = generate_random_d_diagonal(n, seed=seed + attempt + num_samples)
            
            # Define objective function (minimize difference from target c)
            def objective(scale):
                # Scale one element while keeping sum = n
                d_test = d_diagonal_init.copy()
                d_test[0] *= scale
                d_test = d_test * n / np.sum(d_test)  # Renormalize
                
                c_test = compute_c_from_d_diagonal(d_test, pi_a, pi_b, n)
                return abs(c_test - target_c)
            
            # Optimize
            result = minimize_scalar(objective, bounds=(0.1, 10), method='bounded')
            
            # Get the optimized D diagonal
            d_diagonal = d_diagonal_init.copy()
            d_diagonal[0] *= result.x
            d_diagonal = d_diagonal * n / np.sum(d_diagonal)
            
            # Compute actual c value
            c_actual = compute_c_from_d_diagonal(d_diagonal, pi_a, pi_b, n)
            c_diff = abs(c_actual - target_c)
            
            if c_diff < best_c_diff:
                best_c_diff = c_diff
                best_d_diagonal = d_diagonal
        
        # Compute final c value for best D diagonal
        c_final = compute_c_from_d_diagonal(best_d_diagonal, pi_a, pi_b, n)
        results.append((best_d_diagonal, c_final))
    
    # Sort by c value
    results.sort(key=lambda x: x[1])
    
    return results


def generate_specific_d_matrices(
    A: np.ndarray,
    B: np.ndarray,
    strategies: List[str] = ["uniform", "pi_a_inverse", "pi_b_inverse", "intermediate"]
) -> List[Tuple[np.ndarray, float, str]]:
    """
    Generate specific D matrices based on common strategies.
    
    Args:
        A: Row-stochastic matrix
        B: Column-stochastic matrix
        strategies: List of strategies to generate
        
    Returns:
        List of tuples (d_diagonal, c_value, strategy_name)
    """
    n = A.shape[0]
    pi_a = get_left_perron(A)
    pi_b = get_right_perron(B)
    
    results = []
    
    for strategy in strategies:
        if strategy == "uniform":
            d_diagonal = np.ones(n)
        elif strategy == "pi_a_inverse":
            d_diagonal = 1.0 / pi_a
        elif strategy == "pi_b_inverse":
            d_diagonal = 1.0 / pi_b
        elif strategy == "intermediate":
            # Average of pi_a_inverse and pi_b_inverse
            d_diagonal = 0.5 * (1.0 / pi_a + 1.0 / pi_b)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Normalize to sum to n
        d_diagonal = d_diagonal * n / np.sum(d_diagonal)
        
        # Compute c value
        c = compute_c_from_d_diagonal(d_diagonal, pi_a, pi_b, n)
        
        results.append((d_diagonal, c, strategy))
    
    # Sort by c value
    results.sort(key=lambda x: x[1])
    
    return results