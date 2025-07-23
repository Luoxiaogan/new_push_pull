"""
Utility functions for generating diagonal matrices D with varying convergence factors c.

Theoretical Foundation:
-----------------------
For distributed optimization with the PushPull algorithm, the convergence rate depends on
the convergence factor c = n * pi_A^T * D * pi_B, where:
- n is the number of nodes
- pi_A is the left Perron vector of the row-stochastic mixing matrix A
- pi_B is the right Perron vector of the column-stochastic mixing matrix B
- D is a diagonal matrix with positive diagonal elements d_i satisfying sum(d_i) = n

The constraint sum(d_i) = n with d_i > 0 defines an (n-1)-dimensional simplex in R^n.

Key Insight (Simplex Method):
The function c(d) = n * sum(pi_A[i] * d[i] * pi_B[i]) is linear in d.
Linear functions on convex polytopes (like simplices) achieve their extrema at vertices.

The vertices of our simplex are points where one d_j = n and all other d_i = 0.
At vertex j: c_j = n^2 * pi_A[j] * pi_B[j]

Therefore:
- c_min = min_j(n^2 * pi_A[j] * pi_B[j])
- c_max = max_j(n^2 * pi_A[j] * pi_B[j])

This gives us the exact theoretical range of c without any sampling!

Constructing Intermediate Values:
Any d in the simplex can be expressed as a convex combination of vertices.
For two vertices d_min and d_max, we can generate intermediate values via:
d(alpha) = (1 - alpha) * d_min + alpha * d_max, where alpha in [0, 1]

This produces c values that vary linearly from c_min to c_max.
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


def generate_d_from_convex_combination(
    d_min: np.ndarray,
    d_max: np.ndarray,
    alpha: float,
    n: int
) -> np.ndarray:
    """
    Generate D diagonal from convex combination of two extreme D vectors.
    
    Args:
        d_min: D vector corresponding to minimum c
        d_max: D vector corresponding to maximum c
        alpha: Interpolation parameter in [0, 1]
        n: Number of nodes
        
    Returns:
        Interpolated D diagonal that sums to n
    """
    # Compute convex combination
    d_new = (1 - alpha) * d_min + alpha * d_max
    
    # Ensure it sums to n (should already be the case, but for numerical stability)
    d_new = d_new * n / np.sum(d_new)
    
    return d_new


def generate_d_matrices_theoretical(
    A: np.ndarray,
    B: np.ndarray,
    num_c_values: int,
    distribution: str = "uniform"
) -> List[Tuple[np.ndarray, float]]:
    """
    Generate D matrices using theoretical approach based on extreme points of the simplex.
    
    This method leverages the fact that c(d) is a linear function on a simplex:
    1. Computes c at all vertices (where one d_j = n, others = 0)
    2. Identifies exact c_min and c_max without any sampling
    3. Generates intermediate values via convex combinations
    
    Mathematical Basis:
    - The constraint {d: d_i > 0, sum(d_i) = n} forms a simplex
    - c(d) = n * pi_A^T * diag(d) * pi_B is linear in d
    - Linear functions on simplices achieve extrema at vertices
    - At vertex j: c = n^2 * pi_A[j] * pi_B[j]
    
    Distribution Modes:
    - "uniform": Generate uniformly spaced c values between c_min and c_max
                Uses linear interpolation between the two extreme vertices
    - "vertices": Use actual simplex vertices (sorted by c value)
                 May produce non-uniform c spacing depending on pi_A and pi_B
    
    Args:
        A: Row-stochastic matrix
        B: Column-stochastic matrix
        num_c_values: Number of different c values to generate
        distribution: "uniform" or "vertices" - distribution mode for c values
        
    Returns:
        List of tuples (d_diagonal, c_value) sorted by increasing c
    """
    n = A.shape[0]
    pi_a = get_left_perron(A)
    pi_b = get_right_perron(B)
    
    # Step 1: Compute all extreme c values
    # c_j = n^2 * pi_A[j] * pi_B[j] when D has all mass on j-th diagonal
    c_extremes = []
    for j in range(n):
        c_j = n * n * pi_a[j] * pi_b[j]
        d_j = np.zeros(n)
        d_j[j] = n
        c_extremes.append((c_j, j, d_j))
    
    # Sort by c value
    c_extremes.sort(key=lambda x: x[0])
    
    if distribution == "vertices":
        # Return vertices mode
        results = []
        
        if num_c_values >= n:
            # Return all vertices
            for c_j, j, d_j in c_extremes:
                results.append((d_j, c_j))
        else:
            # Select num_c_values vertices with most diverse c values
            # Use a greedy approach to maximize spacing
            selected_indices = []
            
            # Always include min and max
            selected_indices.append(0)
            if num_c_values > 1:
                selected_indices.append(n - 1)
            
            # Fill in remaining slots by maximizing minimum distance
            while len(selected_indices) < num_c_values:
                best_idx = -1
                best_min_dist = -1
                
                for i in range(1, n - 1):
                    if i not in selected_indices:
                        # Calculate minimum distance to already selected points
                        min_dist = float('inf')
                        for j in selected_indices:
                            dist = abs(c_extremes[i][0] - c_extremes[j][0])
                            min_dist = min(min_dist, dist)
                        
                        if min_dist > best_min_dist:
                            best_min_dist = min_dist
                            best_idx = i
                
                if best_idx >= 0:
                    selected_indices.append(best_idx)
            
            # Sort selected indices and create results
            selected_indices.sort()
            for idx in selected_indices:
                c_j, j, d_j = c_extremes[idx]
                results.append((d_j, c_j))
                
    else:  # distribution == "uniform"
        # Use uniform interpolation between min and max
        c_min, j_min, d_min = c_extremes[0]
        c_max, j_max, d_max = c_extremes[-1]
        
        results = []
        
        if num_c_values == 1:
            # Single value: use midpoint
            alpha = 0.5
            d_diagonal = generate_d_from_convex_combination(d_min, d_max, alpha, n)
            c = compute_c_from_d_diagonal(d_diagonal, pi_a, pi_b, n)
            results.append((d_diagonal, c))
        else:
            # Multiple values: use linear interpolation
            alphas = np.linspace(0, 1, num_c_values)
            
            for alpha in alphas:
                d_diagonal = generate_d_from_convex_combination(d_min, d_max, alpha, n)
                c = compute_c_from_d_diagonal(d_diagonal, pi_a, pi_b, n)
                results.append((d_diagonal, c))
    
    return results


def generate_d_matrices_with_increasing_c(
    A: np.ndarray,
    B: np.ndarray,
    num_c_values: int,
    c_min: float = None,
    c_max: float = None,
    num_samples: int = 2000,
    seed: int = 42,
    distribution: str = "uniform"
) -> List[Tuple[np.ndarray, float]]:
    """
    Generate D matrices (as diagonal arrays) that produce increasing c values.
    
    This function now uses the theoretical approach by default, which is more
    efficient and accurate than the previous sampling-based method.
    
    Args:
        A: Row-stochastic matrix
        B: Column-stochastic matrix
        num_c_values: Number of different c values to generate
        c_min: Minimum c value (if None, will be determined automatically)
        c_max: Maximum c value (if None, will be determined automatically)
        num_samples: Number of random samples to generate for finding c range (unused in theoretical approach)
        seed: Random seed (unused in theoretical approach)
        distribution: "uniform" for uniformly spaced c values, "vertices" for simplex vertices
        
    Returns:
        List of tuples (d_diagonal, c_value) sorted by increasing c
    """
    # Use theoretical approach
    return generate_d_matrices_theoretical(A, B, num_c_values, distribution)


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