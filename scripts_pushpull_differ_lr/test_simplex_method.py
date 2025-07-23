#!/usr/bin/env python3
"""
Test script to verify the theoretical simplex method for generating D matrices.

This script demonstrates that:
1. The theoretical method finds exact c_min and c_max
2. Convex combinations produce monotonically increasing c values
3. The method is more efficient than sampling-based approaches
"""

import os
import sys
import numpy as np
import time

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiment_utils import generate_topology_matrices
from utils.algebra_utils import get_left_perron, get_right_perron
from utils.d_matrix_utils import (
    generate_d_matrices_theoretical,
    compute_c_from_d_diagonal,
    generate_random_d_diagonal
)


def test_theoretical_extrema():
    """Test that theoretical method finds exact extrema."""
    print("=" * 60)
    print("Test 1: Theoretical Method Finds Exact Extrema")
    print("=" * 60)
    
    # Generate a small topology for testing
    n = 8
    A, B = generate_topology_matrices("ring", n, matrix_seed=42)
    pi_a = get_left_perron(A)
    pi_b = get_right_perron(B)
    
    # Method 1: Theoretical computation
    print("\nMethod 1: Theoretical (Simplex Vertices)")
    start_time = time.time()
    
    # Compute c at all vertices
    c_values = []
    for j in range(n):
        c_j = n * n * pi_a[j] * pi_b[j]
        c_values.append(c_j)
    
    theoretical_c_min = min(c_values)
    theoretical_c_max = max(c_values)
    theoretical_time = time.time() - start_time
    
    print(f"  c_min = {theoretical_c_min:.6f}")
    print(f"  c_max = {theoretical_c_max:.6f}")
    print(f"  Time: {theoretical_time:.6f} seconds")
    
    # Method 2: Sampling approach (for comparison)
    print("\nMethod 2: Random Sampling (1000 samples)")
    start_time = time.time()
    
    sampled_c_values = []
    for i in range(1000):
        d_random = generate_random_d_diagonal(n, seed=i)
        c = compute_c_from_d_diagonal(d_random, pi_a, pi_b, n)
        sampled_c_values.append(c)
    
    sampled_c_min = min(sampled_c_values)
    sampled_c_max = max(sampled_c_values)
    sampling_time = time.time() - start_time
    
    print(f"  c_min = {sampled_c_min:.6f}")
    print(f"  c_max = {sampled_c_max:.6f}")
    print(f"  Time: {sampling_time:.6f} seconds")
    
    # Compare results
    print("\nComparison:")
    print(f"  c_min error: {abs(sampled_c_min - theoretical_c_min):.6e}")
    print(f"  c_max error: {abs(sampled_c_max - theoretical_c_max):.6e}")
    print(f"  Speedup: {sampling_time / theoretical_time:.1f}x")
    print(f"  Note: Sampling may miss true extrema!")


def test_monotonic_generation():
    """Test that generated c values are monotonically increasing."""
    print("\n" + "=" * 60)
    print("Test 2: Monotonic c Value Generation")
    print("=" * 60)
    
    # Generate topology
    n = 16
    A, B = generate_topology_matrices("neighbor", n, matrix_seed=123, k=3)
    
    # Generate D matrices with uniform distribution
    num_c_values = 10
    d_matrices = generate_d_matrices_theoretical(A, B, num_c_values, distribution="uniform")
    
    print(f"\nGenerated {num_c_values} D matrices with uniform c distribution:")
    
    # Check monotonicity
    c_values = [c for _, c in d_matrices]
    is_monotonic = all(c_values[i] <= c_values[i+1] for i in range(len(c_values)-1))
    
    for i, c in enumerate(c_values):
        print(f"  {i+1:2d}. c = {c:.6f}")
    
    print(f"\nMonotonic: {is_monotonic}")
    print(f"c_min = {c_values[0]:.6f}")
    print(f"c_max = {c_values[-1]:.6f}")
    print(f"Range = {c_values[-1] - c_values[0]:.6f}")


def test_vertices_vs_uniform():
    """Compare vertices distribution vs uniform distribution."""
    print("\n" + "=" * 60)
    print("Test 3: Vertices vs Uniform Distribution")
    print("=" * 60)
    
    # Generate topology
    n = 12
    A, B = generate_topology_matrices("grid", n, matrix_seed=456)
    
    # Generate with vertices distribution
    print("\nVertices Distribution (up to 12 values):")
    d_matrices_vertices = generate_d_matrices_theoretical(A, B, 12, distribution="vertices")
    
    for i, (_, c) in enumerate(d_matrices_vertices[:6]):  # Show first 6
        print(f"  {i+1:2d}. c = {c:.6f}")
    if len(d_matrices_vertices) > 6:
        print(f"  ... ({len(d_matrices_vertices)-6} more values)")
    
    # Generate with uniform distribution
    print("\nUniform Distribution (6 values):")
    d_matrices_uniform = generate_d_matrices_theoretical(A, B, 6, distribution="uniform")
    
    for i, (_, c) in enumerate(d_matrices_uniform):
        print(f"  {i+1:2d}. c = {c:.6f}")
    
    # Analyze spacing
    print("\nSpacing Analysis:")
    
    # Vertices spacing
    if len(d_matrices_vertices) > 1:
        vertices_spacings = [d_matrices_vertices[i+1][1] - d_matrices_vertices[i][1] 
                           for i in range(len(d_matrices_vertices)-1)]
        print(f"  Vertices: min spacing = {min(vertices_spacings):.6f}, "
              f"max spacing = {max(vertices_spacings):.6f}")
    
    # Uniform spacing
    if len(d_matrices_uniform) > 1:
        uniform_spacings = [d_matrices_uniform[i+1][1] - d_matrices_uniform[i][1] 
                          for i in range(len(d_matrices_uniform)-1)]
        print(f"  Uniform:  min spacing = {min(uniform_spacings):.6f}, "
              f"max spacing = {max(uniform_spacings):.6f}")


def test_d_matrix_properties():
    """Verify that generated D matrices satisfy all constraints."""
    print("\n" + "=" * 60)
    print("Test 4: D Matrix Properties")
    print("=" * 60)
    
    # Generate topology
    n = 10
    A, B = generate_topology_matrices("random", n, matrix_seed=789)
    
    # Generate D matrices
    d_matrices = generate_d_matrices_theoretical(A, B, 5, distribution="uniform")
    
    print(f"\nChecking properties of {len(d_matrices)} generated D matrices:")
    
    all_valid = True
    for i, (d_diagonal, c) in enumerate(d_matrices):
        # Check positivity
        all_positive = np.all(d_diagonal > 0)
        
        # Check sum constraint
        d_sum = np.sum(d_diagonal)
        sum_correct = np.isclose(d_sum, n, rtol=1e-10)
        
        # Verify c value
        pi_a = get_left_perron(A)
        pi_b = get_right_perron(B)
        c_computed = compute_c_from_d_diagonal(d_diagonal, pi_a, pi_b, n)
        c_matches = np.isclose(c, c_computed, rtol=1e-10)
        
        is_valid = all_positive and sum_correct and c_matches
        all_valid = all_valid and is_valid
        
        print(f"  D_{i+1}: positive={all_positive}, sum={d_sum:.6f}, "
              f"c_matches={c_matches} {'✓' if is_valid else '✗'}")
    
    print(f"\nAll D matrices valid: {all_valid}")


if __name__ == "__main__":
    # Run all tests
    test_theoretical_extrema()
    test_monotonic_generation()
    test_vertices_vs_uniform()
    test_d_matrix_properties()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)