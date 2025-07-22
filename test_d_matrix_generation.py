#!/usr/bin/env python3
"""
Test script for D matrix generation with uniform and vertices modes.
"""

import numpy as np
import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils.d_matrix_utils import generate_d_matrices_with_increasing_c
from scripts_pushpull_differ_lr.network_utils import generate_nearest_neighbor_matrices

def test_d_matrix_generation():
    """Test D matrix generation with both uniform and vertices modes."""
    # Generate a small topology for testing
    n = 8  # Small size for easy visualization
    A, B = generate_nearest_neighbor_matrices(n=n, k=3, seed=42)
    
    print("=== Testing D Matrix Generation ===")
    print(f"Network size: n={n}")
    
    # Test 1: Uniform distribution with 5 values
    print("\n--- Test 1: Uniform distribution (k=5) ---")
    d_matrices_uniform = generate_d_matrices_with_increasing_c(
        A, B, num_c_values=5, distribution="uniform"
    )
    
    print(f"Generated {len(d_matrices_uniform)} D matrices")
    for i, (d_diag, c_val) in enumerate(d_matrices_uniform):
        print(f"  {i+1}. c = {c_val:.6f}, sum(d) = {np.sum(d_diag):.6f}")
        print(f"     Non-zero elements: {np.count_nonzero(d_diag)}")
    
    # Check if c values are uniformly spaced
    c_values_uniform = [c for _, c in d_matrices_uniform]
    if len(c_values_uniform) > 1:
        diffs = [c_values_uniform[i+1] - c_values_uniform[i] for i in range(len(c_values_uniform)-1)]
        print(f"  C value differences: {[f'{d:.6f}' for d in diffs]}")
        print(f"  Std of differences: {np.std(diffs):.6f} (should be close to 0 for uniform)")
    
    # Test 2: Vertices distribution with all vertices
    print("\n--- Test 2: Vertices distribution (k=n={n}) ---")
    d_matrices_vertices_all = generate_d_matrices_with_increasing_c(
        A, B, num_c_values=n, distribution="vertices"
    )
    
    print(f"Generated {len(d_matrices_vertices_all)} D matrices")
    for i, (d_diag, c_val) in enumerate(d_matrices_vertices_all):
        non_zero_idx = np.nonzero(d_diag)[0]
        print(f"  {i+1}. c = {c_val:.6f}, non-zero at index: {non_zero_idx}")
    
    # Test 3: Vertices distribution with k < n
    print(f"\n--- Test 3: Vertices distribution (k=5, n={n}) ---")
    d_matrices_vertices_subset = generate_d_matrices_with_increasing_c(
        A, B, num_c_values=5, distribution="vertices"
    )
    
    print(f"Generated {len(d_matrices_vertices_subset)} D matrices")
    for i, (d_diag, c_val) in enumerate(d_matrices_vertices_subset):
        non_zero_idx = np.nonzero(d_diag)[0]
        print(f"  {i+1}. c = {c_val:.6f}, non-zero at index: {non_zero_idx}")
    
    # Verify properties
    print("\n=== Verification ===")
    
    # Check that all D matrices sum to n
    all_matrices = [
        ("Uniform", d_matrices_uniform),
        ("Vertices (all)", d_matrices_vertices_all),
        ("Vertices (subset)", d_matrices_vertices_subset)
    ]
    
    for name, matrices in all_matrices:
        sums = [np.sum(d) for d, _ in matrices]
        print(f"{name}: All D matrices sum to n? {all(abs(s - n) < 1e-10 for s in sums)}")
    
    # Check monotonicity
    for name, matrices in all_matrices:
        c_vals = [c for _, c in matrices]
        is_monotonic = all(c_vals[i] <= c_vals[i+1] for i in range(len(c_vals)-1))
        print(f"{name}: C values monotonically increasing? {is_monotonic}")
    
    print("\n=== Test completed successfully! ===")

if __name__ == "__main__":
    test_d_matrix_generation()