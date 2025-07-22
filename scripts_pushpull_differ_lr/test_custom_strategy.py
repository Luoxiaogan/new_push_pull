#!/usr/bin/env python3
"""
Example script demonstrating how to use the custom strategy functionality.
"""

import os
import sys
import numpy as np

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiment_utils import generate_topology_matrices
from utils.d_matrix_utils import (
    generate_d_matrices_with_increasing_c,
    generate_specific_d_matrices,
    compute_c_from_d_diagonal
)
from utils.algebra_utils import get_left_perron, get_right_perron
from run_experiment import run_distributed_optimization_experiment


def example_1_specific_strategies():
    """Example 1: Test specific well-known strategies."""
    print("\n" + "="*60)
    print("Example 1: Testing specific strategies")
    print("="*60)
    
    # Configuration
    n = 16
    topology = "neighbor"
    matrix_seed = 42
    
    # Generate topology
    A, B = generate_topology_matrices(topology, n, matrix_seed, k=3)
    
    # Generate D matrices for specific strategies
    d_results = generate_specific_d_matrices(A, B, 
        strategies=["uniform", "pi_a_inverse", "pi_b_inverse", "intermediate"])
    
    # Print results
    print(f"\nGenerated {len(d_results)} strategies:")
    for d_diagonal, c_value, strategy in d_results:
        print(f"  {strategy}: c = {c_value:.6f}")
        print(f"    D diagonal (first 5 values): {d_diagonal[:5]}")
        print(f"    D diagonal stats: min={d_diagonal.min():.4f}, max={d_diagonal.max():.4f}, mean={d_diagonal.mean():.4f}")
    
    # Run a quick experiment with the uniform strategy
    print("\nRunning quick test with uniform strategy...")
    d_diagonal_uniform = d_results[0][0]  # First strategy (sorted by c)
    
    df = run_distributed_optimization_experiment(
        topology=topology,
        n=n,
        matrix_seed=matrix_seed,
        lr_basic=0.007,
        strategy="custom",
        d_diagonal=d_diagonal_uniform,
        dataset_name="MNIST",
        batch_size=128,
        num_epochs=10,  # Just 10 epochs for testing
        alpha=1000,
        use_hetero=True,
        repetitions=1,
        remark="test_custom_uniform",
        device="cuda:0" if sys.platform != "darwin" else "cpu",  # Use CPU on Mac
        output_dir="./test_custom_output"
    )
    
    print(f"\nTest completed. Final gradient norm: {df['grad_norm'].iloc[-1]:.6f}")


def example_2_varying_c():
    """Example 2: Generate D matrices with increasing c values."""
    print("\n" + "="*60)
    print("Example 2: Generating D matrices with varying c values")
    print("="*60)
    
    # Configuration
    n = 8  # Smaller network for demonstration
    topology = "ring"
    matrix_seed = 123
    
    # Generate topology
    A, B = generate_topology_matrices(topology, n, matrix_seed)
    
    # Compute Perron vectors
    pi_a = get_left_perron(A)
    pi_b = get_right_perron(B)
    
    print(f"\nTopology: {topology}, n={n}")
    print(f"pi_a: {pi_a}")
    print(f"pi_b: {pi_b}")
    
    # Generate 5 D matrices with increasing c values
    d_results = generate_d_matrices_with_increasing_c(A, B, num_c_values=5)
    
    print(f"\nGenerated {len(d_results)} D matrices with increasing c:")
    for i, (d_diagonal, c_value) in enumerate(d_results):
        print(f"\n  {i+1}. c = {c_value:.6f}")
        print(f"     D diagonal: {d_diagonal}")
        
        # Verify c computation
        c_verified = compute_c_from_d_diagonal(d_diagonal, pi_a, pi_b, n)
        print(f"     Verified c: {c_verified:.6f}")


def example_3_direct_usage():
    """Example 3: Direct usage of custom strategy with specific D diagonal."""
    print("\n" + "="*60)
    print("Example 3: Direct usage with specific D diagonal")
    print("="*60)
    
    # Configuration
    n = 4
    
    # Create a simple custom D diagonal
    # Let's create one that emphasizes certain nodes
    d_diagonal = np.array([2.0, 1.0, 0.5, 0.5])  # Must sum to n=4
    
    print(f"Custom D diagonal: {d_diagonal}")
    print(f"Sum: {d_diagonal.sum()} (should be {n})")
    
    # Generate a simple topology
    A, B = generate_topology_matrices("ring", n, seed=42)
    
    # Compute c value
    pi_a = get_left_perron(A)
    pi_b = get_right_perron(B)
    c_value = compute_c_from_d_diagonal(d_diagonal, pi_a, pi_b, n)
    
    print(f"\nComputed c value: {c_value:.6f}")
    print(f"pi_a: {pi_a}")
    print(f"pi_b: {pi_b}")


if __name__ == "__main__":
    # Run examples
    print("Custom Strategy Examples")
    print("========================")
    
    # Example 1: Test specific strategies
    example_1_specific_strategies()
    
    # Example 2: Generate varying c values
    example_2_varying_c()
    
    # Example 3: Direct usage
    example_3_direct_usage()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)