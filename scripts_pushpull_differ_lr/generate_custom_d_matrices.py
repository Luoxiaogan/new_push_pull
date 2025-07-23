#!/usr/bin/env python3
"""
Standalone script for generating custom D matrices with varying convergence factor c values.

This script uses the theoretical simplex method to:
1. Find the exact range of possible c values [c_min, c_max]
2. Generate D matrices that produce monotonically increasing c values
3. Save the results for use in experiments

The theoretical basis:
- c = n * pi_A^T * D * pi_B where D is diagonal with d_i > 0 and sum(d_i) = n
- The constraint defines a simplex in n-dimensional space
- Linear functions on simplices achieve extrema at vertices
- Vertices are points where one d_i = n and others = 0
- This gives us exact c_min and c_max without sampling
"""

import os
import sys
import argparse
import numpy as np
import json
from datetime import datetime
from typing import List, Tuple, Dict

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
    generate_d_from_convex_combination
)


def analyze_c_range(A: np.ndarray, B: np.ndarray) -> Dict[str, any]:
    """
    Analyze the theoretical range of c values for given matrices A and B.
    
    This function computes all possible extreme c values and identifies
    which nodes correspond to c_min and c_max.
    
    Args:
        A: Row-stochastic matrix
        B: Column-stochastic matrix
        
    Returns:
        Dictionary containing analysis results
    """
    n = A.shape[0]
    pi_a = get_left_perron(A)
    pi_b = get_right_perron(B)
    
    # Compute c values at all vertices
    c_values = []
    for j in range(n):
        # At vertex j, d_j = n and d_i = 0 for i != j
        c_j = n * n * pi_a[j] * pi_b[j]
        c_values.append({
            'node': j,
            'c_value': c_j,
            'pi_a': pi_a[j],
            'pi_b': pi_b[j],
            'product': pi_a[j] * pi_b[j]
        })
    
    # Sort by c value
    c_values.sort(key=lambda x: x['c_value'])
    
    analysis = {
        'n': n,
        'c_min': c_values[0]['c_value'],
        'c_max': c_values[-1]['c_value'],
        'c_range': c_values[-1]['c_value'] - c_values[0]['c_value'],
        'c_ratio': c_values[-1]['c_value'] / c_values[0]['c_value'],
        'min_node': c_values[0]['node'],
        'max_node': c_values[-1]['node'],
        'all_vertices': c_values,
        'pi_a': pi_a.tolist(),
        'pi_b': pi_b.tolist()
    }
    
    return analysis


def generate_custom_d_matrices(
    A: np.ndarray,
    B: np.ndarray,
    num_c_values: int,
    distribution: str = "uniform",
    c_targets: List[float] = None
) -> List[Tuple[np.ndarray, float]]:
    """
    Generate custom D matrices with specified c value distribution.
    
    Args:
        A: Row-stochastic matrix
        B: Column-stochastic matrix
        num_c_values: Number of c values to generate
        distribution: Distribution mode
            - "uniform": Uniformly spaced c values between c_min and c_max
            - "vertices": Use actual simplex vertices (may be non-uniform)
            - "log": Logarithmically spaced c values
            - "custom": Use provided c_targets
        c_targets: Custom target c values (only used if distribution="custom")
        
    Returns:
        List of (d_diagonal, c_value) tuples sorted by increasing c
    """
    n = A.shape[0]
    pi_a = get_left_perron(A)
    pi_b = get_right_perron(B)
    
    # Get analysis for c range
    analysis = analyze_c_range(A, B)
    c_min = analysis['c_min']
    c_max = analysis['c_max']
    
    if distribution == "vertices":
        # Use theoretical method with vertices mode
        return generate_d_matrices_theoretical(A, B, num_c_values, distribution="vertices")
    
    elif distribution == "uniform":
        # Uniformly spaced c values
        return generate_d_matrices_theoretical(A, B, num_c_values, distribution="uniform")
    
    elif distribution == "log":
        # Logarithmically spaced c values
        # This is useful when c_max >> c_min
        if c_min <= 0:
            raise ValueError("Cannot use log spacing with non-positive c_min")
        
        log_c_min = np.log(c_min)
        log_c_max = np.log(c_max)
        log_c_values = np.linspace(log_c_min, log_c_max, num_c_values)
        c_values = np.exp(log_c_values)
        
        # Find d vectors for these c values using interpolation
        results = []
        
        # Get min and max d vectors
        min_node = analysis['min_node']
        max_node = analysis['max_node']
        d_min = np.zeros(n)
        d_min[min_node] = n
        d_max = np.zeros(n)
        d_max[max_node] = n
        
        for target_c in c_values:
            # Find alpha such that c(alpha) = target_c
            # Since c is linear in alpha: c(alpha) = (1-alpha)*c_min + alpha*c_max
            alpha = (target_c - c_min) / (c_max - c_min)
            alpha = np.clip(alpha, 0, 1)  # Ensure alpha is in [0, 1]
            
            d_diagonal = generate_d_from_convex_combination(d_min, d_max, alpha, n)
            actual_c = compute_c_from_d_diagonal(d_diagonal, pi_a, pi_b, n)
            results.append((d_diagonal, actual_c))
        
        return results
    
    elif distribution == "custom":
        if c_targets is None:
            raise ValueError("c_targets must be provided when distribution='custom'")
        
        # Similar to log case, but with custom targets
        results = []
        
        # Get min and max d vectors
        min_node = analysis['min_node']
        max_node = analysis['max_node']
        d_min = np.zeros(n)
        d_min[min_node] = n
        d_max = np.zeros(n)
        d_max[max_node] = n
        
        for target_c in sorted(c_targets):
            if target_c < c_min or target_c > c_max:
                print(f"Warning: target c={target_c:.6f} is outside range [{c_min:.6f}, {c_max:.6f}], clipping")
                target_c = np.clip(target_c, c_min, c_max)
            
            alpha = (target_c - c_min) / (c_max - c_min)
            d_diagonal = generate_d_from_convex_combination(d_min, d_max, alpha, n)
            actual_c = compute_c_from_d_diagonal(d_diagonal, pi_a, pi_b, n)
            results.append((d_diagonal, actual_c))
        
        return results
    
    else:
        raise ValueError(f"Unknown distribution mode: {distribution}")


def save_d_matrices(
    d_matrices: List[Tuple[np.ndarray, float]],
    analysis: Dict[str, any],
    topology_info: Dict[str, any],
    output_path: str
):
    """
    Save D matrices and metadata to a JSON file.
    
    Args:
        d_matrices: List of (d_diagonal, c_value) tuples
        analysis: C range analysis results
        topology_info: Information about the topology
        output_path: Path to save the JSON file
    """
    data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "num_matrices": len(d_matrices),
            "topology": topology_info
        },
        "analysis": analysis,
        "d_matrices": []
    }
    
    for i, (d_diagonal, c_value) in enumerate(d_matrices):
        data["d_matrices"].append({
            "index": i,
            "c_value": float(c_value),
            "d_diagonal": d_diagonal.tolist()
        })
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(d_matrices)} D matrices to: {output_path}")


def load_d_matrices(input_path: str) -> Tuple[List[Tuple[np.ndarray, float]], Dict[str, any]]:
    """
    Load D matrices from a JSON file.
    
    Args:
        input_path: Path to the JSON file
        
    Returns:
        Tuple of (d_matrices, metadata)
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    d_matrices = []
    for item in data["d_matrices"]:
        d_diagonal = np.array(item["d_diagonal"])
        c_value = item["c_value"]
        d_matrices.append((d_diagonal, c_value))
    
    return d_matrices, data


def visualize_c_distribution(
    d_matrices: List[Tuple[np.ndarray, float]],
    analysis: Dict[str, any],
    output_path: str = None
):
    """
    Visualize the distribution of c values.
    
    Args:
        d_matrices: List of (d_diagonal, c_value) tuples
        analysis: C range analysis results
        output_path: Path to save the plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
        
        c_values = [c for _, c in d_matrices]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: C values distribution
        ax1.scatter(range(len(c_values)), c_values, s=50)
        ax1.axhline(y=analysis['c_min'], color='r', linestyle='--', label=f'c_min = {analysis["c_min"]:.4f}')
        ax1.axhline(y=analysis['c_max'], color='g', linestyle='--', label=f'c_max = {analysis["c_max"]:.4f}')
        ax1.set_xlabel('Matrix Index')
        ax1.set_ylabel('Convergence Factor c')
        ax1.set_title('Distribution of c Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Vertex c values
        vertices = analysis['all_vertices']
        node_indices = [v['node'] for v in vertices]
        vertex_c_values = [v['c_value'] for v in vertices]
        
        ax2.bar(node_indices, vertex_c_values)
        ax2.set_xlabel('Node Index')
        ax2.set_ylabel('c Value at Vertex')
        ax2.set_title('c Values at Simplex Vertices')
        ax2.grid(True, alpha=0.3)
        
        # Highlight min and max
        ax2.bar(analysis['min_node'], analysis['c_min'], color='red', label='c_min')
        ax2.bar(analysis['max_node'], analysis['c_max'], color='green', label='c_max')
        ax2.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {output_path}")
        else:
            plt.show()
            
    except ImportError:
        print("Matplotlib not available, skipping visualization")


def main():
    """Command line interface for generating custom D matrices."""
    parser = argparse.ArgumentParser(
        description="Generate custom D matrices with varying convergence factor c values"
    )
    
    # Topology parameters
    parser.add_argument("--topology", type=str, required=True,
                        choices=["exp", "grid", "ring", "random", "geometric", "neighbor"],
                        help="Network topology type")
    parser.add_argument("--n", type=int, required=True, help="Number of nodes")
    parser.add_argument("--matrix_seed", type=int, required=True,
                        help="Seed for topology generation")
    
    # D matrix generation parameters
    parser.add_argument("--num_c_values", type=int, default=10,
                        help="Number of c values to generate")
    parser.add_argument("--distribution", type=str, default="uniform",
                        choices=["uniform", "vertices", "log", "custom"],
                        help="Distribution mode for c values")
    parser.add_argument("--c_targets", type=float, nargs="+", default=None,
                        help="Custom target c values (only for distribution='custom')")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./d_matrices",
                        help="Output directory")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization plots")
    
    # Additional topology parameters
    parser.add_argument("--k", type=int, default=3,
                        help="Number of neighbors for 'neighbor' topology")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate topology matrices
    print(f"\n=== Generating {args.topology} topology with n={args.n}, seed={args.matrix_seed} ===")
    A, B = generate_topology_matrices(args.topology, args.n, args.matrix_seed, k=args.k)
    
    # Analyze c range
    print("\n=== Analyzing c value range ===")
    analysis = analyze_c_range(A, B)
    print(f"c_min = {analysis['c_min']:.6f} (at node {analysis['min_node']})")
    print(f"c_max = {analysis['c_max']:.6f} (at node {analysis['max_node']})")
    print(f"c_range = {analysis['c_range']:.6f}")
    print(f"c_ratio = {analysis['c_ratio']:.2f}")
    
    # Generate D matrices
    print(f"\n=== Generating {args.num_c_values} D matrices with {args.distribution} distribution ===")
    d_matrices = generate_custom_d_matrices(
        A, B, 
        args.num_c_values, 
        distribution=args.distribution,
        c_targets=args.c_targets
    )
    
    # Display generated c values
    print("\nGenerated c values:")
    for i, (_, c) in enumerate(d_matrices):
        print(f"  {i+1:2d}. c = {c:.6f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"d_matrices_{args.topology}_n{args.n}_{args.distribution}_{timestamp}.json"
    output_path = os.path.join(args.output_dir, filename)
    
    topology_info = {
        "type": args.topology,
        "n": args.n,
        "matrix_seed": args.matrix_seed,
        "additional_params": {"k": args.k} if args.topology == "neighbor" else {}
    }
    
    save_d_matrices(d_matrices, analysis, topology_info, output_path)
    
    # Visualize if requested
    if args.visualize:
        plot_filename = f"c_distribution_{args.topology}_n{args.n}_{args.distribution}_{timestamp}.png"
        plot_path = os.path.join(args.output_dir, plot_filename)
        visualize_c_distribution(d_matrices, analysis, plot_path)
    
    print(f"\n=== Generation complete ===")
    print(f"Results saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    main()