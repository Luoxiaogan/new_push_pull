#!/usr/bin/env python3
"""
Example script demonstrating how to use custom D matrices with varying c values in experiments.

This script shows:
1. How to generate D matrices with increasing c values
2. How to run experiments using these D matrices
3. How to analyze results based on convergence factor c

The workflow:
1. Generate D matrices using the theoretical simplex method
2. Run experiments with each D matrix using the "custom" strategy
3. Analyze how convergence behavior changes with c
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from run_experiment import run_distributed_optimization_experiment
from generate_custom_d_matrices import (
    generate_custom_d_matrices,
    analyze_c_range,
    save_d_matrices,
    load_d_matrices,
    visualize_c_distribution
)
from experiment_utils import generate_topology_matrices


def run_example_experiment():
    """
    Example: Run experiments with custom D matrices on a nearest neighbor topology.
    """
    # Configuration
    topology = "neighbor"
    n = 16
    matrix_seed = 42
    lr_basic = 7e-3
    num_c_values = 5  # Generate 5 different c values
    
    # Training parameters
    dataset_name = "MNIST"
    batch_size = 128
    num_epochs = 50  # Reduced for example
    alpha = 1000
    use_hetero = True
    device = "cuda:0"
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./example_custom_d_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Custom D Matrix Experiment Example")
    print("=" * 80)
    
    # Step 1: Generate topology
    print(f"\n1. Generating {topology} topology with n={n}")
    A, B = generate_topology_matrices(topology, n, matrix_seed, k=3)
    
    # Step 2: Analyze c range
    print("\n2. Analyzing theoretical c value range")
    analysis = analyze_c_range(A, B)
    print(f"   c_min = {analysis['c_min']:.6f} (at node {analysis['min_node']})")
    print(f"   c_max = {analysis['c_max']:.6f} (at node {analysis['max_node']})")
    print(f"   c_ratio = {analysis['c_ratio']:.2f}")
    
    # Step 3: Generate D matrices with uniform c distribution
    print(f"\n3. Generating {num_c_values} D matrices with uniform c distribution")
    d_matrices = generate_custom_d_matrices(A, B, num_c_values, distribution="uniform")
    
    print("\n   Generated c values:")
    for i, (_, c) in enumerate(d_matrices):
        print(f"   {i+1}. c = {c:.6f}")
    
    # Save D matrices for reproducibility
    d_matrix_file = os.path.join(output_dir, "d_matrices.json")
    topology_info = {"type": topology, "n": n, "matrix_seed": matrix_seed, "k": 3}
    save_d_matrices(d_matrices, analysis, topology_info, d_matrix_file)
    
    # Step 4: Run experiments
    print(f"\n4. Running experiments with each D matrix")
    results = []
    
    for i, (d_diagonal, c_value) in enumerate(d_matrices):
        print(f"\n   Experiment {i+1}/{num_c_values}: c = {c_value:.6f}")
        
        # Run experiment with custom D matrix
        exp_remark = f"c_{i+1:02d}_value_{c_value:.4f}"
        
        try:
            df = run_distributed_optimization_experiment(
                topology=topology,
                n=n,
                matrix_seed=matrix_seed,
                lr_basic=lr_basic,
                strategy="custom",  # Use custom strategy
                d_diagonal=d_diagonal,  # Pass the D diagonal
                dataset_name=dataset_name,
                batch_size=batch_size,
                num_epochs=num_epochs,
                alpha=alpha,
                use_hetero=use_hetero,
                repetitions=1,
                remark=exp_remark,
                device=device,
                output_dir=output_dir,
                k=3
            )
            
            # Extract key metrics
            final_grad_norm = df["grad_norm"].iloc[-1] if "grad_norm" in df.columns else None
            final_loss = df["loss"].iloc[-1] if "loss" in df.columns else None
            
            results.append({
                "index": i,
                "c_value": c_value,
                "final_grad_norm": final_grad_norm,
                "final_loss": final_loss,
                "dataframe": df
            })
            
            print(f"   Final gradient norm: {final_grad_norm:.6e}")
            
        except Exception as e:
            print(f"   Error in experiment: {str(e)}")
            results.append({
                "index": i,
                "c_value": c_value,
                "error": str(e)
            })
    
    # Step 5: Analyze results
    print("\n5. Analyzing results")
    
    # Create summary DataFrame
    summary_data = []
    for r in results:
        if "error" not in r:
            summary_data.append({
                "c_value": r["c_value"],
                "final_grad_norm": r["final_grad_norm"],
                "final_loss": r["final_loss"]
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values("c_value")
        
        print("\n   Summary of results:")
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_file = os.path.join(output_dir, "experiment_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"\n   Summary saved to: {summary_file}")
        
        # Create plots if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 1: Final gradient norm vs c
            ax1.scatter(summary_df["c_value"], summary_df["final_grad_norm"], s=100)
            ax1.set_xlabel("Convergence Factor c")
            ax1.set_ylabel("Final Gradient Norm")
            ax1.set_title("Final Gradient Norm vs c")
            ax1.set_yscale("log")
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Convergence curves
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Gradient Norm")
            ax2.set_title("Convergence Curves for Different c Values")
            ax2.set_yscale("log")
            
            # Plot convergence curves
            for r in results:
                if "error" not in r and "dataframe" in r:
                    df = r["dataframe"]
                    if "grad_norm" in df.columns:
                        epochs = range(len(df))
                        ax2.plot(epochs, df["grad_norm"], 
                               label=f"c = {r['c_value']:.4f}", 
                               linewidth=2)
            
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = os.path.join(output_dir, "convergence_analysis.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"\n   Plots saved to: {plot_file}")
            plt.close()
            
        except ImportError:
            print("\n   Matplotlib not available, skipping plots")
    
    print("\n" + "=" * 80)
    print("Example experiment completed!")
    print(f"All results saved in: {output_dir}")
    print("=" * 80)


def example_with_custom_c_targets():
    """
    Example: Generate D matrices for specific target c values.
    """
    print("\n" + "=" * 80)
    print("Example: Custom Target c Values")
    print("=" * 80)
    
    # Setup
    topology = "ring"
    n = 8
    matrix_seed = 123
    
    # Generate topology
    A, B = generate_topology_matrices(topology, n, matrix_seed)
    
    # Analyze range
    analysis = analyze_c_range(A, B)
    c_min = analysis['c_min']
    c_max = analysis['c_max']
    
    print(f"\nFor {topology} topology with n={n}:")
    print(f"c_min = {c_min:.6f}, c_max = {c_max:.6f}")
    
    # Define custom target c values
    # For example: c_min, 25%, 50%, 75%, c_max
    c_targets = [
        c_min,
        c_min + 0.25 * (c_max - c_min),
        c_min + 0.50 * (c_max - c_min),
        c_min + 0.75 * (c_max - c_min),
        c_max
    ]
    
    print("\nTarget c values:")
    for i, c in enumerate(c_targets):
        print(f"  {i+1}. c = {c:.6f} ({(c-c_min)/(c_max-c_min)*100:.1f}% of range)")
    
    # Generate D matrices
    d_matrices = generate_custom_d_matrices(A, B, len(c_targets), 
                                          distribution="custom", 
                                          c_targets=c_targets)
    
    print("\nActual generated c values:")
    for i, (_, c) in enumerate(d_matrices):
        error = abs(c - c_targets[i])
        print(f"  {i+1}. c = {c:.6f} (error = {error:.2e})")


def example_load_and_use():
    """
    Example: Load previously saved D matrices and use them.
    """
    print("\n" + "=" * 80)
    print("Example: Loading and Using Saved D Matrices")
    print("=" * 80)
    
    # First, generate and save some D matrices
    topology = "grid"
    n = 16  # 4x4 grid
    matrix_seed = 456
    
    # Generate topology
    A, B = generate_topology_matrices(topology, n, matrix_seed)
    analysis = analyze_c_range(A, B)
    
    # Generate D matrices with log distribution
    d_matrices = generate_custom_d_matrices(A, B, 7, distribution="log")
    
    # Save them
    output_dir = "./temp_d_matrices"
    os.makedirs(output_dir, exist_ok=True)
    save_file = os.path.join(output_dir, "example_d_matrices.json")
    
    topology_info = {"type": topology, "n": n, "matrix_seed": matrix_seed}
    save_d_matrices(d_matrices, analysis, topology_info, save_file)
    
    # Now demonstrate loading
    print(f"\nLoading D matrices from: {save_file}")
    loaded_d_matrices, metadata = load_d_matrices(save_file)
    
    print(f"\nLoaded {len(loaded_d_matrices)} D matrices")
    print(f"Topology: {metadata['metadata']['topology']['type']}")
    print(f"Number of nodes: {metadata['metadata']['topology']['n']}")
    print(f"c_min: {metadata['analysis']['c_min']:.6f}")
    print(f"c_max: {metadata['analysis']['c_max']:.6f}")
    
    print("\nLoaded c values:")
    for i, (_, c) in enumerate(loaded_d_matrices):
        print(f"  {i+1}. c = {c:.6f}")
    
    # Clean up
    import shutil
    shutil.rmtree(output_dir)


if __name__ == "__main__":
    # Run the main example
    run_example_experiment()
    
    # Run additional examples
    example_with_custom_c_targets()
    example_load_and_use()