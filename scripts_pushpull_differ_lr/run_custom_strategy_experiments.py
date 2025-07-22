#!/usr/bin/env python3
"""
Standalone script for running experiments with custom learning rate strategies
that produce increasing convergence factor c values.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from run_experiment import run_distributed_optimization_experiment
from experiment_utils import generate_topology_matrices
from utils.d_matrix_utils import (
    generate_d_matrices_with_increasing_c,
    generate_specific_d_matrices,
    compute_c_from_d_diagonal
)
from utils.algebra_utils import get_left_perron, get_right_perron


def run_custom_strategy_experiments(
    topology: str,
    n: int,
    matrix_seed: int,
    lr_basic: float,
    num_c_values: int,
    dataset_name: str,
    batch_size: int,
    num_epochs: int,
    alpha: float,
    use_hetero: bool,
    c_min: float = None,
    c_max: float = None,
    repetitions: int = 1,
    device: str = "cuda:0",
    output_dir: str = "./custom_experiments",
    use_specific_strategies: bool = False,
    **topology_kwargs
) -> Dict[str, Any]:
    """
    Run experiments with custom D matrices that produce varying c values.
    
    Args:
        topology: Network topology type
        n: Number of nodes
        matrix_seed: Seed for topology generation
        lr_basic: Base learning rate
        num_c_values: Number of different c values to test
        dataset_name: Dataset to use
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        alpha: Heterogeneity parameter
        use_hetero: Enable heterogeneous data distribution
        c_min: Minimum c value (auto-determined if None)
        c_max: Maximum c value (auto-determined if None)
        repetitions: Number of repetitions per c value
        device: GPU device
        output_dir: Base directory for saving results
        use_specific_strategies: If True, test specific strategies instead of c range
        **topology_kwargs: Additional topology parameters
        
    Returns:
        Dictionary containing experiment results and metadata
    """
    # Create experiment-specific output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"custom_strategy_{timestamp}_{topology}_{n}nodes"
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Generate topology matrices
    print(f"\n=== Generating {topology} topology with n={n}, seed={matrix_seed} ===")
    A, B = generate_topology_matrices(topology, n, matrix_seed, **topology_kwargs)
    
    # Compute Perron vectors
    pi_a = get_left_perron(A)
    pi_b = get_right_perron(B)
    
    # Generate D matrices with different c values
    if use_specific_strategies:
        print("\n=== Generating D matrices for specific strategies ===")
        d_matrix_results = generate_specific_d_matrices(A, B)
        experiments = [(d, c, name) for d, c, name in d_matrix_results]
    else:
        print(f"\n=== Generating {num_c_values} D matrices with increasing c values ===")
        d_matrix_results = generate_d_matrices_with_increasing_c(
            A, B, num_c_values, c_min, c_max
        )
        experiments = [(d, c, f"c={c:.4f}") for d, c in d_matrix_results]
    
    # Print c value range
    c_values = [exp[1] for exp in experiments]
    print(f"Generated c values: min={min(c_values):.4f}, max={max(c_values):.4f}")
    for i, (_, c, name) in enumerate(experiments):
        print(f"  {i+1}. {name}: c = {c:.6f}")
    
    # Run experiments for each D matrix
    results = []
    
    for i, (d_diagonal, c_value, exp_name_suffix) in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"Running experiment {i+1}/{len(experiments)}: {exp_name_suffix}")
        print(f"{'='*60}")
        
        # Create remark for this specific c value
        remark = f"custom_{exp_name_suffix}"
        
        # Run experiment with custom strategy
        try:
            df = run_distributed_optimization_experiment(
                topology=topology,
                n=n,
                matrix_seed=matrix_seed,
                lr_basic=lr_basic,
                strategy="custom",
                dataset_name=dataset_name,
                batch_size=batch_size,
                num_epochs=num_epochs,
                alpha=alpha,
                use_hetero=use_hetero,
                repetitions=repetitions,
                remark=remark,
                device=device,
                output_dir=exp_dir,
                d_diagonal=d_diagonal,
                **topology_kwargs
            )
            
            # Store results
            results.append({
                "c_value": c_value,
                "experiment_name": exp_name_suffix,
                "d_diagonal": d_diagonal.tolist(),
                "dataframe": df,
                "final_grad_norm": df["grad_norm"].iloc[-1] if "grad_norm" in df.columns else None
            })
            
        except Exception as e:
            print(f"Error in experiment {exp_name_suffix}: {str(e)}")
            results.append({
                "c_value": c_value,
                "experiment_name": exp_name_suffix,
                "d_diagonal": d_diagonal.tolist(),
                "error": str(e)
            })
    
    # Create summary
    summary = {
        "experiment_info": {
            "timestamp": timestamp,
            "output_directory": exp_dir,
            "topology": topology,
            "n": n,
            "matrix_seed": matrix_seed
        },
        "c_values": {
            "count": len(experiments),
            "min": min(c_values),
            "max": max(c_values),
            "values": c_values
        },
        "training_parameters": {
            "lr_basic": lr_basic,
            "dataset_name": dataset_name,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "alpha": alpha,
            "use_hetero": use_hetero,
            "repetitions": repetitions
        },
        "results_summary": []
    }
    
    # Add results summary
    for result in results:
        if "error" not in result:
            summary["results_summary"].append({
                "c_value": result["c_value"],
                "experiment_name": result["experiment_name"],
                "final_grad_norm": result["final_grad_norm"]
            })
    
    # Save summary
    import json
    summary_path = os.path.join(exp_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== All experiments completed ===")
    print(f"Results saved in: {exp_dir}")
    print(f"Summary saved to: {summary_path}")
    
    # Create comparison plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        # Plot final gradient norms vs c values
        valid_results = [r for r in results if "error" not in r and r["final_grad_norm"] is not None]
        if valid_results:
            c_vals = [r["c_value"] for r in valid_results]
            grad_norms = [r["final_grad_norm"] for r in valid_results]
            
            plt.figure(figsize=(10, 6))
            plt.scatter(c_vals, grad_norms, s=100)
            plt.xlabel("Convergence Factor c", fontsize=12)
            plt.ylabel("Final Gradient Norm", fontsize=12)
            plt.title(f"Final Gradient Norm vs Convergence Factor c\n{topology} topology, n={n}", fontsize=14)
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            
            # Add labels for each point
            for r in valid_results:
                plt.annotate(r["experiment_name"], 
                           (r["c_value"], r["final_grad_norm"]),
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=8)
            
            plot_path = os.path.join(exp_dir, "c_vs_grad_norm.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Comparison plot saved to: {plot_path}")
            plt.close()
            
    except ImportError:
        print("Matplotlib not available, skipping plot generation")
    
    return {
        "summary": summary,
        "results": results,
        "exp_dir": exp_dir
    }


def main():
    """Command line interface for custom strategy experiments."""
    parser = argparse.ArgumentParser(
        description="Run distributed optimization experiments with custom strategies"
    )
    
    # Topology parameters
    parser.add_argument("--topology", type=str, required=True,
                        choices=["exp", "grid", "ring", "random", "geometric", "neighbor"],
                        help="Network topology type")
    parser.add_argument("--n", type=int, required=True, help="Number of nodes")
    parser.add_argument("--matrix_seed", type=int, required=True, 
                        help="Seed for topology generation")
    
    # Learning rate parameters
    parser.add_argument("--lr_basic", type=float, required=True, 
                        help="Base learning rate")
    parser.add_argument("--num_c_values", type=int, default=5,
                        help="Number of different c values to test")
    parser.add_argument("--c_min", type=float, default=None,
                        help="Minimum c value (auto if not specified)")
    parser.add_argument("--c_max", type=float, default=None,
                        help="Maximum c value (auto if not specified)")
    parser.add_argument("--use_specific_strategies", action="store_true",
                        help="Test specific strategies instead of c range")
    
    # Training parameters
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=["MNIST", "CIFAR10"], help="Dataset to use")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--alpha", type=float, required=True,
                        help="Heterogeneity parameter (higher = more uniform)")
    parser.add_argument("--use_hetero", action="store_true",
                        help="Enable heterogeneous data distribution")
    
    # Experiment parameters
    parser.add_argument("--repetitions", type=int, default=1,
                        help="Number of repetitions per c value")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="GPU device")
    parser.add_argument("--output_dir", type=str, default="./custom_experiments",
                        help="Output directory")
    
    # Additional topology parameters
    parser.add_argument("--k", type=int, default=3,
                        help="Number of neighbors for 'neighbor' topology")
    
    args = parser.parse_args()
    
    # Run experiments
    run_custom_strategy_experiments(
        topology=args.topology,
        n=args.n,
        matrix_seed=args.matrix_seed,
        lr_basic=args.lr_basic,
        num_c_values=args.num_c_values,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        alpha=args.alpha,
        use_hetero=args.use_hetero,
        c_min=args.c_min,
        c_max=args.c_max,
        repetitions=args.repetitions,
        device=args.device,
        output_dir=args.output_dir,
        use_specific_strategies=args.use_specific_strategies,
        k=args.k
    )


if __name__ == "__main__":
    main()