#!/usr/bin/env python3
"""
Main runner script for distributed optimization experiments.
Can be called from command line or imported as a module.
"""

import os
import sys
import argparse
import numpy as np
import torch
import pandas as pd
import yaml
from typing import Optional
from datetime import datetime

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training.training_track_grad_norm_different_learning_rate import train_track_grad_norm_with_hetero_different_learning_rate
from utils.algebra_utils import compute_kappa_row, compute_kappa_col, compute_beta_row, compute_beta_col
from .experiment_utils import (
    generate_topology_matrices,
    compute_learning_rates,
    compute_c_value,
    average_gradient_norm_results
)


def run_distributed_optimization_experiment(
    # Required parameters (no defaults)
    topology: str,
    n: int,
    matrix_seed: int,
    lr_basic: float,
    strategy: str,
    dataset_name: str,
    batch_size: int,
    num_epochs: int,
    alpha: float,
    use_hetero: bool,
    
    # Optional parameters (with defaults)
    random_seed: Optional[int] = None,
    repetitions: int = 1,
    remark: str = "",
    device: str = "cuda:0",
    output_dir: str = "./experiments",
    d_diagonal: Optional[np.ndarray] = None,
    
    # Additional topology parameters
    **kwargs
) -> pd.DataFrame:
    """
    Run distributed optimization experiment with specified configuration.
    
    Args:
        topology: Network topology type ("exp", "grid", "ring", "random", "geometric", "neighbor")
        n: Number of nodes
        matrix_seed: Seed for topology generation
        lr_basic: Base learning rate (total will be lr_basic * n)
        strategy: Learning rate strategy ("uniform", "pi_a_inverse", "pi_b_inverse", "random", "custom")
        random_seed: Random seed for "random" strategy
        dataset_name: Dataset to use ("MNIST" or "CIFAR10")
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        alpha: Heterogeneity parameter (higher = more uniform)
        use_hetero: Enable heterogeneous data distribution
        repetitions: Number of repetitions for averaging
        remark: Experiment identifier
        device: GPU device
        output_dir: Base directory for saving results
        d_diagonal: Diagonal values for "custom" strategy
        **kwargs: Additional parameters for specific topologies
        
    Returns:
        DataFrame containing averaged results across repetitions
    """
    # Create experiment-specific output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"experiment_{timestamp}_{topology}_{strategy}_{n}nodes"
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Generate topology matrices
    print(f"\n=== Generating {topology} topology with n={n}, seed={matrix_seed} ===")
    A, B = generate_topology_matrices(topology, n, matrix_seed, **kwargs)
    
    # Compute and display topology properties
    kappa_a = compute_kappa_row(A)
    kappa_b = compute_kappa_col(B)
    beta_a = compute_beta_row(A)
    beta_b = compute_beta_col(B)
    print(f"Topology properties:")
    print(f"  Matrix A: kappa={kappa_a:.4f}, beta={beta_a:.4f}")
    print(f"  Matrix B: kappa={kappa_b:.4f}, beta={beta_b:.4f}")
    
    # Compute learning rates
    print(f"\n=== Computing learning rates with strategy '{strategy}' ===")
    lr_list = compute_learning_rates(strategy, A, B, lr_basic, n, random_seed, d_diagonal)
    lr_total = sum(lr_list)
    print(f"Learning rate distribution:")
    print(f"  Total: {lr_total:.6f} (expected: {lr_basic * n:.6f})")
    print(f"  Min: {min(lr_list):.6f}, Max: {max(lr_list):.6f}")
    print(f"  Mean: {np.mean(lr_list):.6f}, Std: {np.std(lr_list):.6f}")
    
    # Compute c value
    c_value = compute_c_value(A, B, lr_list, lr_basic)
    print(f"\nTheoretical convergence factor c = {c_value:.6f}")
    
    # Convert matrices to torch tensors
    # A_tensor = torch.tensor(A, dtype=torch.float32)
    # B_tensor = torch.tensor(B, dtype=torch.float32)

    # Create experiment configuration
    config = {
        "experiment_info": {
            "timestamp": timestamp,
            "experiment_name": exp_name,
            "output_directory": exp_dir
        },
        "topology_parameters": {
            "topology": topology,
            "n": n,
            "matrix_seed": matrix_seed,
            "kappa_a": float(kappa_a),
            "kappa_b": float(kappa_b),
            "beta_a": float(beta_a),
            "beta_b": float(beta_b)
        },
        "learning_rate_parameters": {
            "lr_basic": float(lr_basic),
            "strategy": strategy,
            "random_seed": random_seed,
            "lr_total": float(lr_total),
            "c_value": float(c_value),
            "lr_distribution": {
                "min": float(min(lr_list)),
                "max": float(max(lr_list)),
                "mean": float(np.mean(lr_list)),
                "std": float(np.std(lr_list)),
                "values": [float(lr) for lr in lr_list]
            }
        },
        "training_parameters": {
            "dataset_name": dataset_name,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "alpha": float(alpha),
            "use_hetero": use_hetero,
            "algorithm": "PushPull"
        },
        "experiment_parameters": {
            "repetitions": repetitions,
            "remark": remark,
            "device": device,
            "training_seeds": [42 + i for i in range(repetitions)]
        },
        "generated_files": {
            "config_file": "config.yaml",
            "averaged_grad_norm": "averaged_grad_norm.csv" if repetitions > 1 else None,
            "individual_runs": f"{repetitions} pairs of CSV files (loss and grad_norm)"
        }
    }
    
    # Save configuration file
    config_path = os.path.join(exp_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Saved experiment configuration to: {config_path}")
    
    # Run multiple repetitions
    print(f"\n=== Running {repetitions} repetition(s) ===")
    grad_norm_dfs = []
    
    for rep in range(repetitions):
        print(f"\nRepetition {rep + 1}/{repetitions}")
        
        # Use different seed for each repetition
        training_seed = 42 + rep  # Base seed + repetition index
        
        # Run training
        df = train_track_grad_norm_with_hetero_different_learning_rate(
            algorithm="PushPull",
            lr_list=lr_list,
            A=A,
            B=B,
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_epochs=num_epochs,
            remark=remark,
            alpha=alpha,
            root=exp_dir,  # Use experiment-specific directory
            use_hetero=use_hetero,
            device=device,
            seed=training_seed
        )
        
        # The function returns gradient norm data
        grad_norm_dfs.append(df)
        
        # Clear GPU memory if needed
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    
    # Average results across repetitions
    print(f"\n=== Averaging results ===")
    if len(grad_norm_dfs) > 1:
        avg_df = average_gradient_norm_results(grad_norm_dfs)
        # Save averaged results
        avg_output_path = os.path.join(exp_dir, "averaged_grad_norm.csv")
        avg_df.to_csv(avg_output_path, index=False)
        print(f"Saved averaged results to: {avg_output_path}")
    else:
        avg_df = grad_norm_dfs[0]
    
    # Create experiment configuration
    # config = {
    #     "experiment_info": {
    #         "timestamp": timestamp,
    #         "experiment_name": exp_name,
    #         "output_directory": exp_dir
    #     },
    #     "topology_parameters": {
    #         "topology": topology,
    #         "n": n,
    #         "matrix_seed": matrix_seed,
    #         "kappa_a": float(kappa_a),
    #         "kappa_b": float(kappa_b),
    #         "beta_a": float(beta_a),
    #         "beta_b": float(beta_b)
    #     },
    #     "learning_rate_parameters": {
    #         "lr_basic": float(lr_basic),
    #         "strategy": strategy,
    #         "random_seed": random_seed,
    #         "lr_total": float(lr_total),
    #         "c_value": float(c_value),
    #         "lr_distribution": {
    #             "min": float(min(lr_list)),
    #             "max": float(max(lr_list)),
    #             "mean": float(np.mean(lr_list)),
    #             "std": float(np.std(lr_list)),
    #             "values": [float(lr) for lr in lr_list]
    #         }
    #     },
    #     "training_parameters": {
    #         "dataset_name": dataset_name,
    #         "batch_size": batch_size,
    #         "num_epochs": num_epochs,
    #         "alpha": float(alpha),
    #         "use_hetero": use_hetero,
    #         "algorithm": "PushPull"
    #     },
    #     "experiment_parameters": {
    #         "repetitions": repetitions,
    #         "remark": remark,
    #         "device": device,
    #         "training_seeds": [42 + i for i in range(repetitions)]
    #     },
    #     "generated_files": {
    #         "config_file": "config.yaml",
    #         "averaged_grad_norm": "averaged_grad_norm.csv" if repetitions > 1 else None,
    #         "individual_runs": f"{repetitions} pairs of CSV files (loss and grad_norm)"
    #     }
    # }
    
    # # Save configuration file
    # config_path = os.path.join(exp_dir, "config.yaml")
    # with open(config_path, 'w') as f:
    #     yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    # print(f"Saved experiment configuration to: {config_path}")
    
    print(f"\nExperiment completed successfully!")
    print(f"All results saved in: {exp_dir}")
    
    return avg_df


def main():
    """Command line interface for running experiments."""
    parser = argparse.ArgumentParser(description="Run distributed optimization experiment")
    
    # Topology parameters
    parser.add_argument("--topology", type=str, required=True,
                        choices=["exp", "grid", "ring", "random", "geometric", "neighbor"],
                        help="Network topology type")
    parser.add_argument("--n", type=int, required=True, help="Number of nodes")
    parser.add_argument("--matrix_seed", type=int, required=True, help="Seed for topology generation")
    
    # Learning rate parameters
    parser.add_argument("--lr_basic", type=float, required=True, help="Base learning rate")
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["uniform", "pi_a_inverse", "pi_b_inverse", "random", "custom"],
                        help="Learning rate strategy")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed for 'random' strategy")
    
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
                        help="Number of repetitions for averaging")
    parser.add_argument("--remark", type=str, default="",
                        help="Experiment identifier")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="GPU device")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                        help="Output directory")
    
    # Additional topology parameters
    parser.add_argument("--k", type=int, default=3,
                        help="Number of neighbors for 'neighbor' topology")
    
    args = parser.parse_args()
    
    # Run experiment
    run_distributed_optimization_experiment(
        topology=args.topology,
        n=args.n,
        matrix_seed=args.matrix_seed,
        lr_basic=args.lr_basic,
        strategy=args.strategy,
        random_seed=args.random_seed,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        alpha=args.alpha,
        use_hetero=args.use_hetero,
        repetitions=args.repetitions,
        remark=args.remark,
        device=args.device,
        output_dir=args.output_dir,
        k=args.k
    )


if __name__ == "__main__":
    main()