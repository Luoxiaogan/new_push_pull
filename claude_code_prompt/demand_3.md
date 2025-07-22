# Systematic Testing of Different Learning Rate Strategies

## Overview

This document specifies the requirements for implementing a clean interface to systematically test different learning rate strategies in distributed optimization experiments. The goal is to create a more user-friendly wrapper around the existing training functions while maintaining fair comparisons between different strategies.

## Design Principles

1. **Fair Comparison**: When comparing different learning rate strategies, maintain the total learning rate sum `lr_basic * n` constant across all strategies
2. **Modular Design**: Separate topology generation, learning rate strategy, and experiment execution
3. **Reproducibility**: Provide clear seed management for both network topology and random strategies
4. **Clean Output**: Organize results systematically with clear naming conventions

## API Specification

### Main Function

```python
def run_distributed_optimization_experiment(
    # Topology parameters
    topology: str,          # Options: "exp", "grid", "ring", "random", "geometric", "neighbor"
    n: int,                 # Number of nodes
    matrix_seed: int,       # Seed for topology generation
    
    # Learning rate parameters
    lr_basic: float,        # Base learning rate (total will be lr_basic * n)
    strategy: str,          # Options: "uniform", "pi_a_inverse", "pi_b_inverse", "random"
    random_seed: int = None,  # Only used when strategy="random"
    
    # Training parameters
    dataset_name: str,      # "MNIST" or "CIFAR10"
    batch_size: int,
    num_epochs: int,
    alpha: float,           # Heterogeneity parameter (higher = more uniform)
    use_hetero: bool,       # Enable heterogeneous data distribution
    
    # Experiment parameters
    repetitions: int = 1,   # Number of repetitions for averaging
    remark: str = "",       # Experiment identifier
    device: str = "cuda:0", # GPU device
    
    # Output parameters
    output_dir: str,        # Base directory for saving results
) -> pd.DataFrame:
    """
    Run distributed optimization experiment with specified configuration.
    
    Returns:
        DataFrame containing averaged results across repetitions
    """
```

## Parameter Details

### Topology Options

The `topology` parameter supports 6 predefined network structures:
- `"exp"`: Exponential graph (uses `get_matrixs_from_exp_graph`)
- `"grid"`: Grid topology (uses `generate_grid_matrices`)
- `"ring"`: Ring topology with shortcuts (uses `generate_ring_matrices`)
- `"random"`: Random graph with p=1/3 edge probability (uses `generate_random_graph_matrices`)
- `"geometric"`: Stochastic geometric graph (uses `generate_stochastic_geometric_matrices`)
- `"neighbor"`: k-nearest neighbor graph (uses `generate_nearest_neighbor_matrices`)

### Learning Rate Strategy Options

The `strategy` parameter determines how the total learning rate is distributed across nodes:

1. **`"uniform"`**: All nodes use the same learning rate
   - `lr_list = [lr_basic] * n`

2. **`"pi_a_inverse"`**: Learning rates proportional to inverse of left Perron vector of A
   - Compute `pi_a = get_left_perron(A)`
   - Set `D = diag(1/pi_a)`, normalize so trace(D) = n
   - `lr_list[i] = lr_basic * D[i,i]`

3. **`"pi_b_inverse"`**: Learning rates proportional to inverse of right Perron vector of B
   - Compute `pi_b = get_right_perron(B)`
   - Set `D = diag(1/pi_b)`, normalize so trace(D) = n
   - `lr_list[i] = lr_basic * D[i,i]`

4. **`"random"`**: Random distribution maintaining total sum
   - Uses `random_seed` for reproducibility
   - Generate random positive values, normalize to sum to n
   - `lr_list[i] = lr_basic * random_value[i]`

### Repetition Handling

To handle multiple repetitions for statistical averaging:
- Each repetition should use a different seed for the training function
- Suggested approach: `training_seed = base_seed + repetition_index`
- Average the gradient norm CSV results across repetitions
- Save both individual and averaged results

## Output Specifications

### File Naming Convention

Individual run outputs (from `train_track_grad_norm_with_hetero_different_learning_rate`):
- Loss CSV: `{remark}_hetero={use_hetero}, alpha={alpha}, {algorithm}, lr[0]={lr_list[0]}, n_nodes={n}, batch_size={batch_size}, {date}.csv`
- Gradient norm CSV: `{remark}_grad_norm,hetero={use_hetero},s alpha={alpha}, {algorithm}, lr[0]={lr_list[0]}, n_nodes={n}, batch_size={batch_size}, {date}.csv`

Averaged results (after multiple repetitions):
```
{output_dir}/averaged_results/topology={topology}_n={n}_strategy={strategy}_lr_total={lr_basic*n}_seed={matrix_seed}_reps={repetitions}.csv
```

### Computing c Value

The theoretical convergence factor c is defined as:
```
c = n * pi_A^T * D * pi_B
```

Where:
- `pi_A`: Left Perron vector of matrix A
- `pi_B`: Right Perron vector of matrix B
- `D`: Diagonal matrix representing learning rate multipliers
- `n`: Number of nodes

Implementation:
```python
def compute_c_value(A, B, lr_list, lr_basic):
    n = A.shape[0]
    pi_a = get_left_perron(A)
    pi_b = get_right_perron(B)
    
    # Construct D matrix from learning rates
    D = np.diag([lr / lr_basic for lr in lr_list])
    
    # Compute c = n * pi_A^T * D * pi_B
    c = n * pi_a.T @ D @ pi_b
    return c
```

## Example Usage

```python
# Example 1: Uniform learning rate on nearest neighbor topology
df = run_distributed_optimization_experiment(
    topology="neighbor",
    n=16,
    matrix_seed=42,
    lr_basic=0.007,
    strategy="uniform",
    dataset_name="MNIST",
    batch_size=128,
    num_epochs=100,
    alpha=1000,
    use_hetero=True,
    repetitions=5,
    remark="uniform_lr_test",
    device="cuda:0",
    output_dir="./experiments"
)

# Example 2: Pi_b inverse strategy on grid topology
df = run_distributed_optimization_experiment(
    topology="grid",
    n=16,
    matrix_seed=123,
    lr_basic=0.007,
    strategy="pi_b_inverse",
    dataset_name="CIFAR10",
    batch_size=64,
    num_epochs=200,
    alpha=100,
    use_hetero=True,
    repetitions=3,
    remark="pi_b_inverse_test",
    device="cuda:1",
    output_dir="./experiments"
)
```

## Implementation Notes

1. **Matrix Validation**: Ensure generated matrices A and B satisfy:
   - A is row-stochastic (rows sum to 1)
   - B is column-stochastic (columns sum to 1)
   - Both represent strongly connected graphs

2. **Error Handling**: Include validation for:
   - Valid topology names
   - Valid strategy names
   - Appropriate n values for grid topology (perfect squares)
   - random_seed provided when strategy="random"

3. **Logging**: Consider adding logging for:
   - Selected topology and its properties (kappa, beta values)
   - Learning rate distribution
   - Computed c value
   - Progress through repetitions

4. **Memory Management**: For large n or many repetitions, consider:
   - Saving intermediate results to disk
   - Clearing GPU memory between repetitions
   - Batch processing of results