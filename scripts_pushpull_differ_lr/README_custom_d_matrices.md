# Custom D Matrix Strategy with Varying Convergence Factor c

This document explains the new functionality for generating custom D matrices with controlled convergence factor c values using the theoretical simplex method.

## Overview

The convergence rate of the PushPull algorithm depends on the convergence factor:
```
c = n * π_A^T * D * π_B
```

where:
- `n` is the number of nodes
- `π_A` is the left Perron vector of matrix A
- `π_B` is the right Perron vector of matrix B
- `D` is a diagonal matrix with positive elements summing to n

## Theoretical Foundation

### The Simplex Method

The constraint `d_i > 0` and `Σd_i = n` defines an (n-1)-dimensional simplex. Since c(d) is linear in d, it achieves its extrema at the vertices of this simplex.

**Key Insight**: At vertex j (where d_j = n and d_i = 0 for i ≠ j):
```
c_j = n² * π_A[j] * π_B[j]
```

Therefore:
- `c_min = min_j(n² * π_A[j] * π_B[j])`
- `c_max = max_j(n² * π_A[j] * π_B[j])`

This gives us the **exact** theoretical range without any sampling!

### Generating Intermediate Values

For any two vertices, we can create intermediate D matrices via convex combinations:
```
d(α) = (1 - α) * d_min + α * d_max, where α ∈ [0, 1]
```

This produces c values that vary linearly from c_min to c_max.

## Usage

### 1. Generate Custom D Matrices

```bash
# Generate 10 D matrices with uniformly distributed c values
python generate_custom_d_matrices.py \
    --topology neighbor \
    --n 16 \
    --matrix_seed 42 \
    --num_c_values 10 \
    --distribution uniform \
    --visualize

# Generate D matrices using simplex vertices
python generate_custom_d_matrices.py \
    --topology ring \
    --n 8 \
    --matrix_seed 123 \
    --num_c_values 8 \
    --distribution vertices

# Generate D matrices with logarithmic spacing
python generate_custom_d_matrices.py \
    --topology grid \
    --n 16 \
    --matrix_seed 456 \
    --num_c_values 5 \
    --distribution log
```

### 2. Run Experiments with Custom D Matrices

```bash
# Run experiments with varying c values
python run_custom_strategy_experiments.py \
    --topology neighbor \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --num_c_values 5 \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 100 \
    --alpha 1000 \
    --use_hetero \
    --distribution uniform
```

### 3. Use in Python Code

```python
from generate_custom_d_matrices import generate_custom_d_matrices, analyze_c_range
from experiment_utils import generate_topology_matrices

# Generate topology
A, B = generate_topology_matrices("neighbor", n=16, matrix_seed=42, k=3)

# Analyze c range
analysis = analyze_c_range(A, B)
print(f"c_min = {analysis['c_min']:.6f}")
print(f"c_max = {analysis['c_max']:.6f}")

# Generate D matrices
d_matrices = generate_custom_d_matrices(A, B, num_c_values=5, distribution="uniform")

# Use with run_experiment
from run_experiment import run_distributed_optimization_experiment

for d_diagonal, c_value in d_matrices:
    df = run_distributed_optimization_experiment(
        topology="neighbor",
        n=16,
        matrix_seed=42,
        lr_basic=0.007,
        strategy="custom",  # Use custom strategy
        d_diagonal=d_diagonal,  # Pass the D diagonal
        dataset_name="MNIST",
        batch_size=128,
        num_epochs=100,
        alpha=1000,
        use_hetero=True,
        k=3
    )
```

## Distribution Modes

1. **uniform**: Uniformly spaced c values between c_min and c_max
   - Best for systematic analysis of convergence behavior
   - Uses linear interpolation between extreme vertices

2. **vertices**: Use actual simplex vertices
   - Shows c values at "pure" configurations (single node has all weight)
   - May produce non-uniform spacing

3. **log**: Logarithmically spaced c values
   - Useful when c_max >> c_min
   - Better resolution for small c values

4. **custom**: User-specified target c values
   - Full control over which c values to test
   - Automatically clips to feasible range

## Example Workflow

1. **Analyze your topology**:
   ```bash
   python generate_custom_d_matrices.py --topology neighbor --n 16 --matrix_seed 42 --num_c_values 1
   ```
   This shows you the c_min, c_max, and which nodes they correspond to.

2. **Generate D matrices**:
   ```bash
   python generate_custom_d_matrices.py --topology neighbor --n 16 --matrix_seed 42 --num_c_values 10 --visualize
   ```
   This creates 10 D matrices with uniformly distributed c values and saves them.

3. **Run experiments**:
   ```bash
   python example_custom_d_experiment.py
   ```
   This runs a complete experiment with multiple c values and analyzes results.

## Advantages Over Sampling

1. **Exact extrema**: No risk of missing true c_min or c_max
2. **Efficiency**: O(n) computation vs O(samples × n) for sampling
3. **Deterministic**: Results are reproducible without random seeds
4. **Theoretical guarantee**: Based on convex optimization theory

## Files Created

- `generate_custom_d_matrices.py`: Standalone D matrix generator
- `example_custom_d_experiment.py`: Example usage and analysis
- `test_simplex_method.py`: Tests verifying the theoretical approach
- Enhanced `d_matrix_utils.py`: Core implementation with detailed documentation

## Integration with Existing Code

The implementation is fully compatible with the existing experiment framework:
- Uses the existing "custom" strategy in `compute_learning_rates()`
- Works with all analysis and visualization tools
- Independent of `run_experiment.py` as requested
- Can be used programmatically or via command line