#!/bin/bash

# Script to run custom strategy experiments with D matrices
# Tests both uniform and vertices distribution modes

# Set Python environment (modify if needed)
PYTHON=python3

# Base output directory
OUTPUT_BASE="./custom_experiments/batch_$(date +%Y%m%d_%H%M%S)"

echo "Starting custom strategy experiments..."
echo "Output directory: $OUTPUT_BASE"

# ============================================
# Experiment 1: Compare uniform vs vertices distribution
# ============================================
echo ""
echo "=== Experiment 1: Uniform vs Vertices distribution comparison ==="

# Uniform distribution with 5 c values
$PYTHON run_custom_strategy_experiments.py \
    --topology neighbor \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --num_c_values 5 \
    --distribution uniform \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 100 \
    --alpha 1000 \
    --use_hetero \
    --device cuda:0 \
    --output_dir "$OUTPUT_BASE/exp1_distribution_comparison"

# Vertices distribution with all 16 vertices
$PYTHON run_custom_strategy_experiments.py \
    --topology neighbor \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --num_c_values 16 \
    --distribution vertices \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 100 \
    --alpha 1000 \
    --use_hetero \
    --device cuda:0 \
    --output_dir "$OUTPUT_BASE/exp1_distribution_comparison"

# ============================================
# Experiment 2: Fine-grained c value exploration (uniform)
# ============================================
echo ""
echo "=== Experiment 2: Fine-grained c value exploration ==="

# 10 uniformly distributed c values
$PYTHON run_custom_strategy_experiments.py \
    --topology ring \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --num_c_values 10 \
    --distribution uniform \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 150 \
    --alpha 1000 \
    --use_hetero \
    --repetitions 2 \
    --device cuda:0 \
    --output_dir "$OUTPUT_BASE/exp2_fine_grained"

# ============================================
# Experiment 3: Vertices subset exploration
# ============================================
echo ""
echo "=== Experiment 3: Vertices subset (k < n) ==="

# Select 8 out of 16 vertices with maximum spacing
$PYTHON run_custom_strategy_experiments.py \
    --topology neighbor \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --num_c_values 8 \
    --distribution vertices \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 100 \
    --alpha 1000 \
    --use_hetero \
    --device cuda:0 \
    --output_dir "$OUTPUT_BASE/exp3_vertices_subset"

# ============================================
# Experiment 4: Different topologies with uniform distribution
# ============================================
echo ""
echo "=== Experiment 4: Different topologies (uniform) ==="

# Grid topology
$PYTHON run_custom_strategy_experiments.py \
    --topology grid \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --num_c_values 5 \
    --distribution uniform \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 100 \
    --alpha 1000 \
    --use_hetero \
    --device cuda:0 \
    --output_dir "$OUTPUT_BASE/exp4_topologies"

# Random topology
$PYTHON run_custom_strategy_experiments.py \
    --topology random \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --num_c_values 5 \
    --distribution uniform \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 100 \
    --alpha 1000 \
    --use_hetero \
    --device cuda:0 \
    --output_dir "$OUTPUT_BASE/exp4_topologies"

# ============================================
# Experiment 5: Small network with all vertices
# ============================================
echo ""
echo "=== Experiment 5: Small network complete vertices analysis ==="

# n=8 with all 8 vertices
$PYTHON run_custom_strategy_experiments.py \
    --topology neighbor \
    --n 8 \
    --matrix_seed 42 \
    --lr_basic 0.01 \
    --num_c_values 8 \
    --distribution vertices \
    --dataset_name MNIST \
    --batch_size 256 \
    --num_epochs 150 \
    --alpha 1000 \
    --use_hetero \
    --repetitions 3 \
    --device cuda:0 \
    --output_dir "$OUTPUT_BASE/exp5_small_network"

# ============================================
# Experiment 6: Test specific strategies comparison
# ============================================
echo ""
echo "=== Experiment 6: Specific strategies test ==="

$PYTHON run_custom_strategy_experiments.py \
    --topology neighbor \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --use_specific_strategies \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 100 \
    --alpha 1000 \
    --use_hetero \
    --device cuda:0 \
    --output_dir "$OUTPUT_BASE/exp6_specific_strategies"

echo ""
echo "All custom strategy experiments completed!"
echo "Results saved in: $OUTPUT_BASE"
echo ""
echo "To analyze results:"
echo "- Check individual experiment directories for CSV files"
echo "- View experiment_summary.json for overview"
echo "- Look for c_vs_grad_norm.png plots if matplotlib is available"