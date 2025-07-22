#!/bin/bash

# Script to run multiple distributed optimization experiments
# Each experiment has hardcoded parameters for easy modification

# Set Python environment (modify if needed)
PYTHON=python3

# Base output directory
OUTPUT_BASE="./experiments/batch_$(date +%Y%m%d_%H%M%S)"

echo "Starting batch experiments..."
echo "Output directory: $OUTPUT_BASE"

# ============================================
# Experiment 1: Compare strategies on neighbor topology
# ============================================
echo ""
echo "=== Experiment 1: Strategy comparison on neighbor topology ==="

# Uniform strategy
$PYTHON run_experiment.py \
    --topology neighbor \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --strategy uniform \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 100 \
    --alpha 1000 \
    --use_hetero \
    --repetitions 3 \
    --remark "exp1_uniform" \
    --device cuda:0 \
    --output_dir "$OUTPUT_BASE/exp1_strategies"

# Pi_a inverse strategy
$PYTHON run_experiment.py \
    --topology neighbor \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --strategy pi_a_inverse \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 100 \
    --alpha 1000 \
    --use_hetero \
    --repetitions 3 \
    --remark "exp1_pi_a_inverse" \
    --device cuda:0 \
    --output_dir "$OUTPUT_BASE/exp1_strategies"

# Pi_b inverse strategy
$PYTHON run_experiment.py \
    --topology neighbor \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --strategy pi_b_inverse \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 100 \
    --alpha 1000 \
    --use_hetero \
    --repetitions 3 \
    --remark "exp1_pi_b_inverse" \
    --device cuda:0 \
    --output_dir "$OUTPUT_BASE/exp1_strategies"

# ============================================
# Experiment 2: Compare topologies with uniform strategy
# ============================================
echo ""
echo "=== Experiment 2: Topology comparison with uniform strategy ==="

# Ring topology
$PYTHON run_experiment.py \
    --topology ring \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --strategy uniform \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 100 \
    --alpha 1000 \
    --use_hetero \
    --repetitions 3 \
    --remark "exp2_ring" \
    --device cuda:0 \
    --output_dir "$OUTPUT_BASE/exp2_topologies"

# Grid topology (n must be perfect square)
$PYTHON run_experiment.py \
    --topology grid \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --strategy uniform \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 100 \
    --alpha 1000 \
    --use_hetero \
    --repetitions 3 \
    --remark "exp2_grid" \
    --device cuda:0 \
    --output_dir "$OUTPUT_BASE/exp2_topologies"

# Random topology
$PYTHON run_experiment.py \
    --topology random \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --strategy uniform \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 100 \
    --alpha 1000 \
    --use_hetero \
    --repetitions 3 \
    --remark "exp2_random" \
    --device cuda:0 \
    --output_dir "$OUTPUT_BASE/exp2_topologies"

# ============================================
# Experiment 3: Test different heterogeneity levels
# ============================================
echo ""
echo "=== Experiment 3: Different heterogeneity levels ==="

# Low heterogeneity (alpha=10000)
$PYTHON run_experiment.py \
    --topology neighbor \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --strategy uniform \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 50 \
    --alpha 10000 \
    --use_hetero \
    --repetitions 2 \
    --remark "exp3_low_hetero" \
    --device cuda:0 \
    --output_dir "$OUTPUT_BASE/exp3_heterogeneity"

# Medium heterogeneity (alpha=1000)
$PYTHON run_experiment.py \
    --topology neighbor \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --strategy uniform \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 50 \
    --alpha 1000 \
    --use_hetero \
    --repetitions 2 \
    --remark "exp3_med_hetero" \
    --device cuda:0 \
    --output_dir "$OUTPUT_BASE/exp3_heterogeneity"

# High heterogeneity (alpha=100)
$PYTHON run_experiment.py \
    --topology neighbor \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --strategy uniform \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 50 \
    --alpha 100 \
    --use_hetero \
    --repetitions 2 \
    --remark "exp3_high_hetero" \
    --device cuda:0 \
    --output_dir "$OUTPUT_BASE/exp3_heterogeneity"

# ============================================
# Experiment 4: Random strategy with different seeds
# ============================================
echo ""
echo "=== Experiment 4: Random strategy test ==="

# Random seed 123
$PYTHON run_experiment.py \
    --topology neighbor \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --strategy random \
    --random_seed 123 \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 50 \
    --alpha 1000 \
    --use_hetero \
    --repetitions 2 \
    --remark "exp4_random_123" \
    --device cuda:0 \
    --output_dir "$OUTPUT_BASE/exp4_random"

# Random seed 456
$PYTHON run_experiment.py \
    --topology neighbor \
    --n 16 \
    --matrix_seed 42 \
    --lr_basic 0.007 \
    --strategy random \
    --random_seed 456 \
    --dataset_name MNIST \
    --batch_size 128 \
    --num_epochs 50 \
    --alpha 1000 \
    --use_hetero \
    --repetitions 2 \
    --remark "exp4_random_456" \
    --device cuda:0 \
    --output_dir "$OUTPUT_BASE/exp4_random"

echo ""
echo "All experiments completed!"
echo "Results saved in: $OUTPUT_BASE"