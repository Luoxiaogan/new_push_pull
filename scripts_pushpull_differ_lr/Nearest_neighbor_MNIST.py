import sys
import os
import numpy as np
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training import train_track_grad_norm_different_learning_rate
from training import train_track_grad_norm_with_hetero_different_learning_rate
import torch
from utils import ring1, show_row, get_left_perron, get_right_perron
from network_utils import generate_nearest_neighbor_matrices

lr_basic = 7e-3
num_epochs = 100
bs = 128
alpha = 1000 # 对于diricted图，alpha约大, 分布越均匀
use_hetero=True
seed = 51583
remark="seed_51583_uniform"#"seed_51583_pi_b_inverse"
device = "cuda:1"
root = "/home/lg/ICML2025_project/NEW_PROJECT_20250717/init_test_mnist_0717/neighbor"
#. nohup python /home/lg/ICML2025_project/scripts_pushpull_differ_lr/Nearest_neighbor_MNIST.py > output3.log 2>&1 &
n=16
A, B = generate_nearest_neighbor_matrices(n = n, k=3, seed=seed)

pi_a = get_left_perron(A)
pi_b = get_right_perron(B)

lr_total = lr_basic * n
lr_partial = pi_b.tolist()  # Convert numpy array to Python list
lr_partial = [ 1/x for x in lr_partial]
sum_lr_partial = sum(lr_partial)
lr_partial = [x / sum_lr_partial for x in lr_partial]  # Normalize to sum to 1 

lr_partial = [1/n]*n

lr_list = [lr_total * partial for partial in lr_partial]

print(lr_list)
show_row(A)
print(A.shape)
for i in range(1):
    df = train_track_grad_norm_with_hetero_different_learning_rate(
        algorithm="PushPull",
        lr_list=lr_list,
        A=A,
        B=B,
        dataset_name="MNIST",
        batch_size=bs,
        num_epochs=500,
        remark=remark,
        alpha = alpha,
        root = root,
        use_hetero=use_hetero,
        device=device,
        seed = i+2
    )
     
    if i == 0:
        df_sum = df
        sum = 1
    else:
        df_sum = df_sum+df
        sum = sum + 1
    df_output = df_sum/sum
    df_output.to_csv(f"/home/lg/ICML2025_project/NEW_PROJECT_20250717/init_test_mnist_0717/neighbor_repeate/n={n}_lr_total={lr_total}_seed={seed}.csv")