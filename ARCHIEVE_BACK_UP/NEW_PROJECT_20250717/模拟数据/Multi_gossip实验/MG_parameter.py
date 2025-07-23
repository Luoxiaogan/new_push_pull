import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from useful_functions import *
from opt_function import *

def ring1(n=10):  # 生成稀疏环状图。也可以取n=5
    A, B = np.eye(n) / 2, np.eye(n) / 2
    m = int(n / 2)
    for i in range(n - 1):
        A[i][i + 1] = 0.5
        B[i][i + 1] = 0.5
    A[n - 1][0] = 0.5
    B[n - 1][0] = 0.5
    A[0][m] = 1 / 3
    A[m - 1][m] = 1 / 3
    A[m][m] = 1 / 3
    B[0][0] = 1 / 3
    B[0][1] = 1 / 3
    B[0][m] = 1 / 3
    return A.T, B.T  # A.T是行随机，B.T是列随机矩阵，


# MG =1

n=10
d=10
L=100
A, B = ring1(n)
show_row(A)
init_x=init_x_func(n=n,d=d,seed=42)
h,y,x_opt,x_star=init_data(n=n,d=d,L=L,seed=489,sigma_h=0.1)

l1 = PullDiag_GT(
    A=A,
    init_x=init_x,
    h_data=h,
    y_data=y,
    rho=0.1,
    lr=2.8e-2,# MG=1, 最优的学习率是 lr =  1.5e-1
    sigma_n=1e-12,
    max_it=3000,
)

# MG = 2