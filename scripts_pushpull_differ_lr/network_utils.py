import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import networkx as nx

# @Lg 难点在于训练时间和跑收敛。建议现在先把MNIST的跑起来，然后写CIFAR的代码。如果Cifar-10 实验在10个节点无法收敛，可以考虑构造5个节点的稀疏图比如di_ring(n=5)和 Row(get_xinmeng_matrix(n=5))
# 一、 4层神经网络训练MNIST数据集：
# 【1】异质性观察
# 图1：pulldiag在di_ring(n=5)+三种不同异质性的表现(注：现在只弄了两种异质性，均匀分布和完全异质分布。能不能弄一个稍微混合一点的数据分布？比如在完全异质分布条件下，让1，2号节点数据混合一下。）
# 图2：pullsum在di_ring(n=5)+三种不同异质性的表现
# 图3：pulldiag和pullsum都在di_ring(n=5)或者di_ring(n=10)+强异质性下的对比表现

# 【2】拓扑影响
# 图1：在row_and_col_mat(n=10, p=0.5）+强异质性条件下比较pulldiag, pullsum, frsd, frozen
# 图2：在row_and_col_mat(n=10, p=0.2）+强异质性条件下比较pulldiag, pullsum, frsd, frozen
# 图3：只看pullsum, 在row_and_col_mat(n=10, p=0.5），row_and_col_mat(n=10, p=0.2），di_ring(n=10)，grid_10()上的表现。


# 二、Resnet训练CIFAR-10数据

# 图1：di_ring(n=10)+强异质性条件下比较 pulldiag, pullsum, frsd, frozen
# 图2：grid_10()+强异质性条件下比较 pulldiag, pullsum, frsd, frozen


# @cxy: 合成数据的实验难点在于怎么让强异质性真正发挥作用。我的初步建议是提升问题的维度，比如令d=20，在此基础上提高sigma_h。如果不行，要等吴百濠学长的异质性代码。
# 一、【确定性情况+强异质性。对比pullsum 和其他算法（pulldiag, frsd, frozen）】

# 图1：n=20, row_and_col_mat(n=20, p=0.5）
# 图2：n=20, row_and_col_mat(n=20, p=0.2）
# 图3:  n=20, di_ring(n=20)
# 图4：三角网格图 或者我提供的 grid_20()


# 二、【噪声取1e-3, 1e-2, 1e-1 ，强异质性。对比pullsum 和其他算法（pulldiag, frsd, frozen）】

# 图1：n=20, row_and_col_mat(n=20, p=0.5）
# 图2：n=20, row_and_col_mat(n=20, p=0.2）
# 图3:  n=20, di_ring(n=20)
# 图4：三角网格图 或者我提供的 grid_20()


# 三、【噪声取1e-4，强异质性，只看pullsum】
# 图1：pull sum 在row_and_col_mat(n=20, p=0.5), row_and_col_mat(n=20, p=0.2）,di_ring(n=20),grid_20() 这些拓扑下的表现画在一张图内


# 下列所有代码会返回一个行随机矩阵和一个列随机矩阵。它们的对角元可以达到$2^{-n}$量级，$\kappa_\pi$为$n$的量级。


def row_and_col_mat(
    n=10, p=0.3, seed=None, show_graph=None
):  # 生成p-随机图，一般比较好。
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)

    # 生成强连通的随机有向图
    G = nx.gnp_random_graph(n, p, seed=seed, directed=True)

    # 确保图是强连通的
    while not nx.is_strongly_connected(G):
        G = nx.gnp_random_graph(n, p, directed=True)

    # 计算每个节点的入度和出度
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    # 初始化权重矩阵
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    # 给每条边赋予权重 1 / (目标节点的入度 + 1)
    for i, j in G.edges():
        A[i][j] = 1 / (in_degrees[j] + 1)
        B[i][j] = 1 / (out_degrees[i] + 1)
    # 为每个节点添加自环，并计算自环权重 1 / (入度 + 1)
    for j in range(n):
        A[j][j] = 1 / (in_degrees[j] + 1)
        B[j][j] = 1 / (out_degrees[j] + 1)
    if show_graph is not None:
        return A.T, B.T, G
    return A.T, B.T  # A.T是行随机，B.T是列随机矩阵，


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


def grid_10():  # 生成10节点稀疏网格图。
    A = np.eye(10) / 2
    A[0][1] = 1 / 2
    A[3][2] = 1 / 3
    A[2][2] = 1 / 3
    A[1][2] = 1 / 3
    A[2][7] = 1 / 2
    A[7][8] = 1 / 2
    A[8][9] = 1 / 2
    A[9][0] = 1 / 2
    A[7][6] = 1 / 2
    A[6][5] = 1 / 2
    A[4][3] = 1 / 2
    A[5][4] = 1 / 2
    B = np.eye(10) / 2
    B[0][1] = 1 / 2
    B[1][2] = 1 / 2
    B[2][7] = 1 / 2
    B[3][2] = 1 / 2
    B[4][3] = 1 / 2
    B[5][4] = 1 / 2
    B[6][5] = 1 / 2
    B[7][6] = 1 / 3
    B[7][7] = 1 / 3
    B[7][8] = 1 / 3
    B[8][9] = 1 / 2
    B[9][0] = 1 / 2
    return A.T, B.T  # A.T是行随机，B.T是列随机矩阵，


def ring2():  # 生成20节点稀疏网格图
    A = np.eye(20) / 2
    # 定义A矩阵的非对角线元素
    A[0][1] = 1 / 2
    A[1][2] = 1 / 3
    A[2][3] = 1 / 3
    A[3][4] = 1 / 3
    A[4][5] = 1 / 2
    A[5][6] = 1 / 2
    A[6][7] = 1 / 2
    A[7][8] = 1 / 2
    A[8][9] = 1 / 2
    A[9][10] = 1 / 2
    A[10][11] = 1 / 2
    A[11][12] = 1 / 2
    A[12][13] = 1 / 2
    A[13][14] = 1 / 2
    A[14][15] = 1 / 2
    A[15][16] = 1 / 2
    A[16][17] = 1 / 2
    A[17][18] = 1 / 2
    A[18][19] = 1 / 2
    A[19][0] = 1 / 2

    # 再添加一些其他稀疏的连接
    A[2][10] = 1 / 2
    A[10][15] = 1 / 2
    # A[15][5] = 1/2

    B = np.eye(20) / 2
    # 定义B矩阵的非对角线元素
    B[0][1] = 1 / 2
    B[1][2] = 1 / 2
    B[2][3] = 1 / 2
    B[3][4] = 1 / 2
    B[4][5] = 1 / 2
    B[5][6] = 1 / 2
    B[6][7] = 1 / 2
    B[7][8] = 1 / 3
    B[8][9] = 1 / 3
    B[9][10] = 1 / 3
    B[10][11] = 1 / 2
    B[11][12] = 1 / 2
    B[12][13] = 1 / 2
    B[13][14] = 1 / 2
    B[14][15] = 1 / 2
    B[15][16] = 1 / 2
    B[16][17] = 1 / 2
    B[17][18] = 1 / 2
    B[18][19] = 1 / 2
    B[19][0] = 1 / 2

    # 再添加一些其他稀疏的连接
    B[5][10] = 1 / 3
    B[10][15] = 1 / 3
    B[15][5] = 1 / 3

    return Row(A.T), Col(B.T)


def ring3():  # 生成20节点稀疏网格图
    A = np.eye(20) / 2
    # 定义A矩阵的非对角线元素
    A[0][1] = 1 / 2
    A[1][2] = 1 / 3
    A[2][3] = 1 / 3
    A[3][4] = 1 / 3
    A[4][5] = 1 / 2
    A[5][6] = 1 / 2
    A[6][7] = 1 / 2
    A[7][8] = 1 / 2
    A[8][9] = 1 / 2
    A[9][10] = 1 / 2
    A[10][11] = 1 / 2
    A[11][12] = 1 / 2
    A[12][13] = 1 / 2
    A[13][14] = 1 / 2
    A[14][15] = 1 / 2
    A[15][16] = 1 / 2
    A[16][17] = 1 / 2
    A[17][18] = 1 / 2
    A[18][19] = 1 / 2
    A[19][0] = 1 / 2

    # 再添加一些其他稀疏的连接
    A[2][10] = 1 / 2
    A[10][15] = 1 / 2
    A[15][2] = 1 / 2

    B = np.eye(20) / 2
    # 定义B矩阵的非对角线元素
    B[0][1] = 1 / 2
    B[1][2] = 1 / 2
    B[2][3] = 1 / 2
    B[3][4] = 1 / 2
    B[4][5] = 1 / 2
    B[5][6] = 1 / 2
    B[6][7] = 1 / 2
    B[7][8] = 1 / 3
    B[8][9] = 1 / 3
    B[9][10] = 1 / 3
    B[10][11] = 1 / 2
    B[11][12] = 1 / 2
    B[12][13] = 1 / 2
    B[13][14] = 1 / 2
    B[14][15] = 1 / 2
    B[15][16] = 1 / 2
    B[16][17] = 1 / 2
    B[17][18] = 1 / 2
    B[18][19] = 1 / 2
    B[19][0] = 1 / 2

    # 再添加一些其他稀疏的连接
    B[5][10] = 1 / 3
    B[10][15] = 1 / 3
    B[15][5] = 1 / 3

    return Row(A.T), Col(B.T)


def ring4():  # 生成20节点稀疏网格图
    A = np.eye(20) / 2
    # 定义A矩阵的非对角线元素
    A[0][1] = 1 / 2
    A[1][2] = 1 / 3
    A[2][3] = 1 / 3
    A[3][4] = 1 / 3
    A[4][5] = 1 / 2
    A[5][6] = 1 / 2
    A[6][7] = 1 / 2
    A[7][8] = 1 / 2
    A[8][9] = 1 / 2
    A[9][10] = 1 / 2
    A[10][11] = 1 / 2
    A[11][12] = 1 / 2
    A[12][13] = 1 / 2
    A[13][14] = 1 / 2
    A[14][15] = 1 / 2
    A[15][16] = 1 / 2
    A[16][17] = 1 / 2
    A[17][18] = 1 / 2
    A[18][19] = 1 / 2
    A[19][0] = 1 / 2

    # 再添加一些其他稀疏的连接
    A[0][5] = 1 / 2
    A[10][15] = 1 / 2
    A[5][10] = 1 / 2
    A[15][0] = 1 / 3

    B = np.eye(20) / 2
    # 定义B矩阵的非对角线元素
    B[0][1] = 1 / 2
    B[1][2] = 1 / 2
    B[2][3] = 1 / 2
    B[3][4] = 1 / 2
    B[4][5] = 1 / 2
    B[5][6] = 1 / 2
    B[6][7] = 1 / 2
    B[7][8] = 1 / 3
    B[8][9] = 1 / 3
    B[9][10] = 1 / 3
    B[10][11] = 1 / 2
    B[11][12] = 1 / 2
    B[12][13] = 1 / 2
    B[13][14] = 1 / 2
    B[14][15] = 1 / 2
    B[15][16] = 1 / 2
    B[16][17] = 1 / 2
    B[17][18] = 1 / 2
    B[18][19] = 1 / 2
    B[19][0] = 1 / 2

    # 再添加一些其他稀疏的连接
    B[5][10] = 1 / 3
    B[10][15] = 1 / 3
    B[15][5] = 1 / 3

    return Row(A.T), Col(B.T)


def Row(matrix):
    # 计算每一行的和
    M = matrix.copy()
    row_sums = np.sum(M, axis=1)

    # 将每一行除以该行的和
    for i in range(M.shape[0]):
        M[i, :] /= row_sums[i]

    return M


def Col(matrix):
    W = matrix.copy()
    # 计算每一列的和
    col_sums = np.sum(W, axis=0)

    # 将每一列除以该列的和
    for i in range(W.shape[0]):
        W[:, i] /= col_sums[i]

    return W


def get_xinmeng_matrix(n=5):
    M = np.zeros((n, n))

    # 主对角线上的元素
    M[np.diag_indices(n)] = 1 / 3 * np.ones(n)
    M[n - 1, n - 1] = M[n - 1, n - 1] + 1 / 3

    # 次对角线上的元素
    for i in range(n - 1):
        M[i + 1, i] = M[i + 1, i] + 1 / 3

    # 第一行上的元素
    M[0, :] = M[0, :] + 1 / 3

    return M


import numpy as np
#======================== n=10 稀疏环图 ========================#
def ring4_node_10():
    n = 10
    A = np.eye(n) / 2  # 初始化对角线为1/2
    
    # 基础环状连接 (部分权重不同)
    A[0][1] = 1/2
    A[1][2] = 1/3
    A[2][3] = 1/3
    A[3][4] = 1/3
    A[4][5] = 1/2
    
    # 后续节点保持1/2权重
    for i in range(5, n-1):
        A[i][i+1] = 1/2
    A[n-1][0] = 1/2  # 闭环
    
    # 添加2条跨步连接 (步长=5)
    A[0][5] = 1/2
    A[5][0] = 1/2
    
    return Row(A.T)

#======================== n=100 稀疏环图 ========================#
def ring4_node_100():
    n = 100
    A = np.eye(n) / 2  # 初始化对角线为1/2
    
    # 基础环状连接 (部分权重不同)
    A[0][1] = 1/2
    A[1][2] = 1/3
    A[2][3] = 1/3
    A[3][4] = 1/3
    A[4][5] = 1/2
    
    # 后续节点保持1/2权重
    for i in range(5, n):
        next_node = (i + 1) % n
        A[i][next_node] = 1/2
    
    # 添加25条跨步连接 (步长=4)
    step = 4
    for i in range(0, n, step):
        j = (i + step) % n
        A[i][j] = 1/2
    
    return Row(A.T)


import numpy as np

def generate_exponential_weight_matrix(n):
    """
    生成静态指数图的权重矩阵。

    参数：
    n (int): 图中节点的数量。

    返回：
    numpy.ndarray: 形状为(n, n)的权重矩阵，元素类型为float。
    """
    if n < 1:
        raise ValueError("n必须为正整数")
    
    # 计算分母：|log2(n)| + 1
    denominator = np.abs(np.log2(n)) + 1
    
    # 创建索引网格
    i_indices, j_indices = np.indices((n, n))
    
    # 计算mod_val = (j - i) mod n
    mod_vals = (j_indices - i_indices) % n
    
    # 判断是否为2的幂
    is_power_of_two = (mod_vals != 0) & ((mod_vals & (mod_vals - 1)) == 0)
    
    # 判断是否为自环或mod_val是2的幂
    mask = (i_indices == j_indices) | is_power_of_two
    
    # 生成权重矩阵
    weight_matrix = np.where(mask, 1.0 / denominator, 0.0)
    
    return weight_matrix

def get_matrixs_from_exp_graph(n, seed=42):

    original_matrix = generate_exponential_weight_matrix(n)
    np.random.seed(seed)
    random_matrix = np.where(original_matrix != 0, np.random.randint(1, 3, size=original_matrix.shape), 0)
    # 随机赋值 1, 2, 3

    random_matrix = np.array(random_matrix)

    M = random_matrix.copy().astype(float)
    
    A = Row(M)
    B = Col(M)
    
    return A, B


import numpy as np
import math

# Provided helper functions for normalization
def Row(matrix):
    """
    Normalizes the matrix such that the sum of each row is 1.

    Args:
        matrix (np.ndarray): The input matrix.

    Returns:
        np.ndarray: The row-normalized matrix.
    """
    M = matrix.astype(float).copy() # Ensure float type for division
    row_sums = np.sum(M, axis=1)

    # Avoid division by zero for rows that sum to 0 (although unlikely with weights 1-5)
    # If a row sum is 0, keep the row as zeros.
    non_zero_rows = row_sums != 0
    M[non_zero_rows, :] /= row_sums[non_zero_rows, np.newaxis] # Use np.newaxis for broadcasting

    return M

def Col(matrix):
    """
    Normalizes the matrix such that the sum of each column is 1.

    Args:
        matrix (np.ndarray): The input matrix.

    Returns:
        np.ndarray: The column-normalized matrix.
    """
    W = matrix.astype(float).copy() # Ensure float type for division
    # Calculate the sum of each column
    col_sums = np.sum(W, axis=0)

    # Avoid division by zero for columns that sum to 0
    # If a column sum is 0, keep the column as zeros.
    non_zero_cols = col_sums != 0
    W[:, non_zero_cols] /= col_sums[non_zero_cols] # Direct division works column-wise

    return W

# --- Function 1: generate_grid_matrices ---
def generate_grid_matrices(n, seed=42):
    """
    Generates weighted and normalized adjacency matrices for a grid graph.

    Creates an n x n adjacency matrix N for a sqrt(n) x sqrt(n) grid graph
    where each node has a self-loop and bidirectional connections to its
    horizontal and vertical neighbors. Then, it assigns random integer weights
    (1 to 5) to the connections (where N[i,j]=1). Finally, it normalizes
    this weighted matrix using Row and Col normalization.

    Args:
        n (int): The number of nodes, must be a perfect square (e.g., 4, 9, 25).
        seed (int, optional): Seed for the random number generator. Defaults to 42.

    Returns:
        tuple[np.ndarray, np.ndarray] | None: A tuple containing the
                                             row-normalized matrix (A) and the
                                             column-normalized matrix (B).
                                             Returns None if n is not a perfect square.
    """
    # Validate n is a perfect square
    grid_size_f = math.sqrt(n)
    grid_size = int(grid_size_f)
    if grid_size * grid_size != n:
        print(f"Error: n ({n}) must be a perfect square.")
        return None

    # 1. Create the initial 0-1 adjacency matrix N
    N = np.zeros((n, n), dtype=int)

    # Add self-loops and connections
    for r in range(grid_size):
        for c in range(grid_size):
            idx = r * grid_size + c
            N[idx, idx] = 1  # Self-loop

            # Check right neighbor
            if c + 1 < grid_size:
                neighbor_idx = r * grid_size + (c + 1)
                N[idx, neighbor_idx] = 1
                N[neighbor_idx, idx] = 1  # Bidirectional

            # Check down neighbor
            if r + 1 < grid_size:
                neighbor_idx = (r + 1) * grid_size + c
                N[idx, neighbor_idx] = 1
                N[neighbor_idx, idx] = 1  # Bidirectional

    # 2. Assign random weights (1 to 5) where N[i,j] == 1
    np.random.seed(seed)
    # Create the weighted matrix W, initialize with zeros
    W = np.zeros((n, n), dtype=float) # Use float for potential normalization later
    # Find indices where N is 1
    rows, cols = np.where(N == 1)
    # Assign random weights to these positions in W
    random_weights = np.random.randint(1, 3, size=len(rows)) # 6 is exclusive
    W[rows, cols] = random_weights

    # 3. Normalize using Row and Col functions
    A = Row(W)
    B = Col(W)

    return A, B

# --- Function 2: generate_ring_matrices ---
def generate_ring_matrices(n, seed=42):
    """
    Generates weighted and normalized adjacency matrices for a directed ring graph
    with additional shortcuts.

    Creates an n x n adjacency matrix N for a directed ring graph (0->1->...->n-1->0).
    Each node has a self-loop. Additional directed edges are added:
    0 -> int(n/4) and int(n/2) -> int(3n/4).
    Then, it assigns random integer weights (1 to 5) to the connections
    (where N[i,j]=1). Finally, it normalizes this weighted matrix using
    Row and Col normalization.

    Args:
        n (int): The number of nodes, must be >= 2.
        seed (int, optional): Seed for the random number generator. Defaults to 42.

    Returns:
        tuple[np.ndarray, np.ndarray] | None: A tuple containing the
                                             row-normalized matrix (A) and the
                                             column-normalized matrix (B).
                                             Returns None if n < 2.
    """
    if n < 2:
        print(f"Error: n ({n}) must be >= 2.")
        return None

    # 1. Create the initial 0-1 adjacency matrix N
    N = np.zeros((n, n), dtype=int)

    # Add self-loops and ring connections
    for i in range(n):
        N[i, i] = 1  # Self-loop
        N[i, (i + 1) % n] = 1 # Directed edge i -> i+1 (wraps around)

    # Add additional directed edges
    idx_n_4 = int(n / 4)
    idx_n_2 = int(n / 2)
    idx_3n_4 = int(3 * n / 4)

    # Ensure indices are within bounds (although they should be for n>=2)
    if 0 <= idx_n_4 < n:
        N[0, idx_n_4] = 1
    if 0 <= idx_n_2 < n and 0 <= idx_3n_4 < n:
        N[idx_n_2, idx_3n_4] = 1

    # 2. Assign random weights (1 to 5) where N[i,j] == 1
    np.random.seed(seed)
    # Create the weighted matrix W, initialize with zeros
    W = np.zeros((n, n), dtype=float) # Use float for potential normalization later
    # Find indices where N is 1
    rows, cols = np.where(N == 1)
    # Assign random weights to these positions in W
    random_weights = np.random.randint(1, 3, size=len(rows)) # 6 is exclusive
    W[rows, cols] = random_weights

    # 3. Normalize using Row and Col functions
    A = Row(W)
    B = Col(W)

    return A, B

def generate_random_graph_matrices(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    为随机图生成加权和归一化的邻接矩阵。

    1. 创建一个n x n的初始0-1邻接矩阵N。
       - N的对角线元素（自环）始终为1。
       - 对于任何两个不同的节点i和j，边(i,j)以概率p=1/3存在（双向）。
    2. 对于N中为1的位置，随机赋予1到3之间的整数权重，得到加权矩阵W。
    3. 使用Row()和Col()函数对W进行归一化，得到矩阵A（行归一化）和B（列归一化）。

    Args:
        n (int): 节点数。
        seed (int, optional): 随机数生成器的种子。默认为42。

    Returns:
        tuple[np.ndarray, np.ndarray]: 包含行归一化矩阵(A)和列归一化矩阵(B)的元组。
    """
    np.random.seed(seed)

    # 1. 创建初始的0-1邻接矩阵 N
    #    - 对角线元素为1 (自环)
    #    - 非对角线元素 (i,j) 和 (j,i) 以概率 p=1/3 为1
    
    N = np.zeros((n, n), dtype=int)
    prob_edge_exists = 1/3

    # 设置自环 (对角线元素为1)
    for i in range(n):
        N[i, i] = 1

    # 设置非对角线元素 (双向连接)
    # 遍历上三角矩阵（不包括对角线）以避免重复判断和覆盖自环
    for i in range(n):
        for j in range(i + 1, n): # j > i
            if np.random.rand() < prob_edge_exists:
                N[i, j] = 1
                N[j, i] = 1  # 因为是无向图，所以连接是双向的

    # 2. 对N中为1的位置，随机赋予1到3的权重，得到矩阵W
    W = np.zeros((n, n), dtype=float) # 使用float类型以便后续归一化

    # 找到N中所有为1的元素的索引
    rows, cols = np.where(N == 1)
    
    # 为这些位置生成随机权重 (1, 2, 或 3)
    # np.random.randint 的上界是开区间，所以要得到[1,3]之间的整数，上界应为4
    random_weights = np.random.randint(1, 3 + 1, size=len(rows))
    W[rows, cols] = random_weights

    # 3. 使用Row和Col函数进行归一化
    A = Row(W)
    B = Col(W)

    return A, B



def generate_stochastic_geometric_matrices(n, seed, threshold=5):
    """
    生成基于几何图的行随机矩阵 A 和列随机矩阵 B。

    步骤:
    1. 设置随机种子。
    2. 生成 n 个节点的随机2D坐标。
    3. 初始化一个 n x n 的零矩阵 W。
    4. 将 W 的对角元素设置为 1 (保证自环的初始标记)。
    5. 根据几何图定义：如果节点 i 和 j 之间的距离 <= threshold，
       则将 W[i, j] 和 W[j, i] 设置为 1。
    6. 对于 W 中所有值为 1 的元素（包括自环和几何连接），
       将其随机重新赋值为 1、2 或 3。
    7. Row 函数：对 W 进行行归一化得到行随机矩阵 A。
    8. Col 函数：对 W 进行列归一化得到列随机矩阵 B。

    参数:
        n (int): 节点数目 (确保 n >= 6)。
        seed (int): 随机种子。
        threshold (float): 连接节点的距离阈值。

    返回:
        tuple: (A, B)
               A: n x n 的行随机 numpy 数组。
               B: n x n 的列随机 numpy 数组。
    """
    if not isinstance(n, int) or n < 6:
        raise ValueError("节点数目 n 必须是大于等于6的整数。")
    if not isinstance(seed, int):
        raise ValueError("随机种子 seed 必须是整数。")

    np.random.seed(seed)

    # 1. 生成 n 个节点的随机2D坐标
    # 为了使threshold有意义，我们将坐标限制在一个合理的范围内，例如 0 到 10
    # 这个范围可以根据具体应用调整，以获得不同密度的图
    positions = np.random.rand(n, 2) * 10  # 坐标范围 [0, 10) x [0, 10)

    # 2. 初始化一个 n x n 的零矩阵 W
    W = np.zeros((n, n))

    # 3. 将 W 的对角元素设置为 1 (保证自环的初始标记)
    np.fill_diagonal(W, 1)

    # 4. 根据几何图定义连接边
    for i in range(n):
        for j in range(i + 1, n):  # 避免重复计算和i=j的情况
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= threshold:
                W[i, j] = 1
                W[j, i] = 1  # 图是无向的

    #print(W)

    # 5. 对于W中值为1的元素，随机赋值1到3
    # 找到所有值为1的元素的索引
    one_indices = np.where(W == 1)
    # 为这些位置生成随机值 (1, 2, 或 3)
    random_values = np.random.choice([1, 2, 3], size=len(one_indices[0]))
    W[one_indices] = random_values

    # 6. Row函数归一化得到行随机A
    A = Row(W)

    # 7. Col函数归一化得到列随机B
    B = Col(W)

    return A, B

from scipy.spatial.distance import cdist # For efficient distance calculation

def generate_nearest_neighbor_matrices(n: int, k: int = 3, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates加权和归一化的邻接矩阵 for a k-nearest neighbor graph.

    Steps:
    1. Set random seed.
    2. Generate n random 2D coordinates for the nodes.
    3. Initialize an n x n zero matrix W.
    4. Mark self-loops: Set diagonal elements of W to 1.
    5. For each node i, find its k nearest neighbors (excluding itself).
       For each such neighbor j, mark W[i, j] = 1 and W[j, i] = 1 (undirected graph).
       Note: A node might end up with more than k connections if it's a neighbor
             to many other nodes. Self-loops are handled separately.
    6. For all positions in W that are marked as 1 (self-loops or neighbor connections),
       assign a random integer weight between 1 and 3.
    7. Use Row() and Col() functions to normalize W, yielding matrices A (row-normalized)
       and B (column-normalized).

    Args:
        n (int): Number of nodes. Must be >= 1.
        k (int): Number of nearest neighbors to connect to (excluding self).
                 Must be >= 0 and < n if n > 1. If n=1, k must be 0.
        seed (int, optional): Seed for the random number generator. Defaults to 42.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the row-normalized matrix (A)
                                       and the column-normalized matrix (B).
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("节点数目 n 必须是大于等于1的整数。")
    if not isinstance(k, int) or k < 0:
        raise ValueError("邻居数 k 必须是大于等于0的整数。")
    if n == 1 and k != 0:
        raise ValueError("如果 n=1, k 必须为 0。")
    if n > 1 and k >= n:
        raise ValueError("邻居数 k 必须小于节点数 n (当 n > 1)。")
    if not isinstance(seed, int):
        raise ValueError("随机种子 seed 必须是整数。")

    np.random.seed(seed)

    # 1. Generate n random 2D coordinates
    #    Coordinates in a [0, 10) x [0, 10) square for reasonable spacing
    positions = np.random.rand(n, 2) * 10

    # 2. Initialize W and mark self-loops
    W = np.zeros((n, n), dtype=float) # Use float for weights
    np.fill_diagonal(W, 1) # Mark self-loops (will be weighted later)

    # 3. Connect k-nearest neighbors (if n > 1 and k > 0)
    if n > 1 and k > 0:
        # Calculate all-pairs Euclidean distances
        # cdist(XA, XB) computes the distance between each pair of rows in XA and XB.
        # Here, XA and XB are both 'positions'.
        all_distances = cdist(positions, positions)

        for i in range(n):
            # Get distances from node i to all other nodes
            distances_from_i = all_distances[i, :]
            
            # Get indices of nodes sorted by distance from node i
            # The first element (index 0) will be node i itself (distance 0)
            sorted_neighbor_indices = np.argsort(distances_from_i)
            
            # Select the k nearest neighbors (skipping the node itself at index 0)
            # These are sorted_neighbor_indices[1], ..., sorted_neighbor_indices[k]
            for neighbor_idx in sorted_neighbor_indices[1 : k + 1]:
                W[i, neighbor_idx] = 1
                W[neighbor_idx, i] = 1 # Ensure graph is undirected

    # 4. Assign random weights (1, 2, or 3) to all marked edges (where W[r,c] == 1)
    #    This includes self-loops and neighbor connections.
    rows, cols = np.where(W == 1)
    num_edges_to_weight = len(rows)
    
    if num_edges_to_weight > 0: # Only assign weights if there are edges
        random_weights = np.random.randint(1, 3 + 1, size=num_edges_to_weight)
        W[rows, cols] = random_weights

    # 5. Normalize using Row and Col functions
    A = Row(W)
    B = Col(W)

    return A, B