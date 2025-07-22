# training/train_just_per_batch_loss.py

import torch
import os
import torch.nn as nn
import pandas as pd
from datasets.prepare_data import get_dataloaders
from utils.train_utils import get_first_batch, compute_loss_and_accuracy
from utils.train_utils import simple_compute_loss_and_accuracy
from training.optimizer import PullDiag_GT, PullDiag_GD, PushPull
from models.cnn import new_ResNet18
from models.fully_connected import FullyConnectedMNIST, SimpleFCN
from tqdm import tqdm
from datetime import datetime

def train_just_per_batch_loss(
    algorithm: str,
    lr: float,  
    A: torch.Tensor,
    B: torch.Tensor,
    dataset_name: str,
    batch_size: int,
    num_epochs: int = 10,
    remark: str = "",
)-> pd.DataFrame:
    """
    执行训练过程。

    Args:
        algorithm (str): 算法名称 ('PullDiag_GT' 或 'PullDiag_GD' 或 "PushPull")
        lr (float): 学习率
        model_list (list): 模型列表
        A (torch.Tensor): 混合矩阵
        B (torch.Tensor): 混合矩阵
        dataloaders (list): 训练数据加载器列表
        test_dataloader (DataLoader): 测试数据加载器
        num_epochs (int): 训练轮数
        remark (str): 备注
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    n = A.shape[0]
    A = torch.from_numpy(A).float().to(device)
    B = torch.from_numpy(B).float().to(device)

    if dataset_name == "CIFAR10":
        model_list = [new_ResNet18().to(device) for _ in range(n)]
        trainloader_list, testloader, full_trainloader = get_dataloaders(
            n, dataset_name, batch_size, repeat=1
        )
        model_class = new_ResNet18
        #output_root = "/root/GanLuo/ICML2025_project/outputs/logs/CIFAR10_Multi_Gossip"
        output_root = "/root/GanLuo/ICML2025_project/PUSHPULL_PROJECT/real_data_output/CIFAR"
    elif dataset_name == "MNIST":
        model_list = [SimpleFCN().to(device) for _ in range(n)]
        trainloader_list, testloader, full_trainloader = get_dataloaders(
            n, dataset_name, batch_size, repeat=1
        )
        model_class = SimpleFCN
        #output_root = "/root/GanLuo/ICML2025_project/outputs/logs/MNIST"
        output_root = "/root/GanLuo/ICML2025_project/PUSHPULL_PROJECT/real_data_尝试梁学长的建议/output"
    
    torch.backends.cudnn.benchmark = True

    h_data_train, y_data_train = get_first_batch(trainloader_list)
    h_data_train = [
        tensor.to(device, non_blocking=True) for tensor in h_data_train
    ]  # [tensor.to(device) for tensor in h_data_train]
    y_data_train = [
        tensor.to(device, non_blocking=True) for tensor in y_data_train
    ]  # [tensor.to(device) for tensor in y_data_train]

    def closure():
        total_loss = 0
        for i, model in enumerate(model_list):
            for param in model.parameters():
                param.requires_grad = True
            model.zero_grad()
            output = model(h_data_train[i])
            loss = criterion(output, y_data_train[i])
            loss.backward()
            total_loss += loss.item()
        return total_loss / len(model_list)
    
    # 初始化优化器
    if algorithm == "PullDiag_GT":
        optimizer = PullDiag_GT(model_list, lr=lr, A=A, closure=closure)
    elif algorithm == "PullDiag_GD":
        optimizer = PullDiag_GD(model_list, lr=lr, A=A, closure=closure)
    elif algorithm == "PushPull":
        optimizer = PushPull(model_list, lr=lr, A=A, B=B, closure=closure)
    else:   
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    print("optimizer初始化成功!")

    train_loss_history = []
    train_average_loss_history = []
    train_average_accuracy_history = []
    #test_average_loss_history = []
    #test_average_accuracy_history = []
    grad_norm_history = []

    train_loss_history_per_batch = [0.003] # 设置一个正常的初始值

    length = len(trainloader_list[0])

    progress_bar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in progress_bar:
        train_loss = 0.0

        for batch_idx, batch in enumerate(zip(*trainloader_list)):
            inputs = [
                data[0].to(device, non_blocking=True) for data in batch
            ]  # [data[0] for data in batch]
            labels = [
                data[1].to(device, non_blocking=True) for data in batch
            ]  # [data[1] for data in batch]
            h_data_train = inputs  # [tensor.to(device) for tensor in inputs]
            y_data_train = labels  # [tensor.to(device) for tensor in labels]
            loss = optimizer.step(closure=closure, lr=lr)
            train_loss += loss
            train_loss_history_per_batch.append(loss / length )
        train_loss = train_loss / length
        train_loss_history.append(train_loss)

        # train_average_loss, train_accuracy, test_average_loss, test_accuracy, global_gradient_norm = compute_loss_and_accuracy(
        #     model_class=model_class, model_list=model_list, testloader=testloader, full_trainloader=full_trainloader
        # )
        test_average_loss, test_accuracy = simple_compute_loss_and_accuracy(model_class=model_class, model_list=model_list, testloader=testloader)
        # train_average_loss_history.append(train_average_loss)
        # train_average_accuracy_history.append(train_accuracy)
        # test_average_loss_history.append(test_average_loss)
        # test_average_accuracy_history.append(test_accuracy)
        # grad_norm_history.append(global_gradient_norm)

        progress_bar.set_postfix(
            epoch=epoch + 1,
            train_loss=f"{train_loss_history[-1]:.4f}",
            # train_average_accuracy=f"{100 * train_average_accuracy_history[-1]:.4f}%",
            # test_loss=f"{test_average_loss_history[-1]:.4f}",
            # test_accuracy=f"{100 * test_average_accuracy_history[-1]:.4f}%",
            # grad_norm=f"{global_gradient_norm:.4f}",
        )

        today_date = datetime.now().strftime("%Y-%m-%d")
        
        # 在每个 epoch 结束后保存数据到 CSV
        df = pd.DataFrame({
            # "epoch": range(1, epoch + 2),  # epoch 从 1 开始
            # "train_loss(total)": train_loss_history,
            # "train_loss(average)": train_average_loss_history,
            # "train_accuracy(average)": train_average_accuracy_history,
            # "test_loss(average)": test_average_loss_history,
            # "test_accuracy(average)": test_average_accuracy_history,
            # "global_gradient_norm(average)": grad_norm_history,
            "train_loss_per_batch": train_loss_history_per_batch
        })
        csv_filename = f"{remark}, {algorithm}, lr={lr}, n_nodes={n}, batch_size={batch_size}, {today_date}.csv"
        #csv_filename = f"{algorithm}, lr={lr}, n_nodes={n}, batch_size={batch_size}, {today_date}.csv"
        csv_path = os.path.join(output_root, csv_filename)
        df.to_csv(csv_path, index=False)

    return df