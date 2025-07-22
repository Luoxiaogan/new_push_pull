# training/special_train_loop.py

import torch
import os
import torch.nn as nn
import pandas as pd
from datasets.prepare_data import get_dataloaders
from utils.train_utils import get_first_batch
from training.optimizer import PullDiag_GT, PullDiag_GD
from models.cnn import new_ResNet18
from models.fully_connected import FullyConnectedMNIST, SimpleFCN
from tqdm import tqdm
from datetime import datetime
from typing import Tuple
from torch.cuda.amp import autocast

def new_compute_loss_and_accuracy(
    model_class, model_list, testloader, full_trainloader, use_amp=False, train_dataset=None
) -> Tuple[float, float, float, float]:
    # 使用 CrossEntropyLoss 作为默认损失函数
    criterion = nn.CrossEntropyLoss()

    # 确保模型在正确的设备上
    device = next(model_list[0].parameters()).device

    # Step 1: Compute the average of the parameters from all models
    avg_model = model_class().to(device)  # 创建新的模型实例，并将其移动到同一设备上
    avg_state_dict = avg_model.state_dict()  # 获取新模型的状态字典

    # 初始化 sum_state_dict
    sum_state_dict = {
        key: torch.zeros_like(param).to(device) for key, param in avg_state_dict.items()
    }

    # 汇总所有模型的参数
    for model in model_list:
        state_dict = model.state_dict()
        for key in sum_state_dict.keys():
            sum_state_dict[key] += state_dict[key].to(device)

    # 计算平均值
    num_models = len(model_list)
    avg_state_dict = {key: value / num_models for key, value in sum_state_dict.items()}

    # 将平均参数加载到新模型中
    avg_model.load_state_dict(avg_state_dict)
    return new_compute_full_gradient_norm(avg_model, train_dataset, criterion, batch_size=None, device="cuda")

import copy
import torch
from torch.utils.data import DataLoader

def new_compute_full_gradient_norm(model, train_dataset, criterion, batch_size=None, device="cuda"):
    """
    计算模型在整个训练集上的梯度F范数（所有参数梯度元素的平方和开根号）
    
    参数:
        model: 原始模型（不会被修改）
        train_dataset: 训练集数据集
        criterion: 损失函数
        batch_size: 如果为None则使用全量数据，否则使用指定batch_size（内存不足时使用）
        device: 计算设备
    
    返回:
        float: 全局梯度范数（Frobenius范数）
    """
    # 深拷贝模型以避免影响原始模型
    model_copy = copy.deepcopy(model).to(device)
    model_copy.train()
    
    # 自动确定batch_size（全量数据或用户指定）
    use_full_batch = batch_size is None
    effective_bs = len(train_dataset) if use_full_batch else batch_size
    
    # 创建数据加载器（全量数据时关闭shuffle和drop_last）
    loader = DataLoader(train_dataset,
                        batch_size=effective_bs,
                        shuffle=False,
                        drop_last=False,
                        num_workers=12,
                        pin_memory=True)
    
    # 初始化梯度缓冲区
    for param in model_copy.parameters():
        param.grad = torch.zeros_like(param.data)
    
    total_samples = 0  # 已处理样本计数
    for inputs, labels in loader:
        # 将数据转移到设备
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # 展平图像（适配全连接网络）
        batch_size = inputs.shape[0]
        inputs = inputs.view(batch_size, -1)
        
        # 前向传播
        outputs = model_copy(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播（计算梯度并累加）
        loss.backward()
        
        # 记录已处理样本数
        total_samples += batch_size
    
    # 验证数据完整性
    assert total_samples == len(train_dataset), "数据存在缺失"
    
    # 计算全局梯度范数（所有参数梯度的平方和开根号）
    total_norm_sq = 0.0
    for param in model_copy.parameters():
        if param.grad is not None:
            grad = param.grad.data
            total_norm_sq += torch.sum(grad ** 2).item()
    
    return total_norm_sq ** 0.5



def special_train(
    algorithm: str,
    lr: float,  
    A: torch.Tensor,  
    dataset_name: str,
    batch_size: int,
    num_epochs: int = 10,
    remark: str = "",
)-> pd.DataFrame:
    """
    执行训练过程。

    Args:
        algorithm (str): 算法名称 ('PullDiag_GT' 或 'PullDiag_GD')
        lr (float): 学习率
        model_list (list): 模型列表
        A (torch.Tensor): 混合矩阵
        dataloaders (list): 训练数据加载器列表
        test_dataloader (DataLoader): 测试数据加载器
        num_epochs (int): 训练轮数
        remark (str): 备注
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    n = A.shape[0]
    A = torch.from_numpy(A).float().to(device)

    if dataset_name == "CIFAR10":
        model_list = [new_ResNet18().to(device) for _ in range(n)]
        trainloader_list, testloader, full_trainloader, trainset = get_dataloaders(
            n, dataset_name, batch_size, repeat=1
        )
        model_class = new_ResNet18
        output_root = "/root/GanLuo/ICML2025_project/outputs/logs/CIFAR10"
    elif dataset_name == "MNIST":
        model_list = [SimpleFCN().to(device) for _ in range(n)]
        trainloader_list, testloader, full_trainloader, trainset = get_dataloaders(
            n, dataset_name, batch_size, repeat=1
        )
        model_class = SimpleFCN
        #output_root = "/root/GanLuo/ICML2025_project/outputs/logs/MNIST"
        output_root = "/root/GanLuo/ICML2025_project/outputs/linear_speedup/test_for_best_lr"
    
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
    else:   
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    print("optimizer初始化成功!")

    grad_norm_history = []

    progress_bar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in progress_bar:

        for batch_idx, batch in enumerate(zip(*trainloader_list)):
            inputs = [
                data[0].to(device, non_blocking=True) for data in batch
            ]  # [data[0] for data in batch]
            labels = [
                data[1].to(device, non_blocking=True) for data in batch
            ]  # [data[1] for data in batch]
            h_data_train = inputs  # [tensor.to(device) for tensor in inputs]
            y_data_train = labels  # [tensor.to(device) for tensor in labels]
            optimizer.step(closure=closure, lr=lr)

        global_gradient_norm = new_compute_loss_and_accuracy(
            model_class=model_class, model_list=model_list, testloader=testloader, full_trainloader=full_trainloader, train_dataset=trainset
        )
        grad_norm_history.append(global_gradient_norm)

        progress_bar.set_postfix(
            epoch=epoch + 1,
            grad_norm=f"{global_gradient_norm}",
        )

        today_date = datetime.now().strftime("%Y-%m-%d")
        
        # 在每个 epoch 结束后保存数据到 CSV
        df = pd.DataFrame({
            "epoch": range(1, epoch + 2),  # epoch 从 1 开始
            "global_gradient_norm(average)": grad_norm_history,
        })
        csv_filename = f"只含grad_norm, {remark}, {algorithm}, lr={lr}, n_nodes={n}, batch_size={batch_size}, {today_date}.csv"
        #csv_filename = f"{algorithm}, lr={lr}, n_nodes={n}, batch_size={batch_size}, {today_date}.csv"
        csv_path = os.path.join(output_root, csv_filename)
        df.to_csv(csv_path, index=False)

    return df