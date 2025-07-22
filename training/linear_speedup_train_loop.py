# training/linear_speedup_train_loop.py

import torch
import os
import torch.nn as nn
import pandas as pd
from datasets.prepare_data import get_dataloaders
from utils.train_utils import get_first_batch, compute_loss_and_accuracy
from training.optimizer import PullDiag_GT, PullDiag_GD
from models.cnn import new_ResNet18
from models.fully_connected import FullyConnectedMNIST, two_layer_fc
from tqdm import tqdm
from datetime import datetime

def train_per_iteration(
    algorithm: str,
    lr: float,  
    A: torch.Tensor,  
    dataset_name: str,
    batch_size: int,
    num_epochs: int = 10,
    remark: str = "",
)-> pd.DataFrame:
    """
    执行逻辑和train函数相同, 只是在每个batch执行结束之后都计算一次avaerage loss
    即输出的变量是per_iteration记录的, 而不是每个epoch记录的

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

    print("每个节点分配到的图片数目是",50000//A.shape[0])
    today_date = datetime.now().strftime("%Y-%m-%d")
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    n = A.shape[0]
    A = torch.from_numpy(A).float().to(device)

    if dataset_name == "CIFAR10":
        model_list = [new_ResNet18().to(device) for _ in range(n)]
        trainloader_list, testloader, full_trainloader = get_dataloaders(
            n, dataset_name, batch_size
        )
        model_class = new_ResNet18
        output_root = "/root/GanLuo/ICML2025_project/outputs/CIFAR10_MG"
    elif dataset_name == "MNIST":
        model_list = [FullyConnectedMNIST().to(device) for _ in range(n)]
        trainloader_list, testloader, full_trainloader = get_dataloaders(
            n, dataset_name, batch_size
        )
        model_class = FullyConnectedMNIST
        #output_root = "/root/GanLuo/ICML2025_project/outputs/logs/MNIST"
        output_root = "/root/GanLuo/ICML2025_project/outputs/MNIST数据_MG/csv"
    
    batches_per_epoch = len(trainloader_list[0])
    print("每个epoch执行的iteration次数是", batches_per_epoch)
    
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

    epoch_list = []
    batch_list = []
    iteration_list = []
    train_loss_history = []
    train_average_loss_history = []
    train_average_accuracy_history = []
    test_average_loss_history = []
    test_average_accuracy_history = []

    progress_bar = tqdm(range(num_epochs), desc="Training Progress")

    total_iterations = 0

    for epoch in progress_bar:
        #train_loss = 0.0

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
            train_loss_history.append(round(loss, 4)) 
            # 这里每个batch结束后都记录一次loss
        #train_loss = train_loss / len(trainloader_list[0])
        #train_loss_history.append(train_loss)

            train_average_loss, train_accuracy, test_average_loss, test_accuracy = compute_loss_and_accuracy(
                model_class=model_class, model_list=model_list, testloader=testloader, full_trainloader=full_trainloader
            )

            total_iterations += 1
            epoch_list.append(epoch + 1)
            batch_list.append(batch_idx + 1)
            iteration_list.append(total_iterations)
            train_average_loss_history.append(round(train_average_loss, 4))
            train_average_accuracy_history.append(round(train_accuracy, 4))
            test_average_loss_history.append(round(test_average_loss, 4))
            test_average_accuracy_history.append(round(test_accuracy, 4))
            # 这里每个batch结束后都计算一次average loss, 先看看会不会让计算明显变慢吧

            df = pd.DataFrame({
                        "epoch": epoch_list,
                        "batch": batch_list,
                        "iteration": iteration_list,
                        "train_loss(total)": train_loss_history,
                        "train_loss(average)": train_average_loss_history,
                        "train_accuracy(average)": train_average_accuracy_history,
                        "test_loss(average)": test_average_loss_history,
                        "test_accuracy(average)": test_average_accuracy_history,
                    })
            csv_filename = f"{remark}_{algorithm}_lr={lr}_n={n}_bs={batch_size}_{today_date}.csv".replace(" ", "_")
            csv_path = os.path.join(output_root, csv_filename)
            df.to_csv(csv_path, index=False)


            progress_bar.set_postfix(
                epoch=epoch + 1,
                train_loss=f"{train_loss_history[-1]:.4f}",
                train_average_accuracy=f"{100 * train_average_accuracy_history[-1]:.4f}%",
                test_loss=f"{test_average_loss_history[-1]:.4f}",
                test_accuracy=f"{100 * test_average_accuracy_history[-1]:.4f}%",
            )
    return df