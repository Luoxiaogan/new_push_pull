# datasets/prepare_data.py

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from typing import Tuple, List

# MNIST transforms
""" MNIST_transform_train = transforms.Compose(
    [
        transforms.RandomCrop(28, padding=4),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
) """

MNIST_transform_train = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

MNIST_transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

# CIFAR-10 transforms
cifar10_transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

cifar10_transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


def get_dataloaders(
    n: int, dataset_name: str, batch_size: int, repeat: int = 1
) -> Tuple[List[torch.utils.data.DataLoader], torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    if dataset_name == "CIFAR10":
        transform_train, transform_test = (
            cifar10_transform_train,
            cifar10_transform_test,
        )
        trainset = torchvision.datasets.CIFAR10(
            root="/home/lg/ICML2025_project/data/raw/CIFAR10",
            train=True,
            download=False,
            transform=transform_train,
        )
        testset = torchvision.datasets.CIFAR10(
            root="/home/lg/ICML2025_project/data/raw/CIFAR10",
            train=False,
            download=False,
            transform=transform_test,
        )
    elif dataset_name == "MNIST":
        transform_train, transform_test = MNIST_transform_train, MNIST_transform_test
        trainset = torchvision.datasets.MNIST(
            root="/home/lg/ICML2025_project/data/raw/MNIST",
            train=True,
            download=False,
            transform=transform_train,
        )
        testset = torchvision.datasets.MNIST(
            root="/home/lg/ICML2025_project/data/raw/MNIST",
            train=False,
            download=False,
            transform=transform_test,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Save the original trainset for full_trainloader
    original_trainset = trainset

    # Repeat the training dataset if repeat > 1
    if repeat > 1:
        trainset = torch.utils.data.ConcatDataset([trainset] * repeat)

    total_train_size = len(trainset)
    subset_sizes = [
        total_train_size // n + (1 if i < total_train_size % n else 0) for i in range(n)
    ]

    subsets = torch.utils.data.random_split(trainset, subset_sizes, generator=generator)

    trainloader_list = [
        torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            generator=generator,
        )
        for subset in subsets
    ]

    # Create a DataLoader for the full training set using the original trainset
    full_trainloader = torch.utils.data.DataLoader(
        original_trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        generator=generator,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        generator=generator,
    )

    return trainloader_list, testloader, full_trainloader

def get_dataloaders_fixed_batch(
    n: int, dataset_name: str, batch_size: int, repeat: int = 1
) -> Tuple[List[torch.utils.data.DataLoader], torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    if dataset_name == "CIFAR10":
        transform_train, transform_test = (
            cifar10_transform_train,
            cifar10_transform_test,
        )
        trainset = torchvision.datasets.CIFAR10(
            root="/home/lg/ICML2025_project/data/raw/CIFAR10",
            train=True,
            download=False,
            transform=transform_train,
        )
        testset = torchvision.datasets.CIFAR10(
            root="/home/lg/ICML2025_project/data/raw/CIFAR10",
            train=False,
            download=False,
            transform=transform_test,
        )
    elif dataset_name == "MNIST":
        transform_train, transform_test = MNIST_transform_train, MNIST_transform_test
        trainset = torchvision.datasets.MNIST(
            root="/home/lg/ICML2025_project/data/raw/MNIST",
            train=True,
            download=False,
            transform=transform_train,
        )
        testset = torchvision.datasets.MNIST(
            root="/home/lg/ICML2025_project/data/raw/MNIST",
            train=False,
            download=False,
            transform=transform_test,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Save the original trainset for full_trainloader
    original_trainset = trainset

    # Repeat the training dataset if repeat > 1
    if repeat > 1:
        trainset = torch.utils.data.ConcatDataset([trainset] * repeat)

    total_train_size = len(trainset)
    subset_sizes = [
        total_train_size // n + (1 if i < total_train_size % n else 0) for i in range(n)
    ]

    subsets = torch.utils.data.random_split(trainset, subset_sizes, generator=generator)

    trainloader_list = [
        torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            generator=generator,
            drop_last=True, # 关键在于设置 drop_last=True
        )
        for subset in subsets
    ]

    # Create a DataLoader for the full training set using the original trainset
    full_trainloader = torch.utils.data.DataLoader(
        original_trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        generator=generator,
        drop_last=True, # 为保证一致性，也设置 drop_last=True
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        generator=generator,
    )

    return trainloader_list, testloader, full_trainloader


def get_dataloaders_high_hetero(
    n: int, dataset_name: str, batch_size: int, repeat: int = 1
) -> Tuple[List[torch.utils.data.DataLoader], torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    # Load dataset
    if dataset_name == "CIFAR10":
        transform_train, transform_test = (
            cifar10_transform_train,
            cifar10_transform_test,
        )
        trainset = torchvision.datasets.CIFAR10(
            root="/root/GanLuo/ICML2025_project/data/raw/CIFAR10",
            train=True,
            download=False,
            transform=transform_train,
        )
        testset = torchvision.datasets.CIFAR10(
            root="/root/GanLuo/ICML2025_project/data/raw/CIFAR10",
            train=False,
            download=False,
            transform=transform_test,
        )
        num_classes = 10
    elif dataset_name == "MNIST":
        transform_train, transform_test = MNIST_transform_train, MNIST_transform_test
        trainset = torchvision.datasets.MNIST(
            root="/root/GanLuo/ICML2025_project/data/raw/MNIST",
            train=True,
            download=False,
            transform=transform_train,
        )
        testset = torchvision.datasets.MNIST(
            root="/root/GanLuo/ICML2025_project/data/raw/MNIST",
            train=False,
            download=False,
            transform=transform_test,
        )
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Save original trainset for full_trainloader
    original_trainset = trainset

    # Repeat the training dataset if repeat > 1
    if repeat > 1:
        trainset = torch.utils.data.ConcatDataset([trainset] * repeat)

    # Get labels and create class-specific indices
    labels = np.array(trainset.targets)
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    # Create heterogeneous distributions for each node
    subsets = []
    total_size = len(trainset)
    base_size = total_size // n
    
    # Generate Dirichlet distribution for class proportions across nodes
    alpha = 0.5  # Lower alpha means higher heterogeneity
    class_dist = np.random.dirichlet([alpha] * n, num_classes)
    
    # Assign samples to each node
    for node in range(n):
        node_indices = []
        node_size = base_size + (1 if node < total_size % n else 0)
        
        # Calculate target number of samples per class for this node
        target_dist = class_dist[:, node] * node_size
        
        for cls in range(num_classes):
            num_samples = int(target_dist[cls])
            available_indices = class_indices[cls]
            
            if len(available_indices) > 0:
                selected = np.random.choice(
                    available_indices,
                    size=min(num_samples, len(available_indices)),
                    replace=False
                )
                node_indices.extend(selected)
                # Remove used indices
                class_indices[cls] = np.setdiff1d(class_indices[cls], selected)
                
        subsets.append(torch.utils.data.Subset(trainset, node_indices))

    # Create DataLoaders for each subset
    trainloader_list = [
        torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            generator=generator,
        )
        for subset in subsets
    ]

    # Full training set DataLoader
    full_trainloader = torch.utils.data.DataLoader(
        original_trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        generator=generator,
    )

    # Test set DataLoader
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        generator=generator,
    )

    return trainloader_list, testloader, full_trainloader

import secrets

# def get_dataloaders_high_hetero_fixed_batch(
#     n: int, dataset_name: str, batch_size: int, alpha: float = 0.5, repeat: int = 1
# ) -> Tuple[List[torch.utils.data.DataLoader], torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
#     # seed = secrets.randbelow(100) + 1  # 限制在1到10000之间 # 确保是正数
#     # print("seed = ", seed)
#     # torch.manual_seed(seed)
#     # np.random.seed(seed)
#     # random.seed(seed)

#     # generator = torch.Generator()
#     # generator.manual_seed(seed)

#     # Load dataset
#     if dataset_name == "CIFAR10":
#         transform_train, transform_test = (
#             cifar10_transform_train,
#             cifar10_transform_test,
#         )
#         trainset = torchvision.datasets.CIFAR10(
#             root="/home/lg/ICML2025_project/data/raw/CIFAR10",
#             train=True,
#             download=False,
#             transform=transform_train,
#         )
#         testset = torchvision.datasets.CIFAR10(
#             root="/home/lg/ICML2025_project/data/raw/CIFAR10",
#             train=False,
#             download=False,
#             transform=transform_test,
#         )
#         num_classes = 10
#     elif dataset_name == "MNIST":
#         transform_train, transform_test = MNIST_transform_train, MNIST_transform_test
#         trainset = torchvision.datasets.MNIST(
#             root="/home/lg/ICML2025_project/data/raw/MNIST",
#             train=True,
#             download=False,
#             transform=transform_train,
#         )
#         testset = torchvision.datasets.MNIST(
#             root="/home/lg/ICML2025_project/data/raw/MNIST",
#             train=False,
#             download=False,
#             transform=transform_test,
#         )
#         num_classes = 10
#     else:
#         raise ValueError(f"Unsupported dataset: {dataset_name}")

#     # Save original trainset for full_trainloader
#     original_trainset = trainset

#     # Repeat the training dataset if repeat > 1
#     if repeat > 1:
#         trainset = torch.utils.data.ConcatDataset([trainset] * repeat)

#     # Get labels and create class-specific indices
#     labels = np.array(trainset.targets)
#     class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

#     # Create heterogeneous distributions for each node
#     subsets = []
#     total_size = len(trainset)
#     base_size = total_size // n

#     # Generate Dirichlet distribution for class proportions across nodes
#     class_dist = np.random.dirichlet([alpha] * n, num_classes)

#     # Assign samples to each node
#     remaining_indices = list(range(total_size))
#     for node in range(n):
#         node_indices = []
#         node_size = base_size + (1 if node < total_size % n else 0)

#         # Calculate target number of samples per class for this node
#         target_dist = class_dist[:, node] * node_size

#         available_node_indices = []
#         for cls in range(num_classes):
#             num_samples = int(target_dist[cls])
#             global_available_indices = class_indices[cls]
#             intersected_indices = np.intersect1d(remaining_indices, global_available_indices)

#             if len(intersected_indices) > 0:
#                 selected = np.random.choice(
#                     intersected_indices,
#                     size=min(num_samples, len(intersected_indices)),
#                     replace=False
#                 )
#                 node_indices.extend(selected)
#                 remaining_indices = list(set(remaining_indices) - set(selected))

#         subsets.append(torch.utils.data.Subset(trainset, node_indices))

#     # Create DataLoaders for each subset
#     trainloader_list = [
#         torch.utils.data.DataLoader(
#             subset,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=4,
#             pin_memory=True,
#             prefetch_factor=2,
#             persistent_workers=True,
#             drop_last=True, # 设置 drop_last=True
#             #generator=generator,
#         )
#         for subset in subsets
#     ]

#     # Full training set DataLoader
#     full_trainloader = torch.utils.data.DataLoader(
#         original_trainset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True,
#         prefetch_factor=2,
#         persistent_workers=True,
#         drop_last=True, # 为保证一致性，也设置 drop_last=True
#         #generator=generator,
#     )

#     # Test set DataLoader
#     testloader = torch.utils.data.DataLoader(
#         testset,
#         batch_size=100,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True,
#         prefetch_factor=2,
#         persistent_workers=True,
#         #generator=generator,
#     )

#     return trainloader_list, testloader, full_trainloader



import torch
import torchvision
import numpy as np
import random # 导入 random 模块
from typing import List, Tuple
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset # 显式导入需要的类

def get_dataloaders_high_hetero_fixed_batch( # 函数名稍作修改以示区别
    n: int,
    dataset_name: str,
    batch_size: int,
    alpha: float = 0.5,
    repeat: int = 1,
    seed: int = 42 # 添加 seed 参数，并设置默认值
) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
    """
    生成具有受种子控制的异构数据分布的DataLoader。

    Args:
        n (int): 客户端节点数量。
        dataset_name (str): 数据集名称 ("CIFAR10" 或 "MNIST")。
        batch_size (int): 训练DataLoader的批量大小。
        alpha (float, optional): Dirichlet分布的集中度参数。默认为 0.5。
        repeat (int, optional): 重复训练数据集的次数。默认为 1。
        seed (int, optional): 用于可复现性的随机种子。默认为 42。

    Returns:
        Tuple[List[DataLoader], DataLoader, DataLoader]:
            一个包含以下内容的元组：
            - 每个节点的训练DataLoader列表。
            - 测试DataLoader。
            - 完整的训练DataLoader（使用原始的、未重复的数据集）。
    """
    # --- 设置种子 ---
    print(f"函数内设置种子为: {seed}") # 可选：打印正在使用的种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 如果使用CUDA，也设置GPU种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 适用于多GPU设置
        # 为了确保卷积操作的可复现性（可能会牺牲性能）
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # 为 DataLoader 创建一个生成器对象
    generator = torch.Generator()
    generator.manual_seed(seed)
    # --- 结束设置种子 ---

    # --- 数据集加载 ---
    # 注意：建议将 root 路径也作为参数传入，而不是硬编码
    data_root = "/home/lg/ICML2025_project/data/raw" # 定义根目录

    if dataset_name == "CIFAR10":
        transform_train, transform_test = (
            cifar10_transform_train,
            cifar10_transform_test,
        )
        try:
            trainset = torchvision.datasets.CIFAR10(
                root=f"{data_root}/CIFAR10",
                train=True,
                download=False, # 通常在生产环境中设为False
                transform=transform_train,
            )
            testset = torchvision.datasets.CIFAR10(
                root=f"{data_root}/CIFAR10",
                train=False,
                download=False,
                transform=transform_test,
            )
        except Exception as e:
            print(f"加载 CIFAR10 数据集失败，请确保路径 {data_root}/CIFAR10 正确且包含数据。错误：{e}")
            raise
        num_classes = 10
    elif dataset_name == "MNIST":
        transform_train, transform_test = MNIST_transform_train, MNIST_transform_test
        try:
            trainset = torchvision.datasets.MNIST(
                root=f"{data_root}/MNIST",
                train=True,
                download=False,
                transform=transform_train,
            )
            testset = torchvision.datasets.MNIST(
                root=f"{data_root}/MNIST",
                train=False,
                download=False,
                transform=transform_test,
            )
        except Exception as e:
            print(f"加载 MNIST 数据集失败，请确保路径 {data_root}/MNIST 正确且包含数据。错误：{e}")
            raise
        num_classes = 10
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    # 保存原始训练集用于 full_trainloader
    original_trainset = trainset

    # 如果 repeat > 1，重复训练数据集
    if repeat > 1:
        original_labels = np.array(original_trainset.targets)
        trainset = ConcatDataset([original_trainset] * repeat)
        # 从 ConcatDataset 正确提取标签
        labels = np.concatenate([original_labels] * repeat)
    else:
        # targets 属性可能因 torchvision 版本而异，确保使用正确的方式访问标签
        if hasattr(trainset, 'targets'):
             labels = np.array(trainset.targets)
        elif hasattr(trainset, 'labels'):
             labels = np.array(trainset.labels)
        else:
            # 如果没有直接的标签属性，需要迭代获取
            print("警告：无法直接访问数据集标签，尝试迭代获取...")
            labels = np.array([sample[1] for sample in trainset])


    # 获取类别特定的索引
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    # --- 为每个节点创建异构分布 ---
    subsets = []
    total_size = len(trainset) # 使用可能重复的数据集的总大小
    indices_per_node = [[] for _ in range(n)] # 存储每个节点的索引列表

    # 生成跨节点的类别比例的Dirichlet分布
    # 这里将使用前面 np.random 设置的种子
    class_dist = np.random.dirichlet([alpha] * num_classes, n).T # Shape: (num_classes, n)

    # 按类别分配样本给节点
    all_indices_shuffled = np.arange(total_size)
    # 使用 np.random (已播种) 来打乱索引，确保分配过程也是可复现的
    np.random.shuffle(all_indices_shuffled)
    
    # 计算每个节点每个类别的目标样本数
    node_class_samples_target = (class_dist / class_dist.sum(axis=0, keepdims=True)) \
                               * (total_size / n) # 粗略的目标，后续会调整
    node_class_samples_target = node_class_samples_target.round().astype(int)

    # 确保总数匹配或调整
    current_total = node_class_samples_target.sum()
    diff = total_size - current_total
    # 简单调整：随机增减样本直到总数匹配
    if diff != 0:
        adjustment_indices = np.random.choice(n * num_classes, abs(diff), replace=True)
        adjustments = np.zeros_like(node_class_samples_target.flatten())
        for idx in adjustment_indices:
            adjustments[idx] += np.sign(diff)
        node_class_samples_target = (node_class_samples_target.flatten() + adjustments).reshape(node_class_samples_target.shape)
        node_class_samples_target = np.maximum(0, node_class_samples_target) # 确保非负
    
    # 确保最终目标总数精确匹配
    final_diff = total_size - node_class_samples_target.sum()
    if final_diff != 0:
        # 在某个节点/类别上增加/减少以精确匹配
        adjust_node, adjust_class = np.random.randint(n), np.random.randint(num_classes)
        node_class_samples_target[adjust_class, adjust_node] += final_diff
        node_class_samples_target[adjust_class, adjust_node] = max(0, node_class_samples_target[adjust_class, adjust_node])
        
    # 验证最终目标总数
    assert node_class_samples_target.sum() == total_size, f"目标总数 {node_class_samples_target.sum()} 与数据集大小 {total_size} 不匹配"


    # 按类别划分索引
    indices_by_class = [list(idx) for idx in class_indices]
    # 打乱每个类别内的索引顺序 (使用 random.shuffle，已播种)
    for idx_list in indices_by_class:
        random.shuffle(idx_list)

    # 分配索引到节点
    class_pointers = [0] * num_classes
    for node_idx in range(n):
        node_indices = []
        for class_idx in range(num_classes):
            target_count = node_class_samples_target[class_idx, node_idx]
            start = class_pointers[class_idx]
            end = start + target_count
            
            # 从该类别的打乱索引中取出所需数量
            assigned_indices = indices_by_class[class_idx][start:end]
            node_indices.extend(assigned_indices)
            
            # 更新该类别的指针
            class_pointers[class_idx] = end
            
        # 打乱单个节点的索引顺序（可选，但通常推荐）
        random.shuffle(node_indices)
        
        subsets.append(Subset(trainset, node_indices))
        # print(f"节点 {node_idx}: 分配了 {len(node_indices)} 个样本") # 调试信息

    # --- 创建 DataLoaders ---
    # 检查 persistent_workers 的条件
    use_persistent_workers = True if (torch.cuda.is_available() or num_workers > 0) else False
    num_workers = 4 # 可以考虑也作为参数传入

    trainloader_list = [
        DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True, # 需要随机打乱
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), # 仅在GPU可用时建议开启
            prefetch_factor=2 if num_workers > 0 else None, # 仅在有worker时有效
            persistent_workers=use_persistent_workers if num_workers > 0 else False, # 仅在有worker时有效
            drop_last=True, # 丢弃最后一个不完整的批次
            generator=generator, # 使用播种的生成器
        )
        for subset in subsets
    ]

    # 完整的训练集 DataLoader (使用原始、未重复的数据集)
    full_trainloader = DataLoader(
        original_trainset, # 使用原始数据集
        batch_size=batch_size,
        shuffle=True, # 也打乱这个加载器
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=use_persistent_workers if num_workers > 0 else False,
        drop_last=True, # 保持一致性
        generator=generator, # 使用相同的生成器
    )

    # 测试集 DataLoader
    testloader = DataLoader(
        testset,
        batch_size=100, # 测试时批量大小可以不同
        shuffle=False, # 测试集不需要打乱
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=use_persistent_workers if num_workers > 0 else False,
        # generator=generator, # shuffle=False时不需要generator
    )
    # --- 结束创建 DataLoaders ---

    return trainloader_list, testloader, full_trainloader



import seaborn as sns
import matplotlib.pyplot as plt

def visualize_heatmap(trainloader_list, num_classes=10):
    num_nodes = len(trainloader_list)
    class_counts = np.zeros((num_nodes, num_classes))

    # 统计类别分布
    for node_idx, loader in enumerate(trainloader_list):
        for _, labels in loader:
            for cls in range(num_classes):
                class_counts[node_idx, cls] += (labels == cls).sum().item()

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(class_counts, annot=True, fmt='.0f', cmap='YlOrRd')
    plt.xlabel('类别')
    plt.ylabel('节点')
    plt.title('类别分布热力图')
    plt.show()

# 调用示例
# visualize_heatmap(trainloader_list)

from scipy.stats import entropy

def visualize_kl_divergence(trainloader_list, num_classes=10):
    num_nodes = len(trainloader_list)
    class_counts = np.zeros((num_nodes, num_classes))

    # 统计类别分布
    for node_idx, loader in enumerate(trainloader_list):
        for _, labels in loader:
            for cls in range(num_classes):
                class_counts[node_idx, cls] += (labels == cls).sum().item()

    # 计算比例
    class_ratios = class_counts / class_counts.sum(axis=1, keepdims=True)
    mean_dist = class_ratios.mean(axis=0)  # 平均分布

    # 计算KL散度
    kl_divs = [entropy(class_ratios[i], mean_dist) for i in range(num_nodes)]

    # 绘制
    plt.bar(range(num_nodes), kl_divs)
    plt.xticks(range(num_nodes), [f'节点 {i+1}' for i in range(num_nodes)])
    plt.ylabel('KL散度')
    plt.title('各节点与平均分布的异质性')
    plt.show()

# 调用示例
# visualize_kl_divergence(trainloader_list)