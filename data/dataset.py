"""
dataset.py - Oxford-IIIT Pet 数据集加载与分层划分
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


class TransformWrapper:
    """包装数据集，确保 transform 一定生效。
    不依赖 torchvision 内部属性名，直接在外层做变换。"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def set_seed(seed: int = 42) -> None:
    """固定所有随机种子，确保实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(advanced_aug: bool = False) -> dict:
    """定义训练集和验证/测试集的数据变换。"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_list = [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    if advanced_aug:
        train_list.insert(3, transforms.RandAugment())
        print("已启用 RandAugment 进阶数据增强")

    return {
        "train": transforms.Compose(train_list),
        "val_test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
    }


def get_dataloaders(
    data_root: str = "./data",
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
    advanced_aug: bool = False,
) -> tuple:
    """加载 Oxford-IIIT Pet 数据集并进行 70/15/15 分层划分。"""
    set_seed(seed)

    # 下载并加载完整数据集
    full_dataset = datasets.OxfordIIITPet(
        root=data_root,
        split="trainval",
        download=True,
    )

    # 提取所有样本的类别标签（兼容不同版本）
    if hasattr(full_dataset, '_labels'):
        all_labels = [int(t) for t in full_dataset._labels]
    elif hasattr(full_dataset, '_samples'):
        all_labels = [int(s[1]) for s in full_dataset._samples]
    else:
        all_labels = [int(full_dataset[i][1]) for i in range(len(full_dataset))]

    all_indices = list(range(len(full_dataset)))

    # 分层抽样：70% 训练，15% 验证，15% 测试
    train_idx, temp_idx = train_test_split(
        all_indices, train_size=0.70, stratify=all_labels, random_state=seed,
    )
    temp_labels = [all_labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=0.50, stratify=temp_labels, random_state=seed,
    )

    print(f"数据集划分完成: 训练={len(train_idx)}, 验证={len(val_idx)}, 测试={len(test_idx)}")

    # 防泄漏断言：三个集合必须两两不交
    assert not (set(train_idx) & set(val_idx)), "train 和 val 重叠！"
    assert not (set(train_idx) & set(test_idx)), "train 和 test 重叠！"
    assert not (set(val_idx) & set(test_idx)), "val 和 test 重叠！"

    # 获取 transforms
    transforms_dict = get_transforms(advanced_aug=advanced_aug)

    # 关键修复：必须用 Subset 按索引切分，否则三个 loader 都是完整数据集！
    # Subset(full_dataset, train_idx) = 只取 full_dataset 中索引为 train_idx 的样本
    train_subset = TransformWrapper(Subset(full_dataset, train_idx), transforms_dict["train"])
    val_subset = TransformWrapper(Subset(full_dataset, val_idx), transforms_dict["val_test"])
    test_subset = TransformWrapper(Subset(full_dataset, test_idx), transforms_dict["val_test"])

    # 创建 DataLoader
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    class_names = full_dataset.classes
    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    print(f"类别数量: {len(class_names)}")
    print(f"类别名称（前5个）: {class_names[:5]}")

    # 验证 Batch 形状
    for images, labels in train_loader:
        print(f"Batch 形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"标签示例: {labels[:8]}")
        break

    # 验证划分是否正确（关键！）
    print(f"\n训练集 batch 数: {len(train_loader)}")
    print(f"验证集 batch 数: {len(val_loader)}")
    print(f"测试集 batch 数: {len(test_loader)}")
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
