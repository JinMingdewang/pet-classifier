"""
train.py - 训练主入口（支持 argparse 参数控制）

【训练循环的核心逻辑】
每个 epoch 的训练循环长这样：
    for images, labels in train_loader:
        1. optimizer.zero_grad()   ← 清零梯度（PyTorch默认累积梯度）
        2. outputs = model(images) ← 前向传播，得到预测
        3. loss = criterion(outputs, labels) ← 计算损失
        4. loss.backward()         ← 反向传播，计算每个参数的梯度
        5. optimizer.step()        ← 用梯度更新参数

这四步是 PyTorch 训练的"铁律"，顺序不能乱。
"""

import os
import sys
import time
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 添加项目根目录到路径（方便 Colab 上运行）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import get_dataloaders, set_seed
from models.model import build_model, count_parameters
from utils.metrics import evaluate_model


def parse_args():
    """解析命令行参数。

    为什么用 argparse？
    因为消融实验需要反复修改超参数，如果每次都要改代码再运行，效率太低。
    通过命令行参数，一条命令就能切换实验配置：
        python train.py --label_smoothing 0.1
        python train.py --freeze_backbone
        python train.py --advanced_aug
    """
    parser = argparse.ArgumentParser(description="Oxford-IIIT Pet 分类训练脚本")

    # 数据相关
    parser.add_argument("--data_root", type=str, default="./data", help="数据集根目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")

    # 模型相关
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="冻结backbone，仅训练全连接层（消融实验）")

    # 训练相关
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "sgd"], help="优化器选择")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="权重衰减（L2正则化），缓解过拟合")

    # 消融实验开关
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="标签平滑系数（0=不平滑，0.1=常用值）")
    parser.add_argument("--mixup_alpha", type=float, default=0.0,
                        help="MixUp 的 alpha 参数（0=不用MixUp，0.2=常用值）")
    parser.add_argument("--advanced_aug", action="store_true",
                        help="启用 RandAugment 进阶数据增强")

    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--exp_name", type=str, default="baseline",
                        help="实验名称（用于区分TensorBoard日志）")
    parser.add_argument("--device", type=str, default="auto",
                        help="计算设备: auto / cuda / cpu")

    return parser.parse_args()


def mixup_data(images, labels, alpha=0.2):
    """MixUp 数据增强：将两张图片按比例混合，创造新的训练样本。

    【原理】
    随机取两张图片 A 和 B，按 lambda 比例混合：
        new_image = lambda * A + (1 - lambda) * B
        new_label = lambda * label_A + (1 - lambda) * label_B
    其中 lambda 从 Beta(alpha, alpha) 分布中采样。

    【为什么有效？】
    相当于在训练数据中插值，让模型见过更多"中间态"，
    迫使模型学习更平滑的决策边界，缓解过拟合。

    参数:
        images: [B, 3, 224, 224]
        labels: [B]
        alpha: Beta分布参数，越大混合越均匀

    返回:
        混合后的 images, labels_a, labels_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


def mixup_criterion(criterion, outputs, targets_a, targets_b, lam):
    """MixUp 的损失计算：两个目标的损失加权求和。"""
    return lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    mixup_alpha: float = 0.0,
) -> tuple:
    """训练一个 epoch。

    返回:
        (avg_loss, accuracy)
    """
    model.train()  # 切换到训练模式（开启 Dropout/BatchNorm 的训练统计）

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # ===== 训练循环四步曲 =====
        # 第1步：清零梯度
        # 为什么放在最前面？因为 PyTorch 默认累积梯度（设计初衷是为了支持
        # 梯度累积以节省显存），如果不清零，上一步的梯度会叠加到当前步上，
        # 导致梯度方向完全错误。
        optimizer.zero_grad()

        # MixUp 增强（如果启用）
        if mixup_alpha > 0:
            images, targets_a, targets_b, lam = mixup_data(images, labels, mixup_alpha)

        # 第2步：前向传播
        outputs = model(images)  # [B, 37]

        # 第3步：计算损失
        if mixup_alpha > 0:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, labels)

        # 第4步：反向传播 + 参数更新
        loss.backward()        # 计算梯度
        optimizer.step()       # 用梯度更新参数

        # ===== 统计 =====
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        if mixup_alpha > 0:
            correct += lam * (predicted == targets_a).sum().item() + \
                       (1 - lam) * (predicted == targets_b).sum().item()
        else:
            correct += (predicted == labels).sum().item()

        # 每 50 个 batch 打印一次进度
        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch [{epoch}] Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total

    # 写入 TensorBoard
    writer.add_scalar("Train/Loss", avg_loss, epoch)
    writer.add_scalar("Train/Accuracy", accuracy, epoch)

    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    criterion,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
) -> tuple:
    """在验证集上评估模型。

    返回:
        (avg_loss, top1_acc)
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total

    # 写入 TensorBoard
    writer.add_scalar("Val/Loss", avg_loss, epoch)
    writer.add_scalar("Val/Accuracy", accuracy, epoch)

    return avg_loss, accuracy


def main():
    args = parse_args()

    # 固定随机种子
    set_seed(args.seed)

    # 设备选择
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 打印实验配置
    print(f"\n{'='*50}")
    print(f"实验名称: {args.exp_name}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch Size: {args.batch_size}")
    print(f"优化器: {args.optimizer}, Weight Decay: {args.weight_decay}")
    print(f"标签平滑: {args.label_smoothing}")
    print(f"MixUp: {'开启' if args.mixup_alpha > 0 else '关闭'} (alpha={args.mixup_alpha})")
    print(f"进阶增强: {args.advanced_aug}")
    print(f"冻结Backbone: {args.freeze_backbone}")
    print(f"{'='*50}\n")

    # 1. 加载数据
    print("正在加载数据...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        advanced_aug=args.advanced_aug,
    )

    # Day 1 达标检查点：验证 Batch 形状
    for images, labels in train_loader:
        print(f"✓ Batch 形状验证: {images.shape}")  # 应该是 [32, 3, 224, 224]
        assert images.shape == (args.batch_size, 3, 224, 224), "Batch 形状不对！"
        break

    # 2. 构建模型
    model = build_model(
        num_classes=len(class_names),
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    count_parameters(model)

    # 3. 损失函数
    # 注意：nn.CrossEntropyLoss 内部已经包含了 Softmax，
    # 所以模型的输出不需要再经过 Softmax！这是新手常犯的错误。
    if args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print(f"使用 Label Smoothing CrossEntropy (smoothing={args.label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用标准 CrossEntropyLoss")

    # 4. 优化器
    # 只传 requires_grad=True 的参数，避免浪费计算在冻结的参数上
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )

    # 5. TensorBoard 日志
    log_dir = os.path.join("runs", args.exp_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志目录: {log_dir}")
    print("运行 tensorboard --logdir runs 可查看训练曲线\n")

    # 6. 训练循环
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\n开始训练（共 {args.epochs} 个 Epoch）...")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            mixup_alpha=args.mixup_alpha,
        )

        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer,
        )

        # 记录历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 打印本轮结果
        print(f"Epoch [{epoch:02d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}", end="")

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f" ★ 最优模型已保存 (Val Acc: {val_acc:.4f})")
        else:
            print()

    elapsed = time.time() - start_time
    print(f"\n训练完成！总耗时: {elapsed:.1f} 秒")
    print(f"最佳验证准确率: {best_val_acc:.4f}")

    # 7. 最终测试集评估
    print(f"\n{'='*50}")
    print("在测试集上评估最优模型...")
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    test_results = evaluate_model(model, test_loader, device)

    # 8. 保存实验结果
    results = {
        "exp_name": args.exp_name,
        "config": vars(args),
        "best_val_acc": best_val_acc,
        "test_top1_acc": test_results["top1_acc"],
        "test_top5_acc": test_results["top5_acc"],
        "test_macro_f1": test_results["macro_f1"],
        "training_time_sec": elapsed,
        "history": history,
    }

    result_path = f"results_{args.exp_name}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n实验结果已保存至: {result_path}")

    writer.close()
    print("完成！")


if __name__ == "__main__":
    import numpy as np  # mixup_data 需要
    main()
