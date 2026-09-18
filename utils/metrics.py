"""
metrics.py - 评估指标计算与可视化工具

【核心概念】
1. Top-1 Accuracy：模型预测概率最高的类别恰好是正确答案的比例。
2. Top-5 Accuracy：正确答案出现在模型预测概率前5名中的比例。
   对于37类细粒度分类，Top-5更有意义——如果模型把"金毛"排第2、"拉布拉多"排第1，
   虽然Top-1错了，但Top-5是对的，说明模型"方向没错"。
3. Macro-F1：每个类别分别算F1分数，然后取平均。
   F1 = 2 * Precision * Recall / (Precision + Recall)
   Macro 意味着每个类别权重相同，不会被大类"稀释"。
   这对类别不平衡的数据集很重要。
4. 混淆矩阵：行是真实标签，列是预测标签。对角线越亮越好，
   非对角线的亮块就是模型容易搞混的类别对。
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    f1_score,
    confusion_matrix,
)
from typing import List


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """在给定数据集上评估模型，返回 Top-1 Acc, Top-5 Acc, Macro-F1。

    参数:
        model: 训练好的模型
        dataloader: 数据加载器（验证集或测试集）
        device: 计算设备

    返回:
        包含 top1_acc, top5_acc, macro_f1, all_preds, all_labels 的字典

    【关键细节】
    - model.eval()：切换到评估模式，关闭 Dropout 和 BatchNorm 的随机行为
    - torch.no_grad()：不计算梯度，节省显存+加速
    - 这两个操作缺一不可，是新手最常犯的错误之一
    """
    model.eval()  # 必须！关闭 Dropout/BatchNorm 的训练行为

    all_preds = []    # 存储所有预测结果
    all_labels = []   # 存储所有真实标签
    all_probs = []    # 存储所有预测概率（用于Top-5计算）

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)             # [B, 37] 原始logits
        probs = torch.softmax(outputs, dim=1)  # [B, 37] 转为概率分布

        _, predicted = torch.max(outputs, 1)    # 取概率最大的类别

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算指标
    top1_acc = accuracy_score(all_labels, all_preds)
    top5_acc = top_k_accuracy_score(all_labels, all_probs, k=5)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    print(f"Macro-F1:       {macro_f1:.4f}")

    return {
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "macro_f1": macro_f1,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: List[str],
    save_path: str = "confusion_matrix.png",
    top_k_confused: int = 5,
) -> None:
    """绘制混淆矩阵热力图，并输出最易混淆的类别对。

    参数:
        labels: 真实标签数组
        preds: 预测标签数组
        class_names: 类别名称列表
        save_path: 图片保存路径
        top_k_confused: 输出前K个最易混淆的类别对
    """
    # 计算混淆矩阵：cm[i][j] 表示真实类别i被预测为类别j的次数
    cm = confusion_matrix(labels, preds)

    # 归一化：每行除以该行的总和，得到"真实类别i被预测为j的比例"
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm_normalized,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        fmt=".2f",
        ax=ax,
        annot=False,     # 37类太多，不在格子里标数字
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix (Normalized)", fontsize=14)

    # 旋转x轴标签，避免重叠
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵已保存至: {save_path}")

    # 找出最易混淆的类别对
    # 策略：在对角线为0的情况下，找归一化混淆矩阵中值最大的非对角元素
    np.fill_diagonal(cm_normalized, 0)  # 去掉正确分类的对角线

    print(f"\n最易混淆的 Top-{top_k_confused} 类别对:")
    for _ in range(top_k_confused):
        max_idx = np.unravel_index(np.argmax(cm_normalized), cm_normalized.shape)
        true_cls, pred_cls = max_idx
        score = cm_normalized[max_idx]
        if score == 0:
            break
        print(f"  {class_names[true_cls]:30s} → {class_names[pred_cls]:30s}  混淆率: {score:.2%}")
        cm_normalized[max_idx] = 0  # 标记已输出，找下一个
