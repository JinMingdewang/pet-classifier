"""
model.py - 模型定义与 Head 替换

【核心概念：迁移学习（Transfer Learning）】
ResNet-18 已经在 ImageNet（128万张图，1000类）上预训练过，
它的卷积层已经学会了"边缘检测→纹理识别→局部特征→语义特征"这套通用能力。

我们不需要从零训练一个模型，只需要：
1. 保留预训练的"特征提取器"（卷积层部分，叫 backbone）
2. 把最后的分类头（全连接层）替换成适合我们任务的结构

这叫"微调（Fine-tuning）"，是深度学习最常用的实战技巧之一。

【ResNet-18 结构速览】
输入: [B, 3, 224, 224]
  → conv1(3→64): [B, 64, 112, 112]
  → layer1(64→64): [B, 64, 56, 56]
  → layer2(64→128): [B, 128, 28, 28]
  → layer3(128→256): [B, 256, 14, 14]
  → layer4(256→512): [B, 512, 7, 7]
  → AvgPool: [B, 512, 1, 1]
  → Flatten: [B, 512]
  → fc(512→1000): [B, 1000]   ← 这一层我们要替换成 fc(512→37)
"""

import torch.nn as nn
from torchvision import models


def build_model(
    num_classes: int = 37,
    freeze_backbone: bool = False,
) -> nn.Module:
    """构建 ResNet-18 分类模型。

    参数:
        num_classes: 分类类别数（Oxford-IIIT Pet 有 37 类）
        freeze_backbone: 是否冻结 backbone 参数（消融实验用）
            - True: 只训练最后的全连接层（训练快，但精度上限低）
            - False: 全参数微调（训练慢，但精度更高）

    返回:
        修改后的 ResNet-18 模型

    【为什么要替换 fc 层？】
    原始 ResNet-18 的 fc 层输出 1000 维（对应 ImageNet 的 1000 个类别）。
    我们的任务是 37 类，所以必须替换成 512→37 的全连接层。
    新替换的 fc 层权重是随机初始化的，需要通过训练来学习。
    """
    # 加载 ImageNet 预训练权重
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 【Head 替换】：把最后的全连接层从 1000 维改成 num_classes 维
    # 这是迁移学习的标准操作——保留特征提取能力，重新学习分类头
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # in_features = 512（ResNet-18 最后一层卷积输出512个通道）

    # 【冻结 backbone】：消融实验对比"只训练fc"vs"全参数微调"
    if freeze_backbone:
        for name, param in model.named_parameters():
            # 只保留 fc 层的参数可训练，其他层全部冻结
            if "fc" not in name:
                param.requires_grad = False
        print("已冻结 backbone，仅训练全连接层")
    else:
        print("全参数微调模式")

    return model


def count_parameters(model: nn.Module) -> dict:
    """统计模型参数量（帮助理解模型复杂度）。

    返回:
        包含 total, trainable, frozen 参数量的字典
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"总参数量: {total:,}")
    print(f"可训练参数: {trainable:,}")
    print(f"冻结参数: {frozen:,}")
    print(f"可训练比例: {trainable / total * 100:.1f}%")

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


if __name__ == "__main__":
    # 快速验证模型结构
    model = build_model(num_classes=37)
    count_parameters(model)
    print(f"\n模型结构:\n{model}")

    # 验证输入输出维度
    import torch
    dummy_input = torch.randn(2, 3, 224, 224)  # [batch=2, channels=3, H=224, W=224]
    output = model(dummy_input)
    print(f"\n输入形状: {dummy_input.shape}")    # [2, 3, 224, 224]
    print(f"输出形状: {output.shape}")           # [2, 37]
