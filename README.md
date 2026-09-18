# 基于深度学习的牛津宠物细粒度分类

> **姓名**：王俊晨 | **学号**：W124301195 | **安徽大学 人工智能学院**

## 项目概述

基于 ResNet-18 预训练模型，在 Oxford-IIIT Pet 数据集（37 类细粒度猫狗品种分类）上完成基线训练、消融实验与可视化分析。

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- GPU（推荐 Google Colab / Kaggle 免费 T4）

## 一键复现

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练 Baseline

```bash
python train.py --exp_name baseline --epochs 15 --lr 1e-4 --optimizer adamw
```

### 3. 消融实验

**消融实验 A：冻结 Backbone vs 全参数微调**

```bash
# 仅微调全连接层
python train.py --exp_name freeze_backbone --epochs 15 --lr 1e-3 --freeze_backbone

# 全参数微调（即 Baseline）
python train.py --exp_name baseline --epochs 15 --lr 1e-4
```

**消融实验 B：标签平滑**

```bash
python train.py --exp_name label_smoothing --epochs 15 --lr 1e-4 --label_smoothing 0.1
```

**消融实验 C：进阶数据增强（RandAugment）**

```bash
python train.py --exp_name advanced_aug --epochs 15 --lr 1e-4 --advanced_aug
```

**消融实验 D：MixUp**

```bash
python train.py --exp_name mixup --epochs 15 --lr 1e-4 --mixup_alpha 0.2
```

### 4. 评估与可视化

```bash
python evaluate.py --model_path best_model.pth --exp_name baseline
```

生成结果：
- `confusion_matrix.png` — 混淆矩阵热力图
- `gradcam_correct.png` — 正确预测样本的 Grad-CAM
- `gradcam_wrong.png` — 错误预测样本的 Grad-CAM

### 5. 查看训练曲线

```bash
tensorboard --logdir runs
```

## 项目结构

```
pet_project/
├── data/
│   └── dataset.py       # 数据加载、70:15:15 分层划分与 Transform
├── models/
│   └── model.py         # ResNet-18 模型定义与 Head 替换
├── utils/
│   ├── metrics.py       # Top-1/Top-5 Acc, Macro-F1, 混淆矩阵
│   └── gradcam.py       # Grad-CAM 特征热力图可视化
├── train.py             # 训练主入口（支持 argparse 参数控制）
├── evaluate.py          # 评估与可视化脚本
├── requirements.txt     # 依赖清单
└── README.md            # 项目说明
```

## 实验结果

| 实验配置 | 可训练参数 | Top-1 Acc | Top-5 Acc | Macro-F1 | 训练耗时 |
|:---|:---|:---|:---|:---|:---|
| Baseline (ResNet-18 全参数微调) | 11.20M (100%) | 89.67% | 99.09% | 89.59% | 343s (~6 min) |
| 冻结 Backbone (仅微调 fc) | 18.9K (0.2%) | 88.95% | 99.28% | 89.01% | 335s (~6 min) |
| + Label Smoothing (0.1) | 11.20M (100%) | **90.58%** | 98.55% | **90.49%** | 373s (~6 min) |

**硬件**：NVIDIA RTX 4070 Laptop GPU · Python 3.13 · PyTorch 2.6.0+cu124

## AI 辅助编程声明

本项目在以下环节使用了 AI 工具辅助：
- **代码框架生成**：使用 AI 辅助搭建项目结构（data/models/utils 模块化）
- **Grad-CAM 可视化**：参考 AI 建议的 hook 机制实现方案
- **训练循环调试**：AI 辅助排查梯度累积与数据划分问题
- **实验方案建议**：消融实验方向（冻结 Backbone、标签平滑）由 AI 建议，本人决定采用
- **Bug 修复**：数据泄漏 Bug 由另一个 AI 工具交叉验证发现，本人确认问题后由 AI 修复

本人完成的部分：环境搭建与调试、执行全部训练与消融实验、观察并记录实验结果、
对结果进行初步分析与解读、撰写报告核心内容、准备答辩材料。
