"""
evaluate.py - 评估与生成可视化结果的独立脚本

功能：
1. 加载训练好的模型
2. 在测试集上计算 Top-1 Acc, Top-5 Acc, Macro-F1
3. 绘制混淆矩阵
4. 生成 Grad-CAM 热力图（正确预测 + 错误预测各一张）

用法：
    python evaluate.py --model_path best_model.pth --exp_name baseline
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import get_dataloaders, set_seed
from models.model import build_model
from utils.metrics import evaluate_model, plot_confusion_matrix
from utils.gradcam import visualize_gradcam


def parse_args():
    parser = argparse.ArgumentParser(description="评估与可视化脚本")
    parser.add_argument("--model_path", type=str, default="best_model.pth",
                        help="模型权重文件路径")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="数据集根目录")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--exp_name", type=str, default="baseline",
                        help="实验名称（需与训练时一致）")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_misclassified_samples(
    model, dataloader, device, class_names, num_correct=1, num_wrong=1
):
    """从测试集中找到正确预测和错误预测的样本。

    返回:
        correct_samples: [(image_tensor, true_label, pred_label), ...]
        wrong_samples: [(image_tensor, true_label, pred_label), ...]
    """
    model.eval()
    correct_samples = []
    wrong_samples = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            for i in range(labels.size(0)):
                if preds[i] == labels[i]:
                    if len(correct_samples) < num_correct:
                        correct_samples.append((
                            images[i].cpu(),
                            labels[i].item(),
                            preds[i].item(),
                        ))
                else:
                    if len(wrong_samples) < num_wrong:
                        wrong_samples.append((
                            images[i].cpu(),
                            labels[i].item(),
                            preds[i].item(),
                        ))

            # 找够了就提前退出
            if len(correct_samples) >= num_correct and len(wrong_samples) >= num_wrong:
                break

    return correct_samples, wrong_samples


def main():
    args = parse_args()
    set_seed(args.seed)

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 加载数据
    print("正在加载数据...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # 构建模型并加载权重
    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    print(f"已加载模型权重: {args.model_path}")

    # 1. 测试集评估
    print(f"\n{'='*50}")
    print("测试集评估结果:")
    print(f"{'='*50}")
    results = evaluate_model(model, test_loader, device)

    # 2. 混淆矩阵
    print(f"\n{'='*50}")
    print("生成混淆矩阵...")
    print(f"{'='*50}")
    plot_confusion_matrix(
        labels=results["all_labels"],
        preds=results["all_preds"],
        class_names=class_names,
        save_path="confusion_matrix.png",
        top_k_confused=5,
    )

    # 3. Grad-CAM 可视化
    print(f"\n{'='*50}")
    print("生成 Grad-CAM 可视化...")
    print(f"{'='*50}")

    # 找一张预测正确的、一张预测错误的
    correct_samples, wrong_samples = find_misclassified_samples(
        model, test_loader, device, class_names,
        num_correct=1, num_wrong=1,
    )

    # 正确预测的 Grad-CAM
    if correct_samples:
        img, true_label, pred_label = correct_samples[0]
        print(f"\n正确预测样本: {class_names[true_label]} → {class_names[pred_label]}")
        visualize_gradcam(
            model=model,
            image_tensor=img.unsqueeze(0).to(device),
            true_label=true_label,
            class_names=class_names,
            save_path="gradcam_correct.png",
        )

    # 错误预测的 Grad-CAM
    if wrong_samples:
        img, true_label, pred_label = wrong_samples[0]
        print(f"\n错误预测样本: {class_names[true_label]} → {class_names[pred_label]}")
        visualize_gradcam(
            model=model,
            image_tensor=img.unsqueeze(0).to(device),
            true_label=true_label,
            class_names=class_names,
            save_path="gradcam_wrong.png",
        )

    # 4. 保存评估结果
    eval_results = {
        "model_path": args.model_path,
        "top1_acc": results["top1_acc"],
        "top5_acc": results["top5_acc"],
        "macro_f1": results["macro_f1"],
    }
    with open(f"eval_results_{args.exp_name}.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n{'='*50}")
    print("所有可视化结果已生成完毕！")
    print(f"  - confusion_matrix.png")
    print(f"  - gradcam_correct.png")
    print(f"  - gradcam_wrong.png")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
