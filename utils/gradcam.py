"""
gradcam.py - Grad-CAM 特征热力图可视化

【核心概念：Grad-CAM（Gradient-weighted Class Activation Mapping）】
Grad-CAM 回答一个问题："模型在做出某个分类决策时，到底在看图片的哪个区域？"

原理：
1. 找到模型最后一个卷积层（ResNet-18 中是 layer4）
2. 取该层输出的特征图（feature maps）
3. 对目标类别的分数求梯度，反向传播到特征图
4. 梯度做全局平均池化 → 得到每个特征图的"重要性权重"
5. 用权重加权组合所有特征图 → 得到一张"注意力热力图"
6. 将热力图叠加到原图上，红色区域 = 模型重点关注的区域

用途：
- 验证模型是否在"看对的地方"（比如分类猫的品种时应该关注脸部/花纹）
- 分析错误预测的原因（比如模型可能在看背景而不是宠物本身）
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from typing import Optional


class GradCAM:
    """Grad-CAM 可视化器。

    用法:
        gradcam = GradCAM(model, target_layer=model.layer4)
        mask = gradcam(input_image, target_class=5)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        参数:
            model: 训练好的模型
            target_layer: 要提取梯度的目标卷积层
                对于 ResNet-18，通常选 layer4（最后一个卷积块）
        """
        self.model = model
        self.target_layer = target_layer

        # 用于存储前向传播和反向传播的中间结果
        self.gradients = None
        self.activations = None

        # 注册钩子（hook）：在前向/反向传播时自动捕获目标层的输入输出
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """前向传播钩子：保存目标层的输出特征图。"""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """反向传播钩子：保存目标层的梯度。"""
        self.gradients = grad_output[0].detach()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """生成 Grad-CAM 热力图。

        参数:
            input_tensor: 单张图片的 tensor，形状为 [1, 3, 224, 224]
            target_class: 要可视化的目标类别。
                如果为 None，则使用模型预测概率最高的类别。

        返回:
            热力图 numpy 数组，形状为 [224, 224]，值域 [0, 1]
        """
        self.model.eval()

        # 前向传播
        output = self.model(input_tensor)  # [1, 37]

        # 确定目标类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 反向传播目标类别的分数
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # 获取梯度和特征图
        gradients = self.gradients[0]      # [512, 7, 7]
        activations = self.activations[0]  # [512, 7, 7]

        # 对梯度做全局平均池化 → 每个通道一个权重
        # 这个权重代表"该通道对目标类别分数的贡献程度"
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [512, 1, 1]

        # 加权组合所有特征图通道
        cam = (weights * activations).sum(dim=0)  # [7, 7]
        cam = F.relu(cam)  # 只保留正贡献（ReLU）

        # 归一化到 [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # 上采样到输入图片大小（224x224）
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        return cam


def visualize_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    true_label: int,
    class_names: list,
    save_path: str = "gradcam_result.png",
    target_layer=None,
) -> None:
    """可视化单张图片的 Grad-CAM 热力图。

    参数:
        model: 训练好的模型
        image_tensor: 单张图片 tensor [1, 3, 224, 224]（已归一化）
        true_label: 真实类别索引
        class_names: 类别名称列表
        save_path: 结果保存路径
        target_layer: 目标卷积层（默认 model.layer4）
    """
    if target_layer is None:
        target_layer = model.layer4

    gradcam = GradCAM(model, target_layer)

    # 生成热力图
    cam = gradcam(image_tensor, target_class=None)

    # 将归一化的输入图片还原为可显示的范围
    # 先反归一化（ImageNet 统计量）
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_denorm = image_tensor[0].cpu() * std + mean
    image_denorm = image_denorm.clamp(0, 1).permute(1, 2, 0).numpy()

    # 获取模型预测结果
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_class = probs.argmax().item()
        pred_prob = probs[pred_class].item()

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原图
    axes[0].imshow(image_denorm)
    axes[0].set_title(f"Original\nTrue: {class_names[true_label]}")
    axes[0].axis("off")

    # Grad-CAM 热力图
    axes[1].imshow(cam, cmap="jet", alpha=0.7)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    # 叠加图
    axes[2].imshow(image_denorm)
    axes[2].imshow(cam, cmap="jet", alpha=0.4)
    pred_status = "✓" if pred_class == true_label else "✗"
    axes[2].set_title(
        f"Overlay {pred_status}\n"
        f"Pred: {class_names[pred_class]} ({pred_prob:.1%})"
    )
    axes[2].axis("off")

    plt.suptitle(
        f"Grad-CAM Visualization | True: {class_names[true_label]} | "
        f"Pred: {class_names[pred_class]}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grad-CAM 可视化已保存至: {save_path}")
