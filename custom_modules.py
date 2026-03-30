#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义 YOLOv8 backbone 插入模块

模块1 - BackgroundReconstruct (BGRecon)
  输入 : feat_early  (B, C, H, W)  —— 原图经过两次 CBS 下采样后的特征图
  输出 : diff_map    (B, 1, H, W)  —— 背景差分图（前景显著性空间图）
  原理 : 轻量 encoder-decoder 预测背景强度图 bg_pred，
         再用输入特征图的通道均值 feat_mean 与之作差，
         得到前景响应 diff_map = feat_mean - bg_pred

模块2 - FeatureEnhance (FeatEnh)
  输入 : [feat_early (B, C, H, W), diff_map (B, 1, H, W)]
  输出 : enhanced    (B, C, H, W)  —— 特征增强后的特征图，通道数与 feat_early 相同
  原理 : Concat([feat_early, diff_map], dim=1) -> Conv(C+1, C, 1) -> BN -> SiLU
         再通过残差连接叠加原 feat_early，保留原始信息
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# 工具：带 BN+SiLU 的标准卷积（与 ultralytics Conv 等价，但不依赖它）
# ─────────────────────────────────────────────────────────────
class CBS(nn.Module):
    """Conv + BN + SiLU，自包含，不依赖 ultralytics."""
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int = None):
        super().__init__()
        if p is None:
            p = k // 2  # same padding
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ─────────────────────────────────────────────────────────────
# 模块1：BackgroundReconstruct
# ─────────────────────────────────────────────────────────────
class BackgroundReconstruct(nn.Module):
    """
    背景重建模块。

    轻量 encoder-decoder 从特征图中预测背景强度，
    与特征图通道均值作差，输出前景显著性图 diff_map (1ch)。

    Args:
        c1 (int): 输入通道数（feat_early 的通道数，通常为 128）
        c_mid (int): 中间隐层通道数，默认 c1 // 4
    """

    def __init__(self, c1: int, c_mid: int = None):
        super().__init__()
        c_mid = c_mid or max(c1 // 4, 16)

        # 轻量 encoder：降低通道维度，捕获全局背景统计
        self.encoder = nn.Sequential(
            CBS(c1,    c_mid, k=3, s=1),   # 保持空间分辨率
            CBS(c_mid, c_mid, k=3, s=1),
        )

        # 全局上下文：用 GAP + FC 捕获全局背景均值（类 SE）
        self.global_ctx = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                  # (B, c_mid, 1, 1)
            nn.Flatten(),                              # (B, c_mid)
            nn.Linear(c_mid, c_mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_mid, c_mid, bias=False),
            nn.Sigmoid(),
        )

        # decoder：还原到 1 通道背景预测图
        self.decoder = nn.Sequential(
            CBS(c_mid, c_mid // 2, k=3, s=1),
            nn.Conv2d(c_mid // 2, 1, kernel_size=1, bias=True),  # 输出 1ch bg_pred
        )

    def forward(self, feat_early: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_early: (B, C, H, W)  两次 CBS 下采样后的特征图
        Returns:
            diff_map:   (B, 1, H, W)  前景显著性差分图
        """
        # encoder
        x = self.encoder(feat_early)                              # (B, c_mid, H, W)

        # 全局背景上下文调制
        ctx = self.global_ctx(x)                                  # (B, c_mid)
        x = x * ctx.view(ctx.shape[0], ctx.shape[1], 1, 1)       # channel-wise modulation

        # decoder：预测背景图
        bg_pred = self.decoder(x)                                 # (B, 1, H, W)

        # 前景差分：特征图通道均值 - 背景预测
        # feat_mean 代表该空间位置的"综合特征强度"
        feat_mean = feat_early.mean(dim=1, keepdim=True)          # (B, 1, H, W)
        diff_map  = feat_mean - bg_pred                           # (B, 1, H, W)

        return diff_map


# ─────────────────────────────────────────────────────────────
# 模块2：FeatureEnhance
# ─────────────────────────────────────────────────────────────
class FeatureEnhance(nn.Module):
    """
    特征增强模块。

    将 feat_early 与 diff_map concat 后，通过卷积融合并压回原通道数，
    再以残差方式叠加原 feat_early，保留原始特征信息。

    Args:
        c1 (int): feat_early 的通道数（通常为 128）
        c_diff (int): diff_map 的通道数（通常为 1）
    """

    def __init__(self, c1: int, c_diff: int = 1):
        super().__init__()
        c_in = c1 + c_diff  # concat 后通道数

        # 融合卷积：将 concat 特征压回 c1 通道
        self.fuse = nn.Sequential(
            CBS(c_in, c1, k=1, s=1),   # 1x1 通道对齐
            CBS(c1,   c1, k=3, s=1),   # 3x3 空间融合
        )

        # 可学习残差权重（初始化为 0，让模块从 identity 开始学习）
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: list) -> torch.Tensor:
        """
        Args:
            inputs: [feat_early (B, C, H, W), diff_map (B, 1, H, W)]
        Returns:
            enhanced: (B, C, H, W)
        """
        feat_early, diff_map = inputs[0], inputs[1]

        # Concat + 融合卷积
        x = torch.cat([feat_early, diff_map], dim=1)   # (B, C+1, H, W)
        x = self.fuse(x)                                # (B, C, H, W)

        # 残差：增强幅度由 alpha 控制（初始为 0，逐渐学习）
        enhanced = feat_early + self.alpha * x

        return enhanced


# ─────────────────────────────────────────────────────────────
# 注册函数：将自定义模块注入 parse_model 的命名空间
# 必须在构建 DetectionModel 之前调用
# ─────────────────────────────────────────────────────────────
def register_custom_modules():
    """
    将自定义模块注入 ultralytics.nn.tasks 的全局命名空间，
    使 parse_model 能通过模块名字符串找到它们。

    在 train.py 中，于构建模型之前调用此函数：
        from custom_modules import register_custom_modules
        register_custom_modules()
    """
    import ultralytics.nn.tasks as _tasks
    _tasks.BackgroundReconstruct = BackgroundReconstruct
    _tasks.FeatureEnhance        = FeatureEnhance
    print("[custom_modules] BackgroundReconstruct, FeatureEnhance 已注册到 parse_model")
