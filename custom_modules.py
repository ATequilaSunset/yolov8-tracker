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
import torch.nn.functional as F
from typing import Tuple


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
# 模块1：BackgroundReconstruct (替换为 reconstruction.py 逻辑)
# ─────────────────────────────────────────────────────────────
class UpSampleBlock(nn.Module):
    """对应 reconstruction.py 公式 (1): UpSample(X) = ReLU(Conv1(ReLU(Conv1(TransConv(X)))))"""
    def __init__(self, in_channels):
        super().__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.conv1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.trans_conv(x)
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        return x

class ReconstructionModule(nn.Module):
    """对应 reconstruction.py 中的重建模块 (RM) 和公式 (2)"""
    def __init__(self, feature_channels, output_channels=3):
        super().__init__()
        self.up1 = UpSampleBlock(feature_channels)
        self.up2 = UpSampleBlock(feature_channels)
        self.final_conv = nn.Conv2d(feature_channels, output_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_p2):
        x = self.up1(x_p2)
        x = self.up2(x)
        x = self.final_conv(x)
        return self.sigmoid(x)

class BackgroundReconstruct(nn.Module):
    """
    基于 reconstruction.py 的自重建差异图生成模块。
    输入 : feat_early  (B, C, H, W)  —— 原图经过两次 CBS 下采样后的特征图
    输出 : diff_map    (B, c_out, H, W) —— 重建图与原始特征的绝对差异图
    """
    def __init__(self, c1: int, c_mid: int = None, c_out: int = 1):
        super().__init__()
        self.c_out = c_out
        # 使用 reconstruction.py 的重建结构，输入通道为 c1，输出通道由 c_out 控制
        self.recon_module = ReconstructionModule(feature_channels=c1, output_channels=c_out)

    def forward(self, feat_early: torch.Tensor) -> torch.Tensor:
        H_in, W_in = feat_early.shape[2:]
        
        # 1. 执行重建 (因 2 次 stride=2 的 TransConv，尺寸会变为 4H x 4W)
        recon_out = self.recon_module(feat_early)
        
        # 2. 空间尺寸对齐回 feat_early 的 (H, W)
        if recon_out.shape[2:] != (H_in, W_in):
            recon_out = F.interpolate(recon_out, size=(H_in, W_in), mode='bilinear', align_corners=False)
            
        # 3. 通道对齐：若 feat_early 通道数 != c_out，取均值并扩展
        feat_aligned = feat_early
        if feat_early.shape[1] != self.c_out:
            feat_mean = feat_early.mean(dim=1, keepdim=True).expand(-1, self.c_out, -1, -1)
            feat_aligned = feat_mean
            
        # 4. 计算差异图 (对应 reconstruction.py 的 get_difference_map 逻辑)
        diff_map = torch.abs(recon_out - feat_aligned)
        return diff_map
# ─────────────────────────────────────────────────────────────
# 模块2：FeatureEnhance（替换为 feature_enhancement.py 实现）
# ─────────────────────────────────────────────────────────────
class Filtration(nn.Module):
    """
    差异图过滤模块 (公式5)
    用于滤除背景噪声，增强目标-背景差异
    """
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=True)

    def forward(self, I_d: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        sign_output = torch.sign(I_d - self.threshold)
        filtered = (sign_output + 1) * 0.5 + 1
        I_D = F.interpolate(filtered, size=target_size, mode='bilinear', align_corners=False)
        return I_D


class FeatureEnhance(nn.Module):
    """
    基于目标-背景差异的特征增强模块
    对应论文中的 Fig.3 结构

    输入:
        [feat_early (B, C, H, W), diff_map (B, 1, H_orig, W_orig)]
    输出:
        enhanced (B, C, H, W)
    """

    def __init__(self, c1: int, groups: int = 4, reduction_ratio: int = 2):
        super().__init__()

        self.in_channels = c1
        self.groups = groups
        self.reduction_ratio = reduction_ratio
        self.mid_channels = c1 // reduction_ratio

        # 分支一：特征图处理
        self.group_conv = nn.Conv2d(
            c1, c1,
            kernel_size=1,
            groups=groups,
            bias=False
        )

        self.channel_interaction = nn.Sequential(
            nn.Conv2d(c1, self.mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.Sigmoid()
        )

        self.height_attention = nn.Sequential(
            nn.Conv2d(self.mid_channels, c1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.width_attention = nn.Sequential(
            nn.Conv2d(self.mid_channels, c1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 分支二：差异图过滤
        self.filtration = Filtration(threshold=0.5)

    def forward(self, inputs: list) -> torch.Tensor:
        feat_early, diff_map = inputs[0], inputs[1]
        _, _, H, W = feat_early.shape

        # 分支一：生成注意力权重
        x_grouped = self.group_conv(feat_early)

        x_h = F.adaptive_avg_pool2d(x_grouped, (H, 1))
        x_w = F.adaptive_avg_pool2d(x_grouped, (1, W))

        x_w = x_w.permute(0, 1, 3, 2)
        pooled = torch.cat([x_h, x_w], dim=2)

        interacted = self.channel_interaction(pooled)

        h_feat, w_feat = torch.split(interacted, [H, W], dim=2)
        w_feat = w_feat.permute(0, 1, 3, 2)

        attention_h = self.height_attention(h_feat)
        attention_w = self.width_attention(w_feat)

        # 分支二：差异图过滤（保持与 feature_enhancement.py 一致，当前不参与输出）
        filtered_diff_map = self.filtration(diff_map, target_size=(H, W))
        # ✅ 核心修复：兼容 1ch/3ch 差异图，统一转为 (B, 1, H, W) 空间权重
        if filtered_diff_map.shape[1] > 1:
            diff_weight = filtered_diff_map.mean(dim=1, keepdim=True)
        else:
            diff_weight = filtered_diff_map

        weighted_feature = feat_early * attention_h * attention_w
        output = weighted_feature * feat_early * diff_weight


        return output


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
