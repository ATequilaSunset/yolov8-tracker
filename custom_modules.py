#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义 YOLOv8 backbone 插入模块。

模块1 - BackgroundReconstruct (BGRecon)
  图结构位置：插在 YOLO 早期 backbone 特征之后
  实际输入    ：当前 forward 的原始输入图像 raw_input
  输出        ：diff_map (B, 1, H, W)
  原理        ：复用 reconstruction.py 中的 SelfReconstructionNetwork，
               先重建背景图，再与原图计算 1 通道差异图

模块2 - FeatureEnhance (FeatEnh)
  输入 : [feat_early (B, C, H, W), diff_map (B, 1, H_orig, W_orig)]
  输出 : enhanced    (B, C, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from reconstruction import SelfReconstructionNetwork


class BackgroundReconstruct(nn.Module):
    """
    基于 SelfReconstructionNetwork 的并行背景重建模块。

    说明：
    - 保留在 YOLO 图结构中的位置，用于产出 diff_map。
    - 真正的输入来自当前 forward 的原始图像 raw_input，而不是图上的 feat_early。
    - 该模块对外只输出 1 通道 diff_map，供 FeatureEnhance 使用。
    """

    def __init__(self, c1: int, *_args):
        super().__init__()
        print(f"[BGRecon] init start, c1={c1}")
        self.in_channels = c1  # 仅用于兼容 parse_model / YAML 现有参数语义
        self.out_channels = 1
        self.recon_net = SelfReconstructionNetwork(backbone_name="resnet50", pretrained=False, img_channels=3)
        print("[BGRecon] init done")
        self._raw_input: Optional[torch.Tensor] = None
        self.last_reconstructed_img: Optional[torch.Tensor] = None
        self.last_diff_map: Optional[torch.Tensor] = None

    def set_raw_input(self, raw_input: Optional[torch.Tensor]) -> None:
        self._raw_input = raw_input

    def clear_raw_input(self) -> None:
        self._raw_input = None

    def train(self, mode: bool = True):
        super().train(mode)
        # 重建分支在检测训练中始终冻结，避免 BN running stats 漂移
        self.recon_net.eval()
        return self

    def forward(self, feat_early: torch.Tensor) -> torch.Tensor:
        print(f"[BGRecon] forward enter, feat_early={tuple(feat_early.shape)}, has_raw_input={self._raw_input is not None}")
        if self._raw_input is None:
            # DetectionModel 构建阶段会先跑一次 dummy forward 来推导 stride，
            # 此时尚未注入 raw_input。这里返回原图尺度的 1 通道占位 diff_map，
            # 仅用于让建模流程完成形状推导。
            b, _, h, w = feat_early.shape
            print(f"[BGRecon] fallback diff_map for model build -> {(b, 1, h * 4, w * 4)}")
            return feat_early.new_zeros((b, 1, h * 4, w * 4))

        raw_input = self._raw_input
        print(f"[BGRecon] run recon_net, raw_input={tuple(raw_input.shape)}")
        reconstructed_img = self.recon_net(raw_input)
        diff_map = self.recon_net.get_difference_map(reconstructed_img, raw_input)
        print(f"[BGRecon] recon done, reconstructed={tuple(reconstructed_img.shape)}, diff_map={tuple(diff_map.shape)}")

        self.last_reconstructed_img = reconstructed_img
        self.last_diff_map = diff_map
        return diff_map


class Filtration(nn.Module):
    """
    差异图过滤模块 (公式5)
    用于滤除背景噪声，增强目标-背景差异
    """

    def __init__(self, threshold: float = 2):
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

        self.filtration = Filtration(threshold=0.5)

    def forward(self, inputs: list) -> torch.Tensor:
        feat_early, diff_map = inputs[0], inputs[1]
        _, _, H, W = feat_early.shape

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

        filtered_diff_map = self.filtration(diff_map, target_size=(H, W))
        if filtered_diff_map.shape[1] > 1:
            diff_weight = filtered_diff_map.mean(dim=1, keepdim=True)
        else:
            diff_weight = filtered_diff_map

        weighted_feature = feat_early * attention_h * attention_w
        output = weighted_feature * feat_early * diff_weight
        return output


def register_custom_modules():
    """
    将自定义模块注入 ultralytics.nn.tasks 的全局命名空间，
    使 parse_model 能通过模块名字符串找到它们。
    """
    import ultralytics.nn.tasks as _tasks

    _tasks.BackgroundReconstruct = BackgroundReconstruct
    _tasks.FeatureEnhance = FeatureEnhance
    print("[custom_modules] BackgroundReconstruct, FeatureEnhance 已注册到 parse_model")
