import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class Filtration(nn.Module):
    """
    差异图过滤模块 (公式5)
    用于滤除背景噪声，增强目标-背景差异
    """
    def __init__(self, threshold: float = 0.5):
        super(Filtration, self).__init__()
        # 可学习阈值参数
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=True)
    
    def forward(self, I_d: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            I_d: 原始差异图 [B, 1, H, W]
            target_size: 目标尺寸 (H, W)
        Returns:
            I_D: 过滤后的差异图 [B, 1, H_target, W_target]
        """
        # 应用公式: I_D = Resize((Sign(I_d - t) + 1) × 0.5) + 1
        sign_output = torch.sign(I_d - self.threshold)
        filtered = (sign_output + 1) * 0.5 + 1
        
        # 调整尺寸以匹配P2特征图
        I_D = F.interpolate(filtered, size=target_size, mode='bilinear', align_corners=False)
        
        return I_D


class FeatureEnhancementModule(nn.Module):
    """
    基于目标-背景差异的特征增强模块
    对应论文中的 Fig.3 结构
    """
    def __init__(
        self, 
        in_channels: int, 
        groups: int = 4,
        reduction_ratio: int = 2
    ):
        super(FeatureEnhancementModule, self).__init__()
        
        self.in_channels = in_channels
        self.groups = groups
        self.reduction_ratio = reduction_ratio
        self.mid_channels = in_channels // reduction_ratio
        
        # ========== 分支一：特征图处理 ==========
        # 分组卷积/处理
        self.group_conv = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=1, 
            groups=groups,
            bias=False
        )
        
        # 通道交互 (1×1卷积)
        self.channel_interaction = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.Sigmoid()
        )
        
        # 注意力权重生成 (高度和宽度方向)
        self.height_attention = nn.Sequential(
            nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        self.width_attention = nn.Sequential(
            nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # ========== 分支二：差异图过滤 ==========
        self.filtration = Filtration(threshold=0.5)
        
        # ========== 融合层 ==========
        self.reweight_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(
        self, 
        x_p2: torch.Tensor, 
        diff_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_p2: P2层特征图 [B, C, H, W]
            diff_map: 差异图 [B, 1, H_orig, W_orig]
        Returns:
            enhanced_feature: 增强后的特征图 [B, C, H, W]
        """
        B, C, H, W = x_p2.shape
        
        # ========== 分支一：生成注意力权重 ==========
        # 1. 分组处理
        x_grouped = self.group_conv(x_p2)  # [B, C, H, W]
        
        # 2. 平均池化 (X和Y方向)
        x_h = F.adaptive_avg_pool2d(x_grouped, (H, 1))   # [B, C, H, 1]
        x_w = F.adaptive_avg_pool2d(x_grouped, (1, W))   # [B, C, 1, W]
        
        # 3. 宽度分支交换后两个维度，再拼接
        x_w = x_w.permute(0, 1, 3, 2)                    # [B, C, W, 1]
        pooled = torch.cat([x_h, x_w], dim=2)            # [B, C, H+W, 1]
        # 4. 通道交互
        interacted = self.channel_interaction(pooled)    # [B, C/2, H+W, 1]
        # 5. 按空间长度拆分，再把宽度分支转回去
        h_feat, w_feat = torch.split(interacted, [H, W], dim=2)  # [B, C/2, H,1], [B,C/2,W,1]
        w_feat = w_feat.permute(0, 1, 3, 2)                      # [B, C/2, 1, W]
        attention_h = self.height_attention(h_feat)              # [B, C, H, 1]
        attention_w = self.width_attention(w_feat)               # [B, C, 1, W]
        
        # # ========== 分支二：差异图过滤 ==========
        # diff_filtered = self.filtration(diff_map, target_size=(H, W))  # [B, 1, H, W]
        
        # ========== 融合输出（暂不使用差异图分支） ==========
        weighted_feature = x_p2 * attention_h * attention_w

        # 直接与原特征再相乘作为输出
        output = weighted_feature * x_p2

        return output


