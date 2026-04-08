import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class UpSampleBlock(nn.Module):
    """
    对应论文公式 (1): UpSample(X) = ReLU(Conv1(ReLU(Conv1(TransConv(X)))))
    """
    def __init__(self, in_channels):
        super(UpSampleBlock, self).__init__()
        # TransConv: kernel 4x4, stride 2, padding 1 (实现 2 倍上采样)
        self.trans_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        # Conv1: kernel 3x3, padding 1
        self.conv1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.trans_conv(x)
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        return x

class ReconstructionModule(nn.Module):
    """
    对应论文中的重建模块 (RM) 和公式 (2)
    """
    def __init__(self, feature_channels, output_channels=3):
        super(ReconstructionModule, self).__init__()
        self.up1 = UpSampleBlock(feature_channels)
        self.up2 = UpSampleBlock(feature_channels)
        
        # 最终卷积将通道数还原为原图通道数 (例如 3)
        self.final_conv = nn.Conv2d(feature_channels, output_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_p2):
        x = self.up1(x_p2)
        x = self.up2(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x

class SelfReconstructionNetwork(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True, img_channels=3):
        super(SelfReconstructionNetwork, self).__init__()
        
        # 1. 构建 Backbone (ResNet50)
        resnet = resnet50(pretrained=pretrained)
        
        # 手动提取 ResNet 的前半部分作为特征提取器 (Stem)
        # 输出尺寸：H/4, W/4
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        # P2 特征提取层 (layer1)
        # 输入尺寸：H/4, W/4, 输出尺寸：H/4, W/4, 通道数：256
        self.p2_layer = resnet.layer1 
        self.p2_channels = 256 
        
        # 2. 重建模块
        self.reconstruction_module = ReconstructionModule(self.p2_channels, output_channels=img_channels)

    def forward(self, x):
        """
        x: 原始输入图像 [N, C, H, W]
        返回：重建后的图像 [N, C, H, W]
        """
        # 1. 提取 Stem 特征 (H/4, W/4)
        x_stem = self.stem(x)
        
        # 2. 提取 P2 层特征 (H/4, W/4, C=256)
        x_p2 = self.p2_layer(x_stem)
        
        # 3. 重建图像 (H, W, C=3)
        reconstructed_img = self.reconstruction_module(x_p2)
        
        # 注意：这里不再计算 diff_map，因为 x 已经变成了 x_stem (尺寸变小了)，无法与 reconstructed_img 相减
        # 差异图计算应在外部使用原始输入 x 和 reconstructed_img 进行
        return reconstructed_img

    def get_difference_map(self, reconstructed_img, original_img):
        """
        外部调用此函数计算差异图 (公式 3)
        I_d = M_C(|X'_P2 - I|)
        """
        # 确保尺寸一致
        if reconstructed_img.shape != original_img.shape:
            # 如果 original_img 尺寸不同，这里可以加插值，但通常训练时已保证一致
            pass
            
        # 计算绝对差值
        diff = torch.abs(reconstructed_img - original_img)
        # M_C: 沿通道维度求均值 (dim=1)
        diff_map = torch.mean(diff, dim=1, keepdim=True)
        return diff_map
    

# --- 1. 定义 Loss 函数 (公式 4) ---
class RSELoss(nn.Module):
    """
    Root Mean Squared Error (RMSE)
    RSELoss = sqrt( 1/n * sum( (X'_P2 - I_i)^2 ) )
    """
    def __init__(self):
        super(RSELoss, self).__init__()

    def forward(self, reconstructed, original):
        # 确保形状一致
        if reconstructed.shape != original.shape:
            # 如果原图是 3 通道，重建图也是 3 通道，直接计算
            # 如果原图是高光谱多通道，需确保通道对齐
            pass
            
        mse = torch.mean((reconstructed - original) ** 2)
        rmse = torch.sqrt(mse)
        return rmse