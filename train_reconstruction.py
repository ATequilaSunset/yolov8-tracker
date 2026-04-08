import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import glob

# 假设 reconstruction.py 与当前脚本在同一目录
# 请确保该文件中的 def init 已修正为 def __init__
from reconstruction import SelfReconstructionNetwork, RSELoss

# ================= 1. 自定义数据集 =================
class ReconDataset(Dataset):
    """支持扁平或嵌套目录的图像加载器"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        # 递归查找常见图像格式
        self.image_paths = glob.glob(os.path.join(root_dir, '**/*'), recursive=True)
        self.image_paths = [
            p for p in self.image_paths 
            if os.path.isfile(p) and p.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))
        ]
        self.transform = transform
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"未在 {root_dir} 中找到任何图像文件。")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # 自重建任务：输入和目标均为原图
        return image, image

# ================= 2. 训练主流程 =================
def main():
    # ---------------- 基础配置 ----------------
    TRAIN_DIR = '/data/lz/SatVideoDT/train_data'
    VAL_DIR   = '/data/lz/SatVideoDT/val_data'
    
    # ✅ 新增：本地保存路径配置
    CHECKPOINT_DIR = './checkpoints'            # 权重保存目录
    VISUALIZATION_DIR = './visualization_results' # 可视化图片保存目录
    
    # 创建目录（若不存在）
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LR = 1e-4
    IMG_SIZE = (256, 256)
    LOG_DIR = './runs/reconstruction_tb'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🌐 使用设备: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])

    train_dataset = ReconDataset(TRAIN_DIR, transform=transform)
    val_dataset   = ReconDataset(VAL_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=4, pin_memory=True, drop_last=False)

    # ---------------- 模型、多GPU、优化器 ----------------
    model = SelfReconstructionNetwork(backbone_name='resnet50', pretrained=True, img_channels=3)
    
    # ✅ 多GPU训练支持 (DataParallel)
    if torch.cuda.device_count() > 1:
        print(f"🚀 检测到 {torch.cuda.device_count()} 个GPU，启用 DataParallel 分布式训练")
        model = nn.DataParallel(model)
    model.to(device)

    criterion = RSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    writer = SummaryWriter(log_dir=LOG_DIR)

    # 固定验证集图片
    fixed_val_img = None
    if len(val_dataset) > 0:
        fixed_val_img, _ = val_dataset[0]
        fixed_val_img = fixed_val_img.unsqueeze(0).to(device)

    print("📈 开始训练...")
    for epoch in range(NUM_EPOCHS):
        # ---------- 训练阶段 ----------
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)

        # ---------- 验证阶段 ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
        val_loss /= len(val_loader.dataset)
        scheduler.step()

        # 1. TensorBoard 记录 Loss
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] | Train RMSE: {train_loss:.5f} | Val RMSE: {val_loss:.5f}')

        # ✅ 周期性任务：每 2 个 Epoch 保存权重 & 可视化
        if (epoch + 1) % 2 == 0 and fixed_val_img is not None:
            with torch.no_grad():
                recon_out = model(fixed_val_img)
            
            # --- A. 可视化并保存本地 ---
            orig_np = fixed_val_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            recon_np = recon_out.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(orig_np)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            axes[1].imshow(recon_np)
            axes[1].set_title(f'Reconstructed (Epoch {epoch+1})')
            axes[1].axis('off')
            plt.tight_layout()

            # 保存到 TensorBoard
            writer.add_figure('Visualization/Reconstruction_Comparison', fig, global_step=epoch)
            
            # 保存到本地文件夹
            vis_path = os.path.join(VISUALIZATION_DIR, f'result_epoch_{epoch+1:04d}.png')
            fig.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✅ 可视化图片已保存至: {vis_path}")

            # --- B. 保存模型权重到本地 ---
            # 处理 DataParallel 的 state_dict
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'recon_epoch_{epoch+1:04d}.pth')
            torch.save(state_dict, ckpt_path)
            print(f"  💾 权重模型已保存至: {ckpt_path}")

    writer.close()
    print("🎉 训练完成！")
    
    # 保存最终权重
    final_path = os.path.join(CHECKPOINT_DIR, 'recon_final.pth')
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, final_path)
    print(f"💾 最终权重已保存至 {final_path}")

if __name__ == '__main__':
    main()