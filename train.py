#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 自定义训练主循环
功能：DataLoader | AdamW + Cosine LR | AMP 混合精度 | DDP 多卡 | 断点续训 | TensorBoard 日志
用法（单卡）：
    python train.py
用法（多卡 DDP，4 GPU）：
    torchrun --nproc_per_node=4 train.py --ddp
"""

import argparse
import os
import time
import math
from pathlib import Path

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from ultralytics import YOLO
from ultralytics.data import build_yolo_dataset
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss


def de_parallel(model):
    """返回单卡模型（去掉 DDP/DataParallel 包装）。"""
    return model.module if isinstance(model, (DDP, torch.nn.DataParallel)) else model

# ─────────────────────────── 超参数配置 ───────────────────────────
CFG = {
    # 数据集
    "data_yaml":     "/data/lz/pythonProjects/HP-tracker/yolov8/coco128.yaml",
    "imgsz":         640,
    "batch":         16,
    "workers":       4,
    # 模型
    "model_weights": "yolov8n.pt",
    "nc":            80,
    # 训练
    "epochs":        50,
    "lr0":           1e-3,      # 初始学习率
    "lrf":           0.01,      # Cosine 最终 lr = lr0 * lrf
    "weight_decay":  5e-4,
    "warmup_epochs": 3,
    # 保存 & 日志
    "save_dir":      "/data/lz/pythonProjects/HP-tracker/yolov8/runs/detect/custom_train",
    "save_period":   5,         # 每 N epoch 保存一次 checkpoint
    # 断点续训：填写 checkpoint 路径可续训，留空则从头训练
    "resume":        "",
}


# ─────────────────────────── DDP 工具函数 ───────────────────────────
def is_main(rank): return rank in (-1, 0)


def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ─────────────────────────── 学习率调度 ───────────────────────────
def cosine_lr_lambda(epoch, total_epochs, warmup_epochs, lrf):
    """Warmup + Cosine Annealing，返回相对于 lr0 的倍率。"""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return lrf + 0.5 * (1 - lrf) * (1 + math.cos(math.pi * progress))


# ─────────────────────────── 数据加载 ───────────────────────────
def get_dataloader(cfg, rank, mode="train"):
    """
    利用 ultralytics 内置 build_yolo_dataset / build_dataloader
    构建带数据增强的 DataLoader，支持 DDP DistributedSampler。
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.data.utils import check_det_dataset

    # 合并默认配置
    train_cfg = get_cfg(DEFAULT_CFG)
    train_cfg.data    = cfg["data_yaml"]
    train_cfg.imgsz   = cfg["imgsz"]
    train_cfg.batch   = cfg["batch"]
    train_cfg.workers = cfg["workers"]
    train_cfg.rect    = False
    train_cfg.task    = "detect"

    # 将 yaml 路径解析为 dict（包含 train/val/nc/names 等键）
    data_dict = check_det_dataset(cfg["data_yaml"])
    img_path  = data_dict[mode if mode in data_dict else "train"]

    dataset = build_yolo_dataset(
        train_cfg,
        img_path=img_path,
        batch=cfg["batch"],
        data=data_dict,   # ← 传解析后的 dict，而非字符串
        mode=mode,
        rect=False,
        stride=32,
    )

    shuffle  = (mode == "train")
    sampler  = DistributedSampler(dataset, shuffle=shuffle) if rank != -1 else None
    loader   = DataLoader(
        dataset,
        batch_size=cfg["batch"],
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=cfg["workers"],
        pin_memory=True,
        collate_fn=getattr(dataset, "collate_fn", None),
    )
    return loader, sampler


class TupleCompatibleDetectionLoss:#对Yolov8官方计算Loss的流程进行包装
    def __init__(self, model, de_parallel_fn):
        self.model = model
        self.de_parallel = de_parallel_fn
        self.base_loss = v8DetectionLoss(self.de_parallel(model))
    def _normalize_preds(self, preds):
        # 已是 dict，直接返回
        if isinstance(preds, dict):
            return preds
        # tuple 情况：有些版本是 (inference_out, train_feats)
        if isinstance(preds, tuple):
            # 如果第二项是 dict（新格式），直接用
            if len(preds) > 1 and isinstance(preds[1], dict):
                return preds[1]
            # 如果第二项是 list/tuple（旧特征图格式），取它
            if len(preds) > 1 and isinstance(preds[1], (list, tuple)):
                preds = preds[1]
            else:
                preds = preds[0]
        # list/tuple 特征图 -> dict（按 Detect head 通道规则拆分）
        if isinstance(preds, (list, tuple)):
            feats = list(preds)
            m = self.de_parallel(self.model).model[-1]  # Detect head
            bs = feats[0].shape[0]
            reg_max, nc = m.reg_max, m.nc
            no = 4 * reg_max + nc
            boxes, scores = [], []
            for f in feats:
                f = f.view(bs, no, -1)
                boxes.append(f[:, : 4 * reg_max, :])
                scores.append(f[:, 4 * reg_max :, :])
            return {
                "boxes": torch.cat(boxes, dim=2),
                "scores": torch.cat(scores, dim=2),
                "feats": feats
            }
        raise TypeError(f"Unsupported preds type: {type(preds)}")
    def __call__(self, preds, batch):
        preds_dict = self._normalize_preds(preds)
        return self.base_loss(preds_dict, batch)

    

# ─────────────────────────── 主训练函数 ───────────────────────────
def train(rank=-1, world_size=1):
    cfg      = CFG
    ddp      = rank != -1
    main     = is_main(rank)
    device   = torch.device(f"cuda:{rank}" if ddp else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    save_dir = Path(cfg["save_dir"])
    writer = None
    if main:
        save_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(save_dir / "tensorboard"))
        LOGGER.info(f"[TensorBoard] tensorboard --logdir {save_dir / 'tensorboard'}")

    # ── 构建模型 ──────────────────────────────────────────────────
    yolo   = YOLO(cfg["model_weights"])
    model  = yolo.model.to(device)
    # 关键修复：确保参数可训练（某些权重可能默认全冻结）
    for p in model.parameters():
        p.requires_grad_(True)    

    # 确保 model.args 是带属性访问的 IterableSimpleNamespace
    # v8DetectionLoss 通过 model.args.box / .cls / .dfl 读取超参
    from ultralytics.utils import IterableSimpleNamespace
    if not isinstance(getattr(model, 'args', None), IterableSimpleNamespace):
        default_hyp = {
            'box': 7.5, 'cls': 0.5, 'dfl': 1.5,
            'pose': 12.0, 'kobj': 2.0,
            'label_smoothing': 0.0, 'nbs': 64,
        }
        if isinstance(getattr(model, 'args', None), dict):
            default_hyp.update(model.args)
        model.args = IterableSimpleNamespace(**default_hyp)

    # ── 优化器：AdamW ──────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=cfg["lr0"],
        betas=(0.937, 0.999),
        weight_decay=cfg["weight_decay"],
    )

    # ── Cosine LR 调度器 ───────────────────────────────────────────
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: cosine_lr_lambda(
            ep, cfg["epochs"], cfg["warmup_epochs"], cfg["lrf"]
        ),
    )

    # ── AMP Scaler ────────────────────────────────────────────────
    scaler = GradScaler(device=device.type)

    # ── 断点续训 ───────────────────────────────────────────────────
    start_epoch  = 0
    best_loss    = float("inf")
    if cfg["resume"]:
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss   = ckpt.get("best_loss", best_loss)
        if main:
            LOGGER.info(f"断点续训：从 epoch {start_epoch} 继续，最佳 loss={best_loss:.4f}")

    # ── DDP 包装 ───────────────────────────────────────────────────
    if ddp:
        model = DDP(model, device_ids=[rank], output_device=rank)

    # ── 损失函数（ultralytics 内置 v8DetectionLoss） ────────────────
    compute_loss = TupleCompatibleDetectionLoss(model, de_parallel)

    # ── DataLoader ────────────────────────────────────────────────
    train_loader, train_sampler = get_dataloader(cfg, rank, mode="train")
    if main:
        LOGGER.info(f"训练集批次数: {len(train_loader)}")
        LOGGER.info(f"可训练参数张量数量: {len(trainable_params)}")

    # ─────────────────────────── 训练主循环 ────────────────────────
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss  = 0.0
        box_loss_ep = 0.0
        cls_loss_ep = 0.0
        dfl_loss_ep = 0.0
        t0          = time.time()

        for i, batch in enumerate(train_loader):
            # 将数据移至设备
            batch["img"]    = batch["img"].to(device, non_blocking=True).float() / 255.0
            batch["bboxes"] = batch["bboxes"].to(device)
            batch["cls"]    = batch["cls"].to(device)
            batch["batch_idx"] = batch["batch_idx"].to(device)

            optimizer.zero_grad()

            # ── 前向传播（AMP）─────────────────────────────────────
            with autocast(device_type=device.type):
                preds = model(batch["img"])  #这个地方会进入到ultralytics的源码包。preds的输出形式是tuple的形式

            # ── 损失计算（在 autocast 外，保留完整计算图）──────────────
            loss, loss_items = compute_loss(preds, batch)
            # loss 是形状 [3] 的向量(box/cls/dfl)，sum() 得标量
            loss_scalar = loss.sum()

            print("model.training =", model.training)
            print("loss.requires_grad =", loss.requires_grad, "loss.grad_fn =", loss.grad_fn)
            print("loss_scalar.requires_grad =", loss_scalar.requires_grad, "loss_scalar.grad_fn =", loss_scalar.grad_fn)

            n_trainable = sum(p.requires_grad for p in model.parameters())
            print("trainable params:", n_trainable)

            # ── 反向传播（AMP Scaler）──────────────────────────────
            scaler.scale(loss_scalar).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += loss_scalar.item()
            box_loss_ep += loss_items[0].item()
            cls_loss_ep += loss_items[1].item()
            dfl_loss_ep += loss_items[2].item()

            if main and (i % 20 == 0 or i == len(train_loader) - 1):
                lr_now = optimizer.param_groups[0]["lr"]
                LOGGER.info(
                    f"Epoch [{epoch+1}/{cfg['epochs']}] "
                    f"Step [{i+1}/{len(train_loader)}]  "
                    f"loss={loss_scalar.item():.4f}  lr={lr_now:.6f}"
                )

        scheduler.step()

        # ── 汇总本 epoch 日志 ─────────────────────────────────────
        n_batches   = len(train_loader)
        avg_loss    = epoch_loss  / n_batches
        avg_box     = box_loss_ep / n_batches
        avg_cls     = cls_loss_ep / n_batches
        avg_dfl     = dfl_loss_ep / n_batches
        elapsed     = time.time() - t0
        lr_cur      = optimizer.param_groups[0]["lr"]

        if main:
            LOGGER.info(
                f"\n=== Epoch {epoch+1}/{cfg['epochs']}  "
                f"loss={avg_loss:.4f}  box={avg_box:.4f}  "
                f"cls={avg_cls:.4f}  dfl={avg_dfl:.4f}  "
                f"lr={lr_cur:.6f}  time={elapsed:.1f}s ==="
            )
            # TensorBoard
            global_step = epoch + 1
            writer.add_scalar("Loss/train",     avg_loss, global_step)
            writer.add_scalar("Loss/box",       avg_box,  global_step)
            writer.add_scalar("Loss/cls",       avg_cls,  global_step)
            writer.add_scalar("Loss/dfl",       avg_dfl,  global_step)
            writer.add_scalar("LR/lr0",         lr_cur,   global_step)

            # ── 保存 checkpoint ───────────────────────────────────
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss

            ckpt = {
                "epoch":     epoch,
                "model":     de_parallel(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler":    scaler.state_dict(),
                "best_loss": best_loss,
            }

            # 每 save_period epoch 保存一次
            if (epoch + 1) % cfg["save_period"] == 0:
                torch.save(ckpt, save_dir / f"epoch{epoch+1}.pt")
                LOGGER.info(f"Checkpoint 已保存：{save_dir}/epoch{epoch+1}.pt")

            # 保存最新 & 最佳
            torch.save(ckpt, save_dir / "last.pt")
            if is_best:
                torch.save(ckpt, save_dir / "best.pt")
                LOGGER.info(f"最佳模型已更新：{save_dir}/best.pt  (loss={best_loss:.4f})")

    # ── 训练结束 ──────────────────────────────────────────────────
    if main:
        writer.close()
        LOGGER.info(f"\n训练完成！最佳 loss={best_loss:.4f}")
        LOGGER.info(f"模型保存目录：{save_dir}")
        LOGGER.info(f"TensorBoard：tensorboard --logdir {save_dir / 'tensorboard'}")

    cleanup_ddp()


# ─────────────────────────── 入口 ───────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp", action="store_true", help="启用 DDP 多卡训练")
    args = parser.parse_args()

    if args.ddp:
        # DDP 模式：由 torchrun 注入 LOCAL_RANK / WORLD_SIZE
        local_rank  = int(os.environ.get("LOCAL_RANK", 0))
        world_size  = int(os.environ.get("WORLD_SIZE", 1))
        setup_ddp(local_rank, world_size)
        train(rank=local_rank, world_size=world_size)
    else:
        # 单卡模式
        train(rank=-1, world_size=1)
