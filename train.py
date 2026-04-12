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

# ── 自定义模块注册（必须在 ultralytics 模型构建之前完成）──────────────
# 将 BackgroundReconstruct 和 FeatureEnhance 注入 parse_model 的
# 命名空间，同时 patch parse_model 以正确推导这两个模块的输出通道数。
from custom_modules import BackgroundReconstruct, FeatureEnhance, register_custom_modules


def patch_parse_model():
    """
    Monkey-patch ultralytics.nn.tasks.parse_model，在 else 分支前
    插入对 BackgroundReconstruct 和 FeatureEnhance 的通道处理逻辑：

      BackgroundReconstruct(c1)                     -> 输出 1ch  (diff_map)
      FeatureEnhance(c1, groups=4, reduction_ratio=2) -> 输出 c1 ch (与 feat_early 相同)

    parse_model 的 else 分支默认 c2 = ch[f]，对多输入层取最后一个 from
    的通道数。这对我们两个模块均不正确，需要显式覆盖。
    """
    import ultralytics.nn.tasks as _tasks

    def _patched_parse_model(d, ch, verbose=True):
        import ast
        import contextlib
        from ultralytics.nn.modules import (
            AIFI, C1, C2, C2PSA, C3, C3TR, ELAN1, OBB, OBB26, PSA, SPP,
            SPPELAN, SPPF, A2C2f, AConv, ADown, Bottleneck, BottleneckCSP,
            C2f, C2fAttn, C2fCIB, C2fPSA, C3Ghost, C3k2, C3x, CBFuse,
            CBLinear, Classify, Concat, Conv, Conv2, ConvTranspose, Detect,
            DWConv, DWConvTranspose2d, Focus, GhostBottleneck, GhostConv,
            HGBlock, HGStem, ImagePoolingAttn, Index, Pose, Pose26, RepC3,
            RepConv, RepNCSPELAN4, RepVGGDW, ResNetLayer, RTDETRDecoder,
            SCDown, Segment, Segment26, TorchVision, WorldDetect,
            YOLOEDetect, YOLOESegment, YOLOESegment26, v10Detect,
        )
        from ultralytics.utils import LOGGER, colorstr
        from ultralytics.utils.ops import make_divisible

        legacy = True
        max_channels = float("inf")
        nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
        reg_max = d.get("reg_max", 16)
        depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
        scale = d.get("scale")
        if scales:
            if not scale:
                scale = next(iter(scales.keys()))
                LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
            depth, width, max_channels = scales[scale]

        if act:
            Conv.default_act = eval(act)
            if verbose:
                LOGGER.info(f"{colorstr('activation:')} {act}")

        if verbose:
            LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

        ch = [ch]
        layers, save, c2 = [], [], ch[-1]

        base_modules = frozenset({
            Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck,
            SPP, SPPF, C2fPSA, C2PSA, DWConv, Focus, BottleneckCSP, C1, C2, C2f,
            C3k2, RepNCSPELAN4, ELAN1, ADown, AConv, SPPELAN, C2fAttn, C3, C3TR,
            C3Ghost, torch.nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3, PSA,
            SCDown, C2fCIB, A2C2f,
        })
        repeat_modules = frozenset({
            BottleneckCSP, C1, C2, C2f, C3k2, C2fAttn, C3, C3TR, C3Ghost, C3x,
            RepC3, C2fPSA, C2fCIB, C2PSA, A2C2f,
        })

        # ── 自定义模块的通道映射表 ─────────────────────────────────────
        # key  : 模块类
        # value: callable(args, ch_in_list) -> (c2, new_args)
        #   c2        : 该模块的输出通道数（用于更新 ch 列表）
        #   new_args  : 传给模块 __init__ 的最终 args
        # train.py -> patch_parse_model() 内部
        def _bgr_channels(args, ch_in_list):
            c1 = ch_in_list[0]
            c_mid = args[0] if len(args) > 0 else None
            c_out = args[1] if len(args) > 1 else 3  # 默认 3 通道（与你的权重一致）
            new_args = [c1, c_mid, c_out]
            return c_out, new_args  # ✅ 必须返回 c_out，不能写死 1

        def _feh_channels(args, ch_in_list):
            # FeatureEnhance(c1, groups=4, reduction_ratio=2)
            # c1 = ch[from[0]] (feat_early)，diff_map 仍由 from[1] 传入 forward
            c1 = ch_in_list[0]

            # 兼容 yaml 可选参数：args = [c1, groups, reduction_ratio]
            # 若未提供则使用模块默认值
            groups = args[1] if len(args) > 1 else 4
            reduction_ratio = args[2] if len(args) > 2 else 2
            new_args = [c1, groups, reduction_ratio]
            return c1, new_args   # 输出与 feat_early 通道数相同

        custom_channel_handlers = {
            BackgroundReconstruct: _bgr_channels,
            FeatureEnhance:        _feh_channels,
        }
        # ────────────────────────────────────────────────────────────────

        for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
            m = (
                getattr(torch.nn, m[3:])
                if "nn." in m
                else getattr(__import__("torchvision").ops, m[16:])
                if "torchvision.ops." in m
                else vars(_tasks)[m]
            )
            for j, a in enumerate(args):
                if isinstance(a, str):
                    with contextlib.suppress(ValueError):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

            n = n_ = max(round(n * depth), 1) if n > 1 else n

            if m in base_modules:
                c1, c2 = ch[f], args[0]
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                if m is C2fAttn:
                    args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                    args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])
                args = [c1, c2, *args[1:]]
                if m in repeat_modules:
                    args.insert(2, n)
                    n = 1
                if m is C3k2:
                    legacy = False
                    if scale in "mlx":
                        args[3] = True
                if m is A2C2f:
                    legacy = False
                    if scale in "lx":
                        args.extend((True, 1.2))
                if m is C2fCIB:
                    legacy = False
            elif m is AIFI:
                args = [ch[f], *args]
            elif m in frozenset({HGStem, HGBlock}):
                c1, cm, c2 = ch[f], args[0], args[1]
                args = [c1, cm, c2, *args[2:]]
                if m is HGBlock:
                    args.insert(4, n)
                    n = 1
            elif m is ResNetLayer:
                c2 = args[1] if args[3] else args[1] * 4
            elif m is torch.nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            elif m in frozenset({
                Detect, WorldDetect, YOLOEDetect, Segment, Segment26,
                YOLOESegment, YOLOESegment26, Pose, Pose26, OBB, OBB26,
            }):
                args.extend([reg_max, end2end, [ch[x] for x in f]])
                if m is Segment or m is YOLOESegment or m is Segment26 or m is YOLOESegment26:
                    args[2] = make_divisible(min(args[2], max_channels) * width, 8)
                if m in {Detect, YOLOEDetect, Segment, Segment26, YOLOESegment, YOLOESegment26, Pose, Pose26, OBB, OBB26}:
                    m.legacy = legacy
            elif m is v10Detect:
                args.append([ch[x] for x in f])
            elif m is ImagePoolingAttn:
                args.insert(1, [ch[x] for x in f])
            elif m is RTDETRDecoder:
                args.insert(1, [ch[x] for x in f])
            elif m is CBLinear:
                c2 = args[0]
                c1 = ch[f]
                args = [c1, c2, *args[1:]]
            elif m is CBFuse:
                c2 = ch[f[-1]]
            elif m in frozenset({TorchVision, Index}):
                c2 = args[0]
                c1 = ch[f]
                args = [*args[1:]]
            # ── 自定义模块通道处理 ────────────────────────────────────
            elif m in custom_channel_handlers:
                ch_in_list = [ch[x] for x in f] if isinstance(f, list) else [ch[f]]
                c2, args = custom_channel_handlers[m](args, ch_in_list)
            # ────────────────────────────────────────────────────────
            else:
                c2 = ch[f]

            m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace("__main__.", "")
            m_.np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type = i, f, t
            if verbose:
                LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)

        return torch.nn.Sequential(*layers), sorted(save)

    # 替换原始 parse_model
    _tasks.parse_model = _patched_parse_model
    # 同时让 patched 版本能访问自定义模块（globals() 查找用）
    import builtins
    builtins.BackgroundReconstruct = BackgroundReconstruct
    builtins.FeatureEnhance        = FeatureEnhance
    print("[patch_parse_model] parse_model 已 patch，自定义模块通道推导已就绪")


# 执行注册 & patch（模块导入后立即生效）
register_custom_modules()
patch_parse_model()

# ── 此后才导入依赖 parse_model 的 ultralytics 组件 ────────────────────
from ultralytics.nn.tasks import DetectionModel
from ultralytics.data import build_yolo_dataset
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss
# intersect_dicts 已不需要（使用自定义 load_pretrained_weights 代替）


def de_parallel(model):
    """返回单卡模型（去掉 DDP/DataParallel 包装）。"""
    return model.module if isinstance(model, (DDP, torch.nn.DataParallel)) else model


# ─────────────────────────── 超参数配置 ───────────────────────────
CFG = {
    # 数据集
    "data_yaml":     "/data/lz/pythonProjects/HP-tracker/yolov8/satvideodt.yaml",
    "imgsz":         1024,
    "batch":         16,
    "workers":       4,
    # 模型
    "recon_weights": "/data/lz/pythonProjects/HP-tracker/yolov8/recon_checkpoints/recon_epoch_0038.pth", 
    "model_yaml":    "/data/lz/pythonProjects/HP-tracker/yolov8/yolov8n_custom.yaml",
    "model_weights": "/data/lz/pythonProjects/HP-tracker/yolov8/yolov8n.pt",
    "nc":            4,
    # 训练
    "epochs":        50,
    "lr0":           1e-3,
    "lrf":           0.01,
    "weight_decay":  5e-4,
    "warmup_epochs": 3,
    # 保存 & 日志
    "save_dir":      "/data/lz/pythonProjects/HP-tracker/yolov8/reconstruction-enhance-detection",
    "save_period":   5,
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

    train_cfg = get_cfg(DEFAULT_CFG)
    train_cfg.data    = cfg["data_yaml"]
    train_cfg.imgsz   = cfg["imgsz"]
    train_cfg.batch   = cfg["batch"]
    train_cfg.workers = cfg["workers"]
    train_cfg.rect    = False
    train_cfg.task    = "detect"

    data_dict = check_det_dataset(cfg["data_yaml"])
    img_path  = data_dict[mode if mode in data_dict else "train"]

    dataset = build_yolo_dataset(
        train_cfg,
        img_path=img_path,
        batch=cfg["batch"],
        data=data_dict,
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


class TupleCompatibleDetectionLoss:
    """对 v8DetectionLoss 的输出格式进行包装兼容。"""
    def __init__(self, model, de_parallel_fn):
        self.model        = model
        self.de_parallel  = de_parallel_fn
        self.base_loss    = v8DetectionLoss(self.de_parallel(model))

    def _normalize_preds(self, preds):
        if isinstance(preds, dict):
            return preds
        if isinstance(preds, tuple):
            if len(preds) > 1 and isinstance(preds[1], dict):
                return preds[1]
            if len(preds) > 1 and isinstance(preds[1], (list, tuple)):
                preds = preds[1]
            else:
                preds = preds[0]
        if isinstance(preds, (list, tuple)):
            feats   = list(preds)
            m       = self.de_parallel(self.model).model[-1]
            bs      = feats[0].shape[0]
            reg_max, nc = m.reg_max, m.nc
            no      = 4 * reg_max + nc
            boxes, scores = [], []
            for f in feats:
                f = f.view(bs, no, -1)
                boxes.append(f[:, : 4 * reg_max, :])
                scores.append(f[:, 4 * reg_max :, :])
            return {
                "boxes":  torch.cat(boxes,  dim=2),
                "scores": torch.cat(scores, dim=2),
                "feats":  feats,
            }
        raise TypeError(f"Unsupported preds type: {type(preds)}")

    def __call__(self, preds, batch):
        return self.base_loss(self._normalize_preds(preds), batch)


# ─────────────────────────── 权重迁移工具 ───────────────────────────
def load_pretrained_weights(model, weights_path, device):
    """
    将预训练权重迁移到自定义模型。

    因为自定义 YAML 在 backbone 插入了新层，层序号整体偏移，
    原始权重的 key（如 model.2.xxx）与新模型的 key（model.4.xxx）不匹配。
    此函数通过「按结构顺序对齐」的方式完成迁移：
      1. 收集预训练模型的 backbone 各层 state_dict（按层序）
      2. 收集自定义模型中对应 backbone 层的 state_dict（跳过新增层）
      3. 按顺序映射并加载
    新增的 BackgroundReconstruct 和 FeatureEnhance 层保持随机初始化。
    """
    ckpt = torch.load(weights_path, map_location=device)
    pretrained_model = (ckpt.get("ema") or ckpt["model"]).float()
    pretrained_sd    = pretrained_model.state_dict()

    # 原版 yolov8n backbone 层索引：0~9（10层）
    # 自定义模型 backbone 层索引：0,1 不变；2,3 是新增层；4~11 对应原 2~9
    # 原层号 -> 新层号 的映射（仅 backbone 部分）
    orig_to_new = {
        0: 0,    # Conv P1/2
        1: 1,    # Conv P2/4  (feat_early)
        # 原 2 -> 新 4  (C2f[128]，层号+2)
        2: 4,
        3: 5,
        4: 6,
        5: 7,
        6: 8,
        7: 9,
        8: 10,
        9: 11,   # SPPF
        # head 部分：原 10~22 -> 新 12~24（+2）
    }
    # head 层偏移量（统一 +2）
    n_orig_backbone = 10
    n_new_inserted  = 2   # 插入的新模块数量

    new_sd = model.state_dict()
    load_sd = {}

    # 迁移 backbone
    for orig_idx, new_idx in orig_to_new.items():
        prefix_orig = f"model.{orig_idx}."
        prefix_new  = f"model.{new_idx}."
        for k, v in pretrained_sd.items():
            if k.startswith(prefix_orig):
                new_k = prefix_new + k[len(prefix_orig):]
                if new_k in new_sd and new_sd[new_k].shape == v.shape:
                    load_sd[new_k] = v

    # 迁移 head（原 10~22 -> 新 12~24）
    n_orig_total = len(set(k.split(".")[1] for k in pretrained_sd if k.startswith("model.")))
    for orig_idx in range(n_orig_backbone, n_orig_total):
        new_idx     = orig_idx + n_new_inserted
        prefix_orig = f"model.{orig_idx}."
        prefix_new  = f"model.{new_idx}."
        for k, v in pretrained_sd.items():
            if k.startswith(prefix_orig):
                new_k = prefix_new + k[len(prefix_orig):]
                if new_k in new_sd and new_sd[new_k].shape == v.shape:
                    load_sd[new_k] = v

    missing = [k for k in new_sd if k not in load_sd]
    model.load_state_dict(load_sd, strict=False)
    LOGGER.info(
        f"[权重迁移] 成功迁移 {len(load_sd)}/{len(new_sd)} 个参数张量\n"
        f"           未迁移（新增/不匹配）: {len(missing)} 个张量"
    )
    # 打印新增层的 key，便于核查
    new_layer_keys = [k for k in missing if any(
        f"model.{i}." in k for i in [2, 3]  # BGRecon, FeatEnh 层
    )]
    if new_layer_keys:
        LOGGER.info(f"           新增层参数（随机初始化）: {new_layer_keys[:5]}{'...' if len(new_layer_keys)>5 else ''}")
    return model


def load_recon_pretrained(model, recon_ckpt_path, device):
    """将独立的 BackgroundReconstruct 权重精准注入 YOLO 的 layer 2"""
    if not recon_ckpt_path or not os.path.exists(recon_ckpt_path):
        LOGGER.info("⚠️  未指定或找不到重建模块权重，跳过加载")
        return model

    ckpt = torch.load(recon_ckpt_path, map_location=device)
    # 兼容 torch.save(state_dict) 或 torch.save({'model': state_dict})
    recon_sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    # 1. 清理 DataParallel 遗留的 module. 前缀
    cleaned_sd = {k.replace("module.", ""): v for k, v in recon_sd.items()}

    # 2. 拼接 YOLO 模型中的正确前缀 (单卡: model.2. | DDP: module.model.2.)
    is_ddp = isinstance(model, DDP)
    prefix = "module.model.2." if is_ddp else "model.2."
    mapped_sd = {prefix + k: v for k, v in cleaned_sd.items()}

    # 3. 通道形状预检（防止 1ch/3ch 不一致导致崩溃）
    current_sd = model.state_dict()
    shape_mismatch = []
    for k, v in mapped_sd.items():
        if k in current_sd and current_sd[k].shape != v.shape:
            shape_mismatch.append(f"  {k} -> 模型{current_sd[k].shape} vs 权重{v.shape}")
    if shape_mismatch:
        LOGGER.error("❌ 权重通道形状不匹配，已中止加载：\n" + "\n".join(shape_mismatch))
        LOGGER.error("请确保 YAML 中 BackgroundReconstruct 的 c_out 与预训练权重一致！")
        return model

    # 4. 注入权重
    model.load_state_dict(mapped_sd, strict=False)
    LOGGER.info(f"✅ 成功加载 BackgroundReconstruct 权重至 layer 2: {recon_ckpt_path}")
    return model

# ─────────────────────────── 主训练函数 ───────────────────────────
def train(rank=-1, world_size=1):
    cfg    = CFG
    ddp    = rank != -1
    main   = is_main(rank)
    device = torch.device(f"cuda:{rank}" if ddp else ("cuda:3" if torch.cuda.is_available() else "cpu"))

    save_dir = Path(cfg["save_dir"])
    writer   = None
    if main:
        save_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(save_dir / "tensorboard"))
        LOGGER.info(f"[TensorBoard] tensorboard --logdir {save_dir / 'tensorboard'}")

    # ── 构建自定义模型 ────────────────────────────────────────────────
    # 直接从 YAML 构建 DetectionModel，nc 在此指定为 4
    model = DetectionModel(cfg["model_yaml"], ch=3, nc=cfg["nc"], verbose=main)

    # ── 迁移预训练权重（backbone/neck 层按结构顺序对齐迁移）────────────
    model = load_pretrained_weights(model, cfg["model_weights"], device="cpu")

    # ✅ 新增：加载 BackgroundReconstruct 独立预训练权重
    model = load_recon_pretrained(model, cfg.get("recon_weights"), device=device)

    if main:
        recon_layer = de_parallel(model).model[2]
        LOGGER.info(f"[验证] BackgroundReconstruct 输出通道: {recon_layer.c_out}")

    model = model.to(device)

    # 冻结 BackgroundReconstruct 参数，不参与反向传播更新
    background_reconstruct = model.model[2]
    for p in background_reconstruct.parameters():
        p.requires_grad_(False)

    # 其余参数保持可训练
    for name, p in model.named_parameters():
        if not name.startswith("model.2."):
            p.requires_grad_(True)

    # 确保 model.args 是带属性访问的 IterableSimpleNamespace
    from ultralytics.utils import IterableSimpleNamespace
    if not isinstance(getattr(model, "args", None), IterableSimpleNamespace):
        default_hyp = {
            "box": 7.5, "cls": 0.5, "dfl": 1.5,
            "pose": 12.0, "kobj": 2.0,
            "label_smoothing": 0.0, "nbs": 64,
        }
        if isinstance(getattr(model, "args", None), dict):
            default_hyp.update(model.args)
        model.args = IterableSimpleNamespace(**default_hyp)

    # ── 优化器：AdamW ─────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=cfg["lr0"],
        betas=(0.937, 0.999),
        weight_decay=cfg["weight_decay"],
    )

    # ── Cosine LR 调度器 ───────────────────────────────────────────────
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: cosine_lr_lambda(
            ep, cfg["epochs"], cfg["warmup_epochs"], cfg["lrf"]
        ),
    )

    # ── AMP Scaler ────────────────────────────────────────────────────
    scaler = GradScaler(device=device.type)

    # ── 断点续训 ───────────────────────────────────────────────────────
    start_epoch = 0
    best_loss   = float("inf")
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

    # ── DDP 包装 ───────────────────────────────────────────────────────
    if ddp:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # ── 损失函数 ───────────────────────────────────────────────────────
    compute_loss = TupleCompatibleDetectionLoss(model, de_parallel)

    # ── DataLoader ────────────────────────────────────────────────────
    train_loader, train_sampler = get_dataloader(cfg, rank, mode="train")
    val_loader, val_sampler     = get_dataloader(cfg, rank, mode="val")
    if main:
        LOGGER.info(f"训练集批次数: {len(train_loader)}")
        LOGGER.info(f"验证集批次数: {len(val_loader)}")
        LOGGER.info(f"可训练参数张量数量: {len(trainable_params)}")

    # ─────────────────────────── 训练主循环 ───────────────────────────
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        epoch_loss  = 0.0
        box_loss_ep = 0.0
        cls_loss_ep = 0.0
        dfl_loss_ep = 0.0
        t0          = time.time()

        for i, batch in enumerate(train_loader):
            batch["img"]       = batch["img"].to(device, non_blocking=True).float() / 255.0
            batch["bboxes"]    = batch["bboxes"].to(device)
            batch["cls"]       = batch["cls"].to(device)
            batch["batch_idx"] = batch["batch_idx"].to(device)

            optimizer.zero_grad()

            with autocast(device_type=device.type):
                preds = model(batch["img"])

            loss, loss_items = compute_loss(preds, batch)
            loss_scalar = loss.sum()

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

        n_batches = len(train_loader)
        avg_loss  = epoch_loss  / n_batches
        avg_box   = box_loss_ep / n_batches
        avg_cls   = cls_loss_ep / n_batches
        avg_dfl   = dfl_loss_ep / n_batches

        # ── 验证集 loss（仅前向，不更新参数）─────────────────────────────
        model.eval()
        val_loss_ep = 0.0
        val_box_ep  = 0.0
        val_cls_ep  = 0.0
        val_dfl_ep  = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch["img"]       = batch["img"].to(device, non_blocking=True).float() / 255.0
                batch["bboxes"]    = batch["bboxes"].to(device)
                batch["cls"]       = batch["cls"].to(device)
                batch["batch_idx"] = batch["batch_idx"].to(device)

                with autocast(device_type=device.type):
                    preds = model(batch["img"])

                v_loss, v_items = compute_loss(preds, batch)
                v_scalar = v_loss.sum()
                val_loss_ep += v_scalar.item()
                val_box_ep  += v_items[0].item()
                val_cls_ep  += v_items[1].item()
                val_dfl_ep  += v_items[2].item()

        n_val_batches = len(val_loader)
        avg_val_loss  = val_loss_ep / n_val_batches
        avg_val_box   = val_box_ep  / n_val_batches
        avg_val_cls   = val_cls_ep  / n_val_batches
        avg_val_dfl   = val_dfl_ep  / n_val_batches

        elapsed   = time.time() - t0
        lr_cur    = optimizer.param_groups[0]["lr"]

        if main:
            LOGGER.info(
                f"\n=== Epoch {epoch+1}/{cfg['epochs']}  "
                f"train_loss={avg_loss:.4f}  train_box={avg_box:.4f}  "
                f"train_cls={avg_cls:.4f}  train_dfl={avg_dfl:.4f}  "
                f"val_loss={avg_val_loss:.4f}  val_box={avg_val_box:.4f}  "
                f"val_cls={avg_val_cls:.4f}  val_dfl={avg_val_dfl:.4f}  "
                f"lr={lr_cur:.6f}  time={elapsed:.1f}s ==="
            )
            global_step = epoch + 1
            writer.add_scalar("Loss/train", avg_loss, global_step)
            writer.add_scalar("Loss/box",   avg_box,  global_step)
            writer.add_scalar("Loss/cls",   avg_cls,  global_step)
            writer.add_scalar("Loss/dfl",   avg_dfl,  global_step)
            writer.add_scalar("Loss/val",   avg_val_loss, global_step)
            writer.add_scalar("Val/box",    avg_val_box,  global_step)
            writer.add_scalar("Val/cls",    avg_val_cls,  global_step)
            writer.add_scalar("Val/dfl",    avg_val_dfl,  global_step)
            writer.add_scalar("LR/lr0",     lr_cur,   global_step)

            is_best = avg_val_loss < best_loss
            if is_best:
                best_loss = avg_val_loss

            ckpt = {
                "epoch":     epoch,
                "model":     de_parallel(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler":    scaler.state_dict(),
                "best_loss": best_loss,
            }

            if (epoch + 1) % cfg["save_period"] == 0:
                torch.save(ckpt, save_dir / f"epoch{epoch+1}.pt")
                LOGGER.info(f"Checkpoint 已保存：{save_dir}/epoch{epoch+1}.pt")

            torch.save(ckpt, save_dir / "last.pt")
            if is_best:
                torch.save(ckpt, save_dir / "best.pt")
                LOGGER.info(f"最佳模型已更新：{save_dir}/best.pt  (loss={best_loss:.4f})")

    # ── 训练结束 ──────────────────────────────────────────────────────
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
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        setup_ddp(local_rank, world_size)
        train(rank=local_rank, world_size=world_size)
    else:
        train(rank=-1, world_size=1)
  