#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机抽取验证集 10 张图像，测试 ./yolo-enhance-results/best.pt 的检测效果并保存可视化结果。

说明：
- 兼容自定义 train.py 保存的 checkpoint 格式（ckpt["model"] 为 state_dict）。

用法示例：
    python test_best.py
    python test_best.py --num-images 10 --conf 0.2 --imgsz 640
"""
 
import argparse
import random
from pathlib import Path

import torch
import yaml 
from ultralytics import YOLO

from custom_modules import register_custom_modules


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def get_dataset_nc(data_yaml: Path) -> int:
    with open(data_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    if "nc" in data_cfg:
        return int(data_cfg["nc"])
    if "names" in data_cfg:
        names = data_cfg["names"]
        return len(names) if isinstance(names, (list, tuple)) else len(dict(names))
    raise KeyError("数据集配置中未找到 nc 或 names")


def load_val_images(data_yaml: Path) -> list[Path]:
    with open(data_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = Path(data_cfg["path"]).expanduser().resolve()
    val_rel = data_cfg["val"]
    val_dir = (dataset_root / val_rel).resolve()

    if not val_dir.exists():
        raise FileNotFoundError(f"验证集目录不存在: {val_dir}")

    images = [p for p in val_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not images:
        raise RuntimeError(f"验证集目录中未找到图像: {val_dir}")

    return images


def ensure_custom_parse_model_patched() -> None:
    """确保自定义模块对应的 parse_model patch 已生效。"""
    import ultralytics.nn.tasks as _tasks

    if getattr(_tasks, "_hp_custom_parse_patched", False):
        return

    # 复用训练脚本中的 patch 逻辑，避免测试与训练的模型构建行为不一致。
    from train import patch_parse_model

    patch_parse_model()
    _tasks._hp_custom_parse_patched = True


def load_model_compatible(weights_path: Path, model_yaml: Path, nc: int) -> YOLO:
    """
    兼容两种权重格式：
    1) Ultralytics 标准 .pt（可直接 YOLO(weights)）
    2) 自定义 checkpoint（dict 且 ckpt['model'] 为 state_dict）
    """
    try:
        return YOLO(str(weights_path))
    except AttributeError as e:
        if "OrderedDict" not in str(e):
            raise

        # 回退：按 yaml + 指定 nc 构建 DetectionModel，再手动加载 state_dict
        from ultralytics.nn.tasks import DetectionModel

        yolo = YOLO(str(model_yaml))
        yolo.model = DetectionModel(str(model_yaml), ch=3, nc=nc, verbose=False)

        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise RuntimeError("不支持的 checkpoint 格式，无法解析为 state_dict")

        yolo.model.load_state_dict(state_dict, strict=True)
        return yolo


def main():
    parser = argparse.ArgumentParser(description="随机测试 best.pt 在验证集的检测效果")
    parser.add_argument("--weights", type=str, default="./yolo-enhance-results/epoch15.pt", help="模型权重路径")
    parser.add_argument("--model-yaml", type=str, default="./yolov8n_custom.yaml", help="模型结构 yaml（用于加载自定义 checkpoint）")
    parser.add_argument("--data", type=str, default="./satvideodt.yaml", help="数据集配置 yaml")
    parser.add_argument("--num-images", type=int, default=10, help="随机抽取的验证图像数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--device", type=str, default="", help="推理设备，例如 0 或 cpu；留空自动选择")
    parser.add_argument("--project", type=str, default="/data/lz/pythonProjects/HP-tracker/yolov8/detect_results_vis/val_random10", help="可视化输出目录")
    parser.add_argument("--name", type=str, default="predict", help="本次实验子目录名")
    args = parser.parse_args()

    weights_path = Path(args.weights).expanduser().resolve()
    model_yaml = Path(args.model_yaml).expanduser().resolve()
    data_yaml = Path(args.data).expanduser().resolve()

    if not weights_path.exists():
        raise FileNotFoundError(f"模型权重不存在: {weights_path}")
    if not model_yaml.exists():
        raise FileNotFoundError(f"模型结构 yaml 不存在: {model_yaml}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"数据集配置不存在: {data_yaml}")

    all_val_images = load_val_images(data_yaml)
    dataset_nc = get_dataset_nc(data_yaml)
    k = min(args.num_images, len(all_val_images))

    random.seed(args.seed)
    sampled_images = random.sample(all_val_images, k)

    # 注册自定义模块（构建/加载包含自定义层的模型时必需）
    register_custom_modules()
    ensure_custom_parse_model_patched()
    model = load_model_compatible(weights_path, model_yaml, nc=dataset_nc)

    results = model.predict(
        source=[str(p) for p in sampled_images],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=False,
    )

    save_dir = Path(results[0].save_dir) if results else (Path(args.project) / args.name)
    save_dir.mkdir(parents=True, exist_ok=True)

    sampled_txt = save_dir / "sampled_images.txt"
    sampled_txt.write_text("\n".join(str(p) for p in sampled_images), encoding="utf-8")

    print(f"\n已随机抽取 {k} 张验证图像并完成推理。")
    print(f"可视化结果保存目录: {save_dir}")
    print(f"采样清单: {sampled_txt}")


if __name__ == "__main__":
    main()