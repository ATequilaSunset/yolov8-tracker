#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 best.pt 效果并可视化（默认 10 张图）
兼容：
1) Ultralytics 原生 YOLO 权重
2) 自定义训练保存的 checkpoint/state_dict
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="测试 best.pt 并保存可视化结果")
    parser.add_argument(
        "--model",
        type=str,
        default="/data/lz/pythonProjects/HP-tracker/yolov8/runs/detect/custom_train/best.pt",
        help="模型权重路径",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolov8n.pt",
        help="当 --model 是 state_dict 时，用于构建网络结构的基础模型",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="/data/lz/pythonProjects/HP-tracker/yolov8/coco128/images/train2017",
        help="测试图片目录",
    )
    parser.add_argument("--num-images", type=int, default=10, help="测试图片数量")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/data/lz/pythonProjects/HP-tracker/yolov8/runs/detect/custom_train/test_vis",
        help="可视化输出目录",
    )
    return parser.parse_args()


def collect_images(image_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def load_model_compatible(model_path: Path, base_model: str):
    """先尝试原生加载，失败则按 state_dict 加载。"""
    try:
        return YOLO(str(model_path))
    except Exception:
        yolo = YOLO(base_model)
        ckpt = torch.load(str(model_path), map_location="cpu")

        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise RuntimeError(f"不支持的 checkpoint 格式: {type(ckpt)}")

        missing, unexpected = yolo.model.load_state_dict(state_dict, strict=False)
        print(f"[Info] state_dict 加载完成: missing={len(missing)}, unexpected={len(unexpected)}")
        return yolo


def main():
    args = parse_args()

    model_path = Path(args.model)
    image_dir = Path(args.image_dir)
    save_dir = Path(args.save_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")

    save_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(image_dir)
    if len(image_paths) == 0:
        raise RuntimeError(f"目录下没有可用图片: {image_dir}")

    n = min(args.num_images, len(image_paths))
    selected = image_paths[:n]

    model = load_model_compatible(model_path, args.base_model)
    results = model.predict(
        source=[str(p) for p in selected],
        imgsz=args.imgsz,
        conf=args.conf,
        verbose=False,
    )

    lines = []
    for i, (img_path, result) in enumerate(zip(selected, results), start=1):
        plotted = result.plot()
        out_file = save_dir / f"{i:02d}_{img_path.name}"
        cv2.imwrite(str(out_file), plotted)

        n_det = 0 if result.boxes is None else len(result.boxes)
        lines.append(f"{img_path.name}\tdetections={n_det}\tsaved={out_file.name}")

    (save_dir / "selected_images.txt").write_text("\n".join([str(p) for p in selected]), encoding="utf-8")
    (save_dir / "pred_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("测试完成")
    print(f"模型: {model_path}")
    print(f"测试图片数: {len(selected)}")
    print(f"可视化输出目录: {save_dir}")
    print(f"结果汇总: {save_dir / 'pred_summary.txt'}")


if __name__ == "__main__":
    main()