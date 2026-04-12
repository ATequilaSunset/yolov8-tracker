#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 SatVideoDT 测试集上运行 YOLO + SORT(ByteTrack 风格二阶段匹配) 多目标追踪，
并将可视化结果保存到指定目录。

用法示例：
    python test_best.py
    python test_best.py --conf 0.2 --imgsz 640 --device 0
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

from custom_modules import register_custom_modules
from sort import Sort, TrackingVisualizer


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


def ensure_custom_parse_model_patched() -> None:
    """确保自定义模块对应的 parse_model patch 已生效。"""
    import ultralytics.nn.tasks as _tasks

    if getattr(_tasks, "_hp_custom_parse_patched", False):
        return

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


def load_test_sequences(test_root: Path) -> list[tuple[str, list[Path]]]:
    sequences: list[tuple[str, list[Path]]] = []

    for seq_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        img_dir = seq_dir / "img1"
        if not img_dir.exists():
            continue

        frame_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if frame_paths:
            sequences.append((seq_dir.name, frame_paths))

    if not sequences:
        raise RuntimeError(f"测试集目录中未找到有效序列: {test_root}")

    return sequences


def result_to_detections(result) -> np.ndarray:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 5), dtype=np.float32)

    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1)
    return np.concatenate((xyxy, conf), axis=1)


def save_tracking_txt(track_txt_path: Path, frame_id: int, tracks: np.ndarray) -> None:
    lines = []
    for track in tracks:
        x1, y1, x2, y2, track_id = track[:5]
        w = x2 - x1
        h = y2 - y1
        lines.append(
            f"{frame_id},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1"
        )

    if lines:
        with open(track_txt_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def run_tracking_on_sequence(
    model: YOLO,
    seq_name: str,
    frame_paths: list[Path],
    output_root: Path,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    max_age: int,
    min_hits: int,
    track_iou: float,
    det_thresh: float,
) -> None:
    seq_output_dir = output_root / seq_name
    vis_dir = seq_output_dir / "img1"
    vis_dir.mkdir(parents=True, exist_ok=True)

    track_txt_path = seq_output_dir / f"{seq_name}.txt"
    if track_txt_path.exists():
        track_txt_path.unlink()

    tracker = Sort(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=track_iou,
        det_thresh=det_thresh,
    )
    visualizer = TrackingVisualizer()

    for frame_id, frame_path in enumerate(frame_paths, start=1):
        results = model.predict(
            source=str(frame_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            save=False,
            verbose=False,
        )
        result = results[0]
        dets = result_to_detections(result)
        tracks = tracker.update(dets)

        image = Image.open(frame_path).convert("RGB")
        vis_image = visualizer.draw_tracks(image=image, tracks=tracks, detections=dets)
        vis_image.save(vis_dir / frame_path.name)

        save_tracking_txt(track_txt_path, frame_id, tracks)

    print(f"序列 {seq_name} 处理完成，结果保存到: {seq_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="在 SatVideoDT 测试集上运行 YOLO + SORT 多目标追踪")
    parser.add_argument("--weights", type=str, default="./reconstruction-enhance-detection/best.pt", help="模型权重路径")
    parser.add_argument("--model-yaml", type=str, default="./yolov8n_custom.yaml", help="模型结构 yaml（用于加载自定义 checkpoint）")
    parser.add_argument("--data", type=str, default="./satvideodt.yaml", help="数据集配置 yaml")
    parser.add_argument("--test-root", type=str, default="/data/lz/SatVideoDT/val_data", help="测试集根目录")
    parser.add_argument("--imgsz", type=int, default=1024, help="推理尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="检测 NMS IoU 阈值")
    parser.add_argument("--device", type=str, default="", help="推理设备，例如 0 或 cpu；留空自动选择")
    parser.add_argument("--output", type=str, default="./track_results_vis/sort_bytetrack", help="追踪可视化输出目录")
    parser.add_argument("--max-age", type=int, default=20, help="轨迹最大丢失帧数")
    parser.add_argument("--min-hits", type=int, default=3, help="轨迹确认所需最少命中次数")
    parser.add_argument("--track-iou", type=float, default=0.3, help="SORT 关联 IoU 阈值")
    parser.add_argument("--det-thresh", type=float, default=0.6, help="ByteTrack 风格高分检测阈值")
    args = parser.parse_args()

    weights_path = Path(args.weights).expanduser().resolve()
    model_yaml = Path(args.model_yaml).expanduser().resolve()
    data_yaml = Path(args.data).expanduser().resolve()
    test_root = Path(args.test_root).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not weights_path.exists():
        raise FileNotFoundError(f"模型权重不存在: {weights_path}")
    if not model_yaml.exists():
        raise FileNotFoundError(f"模型结构 yaml 不存在: {model_yaml}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"数据集配置不存在: {data_yaml}")
    if not test_root.exists():
        raise FileNotFoundError(f"测试集根目录不存在: {test_root}")

    test_sequences = load_test_sequences(test_root)
    dataset_nc = get_dataset_nc(data_yaml)

    register_custom_modules()
    ensure_custom_parse_model_patched()
    model = load_model_compatible(weights_path, model_yaml, nc=dataset_nc)

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"共发现 {len(test_sequences)} 个测试序列")
    print(f"结果输出目录: {output_root}")

    for seq_name, frame_paths in test_sequences:
        run_tracking_on_sequence(
            model=model,
            seq_name=seq_name,
            frame_paths=frame_paths,
            output_root=output_root,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            max_age=args.max_age,
            min_hits=args.min_hits,
            track_iou=args.track_iou,
            det_thresh=args.det_thresh,
        )

    print("\n全部测试序列追踪完成。")


if __name__ == "__main__":
    main()
