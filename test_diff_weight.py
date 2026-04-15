#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对单张图片执行真实 YOLO 推理链路，并可视化 FeatureEnhance 中的 diff_weight。

默认行为：
1. 从 DEFAULT_SOURCE_DIR 中按文件名排序取第 DEFAULT_IMAGE_INDEX 张图片
2. 加载完整模型权重
3. 使用 forward hook 捕获 feat_early、recon_out、diff_map 与 filtered_diff_map
4. 按 custom_modules.py 中相同逻辑恢复 diff_weight
5. 将 diff_weight 与 recon_out 保存为图像

用法示例：
    python test_diff_weight.py
    python test_diff_weight.py --image-index 10
    python test_diff_weight.py --device 0 --imgsz 1024
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from test_best import ensure_custom_parse_model_patched, get_dataset_nc, load_model_compatible
from train import clear_background_reconstruct_raw_input, set_background_reconstruct_raw_input


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_WEIGHTS = "./reconstruction-enhance-detection/best.pt"
DEFAULT_MODEL_YAML = "./yolov8n_custom.yaml"
DEFAULT_DATA = "./satvideodt.yaml"
DEFAULT_SOURCE_DIR = "/data/lz/SatVideoDT/val_data/005/img1/"
DEFAULT_IMAGE_INDEX = 0
DEFAULT_OUTPUT = "./diff_weight_vis"
DEFAULT_DEVICE = "0"
DEFAULT_IMGSZ = 1024
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7


def get_source_image(source_dir: Path, image_index: int) -> Path:
    image_paths = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not image_paths:
        raise RuntimeError(f"目录中未找到图片: {source_dir}")
    if image_index < 0 or image_index >= len(image_paths):
        raise IndexError(
            f"image_index={image_index} 超出范围，当前目录共有 {len(image_paths)} 张图片"
        )
    return image_paths[image_index]


def get_backbone_sequential(model: YOLO) -> torch.nn.Sequential:
    detection_model = model.model
    if hasattr(detection_model, "model"):
        backbone = detection_model.model
        if isinstance(backbone, torch.nn.Sequential):
            return backbone
    raise RuntimeError("未能从 YOLO 模型中找到 Sequential 主干结构")


def register_hooks(model: YOLO, captured: dict[str, torch.Tensor]) -> list[torch.utils.hooks.RemovableHandle]:
    backbone = get_backbone_sequential(model)

    if len(backbone) <= 3:
        raise RuntimeError("模型层数不足，无法定位 feat_early / FeatureEnhance")

    feat_layer = backbone[1]
    background_reconstruct = backbone[2]
    diff_map_layer = backbone[2]
    feature_enhance = backbone[3]

    if not hasattr(feature_enhance, "filtration"):
        raise RuntimeError("未在第 3 层找到 filtration 子模块，请检查模型结构")
    if not hasattr(background_reconstruct, "recon_net"):
        raise RuntimeError("未在第 2 层找到 recon_net 子模块，请检查模型结构")

    def feat_hook(_module, _inputs, output):
        captured["feat_early"] = output.detach().cpu()

    def recon_hook(_module, _inputs, output):
        captured["recon_out"] = output.detach().cpu()

    def diff_map_hook(_module, _inputs, output):
        captured["diff_map"] = output.detach().cpu()

    def filtration_hook(_module, _inputs, output):
        captured["filtered_diff_map"] = output.detach().cpu()

    handles = [
        feat_layer.register_forward_hook(feat_hook),
        background_reconstruct.recon_net.register_forward_hook(recon_hook),
        diff_map_layer.register_forward_hook(diff_map_hook),
        feature_enhance.filtration.register_forward_hook(filtration_hook),
    ]
    return handles


def get_filtration_threshold(model: YOLO) -> float:
    backbone = get_backbone_sequential(model)
    feature_enhance = backbone[3]
    if not hasattr(feature_enhance, "filtration"):
        raise RuntimeError("未在第 3 层找到 filtration 子模块，请检查模型结构")
    return float(feature_enhance.filtration.threshold.detach().cpu().item())


def restore_diff_weight(filtered_diff_map: torch.Tensor) -> torch.Tensor:
    if filtered_diff_map.ndim != 4:
        raise ValueError(f"filtered_diff_map 维度异常: {tuple(filtered_diff_map.shape)}")
    if filtered_diff_map.shape[1] > 1:
        return filtered_diff_map.mean(dim=1, keepdim=True)
    return filtered_diff_map


def save_single_channel_image(tensor: torch.Tensor, save_path: Path) -> None:
    tensor_2d = tensor[0, 0].numpy()
    min_val = float(tensor_2d.min())
    max_val = float(tensor_2d.max())

    if max_val > min_val:
        norm = (tensor_2d - min_val) / (max_val - min_val)
    else:
        norm = np.zeros_like(tensor_2d, dtype=np.float32)

    image_uint8 = (norm * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(image_uint8, mode="L").save(save_path)


def save_three_channel_image(tensor: torch.Tensor, save_path: Path) -> None:
    tensor_3d = tensor[0].permute(1, 2, 0).numpy()
    image_uint8 = (tensor_3d * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(image_uint8, mode="RGB").save(save_path)


def print_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    print(
        f"{name}: shape={tuple(tensor.shape)}, "
        f"min={tensor.min().item():.6f}, "
        f"max={tensor.max().item():.6f}, "
        f"mean={tensor.mean().item():.6f}"
    )


def print_threshold_analysis(diff_map: torch.Tensor, threshold: float) -> None:
    ratio_above = (diff_map > threshold).float().mean().item()
    ratio_equal = (diff_map == threshold).float().mean().item()
    ratio_below = (diff_map < threshold).float().mean().item()
    print(f"filtration.threshold: {threshold:.6f}")
    print(
        f"diff_map vs threshold: above={ratio_above:.6f}, "
        f"equal={ratio_equal:.6f}, below={ratio_below:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化单张图片经过真实链路后的 diff_weight")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="模型权重路径")
    parser.add_argument("--model-yaml", type=str, default=DEFAULT_MODEL_YAML, help="模型结构 yaml")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="数据集配置 yaml")
    parser.add_argument("--source-dir", type=str, default=DEFAULT_SOURCE_DIR, help="默认图片目录")
    parser.add_argument("--image-index", type=int, default=DEFAULT_IMAGE_INDEX, help="按排序后的图片索引选择单张图片")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="推理尺寸")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="检测置信度阈值")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="检测 NMS IoU 阈值")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="推理设备，默认使用 GPU 0")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="灰度图输出目录")
    args = parser.parse_args()

    weights_path = Path(args.weights).expanduser().resolve()
    model_yaml = Path(args.model_yaml).expanduser().resolve()
    data_yaml = Path(args.data).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not weights_path.exists():
        raise FileNotFoundError(f"模型权重不存在: {weights_path}")
    if not model_yaml.exists():
        raise FileNotFoundError(f"模型结构 yaml 不存在: {model_yaml}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"数据集配置不存在: {data_yaml}")
    if not source_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {source_dir}")

    source_image = get_source_image(source_dir, args.image_index)
    dataset_nc = get_dataset_nc(data_yaml)

    ensure_custom_parse_model_patched()
    model = load_model_compatible(weights_path, model_yaml, nc=dataset_nc)

    captured: dict[str, torch.Tensor] = {}
    hook_handles = register_hooks(model, captured)

    try:
        from torchvision import transforms

        image = Image.open(source_image).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((args.imgsz, args.imgsz)),
            transforms.ToTensor(),
        ])
        input_tensor = preprocess(image).unsqueeze(0)

        yolo_model = model.model if hasattr(model, "model") else model
        yolo_model = yolo_model.to(args.device if args.device else "cpu")
        input_tensor = input_tensor.to(next(yolo_model.parameters()).device)

        set_background_reconstruct_raw_input(yolo_model, input_tensor)
        try:
            with torch.no_grad():
                _ = yolo_model(input_tensor)
        finally:
            clear_background_reconstruct_raw_input(yolo_model)
    finally:
        for handle in hook_handles:
            handle.remove()

    if "feat_early" not in captured:
        raise RuntimeError("未捕获到 feat_early，请检查 hook 位置是否正确")
    if "recon_out" not in captured:
        raise RuntimeError("未捕获到 recon_out，请检查 hook 位置是否正确")
    if "diff_map" not in captured:
        raise RuntimeError("未捕获到 diff_map，请检查 hook 位置是否正确")
    if "filtered_diff_map" not in captured:
        raise RuntimeError("未捕获到 filtered_diff_map，请检查 hook 位置是否正确")

    feat_early = captured["feat_early"]
    recon_out = captured["recon_out"]
    diff_map = captured["diff_map"]
    filtered_diff_map = captured["filtered_diff_map"]
    diff_weight = restore_diff_weight(filtered_diff_map)
    filtration_threshold = get_filtration_threshold(model)

    output_dir.mkdir(parents=True, exist_ok=True)
    diff_weight_path = output_dir / f"diff_weight_{source_image.stem}.png"
    recon_out_path = output_dir / f"recon_out_{source_image.stem}.png"
    save_single_channel_image(diff_weight, diff_weight_path)
    save_three_channel_image(recon_out, recon_out_path)

    print(f"source_image: {source_image}")
    print_tensor_stats("feat_early", feat_early)
    print_tensor_stats("recon_out", recon_out)
    print_tensor_stats("diff_map", diff_map)
    print_tensor_stats("filtered_diff_map", filtered_diff_map)
    print_tensor_stats("diff_weight", diff_weight)
    print_threshold_analysis(diff_map, filtration_threshold)
    print(f"recon_out 图像已保存到: {recon_out_path}")
    print(f"diff_weight 灰度图已保存到: {diff_weight_path}")


if __name__ == "__main__":
    main()
