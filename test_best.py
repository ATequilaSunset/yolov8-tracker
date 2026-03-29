"""
测试 best.pt 效果并可视化（默认 10 张图）
支持自定义训练保存的 checkpoint（train.py 格式：ckpt['model'] = state_dict）
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
def parse_args():
    parser = argparse.ArgumentParser(description="测试 best.pt 并保存可视化结果")
    parser.add_argument(
        "--model",
        type=str,
        default="/data/lz/pythonProjects/HP-tracker/yolov8/sat_results/best.pt",
        help="模型权重路径",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolov8n.pt",
        help="用于构建网络结构的基础模型（仅当 checkpoint 为 state_dict 时使用）",
    )
    parser.add_argument(
        "--nc",
        type=int,
        default=1,
        help="训练时的类别数（当 checkpoint 为 state_dict 时使用）",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="/data/lz/SatVideoDT/val_data",
        help="测试图片目录（会递归搜索所有 img1/ 子目录下的图片）",
    )
    parser.add_argument("--num-images", type=int, default=10, help="测试图片数量")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/data/lz/pythonProjects/HP-tracker/yolov8/sat_results/test_vis",
        help="可视化输出目录",
    )
    return parser.parse_args()
def collect_images(image_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    # 递归搜索所有 img1/ 子目录下的图片（SatVideoDT 数据集结构）
    img1_images = sorted([
        p for p in image_dir.rglob("img1/*")
        if p.is_file() and p.suffix.lower() in exts
    ])
    if img1_images:
        return img1_images
    # 降级：直接搜索当前目录下的图片
    return sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
def load_model_compatible(model_path: Path, base_model: str, nc: int):
    """
    加载模型，兼容两种格式：
    1. Ultralytics 原生格式（ckpt['model'] 是 nn.Module 对象）
    2. train.py 自定义格式（ckpt['model'] 是 OrderedDict state_dict，nc=1）
    """
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    # 判断是否为原生 Ultralytics 格式：model 字段是 nn.Module
    if isinstance(ckpt, dict) and "model" in ckpt:
        model_obj = ckpt["model"]
        if hasattr(model_obj, "float"):  # nn.Module
            print("[Info] 检测到 Ultralytics 原生格式，直接加载")
            return YOLO(str(model_path))
        # 否则是 state_dict（OrderedDict），走自定义加载
        state_dict = model_obj
        print(f"[Info] 检测到自定义 checkpoint 格式（state_dict），nc={nc}")
    else:
        raise RuntimeError(f"不支持的 checkpoint 格式: {type(ckpt)}")
    # 用 base_model 构建骨架，但覆盖 nc 为训练时的类别数
    yolo = YOLO(base_model)
    # 重新构建正确 nc 的检测头
    yolo.model.model[-1].nc = nc  # 设置 nc
    # 用 nc=1 重新初始化检测头（DetectionModel 会根据 nc 调整输出通道）
    # 直接用 state_dict 覆盖，strict=False 忽略无关键
    missing, unexpected = yolo.model.load_state_dict(state_dict, strict=False)
    # 如果仍有尺寸不匹配，说明需要重建检测头，改用 DetectionModel 直接构建
    if missing or unexpected:
        print(f"[Warning] strict=False 加载: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"  missing keys (前5): {missing[:5]}")
        if unexpected:
            print(f"  unexpected keys (前5): {unexpected[:5]}")
    yolo.model.eval()
    return yolo
def _rebuild_detect_head(inner_model, nc: int):
    """完全复现 train.py 的检测头替换逻辑，物理替换 cv3 最后一层 Conv2d 输出通道为 nc。"""
    detect_head = inner_model.model[-1]
    detect_head.nc = nc
    for i, cv3_seq in enumerate(detect_head.cv3):
        last = cv3_seq[-1]
        if hasattr(last, 'conv') and isinstance(last.conv, torch.nn.Conv2d):
            # ultralytics Conv 包装类
            raw = last.conv
            last.conv = torch.nn.Conv2d(
                raw.in_channels, nc,
                kernel_size=raw.kernel_size,
                stride=raw.stride,
                padding=raw.padding,
                bias=raw.bias is not None,
            )
        elif isinstance(last, torch.nn.Conv2d):
            # 原生 nn.Conv2d
            cv3_seq[-1] = torch.nn.Conv2d(
                last.in_channels, nc,
                kernel_size=last.kernel_size,
                stride=last.stride,
                padding=last.padding,
                bias=last.bias is not None,
            )
        else:
            raise TypeError(f"cv3[{i}][-1] 类型未知: {type(last)}")
    print(f"[Info] Detect head cv3 输出通道已重建为 nc={nc}")


def load_model_rebuild(model_path: Path, base_model: str, nc: int):
    """
    针对 train.py 自定义格式：
      1. 用 base_model 构建骨架
      2. 物理替换检测头（完全复现 train.py 逻辑，shape 才能匹配）
      3. 加载 state_dict
    """
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if not (isinstance(ckpt, dict) and "model" in ckpt):
        raise RuntimeError(f"不支持的 checkpoint 格式: {type(ckpt)}")
    model_field = ckpt["model"]
    if hasattr(model_field, "float"):  # nn.Module -> Ultralytics 原生格式
        print("[Info] Ultralytics 原生格式，直接加载")
        return YOLO(str(model_path))
    state_dict = model_field
    print(f"[Info] 自定义 checkpoint，复现 train.py 检测头替换 (nc={nc})")
    yolo = YOLO(base_model)
    inner_model = yolo.model
    _rebuild_detect_head(inner_model, nc)          # 关键：物理替换卷积层
    missing, unexpected = inner_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Info] missing keys: {len(missing)} 个")
    if unexpected:
        print(f"[Info] unexpected keys: {len(unexpected)} 个")
    print("[Info] state_dict 加载完成")
    inner_model.eval()
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
    print(f"模型: {model_path}")
    print(f"共找到图片: {len(image_paths)} 张，本次测试: {n} 张")
    for p in selected:
        print(f"  {p}")
    # 使用重建方式加载（针对 train.py 自定义 checkpoint）
    model = load_model_rebuild(model_path, args.base_model, args.nc)
    results = model.predict(
        source=[str(p) for p in selected],
        imgsz=args.imgsz,
        conf=args.conf,
        verbose=False,
    )
    lines = []
    for i, (img_path, result) in enumerate(zip(selected, results), start=1):
        plotted = result.plot(labels=False)
        # 用序号+父目录名+文件名避免重名
        out_name = f"{i:02d}_{img_path.parent.parent.name}_{img_path.name}"
        out_file = save_dir / out_name
        cv2.imwrite(str(out_file), plotted)
        n_det = 0 if result.boxes is None else len(result.boxes)
        lines.append(f"{img_path}\tdetections={n_det}\tsaved={out_file.name}")
        print(f"[{i:02d}] {img_path.name}  detections={n_det}  -> {out_file.name}")
    (save_dir / "selected_images.txt").write_text("\n".join([str(p) for p in selected]), encoding="utf-8")
    (save_dir / "pred_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n测试完成")
    print(f"可视化输出目录: {save_dir}")
    print(f"结果汇总: {save_dir / 'pred_summary.txt'}")
if __name__ == "__main__":
    main()