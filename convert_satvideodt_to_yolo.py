import xml.etree.ElementTree as ET
from pathlib import Path

DATASET_ROOT     = Path("/data/lz/SatVideoDT")
TRAIN_DATA       = DATASET_ROOT / "train_data"
VAL_DATA         = DATASET_ROOT / "val_data"
OUT_ROOT         = DATASET_ROOT / "yolo_dataset"
OUT_LABELS_TRAIN = OUT_ROOT / "labels" / "train"
OUT_LABELS_VAL   = OUT_ROOT / "labels" / "val"
OUT_IMAGES_TRAIN = OUT_ROOT / "images" / "train"
OUT_IMAGES_VAL   = OUT_ROOT / "images" / "val"
CLASS_MAP = {"car": 0}


def parse_xml(xml_path):
    tree  = ET.parse(xml_path)
    root  = tree.getroot()
    size  = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name   = obj.find("name").text.strip().lower()
        cls_id = CLASS_MAP.get(name, -1)
        if cls_id == -1:
            continue
        bb   = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        cx = min(max((xmin + xmax) / 2.0 / img_w, 0.0), 1.0)
        cy = min(max((ymin + ymax) / 2.0 / img_h, 0.0), 1.0)
        bw = min(max((xmax - xmin) / img_w, 1e-6), 1.0)
        bh = min(max((ymax - ymin) / img_h, 1e-6), 1.0)
        boxes.append((cls_id, cx, cy, bw, bh))
    return img_w, img_h, boxes


def convert_split(data_dir, out_labels_dir, out_images_dir, split_name):
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    out_images_dir.mkdir(parents=True, exist_ok=True)
    sequences = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    total_imgs = total_boxes = no_xml = 0
    for seq in sequences:
        img_dir = seq / "img1"
        xml_dir = seq / "xml"
        if not img_dir.exists() or not xml_dir.exists():
            continue
        images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        for img_path in images:
            frame_id   = img_path.stem
            xml_path   = xml_dir / f"{frame_id}.xml"
            uname      = f"{seq.name}_{frame_id}"
            label_file = out_labels_dir / f"{uname}.txt"
            img_link   = out_images_dir  / f"{uname}{img_path.suffix}"
            if xml_path.exists():
                _, _, boxes = parse_xml(xml_path)
                with open(label_file, "w") as f:
                    for cls_id, cx, cy, bw, bh in boxes:
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                total_boxes += len(boxes)
            else:
                label_file.write_text("")
                no_xml += 1
            if not img_link.exists():
                img_link.symlink_to(img_path.resolve())
            total_imgs += 1
    print(f"  [{split_name}] {total_imgs} imgs, {total_boxes} boxes, {no_xml} no-xml")


def write_yaml():
    yaml_path = Path("/data/lz/pythonProjects/HP-tracker/yolov8/satvideodt.yaml")
    yaml_path.write_text(
        "# SatVideoDT dataset - satellite video car detection\n"
        f"path: {OUT_ROOT}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "\nnc: 1\n"
        "\nnames:\n"
        "  0: car\n",
        encoding="utf-8",
    )
    print(f"  YAML written: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    print("SatVideoDT -> YOLO conversion")
    print("[1/3] train split...")
    convert_split(TRAIN_DATA, OUT_LABELS_TRAIN, OUT_IMAGES_TRAIN, "train")
    print("[2/3] val split...")
    convert_split(VAL_DATA, OUT_LABELS_VAL, OUT_IMAGES_VAL, "val")
    print("[3/3] writing yaml...")
    yaml_path = write_yaml()
    print("\nDone!")
    print(f"  Set CFG['data_yaml'] = '{yaml_path}' in train.py")
    print("  Set CFG['nc'] = 1 in train.py")
