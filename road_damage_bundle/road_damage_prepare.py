import xml.etree.ElementTree as ET
import zipfile
from collections import OrderedDict
from pathlib import Path
from shutil import copy2


ZIP_FILES = ["habibi.zip", "alvaro.zip", "abi.zip"]


def extract_all_zips(base_dir: Path, raw_root: Path) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    for zip_name in ZIP_FILES:
        zip_path = base_dir / zip_name
        if not zip_path.exists():
            print(f"ZIP not found, skipping: {zip_path}")
            continue
        target_dir = raw_root / zip_path.stem
        if any(target_dir.iterdir()) if target_dir.exists() else False:
            print(f"Already extracted, skipping: {target_dir}")
            continue
        print(f"Extracting {zip_path} -> {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)


def parse_voc_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"No <size> in {xml_path}")
    img_w = float(size.findtext("width"))
    img_h = float(size.findtext("height"))

    filename = root.findtext("filename")
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bnd = obj.find("bndbox")
        if bnd is None or name is None:
            continue
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))

        # YOLO normalized (cx, cy, w, h)
        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        objects.append(
            {
                "class_name": name,
                "cx": cx,
                "cy": cy,
                "w": bw,
                "h": bh,
            }
        )

    return filename, objects


def discover_classes_and_samples(raw_root: Path):
    class_names = OrderedDict()
    samples = []

    for xml_path in raw_root.rglob("*.xml"):
        rel = xml_path.relative_to(raw_root).as_posix()
        if "kerusakan-jalan" not in rel:
            continue

        # Determine split from folder name if possible
        rel_lower = rel.lower()
        if "/train/" in rel_lower:
            split = "train"
        elif "/valid/" in rel_lower or "/val/" in rel_lower:
            split = "val"
        elif "/test/" in rel_lower:
            split = "test"
        else:
            split = "train"

        try:
            filename, objects = parse_voc_xml(xml_path)
        except Exception as e:
            print(f"Failed to parse {xml_path}: {e}")
            continue

        if not objects:
            continue

        for obj in objects:
            if obj["class_name"] not in class_names:
                class_names[obj["class_name"]] = len(class_names)

        samples.append(
            {
                "xml_path": xml_path,
                "filename": filename,
                "objects": objects,
                "split": split,
            }
        )

    return class_names, samples


def prepare_yolo_dataset(base_dir: Path):
    raw_root = base_dir / "data" / "raw"
    yolo_root = base_dir / "data" / "yolo"

    extract_all_zips(base_dir, raw_root)

    class_names, samples = discover_classes_and_samples(raw_root)
    if not samples:
        print("No samples found. Check extracted structure under data/raw.")
        return

    print(f"Discovered {len(class_names)} classes: {list(class_names.keys())}")
    print(f"Total annotated images: {len(samples)}")

    for split in ("train", "val", "test"):
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for sample in samples:
        xml_path = sample["xml_path"]
        split = sample["split"]
        filename = sample["filename"]

        # Image path: same folder as XML, with filename from annotation
        img_path = xml_path.with_name(filename)
        if not img_path.exists():
            # fallback: change suffix to .jpg
            img_path = xml_path.with_suffix(".jpg")
        if not img_path.exists():
            print(f"Image not found for {xml_path}, skipping")
            continue

        out_img_name = img_path.name
        out_img_path = yolo_root / "images" / split / out_img_name
        out_label_path = yolo_root / "labels" / split / (img_path.stem + ".txt")

        copy2(img_path, out_img_path)

        lines = []
        for obj in sample["objects"]:
            cid = class_names[obj["class_name"]]
            lines.append(
                f"{cid} {obj['cx']:.6f} {obj['cy']:.6f} {obj['w']:.6f} {obj['h']:.6f}"
            )
        out_label_path.write_text("\n".join(lines), encoding="utf-8")

    # Write dataset YAML for YOLO (use absolute path so Ultralytics
    # does not prepend its own datasets_dir)
    yaml_path = base_dir / "data" / "road_damage.yaml"
    yaml_lines = [
        f"path: {yolo_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for name, idx in class_names.items():
        yaml_lines.append(f"  {idx}: {name}")
    yaml_path.write_text("\n".join(yaml_lines), encoding="utf-8")

    print(f"YOLO dataset ready under: {yolo_root}")
    print(f"Dataset config written to: {yaml_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    prepare_yolo_dataset(base_dir)

