from pathlib import Path


def find_trained_weights(base_dir: Path):
    """Cari weights hasil training sebelumnya (ultralytics simpan di runs/detect/train*/weights/)."""
    runs = base_dir / "runs" / "detect"
    if not runs.exists():
        return None
    for train_dir in sorted(runs.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not train_dir.is_dir() or not train_dir.name.startswith("train"):
            continue
        weights = train_dir / "weights"
        for name in ("best.pt", "last.pt"):
            p = weights / name
            if p.exists():
                return str(p)
    return None


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics not installed. Please run: pip install -r requirements.txt")
        return

    base_dir = Path(__file__).parent
    data_yaml = base_dir / "data" / "road_damage.yaml"

    if not data_yaml.exists():
        print(f"Dataset config not found at {data_yaml}.")
        print("Run road_damage_prepare.py first to build the YOLO dataset.")
        return

    # Pakai model yang sudah ditraining kalau ada, kalau tidak train dari awal
    weights_path = find_trained_weights(base_dir)
    if weights_path:
        print(f"Menggunakan model yang sudah ada: {weights_path}")
        print("Training dilewati (hanya val + export).")
        model = YOLO(weights_path)
    else:
        model = YOLO("yolov8n.pt")
        model.train(
            data=str(data_yaml),
            epochs=50,
            imgsz=640,
            batch=8,
            workers=2,
        )

    model.val()

    export_dir = base_dir / "models"
    export_dir.mkdir(exist_ok=True)

    try:
        import onnx  # noqa: F401
        model.export(format="onnx", imgsz=640, opset=12)
        print("Model exported to ONNX.")
    except ImportError:
        print("ONNX export skipped (pip install onnx to enable). Trained .pt model is saved in runs/detect/train/weights/.")
    except Exception as e:
        print(f"ONNX export skipped: {e}. Trained .pt model is saved in runs/detect/train/weights/.")


if __name__ == "__main__":
    main()

