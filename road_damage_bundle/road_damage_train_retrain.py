"""
Training ulang YOLO (fine-tuning) agar data baru dari pothole_video ikut dipakai.

Jalankan:
  python road_damage_train_retrain.py --epochs 30

Opsional:
  --weights last/best otomatis (pakai best.pt terbaru)
  --imgsz 640
  --batch 8
"""

from __future__ import annotations

import argparse
from pathlib import Path


def find_trained_weights(base_dir: Path) -> str | None:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8, help="Batch size untuk training.")
    parser.add_argument("--workers", type=int, default=2, help="Jumlah workers dataloader.")
    parser.add_argument("--conf", type=float, default=None)
    args = parser.parse_args()

    from ultralytics import YOLO

    base_dir = Path(__file__).parent
    data_yaml = base_dir / "data" / "road_damage.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"Dataset YAML tidak ditemukan: {data_yaml}")

    weights_path = find_trained_weights(base_dir)
    if not weights_path:
        print("Weights tidak ditemukan, start dari yolov8n.pt")
        weights_path = "yolov8n.pt"
    else:
        print(f"Fine-tuning mulai dari: {weights_path}")

    model = YOLO(weights_path)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        name="train_pothole_video",
    )

    # Validasi singkat
    model.val()


if __name__ == "__main__":
    main()

