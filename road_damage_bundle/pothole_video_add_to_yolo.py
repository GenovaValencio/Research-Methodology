"""
Tambahkan data video pothole_video ke dataset YOLO yang sudah ada.

Asumsi:
- Dataset YOLO sudah ada di: data/yolo/images/train dan data/yolo/labels/train
- Model deteksi sudah terlatih (runs/detect/train*/weights/best.pt) dengan kelas termasuk "lubang"

Cara kerja:
1) Ekstrak frame dari tiap video (sampling rate pakai interval detik)
2) Jalankan YOLO pada frame untuk menghasilkan pseudo-label
3) Simpan frame + label YOLO (cx,cy,w,h normalized) hanya jika ada deteksi kelas target
4) Lalu hasilnya menambah isi folder train (tidak mengubah val/test)

Jalankan:
  python pothole_video_add_to_yolo.py

Parameter opsional:
  --video_dir data/raw/pothole_video
  --interval_seconds 1.0
  --max_frames_per_video 200
  --conf 0.35
  --target_class lubang
  --imgsz 640
  --skip_if_exists
"""

from __future__ import annotations

import argparse
import cv2
import yaml
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


def load_class_id_from_yaml(yaml_path: Path, class_name: str) -> int:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    # names biasanya mapping idx -> name
    inv = {v: int(k) for k, v in names.items()}
    if class_name not in inv:
        raise ValueError(f"Class '{class_name}' tidak ada di {yaml_path}. Names={list(inv.keys())}")
    return inv[class_name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="data/raw/pothole_video")
    parser.add_argument("--max_videos", type=int, default=50, help="Batasi jumlah video untuk diproses (default: 50). Gunakan 0 untuk semua.")
    parser.add_argument("--interval_seconds", type=float, default=1.0)
    parser.add_argument("--max_frames_per_video", type=int, default=50)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--target_class", type=str, default="lubang")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--skip_if_exists", action="store_true")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    # args.video_dir boleh path relatif (terhadap base_dir) atau absolut
    video_dir = Path(args.video_dir)
    if not video_dir.is_absolute():
        video_dir = base_dir / args.video_dir
    if not video_dir.exists():
        raise SystemExit(
            f"Folder video tidak ditemukan: {video_dir}\n"
            f"Silakan pastikan folder itu berisi file video (mp4/mov/mkv/avi/webm)."
        )

    weights_path = find_trained_weights(base_dir)
    if not weights_path:
        raise SystemExit("Weights model tidak ditemukan. Jalankan road_damage_train.py dulu.")

    data_yaml = base_dir / "data" / "road_damage.yaml"
    target_id = load_class_id_from_yaml(data_yaml, args.target_class)

    print(f"Model: {weights_path}")
    print(f"Target class: {args.target_class} (id={target_id})")

    from ultralytics import YOLO

    model = YOLO(weights_path)

    out_img_dir = base_dir / "data" / "yolo" / "images" / "train"
    out_lbl_dir = base_dir / "data" / "yolo" / "labels" / "train"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    video_paths = []
    for p in video_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            video_paths.append(p)
    video_paths = sorted(video_paths, key=lambda p: p.stat().st_mtime)
    if args.max_videos and args.max_videos > 0:
        video_paths = video_paths[: args.max_videos]
    if not video_paths:
        raise SystemExit(f"Tidak ada video ditemukan di {video_dir} (mp4/mov/mkv/avi/webm).")

    total_added = 0
    total_saved = 0

    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"[SKIP] Tidak bisa buka video: {vp}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 25.0
        step_frames = max(1, int(fps * args.interval_seconds))

        max_frames = args.max_frames_per_video
        saved_this_video = 0
        frame_idx = 0

        print(f"\nProses video: {vp.name} | fps~{fps:.2f} | step_frames={step_frames}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step_frames != 0:
                frame_idx += 1
                continue

            if saved_this_video >= max_frames:
                break

            h, w = frame.shape[:2]

            # Inference YOLO
            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                frame_idx += 1
                continue

            # Filter target class
            cls_list = boxes.cls.tolist() if hasattr(boxes.cls, "tolist") else [int(x) for x in boxes.cls]
            kept_rows = []

            for i in range(len(boxes)):
                cls_i = int(cls_list[i])
                if cls_i != target_id:
                    continue
                conf_i = float(boxes.conf[i].item()) if hasattr(boxes.conf[i], "item") else float(boxes.conf[i])
                if conf_i < args.conf:
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                # Convert to normalized cx,cy,w,h
                x1 = max(0.0, min(float(x1), float(w)))
                x2 = max(0.0, min(float(x2), float(w)))
                y1 = max(0.0, min(float(y1), float(h)))
                y2 = max(0.0, min(float(y2), float(h)))

                cx = ((x1 + x2) / 2.0) / float(w)
                cy = ((y1 + y2) / 2.0) / float(h)
                bw = (x2 - x1) / float(w)
                bh = (y2 - y1) / float(h)

                # Guard: YOLO format expects all in [0..1]
                cx = max(0.0, min(cx, 1.0))
                cy = max(0.0, min(cy, 1.0))
                bw = max(0.0, min(bw, 1.0))
                bh = max(0.0, min(bh, 1.0))

                kept_rows.append((target_id, cx, cy, bw, bh))

            if not kept_rows:
                frame_idx += 1
                continue

            # Save frame + labels
            stem = f"{vp.stem}_f{frame_idx:08d}"
            img_path = out_img_dir / f"{stem}.jpg"
            lbl_path = out_lbl_dir / f"{stem}.txt"

            if args.skip_if_exists and img_path.exists() and lbl_path.exists():
                frame_idx += 1
                continue

            ok = cv2.imwrite(str(img_path), frame)
            if not ok:
                print(f"[SKIP] Gagal simpan frame: {img_path}")
                frame_idx += 1
                continue

            lines = [f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cid, cx, cy, bw, bh in kept_rows]
            lbl_path.write_text("\n".join(lines), encoding="utf-8")

            saved_this_video += 1
            total_saved += 1
            total_added += len(kept_rows)
            frame_idx += 1

        cap.release()
        print(f"Disimpan frame: {saved_this_video} | total box baru ditambahkan: {total_added}")

    print("\nSelesai.")
    print(f"Total frame baru disimpan: {total_saved}")
    print(f"Total box/objek pseudo-label ditambahkan: {total_added}")


if __name__ == "__main__":
    main()

