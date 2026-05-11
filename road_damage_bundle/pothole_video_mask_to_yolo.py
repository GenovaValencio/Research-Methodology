"""
Konversi dataset pothole_video (mask video) -> YOLO detection labels.

Struktur yang diharapkan:
  data/raw/pothole_video/<split>/rgb/*.mp4
  data/raw/pothole_video/<split>/mask/*.mp4

Langkah:
1) Pasangkan video rgb & mask berdasarkan nama file (stem).
2) Sampling frame dengan interval_seconds.
3) Untuk tiap frame:
   - Ambil mask frame, ubah jadi grayscale -> binary via threshold.
   - Ambil contour -> bounding boxes -> label YOLO (cx,cy,w,h) untuk class 'lubang'.
   - Simpan frame rgb sebagai image JPEG + label TXT.

Output:
  data/yolo/images/<split>/... (umumnya train)
  data/yolo/labels/<split>/... (umumnya train)

Catatan:
- Ini pseudo-label dari mask (lebih ground-truth dibanding YOLO pseudo-label).
- Karena dataset besar, pakai opsi sampling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def find_class_id(yaml_path: Path, class_name: str) -> int:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    inv = {v: int(k) for k, v in names.items()}
    if class_name not in inv:
        raise ValueError(f"Class '{class_name}' not found in {yaml_path}. Available: {list(inv.keys())}")
    return inv[class_name]


def compute_bboxes_from_mask(mask_bgr_or_gray: np.ndarray, mask_threshold: int, min_area_ratio: float):
    """
    Return list of bboxes (x1, y1, x2, y2) in pixel coordinates.
    mask_threshold: threshold on grayscale (0..255)
    min_area_ratio: area_min = min_area_ratio * (W*H)
    """
    if mask_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(mask_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask_bgr_or_gray

    # Threshold (as pothole region in mask should be bright)
    _, bin_img = cv2.threshold(gray, mask_threshold, 255, cv2.THRESH_BINARY)

    h, w = bin_img.shape[:2]
    min_area = float(min_area_ratio) * float(w * h)

    # Find connected components via contours
    contours, _hier = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        x1, y1, x2, y2 = x, y, x + bw, y + bh
        bboxes.append((x1, y1, x2, y2))
    return bboxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str, default="data/raw/pothole_video")
    parser.add_argument("--splits", type=str, default="train", help="Comma list: train,val,test")
    parser.add_argument("--output_split", type=str, default="train", help="YOLO split folder output (images/labels).")
    parser.add_argument("--interval_seconds", type=float, default=1.0)
    parser.add_argument("--max_videos", type=int, default=30, help="Sampling: max number of video pairs.")
    parser.add_argument("--max_frames_per_video", type=int, default=50, help="Sampling: max frames saved per video pair.")
    parser.add_argument("--mask_threshold", type=int, default=30, help="Threshold grayscale for pothole mask.")
    parser.add_argument("--min_area_ratio", type=float, default=0.0005, help="Min contour area ratio to frame area.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO imgsz (only affects training; we don't resize during saving).")
    parser.add_argument("--target_class", type=str, default="lubang")
    parser.add_argument("--skip_if_exists", action="store_true")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    video_root = base_dir / args.video_root
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    yolo_img_dir = base_dir / "data" / "yolo" / "images" / args.output_split
    yolo_lbl_dir = base_dir / "data" / "yolo" / "labels" / args.output_split
    yolo_img_dir.mkdir(parents=True, exist_ok=True)
    yolo_lbl_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = base_dir / "data" / "road_damage.yaml"
    class_id = find_class_id(yaml_path, args.target_class)
    print(f"Class '{args.target_class}' id={class_id}")

    rgb_ext = ".mp4"
    allowed = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

    # Collect video stems from split/rgb and pair with mask
    pairs = []
    for sp in splits:
        rgb_dir = video_root / sp / "rgb"
        mask_dir = video_root / sp / "mask"
        if not rgb_dir.exists() or not mask_dir.exists():
            print(f"[SKIP] Missing dirs for split '{sp}' -> {rgb_dir} / {mask_dir}")
            continue
        for p in rgb_dir.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in allowed:
                continue
            stem = p.stem
            # Try to find matching mask file by same stem across allowed suffixes
            mask_match = None
            for mp in mask_dir.rglob("*"):
                if mp.is_file() and mp.stem == stem and mp.suffix.lower() in allowed:
                    mask_match = mp
                    break
            if mask_match is None:
                continue
            pairs.append((p, mask_match, stem))

    # Sort for deterministic
    pairs = sorted(pairs, key=lambda t: t[0].name)
    if args.max_videos and args.max_videos > 0:
        pairs = pairs[: args.max_videos]

    if not pairs:
        raise SystemExit(f"No video pairs found under {video_root} with splits={splits}.")

    total_images = 0
    total_labels = 0

    for rgb_path, mask_path, stem in pairs:
        print(f"\nVideo pair: rgb={rgb_path.name} mask={mask_path.name}")
        cap_rgb = cv2.VideoCapture(str(rgb_path))
        cap_mask = cv2.VideoCapture(str(mask_path))
        if not cap_rgb.isOpened() or not cap_mask.isOpened():
            print(f"[SKIP] Cannot open pair: {rgb_path} / {mask_path}")
            continue

        fps = cap_rgb.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 25.0
        step_frames = max(1, int(round(fps * args.interval_seconds)))

        frames_saved = 0
        frame_idx = 0

        while True:
            ret1, frame_rgb = cap_rgb.read()
            ret2, frame_mask = cap_mask.read()
            if not ret1 or not ret2:
                break

            if frame_idx % step_frames != 0:
                frame_idx += 1
                continue

            if frames_saved >= args.max_frames_per_video:
                break

            h, w = frame_rgb.shape[:2]

            bboxes = compute_bboxes_from_mask(
                frame_mask,
                mask_threshold=args.mask_threshold,
                min_area_ratio=args.min_area_ratio,
            )

            if not bboxes:
                frame_idx += 1
                continue

            # Save frame + labels
            img_name = f"{stem}_f{frame_idx:08d}.jpg"
            lbl_name = f"{stem}_f{frame_idx:08d}.txt"
            img_path = yolo_img_dir / img_name
            lbl_path = yolo_lbl_dir / lbl_name

            if args.skip_if_exists and img_path.exists() and lbl_path.exists():
                frame_idx += 1
                continue

            ok = cv2.imwrite(str(img_path), frame_rgb)
            if not ok:
                print(f"[SKIP] Failed save image: {img_path}")
                frame_idx += 1
                continue

            lines = []
            for (x1, y1, x2, y2) in bboxes:
                # Normalize cx,cy,w,h
                x1 = max(0, min(int(x1), w - 1))
                x2 = max(0, min(int(x2), w))
                y1 = max(0, min(int(y1), h - 1))
                y2 = max(0, min(int(y2), h))
                if x2 <= x1 or y2 <= y1:
                    continue

                cx = ((x1 + x2) / 2.0) / float(w)
                cy = ((y1 + y2) / 2.0) / float(h)
                bw = (x2 - x1) / float(w)
                bh = (y2 - y1) / float(h)

                cx = float(np.clip(cx, 0.0, 1.0))
                cy = float(np.clip(cy, 0.0, 1.0))
                bw = float(np.clip(bw, 0.0, 1.0))
                bh = float(np.clip(bh, 0.0, 1.0))

                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if not lines:
                # Remove image if no labels (safety)
                try:
                    img_path.unlink(missing_ok=True)  # py3.8+
                except Exception:
                    pass
                frame_idx += 1
                continue

            lbl_path.write_text("\n".join(lines), encoding="utf-8")

            frames_saved += 1
            total_images += 1
            total_labels += len(lines)
            frame_idx += 1

        cap_rgb.release()
        cap_mask.release()
        print(f"Saved frames: {frames_saved}")

    print("\nSelesai konversi mask->YOLO")
    print(f"Total images saved: {total_images}")
    print(f"Total label boxes saved: {total_labels}")


if __name__ == "__main__":
    main()

