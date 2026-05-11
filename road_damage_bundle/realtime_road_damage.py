"""
Deteksi kerusakan jalan secara real-time dari kamera.
Gunakan model YOLO yang sudah ditraining.

Kamera laptop:     python realtime_road_damage.py
Kamera HP (stream): python realtime_road_damage.py --url http://192.168.1.xxx:8080/video
Tekan 'q' untuk keluar, 's' untuk simpan screenshot.
"""
import argparse
import time
from pathlib import Path

# Nama kelas dari data/road_damage.yaml (urutan harus sama dengan model)
CLASS_NAMES = [
    "retak_memanjang",
    "retak_blok",
    "retak_kulit_buaya",
    "pengelupasan_lapisan_permukaan",
    "lubang",
    "retak_pinggir",
]


def find_trained_weights(base_dir: Path):
    """Cari weights hasil training (runs/detect/train*/weights/best.pt atau last.pt)."""
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
    parser = argparse.ArgumentParser(description="Deteksi kerusakan jalan real-time dari kamera")
    parser.add_argument("--camera", type=int, default=0, help="Index kamera PC (default: 0)")
    parser.add_argument("--url", type=str, default=None, help="URL stream kamera HP (IP Webcam/DroidCam)")
    parser.add_argument("--model", type=str, default=None, help="Path ke model .pt (default: cari otomatis)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    args = parser.parse_args()

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        print("Perlu: pip install ultralytics opencv-python")
        raise SystemExit(1) from e

    base_dir = Path(__file__).parent
    model_path = args.model or find_trained_weights(base_dir)
    if not model_path or not Path(model_path).exists():
        print("Model tidak ditemukan. Jalankan road_damage_train.py dulu untuk training.")
        return 1

    print(f"Memuat model: {model_path}")
    model = YOLO(model_path)

    # Sumber video: URL kamera HP atau index kamera PC
    if args.url:
        source = args.url.strip()
        print(f"Menyambung ke kamera HP: {source}")
    else:
        source = args.camera

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if args.url:
            print("Stream HP tidak bisa dibuka. Cek:")
            print("  1. HP dan PC satu WiFi yang sama")
            print("  2. Di HP: IP Webcam sudah jalan, lalu salin URL (mis. http://192.168.1.x:8080/video)")
            print("  3. Jalankan: python realtime_road_damage.py --url http://IP_HP:8080/video")
        else:
            print(f"Kamera {args.camera} tidak bisa dibuka. Coba --camera 1 atau pakai --url untuk kamera HP.")
        return 1

    print("Kamera aktif. Tekan 'q' untuk keluar, 's' untuk simpan foto.")
    print("Arahkan kamera ke jalan untuk deteksi kerusakan.")

    snapshot_dir = base_dir / "realtime_snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t0 = time.perf_counter()

        # Inferensi YOLO
        results = model.predict(
            frame,
            conf=args.conf,
            imgsz=640,
            verbose=False,
        )

        # Gambar hasil di frame (Ultralytics bisa pakai results[0].plot())
        annotated = results[0].plot()

        # FPS
        elapsed = time.perf_counter() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Jumlah deteksi
        n = len(results[0].boxes)
        cv2.putText(
            annotated,
            f"Deteksi: {n}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Deteksi Kerusakan Jalan - Real-time", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            path = snapshot_dir / f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(str(path), annotated)
            print(f"Screenshot disimpan: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Selesai.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
