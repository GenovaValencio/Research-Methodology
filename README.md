# Deteksi Kerusakan Jalan (YOLOv8)

Repository ini berisi **file yang dimaksudkan untuk GitHub**: kode Python, konfigurasi dataset (`data/road_damage.yaml`), dependensi, dan pengabaian Git (`.gitignore`). **Bukan** penyimpanan dataset gambar/label di GitHub; tautan unduhan diisi di **`LOKASI_DATASET.txt`**.

---

## Lokasi dataset

Gambar dan label YOLO **tidak** ikut di repository ini. Tautan unduhan dan langkah setelah unduh ditulis di file **`LOKASI_DATASET.txt`** (di root folder repo ini). Buka file itu, isi baris URL, lalu commit ke GitHub.

---

## Isi repository (file di GitHub)

| File / folder | Isi |
|---------------|-----|
| `README.md` | Dokumentasi proyek |
| `requirements.txt` | Dependensi Python (Ultralytics, OpenCV, FastAPI, dll.) |
| `.gitignore` | Pola file yang **tidak** di-push (dataset, `runs/`, model `.pt`, venv, dll.) |
| `data/road_damage.yaml` | Konfigurasi YOLO: nama kelas, path relatif ke akar dataset |
| `data/yolo/LETAKKAN_DATASET_YOLO_DI_SINI.txt` | Pengingat struktur folder; folder gambar/label **tidak** ikut di repo |
| `*.py` | Skrip utama proyek (training, demo web, kamera, utilitas video, pothole) |
| `.vscode/` | Konfigurasi editor (opsional, jika ada) |
| `MidExam_ResearchMethodology/` | Dokumen tugas / metodologi (jika disertakan) |
| `CARA_PAKAI.txt` | Catatan singkat penggunaan lokal |
| `LOKASI_DATASET.txt` | Tautan unduhan dataset + ringkasan cara meletakkan ke `data/yolo/` |

**Skrip Python di root repo**

- `road_damage_prepare.py` — alur dari ZIP VOC ke struktur data (sesuaikan nama ZIP di skrip jika perlu).
- `road_damage_train.py` — training / validasi / ekspor ONNX.
- `road_damage_train_retrain.py` — fine-tuning dengan argumen CLI (`--epochs`, `--batch`, dll.).
- `realtime_road_damage.py` — inferensi kamera / stream.
- `web_demo.py` — API FastAPI untuk unggah gambar dari browser.
- `video.py` — utilitas video.
- `pothole_video_mask_to_yolo.py` — mask video → label YOLO.
- `pothole_video_add_to_yolo.py` — tambah frame dari video (pseudo-label).
- `install_onnx.py` — bantuan instal ONNX.

**Kelas di `road_damage.yaml`**

| ID | Nama |
|----|------|
| 0 | retak_memanjang |
| 1 | retak_blok |
| 2 | retak_kulit_buaya |
| 3 | pengelupasan_lapisan_permukaan |
| 4 | lubang |
| 5 | retak_pinggir |

---

## Yang sengaja tidak ada di GitHub

Sesuai `.gitignore`, antara lain: isi `data/yolo/images/` dan `data/yolo/labels/`, `data/raw/`, folder `runs/`, berat model (`*.pt`, `*.onnx` besar), lingkungan virtual, ZIP dataset, snapshot. Itu muncul **hanya di mesin Anda** setelah training atau menyalin data.

---

## Menjalankan setelah clone

1. Buat virtual environment dan pasang dependensi:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Baca **`LOKASI_DATASET.txt`**, unduh dari tautan di sana, lalu letakkan ke `data/yolo/`. Tanpa `images/` dan `labels/` yang berisi data, skrip training tidak bisa jalan.

3. Contoh training ulang:

```bash
python road_damage_train_retrain.py --epochs 30 --batch 8
```

Output training (log, `weights/`, dll.) ada di `runs/` — lokal, tidak perlu di-commit.

Opsional ONNX saat training: `pip install onnx`.

---

## Lisensi / atribusi

Lengkapi sesuai kebijakan kampus atau sumber data Anda.
