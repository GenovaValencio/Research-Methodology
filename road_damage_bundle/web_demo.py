"""
Web demo untuk iPhone (Safari):
- Ambil foto dari kamera atau pilih dari gallery
- Kirim ke laptop/PC untuk inferensi YOLO
- Tampilkan hasil bounding box + label

Jalankan:
  pip install -r requirements.txt
  python web_demo.py

Lalu buka di iPhone (satu WiFi):
  http://IP_LAPTOP:8000
"""

import io
import os
import socket
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response


def get_local_ip() -> str:
    # Best-effort: detect LAN IP
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def find_trained_weights(base_dir: Path):
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


app = FastAPI()
BASE_DIR = Path(__file__).parent

# Basic access control (recommended if exposed publicly)
# Set env var RD_TOKEN to require token: https://.../?token=YOURTOKEN
RD_TOKEN = os.environ.get("RD_TOKEN", "").strip()
MAX_UPLOAD_BYTES = int(os.environ.get("RD_MAX_UPLOAD_BYTES", "15000000"))  # ~15MB default
PORT = int(os.environ.get("RD_PORT", "8000"))

# Lazy-load model
_model = None


def get_model():
    global _model
    if _model is not None:
        return _model
    from ultralytics import YOLO

    weights = find_trained_weights(BASE_DIR)
    if not weights:
        raise RuntimeError("Model weights not found. Train first to create runs/detect/train/weights/best.pt")
    _model = YOLO(weights)
    return _model


INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Road Damage Detector</title>
    <style>
      body { font-family: -apple-system, system-ui, Segoe UI, Roboto, Arial; margin: 16px; }
      .card { max-width: 720px; margin: 0 auto; }
      button { padding: 12px 16px; font-size: 16px; }
      input { font-size: 16px; }
      img { width: 100%; border: 1px solid #ddd; border-radius: 12px; margin-top: 12px; }
      .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
      .muted { color: #666; font-size: 14px; }
      .status { margin-top: 10px; font-size: 14px; white-space: pre-wrap; }
    </style>
  </head>
  <body>
    <div class="card">
      <h2>Road Damage Detector</h2>
      <p class="muted">Pilih sumber gambar: <b>Gallery</b> atau <b>Kamera</b>.</p>
      <p class="muted" id="tokenHint" style="display:none;">Token aktif. Pastikan URL Anda mengandung <code>?token=...</code></p>
      <div class="row">
        <button id="btnGallery">Gallery</button>
        <button id="btnCamera">Kamera</button>
        <button id="btnDetect" disabled>Detect</button>
      </div>
      <div class="status" id="status"></div>
      <img id="out" alt="output" style="display:none;" />
    </div>
    <script>
      // Hidden inputs: one for gallery (no capture), one for camera (capture)
      const inputGallery = document.createElement('input');
      inputGallery.type = 'file';
      inputGallery.accept = 'image/*';
      inputGallery.style.display = 'none';
      document.body.appendChild(inputGallery);

      const inputCamera = document.createElement('input');
      inputCamera.type = 'file';
      inputCamera.accept = 'image/*';
      inputCamera.capture = 'environment';
      inputCamera.style.display = 'none';
      document.body.appendChild(inputCamera);

      const btnGallery = document.getElementById('btnGallery');
      const btnCamera = document.getElementById('btnCamera');
      const btnDetect = document.getElementById('btnDetect');
      const statusEl = document.getElementById('status');
      const outEl = document.getElementById('out');
      const tokenHint = document.getElementById('tokenHint');
      const token = new URLSearchParams(window.location.search).get('token') || '';
      if (token) tokenHint.style.display = "block";

      let selectedFile = null;
      function onPicked(file) {
        selectedFile = file || null;
        if (selectedFile) {
          statusEl.textContent = `Selected: ${selectedFile.name} (${Math.round(selectedFile.size/1024)} KB). Tap Detect.`;
          btnDetect.disabled = false;
        } else {
          statusEl.textContent = "Tidak ada file terpilih.";
          btnDetect.disabled = true;
        }
      }

      inputGallery.addEventListener('change', () => onPicked(inputGallery.files && inputGallery.files[0]));
      inputCamera.addEventListener('change', () => onPicked(inputCamera.files && inputCamera.files[0]));

      btnGallery.onclick = () => inputGallery.click();
      btnCamera.onclick = () => inputCamera.click();

      btnDetect.onclick = async () => {
        if (!selectedFile) {
          statusEl.textContent = "Pilih gambar dulu.";
          return;
        }
        const f = selectedFile;
        statusEl.textContent = "Uploading & running detection...";
        outEl.style.display = "none";

        const form = new FormData();
        form.append("image", f, f.name);
        const t0 = performance.now();
        const url = token ? ("/predict?token=" + encodeURIComponent(token)) : "/predict";
        const resp = await fetch(url, { method: "POST", body: form });
        const t1 = performance.now();
        if (!resp.ok) {
          const txt = await resp.text();
          statusEl.textContent = "Error: " + txt;
          return;
        }
        const blob = await resp.blob();
        outEl.src = URL.createObjectURL(blob);
        outEl.style.display = "block";
        statusEl.textContent = `Done in ${(t1 - t0).toFixed(0)} ms`;
      };
    </script>
  </body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.post("/predict")
async def predict(request: Request, image: UploadFile = File(...), token: str = ""):
    if RD_TOKEN and token != RD_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized (token required)")

    from PIL import Image
    import numpy as np

    content = await image.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_BYTES} bytes)")

    img = Image.open(io.BytesIO(content)).convert("RGB")
    frame = np.array(img)  # HWC RGB

    model = get_model()
    results = model.predict(frame, imgsz=640, conf=0.35, verbose=False)
    plotted = results[0].plot()  # BGR uint8

    # Convert BGR -> RGB for PIL
    plotted_rgb = plotted[:, :, ::-1]
    out_img = Image.fromarray(plotted_rgb)
    buf = io.BytesIO()
    out_img.save(buf, format="JPEG", quality=90)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn

    ip = get_local_ip()
    if RD_TOKEN:
        print(f"Local:  http://{ip}:{PORT}/?token={RD_TOKEN}")
        print("Public: share ONLY the full URL (includes token).")
    else:
        print(f"Local:  http://{ip}:{PORT}")
        print("Tip: set RD_TOKEN env var to protect when public.")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

