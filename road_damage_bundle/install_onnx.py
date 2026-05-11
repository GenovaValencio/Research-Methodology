"""
Pasang paket ONNX (dan onnxruntime) untuk ekspor model YOLO ke format ONNX.
Jalankan: python install_onnx.py
"""
import subprocess
import sys


def main():
    packages = ["onnx>=1.12.0", "onnxruntime"]
    cmd = [sys.executable, "-m", "pip", "install"] + packages
    print("Installing:", " ".join(packages))
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("ONNX installed successfully. You can run road_damage_train.py again to export to ONNX.")
    else:
        print("Installation failed. Try running in terminal: pip install onnx onnxruntime")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
