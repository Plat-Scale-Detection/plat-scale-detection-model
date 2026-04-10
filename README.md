# plat-scale-detection-model

Object detection model that finds **graphical scale bars** on engineering plats.
Weights are trained by the companion
[plat-scale-detection-trainer](https://github.com/Plat-Scale-Detection/plat-scale-detection-trainer)
and published here as GitHub Release assets.

**License:** Apache 2.0 — see [LICENSE](LICENSE).  
**Author:** Justin Kumpe and contributors.

> Training is handled entirely by the trainer repo. This repo contains only
> what is needed to **run inference** using the published weights.

---

## Getting started

### 1. Download weights

Download `best.onnx` from the [latest GitHub Release](../../releases/latest)
and place it at `weights/best.onnx`.

```bash
mkdir -p weights
# then download best.onnx from the Releases page into weights/
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `onnxruntime`, `opencv-python-headless`, `numpy`, `pyyaml` — all MIT/BSD licensed.  
No PyTorch or Ultralytics required.

### 3. Run inference

```bash
# Single image → annotated PNG in runs/inference/
python inference.py --image plat.png

# Directory of images
python inference.py --image plats/ --output results/

# Tiled inference (recommended for large high-res drawings)
python inference.py --image large_plat.tiff --tile

# JSON output
python inference.py --image plat.png --json

# Custom weights or confidence threshold
python inference.py --image plat.png --weights weights/best.onnx --conf 0.4
```

### 4. Python API

```python
from model.detector import ScaleDetector

detector = ScaleDetector("weights/best.onnx", conf_threshold=0.25)

# Single image
detections = detector.predict("plat.png")

# Tiled inference for large plats
detections = detector.predict_tiled("large_plat.tiff")

# Each detection dict:
# {
#   "class_id": 0, "class_name": "graphical_scale",
#   "confidence": 0.92,
#   "x": 412, "y": 308, "width": 154, "height": 22,       # pixels
#   "x_center_norm": 0.31, "y_center_norm": 0.18,          # normalised [0,1]
#   "width_norm": 0.12, "height_norm": 0.02,
# }
```

---

## Repository structure

```
plat-scale-detection-model/
├── config/
│   └── model_config.yaml   # inference defaults (conf, iou, imgsz, tiling)
├── model/
│   └── detector.py         # ScaleDetector — ONNX-based inference wrapper
├── weights/                # place best.onnx here (gitignored, from Releases)
├── inference.py            # CLI: detect scales on one image or a folder
├── requirements.txt        # onnxruntime, opencv, numpy, pyyaml
└── LICENSE                 # Apache 2.0
```

---

## Model details

| Property | Value |
|----------|-------|
| Architecture | YOLOv8s (fine-tuned from COCO) |
| Classes | 1 (`graphical_scale`) |
| Input size | 1280 × 1280 |
| Inference runtime | ONNX Runtime (MIT) |
| Weights format | ONNX (`.onnx`) |

---

## Attribution

If you use this model in your work, please credit:

> **Justin Kumpe** — [Plat-Scale-Detection](https://github.com/Plat-Scale-Detection)


---
