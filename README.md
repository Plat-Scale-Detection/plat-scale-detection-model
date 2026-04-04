# plat-scale-detection-model

YOLOv8-based object detection model that finds **graphical scale bars** on
engineering plats.  Training data is produced by the companion
[plat-scale-detection-trainer](https://github.com/Plat-Scale-Detection/plat-scale-detection-trainer)
annotation tool.

---

## Repository structure

```
plat-scale-detection-model/
├── config/
│   └── model_config.yaml     # all hyper-parameters, paths, inference defaults
├── datasets/
│   └── latest/               # YOLO dataset pushed by the trainer (gitignored images)
├── model/
│   └── detector.py           # ScaleDetector wrapper (single + tiled inference)
├── scripts/
│   ├── import_dataset.py     # import a ZIP or directory exported from the trainer
│   └── evaluate.py           # run validation metrics on a trained model
├── weights/                  # trained .pt / .onnx files (gitignored, stored as Releases)
├── train.py                  # fine-tune YOLOv8 on the dataset
├── inference.py              # detect scales on one image or a folder
└── .github/workflows/
    └── train.yml             # auto-trains when a new dataset is pushed
```

---

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> GPU recommended.  CPU training is supported but slow for large datasets.

### 2. Import training data

**Option A – from a ZIP downloaded from the trainer UI:**
```bash
python scripts/import_dataset.py --zip ~/Downloads/plat_scale_dataset.zip
```

**Option B – from a directory (or if the trainer pushed directly to this repo):**
```bash
python scripts/import_dataset.py --dir datasets/latest
```

### 3. Train

```bash
python train.py                          # uses config/model_config.yaml defaults
python train.py --epochs 50              # override epochs
python train.py --device cuda:0          # explicit GPU
python train.py --resume                 # resume from last checkpoint
```

Best weights are saved to `weights/best.pt` and exported to `weights/best.onnx`.

### 4. Evaluate

```bash
python scripts/evaluate.py
```

Prints mAP50, mAP50-95, precision, and recall on the validation split.

### 5. Run inference

```bash
# Single image → annotated PNG in runs/inference/
python inference.py --image plat.png

# Directory of images → annotated PNGs
python inference.py --image plats/ --output results/

# Tiled inference (recommended for large high-res drawings)
python inference.py --image large_plat.tiff --tile

# JSON output instead of annotated images
python inference.py --image plat.png --json
```

---

## Automated training (GitHub Actions)

Pushing a new dataset to `datasets/` on the `main` branch (or using the
**Push to GitHub** button in the trainer) automatically triggers the
[`train.yml`](.github/workflows/train.yml) workflow, which:

1. Validates the dataset.
2. Fine-tunes YOLOv8 (CPU runner – swap for a self-hosted GPU runner for
   production use).
3. Uploads `weights/` as a GitHub Actions artifact.
4. Creates a GitHub Release with `best.pt` and `best.onnx` attached.

---

## Model details

| Property | Value |
|----------|-------|
| Architecture | YOLOv8s (fine-tuned from COCO) |
| Classes | 1 (`graphical_scale`) |
| Input size | 1280 × 1280 |
| Framework | Ultralytics · PyTorch |
| Export formats | ONNX, TorchScript |

### Why YOLOv8?

Graphical scale bars are small, elongated objects.  YOLOv8's anchor-free
detector with a large receptive field handles them well, and the Ultralytics
library provides a single-command training + export pipeline.  The `s` variant
strikes a good balance between accuracy and speed on CPU for deployment in
scanning workflows.
