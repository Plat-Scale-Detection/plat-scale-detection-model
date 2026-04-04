"""
evaluate.py – Evaluate trained model on the validation split and print metrics.

Usage
-----
python scripts/evaluate.py
python scripts/evaluate.py --weights weights/best.pt --dataset datasets/latest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


REPO_ROOT    = Path(__file__).parent.parent
CONFIG_PATH  = REPO_ROOT / "config" / "model_config.yaml"
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "best.pt"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the scale detection model")
    parser.add_argument("--weights",  default=None)
    parser.add_argument("--dataset",  default="datasets/latest")
    parser.add_argument("--device",   default="")
    args = parser.parse_args()

    from ultralytics import YOLO

    cfg     = load_config()
    weights = Path(args.weights) if args.weights else DEFAULT_WEIGHTS
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    dataset_root = REPO_ROOT / args.dataset
    data_yaml    = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_root}")

    # Patch absolute path
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    data["path"] = str(dataset_root.resolve())
    patched = dataset_root / "_data_patched.yaml"
    with open(patched, "w") as f:
        yaml.dump(data, f)

    model   = YOLO(str(weights))
    metrics = model.val(
        data   = str(patched),
        imgsz  = cfg["inference"]["imgsz"],
        conf   = cfg["inference"]["conf_threshold"],
        iou    = cfg["inference"]["iou_threshold"],
        device = args.device or "",
    )

    print("\n── Evaluation Results ──────────────────────────────")
    print(f"  mAP50     : {metrics.box.map50:.4f}")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")
    print("────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
