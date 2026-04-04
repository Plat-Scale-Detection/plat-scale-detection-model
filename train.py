"""
train.py – Fine-tune YOLOv8 on the plat graphical-scale dataset.

Usage
-----
# Basic (uses defaults from config/model_config.yaml)
python train.py

# Override dataset path and epochs
python train.py --dataset datasets/latest --epochs 50

# Resume from last checkpoint
python train.py --resume
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).parent
CONFIG_PATH = REPO_ROOT / "config" / "model_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def patch_dataset_yaml(dataset_root: Path) -> Path:
    """
    Rewrite the path field in data.yaml so it points to the absolute location
    on disk.  YOLOv8 requires an absolute path or a path relative to the
    data.yaml file itself.
    """
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_root}")

    with open(data_yaml) as f:
        data = yaml.safe_load(f)

    data["path"] = str(dataset_root.resolve())

    # Write a patched copy so we don't dirty the original
    patched = dataset_root / "_data_patched.yaml"
    with open(patched, "w") as f:
        yaml.dump(data, f)
    return patched


def train(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    cfg = load_config()
    tc  = cfg["training"]

    dataset_root = REPO_ROOT / args.dataset
    data_yaml    = patch_dataset_yaml(dataset_root)

    model_base   = args.weights or cfg["model"]["base"]
    output_dir   = REPO_ROOT / cfg["output"]["weights_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] Base model  : {model_base}")
    print(f"[train] Dataset     : {dataset_root}")
    print(f"[train] Epochs      : {args.epochs or tc['epochs']}")
    print(f"[train] Image size  : {tc['imgsz']}")

    model = YOLO(model_base)

    results = model.train(
        data        = str(data_yaml),
        epochs      = args.epochs or tc["epochs"],
        imgsz       = tc["imgsz"],
        batch       = tc["batch"],
        patience    = tc["patience"],
        optimizer   = tc["optimizer"],
        lr0         = tc["lr0"],
        lrf         = tc["lrf"],
        momentum    = tc["momentum"],
        weight_decay= tc["weight_decay"],
        warmup_epochs=tc["warmup_epochs"],
        cos_lr      = tc["cos_lr"],
        augment     = tc["augment"],
        degrees     = tc["degrees"],
        translate   = tc["translate"],
        scale       = tc["scale"],
        shear       = tc["shear"],
        flipud      = tc["flipud"],
        fliplr      = tc["fliplr"],
        mosaic      = tc["mosaic"],
        mixup       = tc["mixup"],
        project     = str(REPO_ROOT / cfg["output"]["runs_dir"]),
        name        = "plat_scale_train",
        resume      = args.resume,
        device      = args.device or "",
        exist_ok    = True,
    )

    # Copy best weights to top-level weights/ dir for easy access
    best_src = Path(results.save_dir) / "weights" / "best.pt"
    last_src = Path(results.save_dir) / "weights" / "last.pt"
    if best_src.exists():
        shutil.copy2(best_src, output_dir / "best.pt")
        print(f"[train] Best weights saved to {output_dir / 'best.pt'}")
    if last_src.exists():
        shutil.copy2(last_src, output_dir / "last.pt")

    # Optionally export to ONNX
    export_fmt = cfg["output"].get("export_format", "onnx")
    if export_fmt and best_src.exists():
        export_model = YOLO(str(output_dir / "best.pt"))
        export_path  = export_model.export(
            format=export_fmt,
            imgsz=tc["imgsz"],
            dynamic=True,
        )
        print(f"[train] Model exported to {export_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the plat scale detection model")
    parser.add_argument("--dataset", default="datasets/latest",
                        help="Path to dataset folder (relative to repo root)")
    parser.add_argument("--epochs",  type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--weights", default=None,
                        help="Path to starting weights (.pt). Defaults to config value.")
    parser.add_argument("--device",  default="",
                        help="Device: '', 'cpu', 'cuda', 'cuda:0', etc.")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume training from the last checkpoint")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
