"""
inference.py – Run the trained scale detector on one or more images.

Usage
-----
# Single image
python inference.py --image path/to/plat.png

# Directory of images
python inference.py --image path/to/plats/ --output results/

# Use a specific weights file
python inference.py --image plat.png --weights weights/best.pt

# Output JSON instead of annotated images
python inference.py --image plat.png --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


REPO_ROOT   = Path(__file__).parent
CONFIG_PATH = REPO_ROOT / "config" / "model_config.yaml"
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "best.pt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def draw_detections(image_path: Path, detections: list[dict], output_path: Path) -> None:
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return

    for det in detections:
        x, y, w, h = det["x"], det["y"], det["width"], det["height"]
        conf = det["confidence"]
        label = f"scale {conf:.2f}"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), (0, 165, 255), -1)
        cv2.putText(img, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def run_inference(args: argparse.Namespace) -> None:
    from model.detector import ScaleDetector

    cfg   = load_config()
    ic    = cfg["inference"]
    wcfg  = cfg["output"]

    weights = Path(args.weights) if args.weights else DEFAULT_WEIGHTS
    if not weights.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weights}. "
            "Run train.py first or provide --weights."
        )

    detector = ScaleDetector(
        weights_path   = weights,
        conf_threshold = args.conf or ic["conf_threshold"],
        iou_threshold  = ic["iou_threshold"],
        imgsz          = ic["imgsz"],
        device         = args.device or "",
    )

    images  = collect_images(Path(args.image))
    if not images:
        print(f"No images found at: {args.image}")
        return

    output_dir = Path(args.output) if args.output else REPO_ROOT / "runs" / "inference"
    all_results: dict[str, list[dict]] = {}

    print(f"Running inference on {len(images)} image(s)…")
    for img_path in images:
        if args.tile:
            dets = detector.predict_tiled(
                img_path,
                tile_size=ic["tile_size"],
                overlap=ic["tile_overlap"],
            )
        else:
            dets = detector.predict(img_path)

        all_results[str(img_path)] = dets
        status = f"{len(dets)} detection(s)"
        print(f"  {img_path.name}: {status}")

        if not args.json:
            out_img = output_dir / img_path.name
            draw_detections(img_path, dets, out_img)

    if args.json:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "detections.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nJSON results written to {json_path}")
    else:
        print(f"\nAnnotated images written to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect graphical scales on engineering plats")
    parser.add_argument("--image",   required=True,
                        help="Path to an image file or directory of images")
    parser.add_argument("--weights", default=None,
                        help="Path to model weights (.pt). Defaults to weights/best.pt")
    parser.add_argument("--output",  default=None,
                        help="Output directory for annotated images or JSON")
    parser.add_argument("--conf",    type=float, default=None,
                        help="Confidence threshold override")
    parser.add_argument("--device",  default="",
                        help="Device override: 'cpu', 'cuda', 'cuda:0', etc.")
    parser.add_argument("--tile",    action="store_true",
                        help="Use tiled inference (recommended for large plats)")
    parser.add_argument("--json",    action="store_true",
                        help="Write JSON detections instead of annotated images")
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
