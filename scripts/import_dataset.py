"""
import_dataset.py – Import a YOLO-format dataset ZIP exported from the trainer.

Usage
-----
# From a ZIP downloaded from the trainer web UI
python scripts/import_dataset.py --zip ~/Downloads/plat_scale_dataset.zip

# From a directory you already extracted
python scripts/import_dataset.py --dir /path/to/extracted/dataset

# Specify a custom destination name
python scripts/import_dataset.py --zip plat_scale_dataset.zip --name v2
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


def import_from_zip(zip_path: Path, dest: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    print(f"Extracting {zip_path} → {dest}/")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    print("Extraction complete.")


def import_from_dir(src: Path, dest: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    print(f"Copying {src} → {dest}/")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    print("Copy complete.")


def validate_dataset(dest: Path) -> None:
    """Basic sanity checks on the imported dataset."""
    issues = []

    data_yaml = dest / "data.yaml"
    if not data_yaml.exists():
        issues.append("Missing data.yaml")

    for split in ("train", "val"):
        img_dir = dest / "images" / split
        lbl_dir = dest / "labels" / split
        if not img_dir.exists():
            issues.append(f"Missing images/{split}/")
        if not lbl_dir.exists():
            issues.append(f"Missing labels/{split}/")
        else:
            imgs = list(img_dir.glob("*.*"))
            lbls = list(lbl_dir.glob("*.txt"))
            print(f"  {split}: {len(imgs)} images, {len(lbls)} label files")

    if issues:
        print("\nWARNING – dataset issues found:")
        for iss in issues:
            print(f"  ✗ {iss}")
    else:
        print("\nDataset looks valid ✓")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a YOLO dataset from the plat-scale-detection-trainer"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip", metavar="ZIP",
                        help="Path to the ZIP file downloaded from the trainer")
    source.add_argument("--dir", metavar="DIR",
                        help="Path to an already-extracted dataset directory")
    parser.add_argument("--name", default="latest",
                        help="Destination name under datasets/ (default: latest)")
    args = parser.parse_args()

    dest = REPO_ROOT / "datasets" / args.name

    if args.zip:
        import_from_zip(Path(args.zip), dest)
    else:
        import_from_dir(Path(args.dir), dest)

    print(f"\nValidating dataset at {dest}/")
    validate_dataset(dest)
    print(f"\nReady to train:  python train.py --dataset datasets/{args.name}")


if __name__ == "__main__":
    main()
