"""
detector.py – wrapper around a YOLOv8 model for graphical-scale detection.

Handles:
  • loading weights (PT or ONNX)
  • single-image and batch inference
  • tiled inference for large engineering plats
  • result post-processing into a clean dict format
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


class ScaleDetector:
    """Thin wrapper around an Ultralytics YOLOv8 model."""

    CLASS_NAMES = ["graphical_scale"]

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 1280,
        device: str = "",
    ) -> None:
        """
        Args:
            weights_path: Path to .pt or .onnx model file.
            conf_threshold: Minimum confidence to report a detection.
            iou_threshold: NMS IoU threshold.
            imgsz: Inference image size (long edge).
            device: Torch device string, e.g. "cpu", "cuda:0", or "" for auto.
        """
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path))
        self.conf  = conf_threshold
        self.iou   = iou_threshold
        self.imgsz = imgsz
        self.device = device

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, image_path: str | Path) -> list[dict[str, Any]]:
        """Run inference on a single image. Returns a list of detections."""
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results[0])

    def predict_tiled(
        self,
        image_path: str | Path,
        tile_size: int = 1280,
        overlap: float = 0.2,
    ) -> list[dict[str, Any]]:
        """
        Tile a large image, run inference on each tile, then merge detections
        with NMS.  Useful for high-res engineering plats.
        """
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        h, w = img.shape[:2]
        step = int(tile_size * (1 - overlap))
        all_boxes: list[list[float]] = []
        all_scores: list[float]      = []

        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                x1 = min(x0 + tile_size, w)
                y1 = min(y0 + tile_size, h)
                tile = img[y0:y1, x0:x1]

                results = self.model.predict(
                    source=tile,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=tile_size,
                    device=self.device,
                    verbose=False,
                )
                for det in self._parse_results(results[0]):
                    # Translate box back to full-image coordinates
                    bx, by, bw, bh = (
                        det["x"] + x0,
                        det["y"] + y0,
                        det["width"],
                        det["height"],
                    )
                    all_boxes.append([bx, by, bx + bw, by + bh])
                    all_scores.append(det["confidence"])

        if not all_boxes:
            return []

        # Apply NMS across all tiles using OpenCV's DNN backend
        import cv2
        boxes_cv = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in all_boxes]
        indices  = cv2.dnn.NMSBoxes(boxes_cv, all_scores, self.conf, self.iou)
        kept     = indices.flatten().tolist() if len(indices) else []

        detections = []
        for i in kept:
            x1, y1, x2, y2 = all_boxes[i]
            detections.append({
                "class_id":   0,
                "class_name": "graphical_scale",
                "confidence": round(all_scores[i], 4),
                "x":          int(x1),
                "y":          int(y1),
                "width":      int(x2 - x1),
                "height":     int(y2 - y1),
                # YOLO-normalised (requires image dimensions)
                "x_center_norm": (x1 + x2) / 2 / w,
                "y_center_norm": (y1 + y2) / 2 / h,
                "width_norm":    (x2 - x1) / w,
                "height_norm":   (y2 - y1) / h,
            })
        return detections

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_results(result) -> list[dict[str, Any]]:
        detections = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            cls_id = int(box.cls.item())
            xyxy   = box.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy
            orig_h, orig_w  = result.orig_shape
            detections.append({
                "class_id":       cls_id,
                "class_name":     ScaleDetector.CLASS_NAMES[cls_id]
                                  if cls_id < len(ScaleDetector.CLASS_NAMES)
                                  else str(cls_id),
                "confidence":     round(float(box.conf.item()), 4),
                "x":              int(x1),
                "y":              int(y1),
                "width":          int(x2 - x1),
                "height":         int(y2 - y1),
                "x_center_norm":  (x1 + x2) / 2 / orig_w,
                "y_center_norm":  (y1 + y2) / 2 / orig_h,
                "width_norm":     (x2 - x1) / orig_w,
                "height_norm":    (y2 - y1) / orig_h,
            })
        return detections
