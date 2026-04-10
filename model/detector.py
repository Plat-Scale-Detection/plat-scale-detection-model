"""
detector.py – ONNX-based graphical-scale detector.

Uses onnxruntime (MIT licensed) for inference — no dependency on Ultralytics
or PyTorch.  Weights are exported to ONNX by the trainer and published as
release assets alongside best.pt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class ScaleDetector:
    """Inference wrapper for the ONNX-exported YOLOv8 scale detection model."""

    CLASS_NAMES = ["graphical_scale"]

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 1280,
    ) -> None:
        """
        Args:
            weights_path:   Path to best.onnx (downloaded from GitHub Releases).
            conf_threshold: Minimum confidence to report a detection.
            iou_threshold:  NMS IoU threshold.
            imgsz:          Inference image size (square, must match export size).
        """
        import onnxruntime as ort

        self._conf  = conf_threshold
        self._iou   = iou_threshold
        self._imgsz = imgsz

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session    = ort.InferenceSession(str(weights_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, image_path: str | Path) -> list[dict[str, Any]]:
        """Run inference on a single image. Returns a list of detection dicts."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self._infer(img)

    def predict_tiled(
        self,
        image_path: str | Path,
        tile_size: int | None = None,
        overlap: float = 0.2,
    ) -> list[dict[str, Any]]:
        """
        Tile a large image, run inference on each tile, then merge with NMS.
        Recommended for high-res engineering plats.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        h, w = img.shape[:2]
        ts   = tile_size or self._imgsz
        step = int(ts * (1 - overlap))

        all_boxes: list[list[float]] = []
        all_scores: list[float]      = []

        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                tile = img[y0:min(y0 + ts, h), x0:min(x0 + ts, w)]
                for det in self._infer(tile):
                    all_boxes.append([
                        det["x"] + x0,
                        det["y"] + y0,
                        det["x"] + x0 + det["width"],
                        det["y"] + y0 + det["height"],
                    ])
                    all_scores.append(det["confidence"])

        if not all_boxes:
            return []

        boxes_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in all_boxes]
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, all_scores, self._conf, self._iou)
        kept = idxs.flatten().tolist() if len(idxs) else []

        results = []
        for i in kept:
            x1, y1, x2, y2 = all_boxes[i]
            bw, bh = x2 - x1, y2 - y1
            results.append({
                "class_id":      0,
                "class_name":    "graphical_scale",
                "confidence":    round(all_scores[i], 4),
                "x":             int(x1),
                "y":             int(y1),
                "width":         int(bw),
                "height":        int(bh),
                "x_center_norm": (x1 + bw / 2) / w,
                "y_center_norm": (y1 + bh / 2) / h,
                "width_norm":    bw / w,
                "height_norm":   bh / h,
            })
        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _preprocess(
        self, img_bgr: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        """Letterbox-resize, normalise, and format for ONNX input."""
        h, w  = img_bgr.shape[:2]
        ratio = min(self._imgsz / h, self._imgsz / w)
        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))
        dw    = (self._imgsz - new_w) / 2
        dh    = (self._imgsz - new_h) / 2

        img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(
            img,
            int(round(dh - 0.1)), int(round(dh + 0.1)),
            int(round(dw - 0.1)), int(round(dw + 0.1)),
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.expand_dims(img_rgb.transpose(2, 0, 1), 0)  # [1, 3, H, W]
        return inp, ratio, (dw, dh)

    def _infer(self, img_bgr: np.ndarray) -> list[dict[str, Any]]:
        """Run inference on a BGR numpy array."""
        orig_h, orig_w = img_bgr.shape[:2]
        inp, ratio, (dw, dh) = self._preprocess(img_bgr)

        # Output shape: [1, 4+nc, num_anchors]; for nc=1 → [1, 5, anchors]
        raw = self._session.run(None, {self._input_name: inp})[0][0].T  # [anchors, 5]

        scores = raw[:, 4]
        mask   = scores >= self._conf
        raw    = raw[mask]
        scores = scores[mask]

        if len(raw) == 0:
            return []

        # Decode cx, cy, w, h (letterboxed pixels) → x1, y1, x2, y2 (orig pixels)
        cx, cy, bw, bh = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        x1 = np.clip((cx - bw / 2 - dw) / ratio, 0, orig_w)
        y1 = np.clip((cy - bh / 2 - dh) / ratio, 0, orig_h)
        x2 = np.clip((cx + bw / 2 - dw) / ratio, 0, orig_w)
        y2 = np.clip((cy + bh / 2 - dh) / ratio, 0, orig_h)

        boxes_xywh = [
            [float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])]
            for i in range(len(x1))
        ]
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), self._conf, self._iou)
        kept = idxs.flatten().tolist() if len(idxs) else []

        detections = []
        for i in kept:
            bx, by   = float(x1[i]), float(y1[i])
            bww, bhh = float(x2[i] - x1[i]), float(y2[i] - y1[i])
            detections.append({
                "class_id":      0,
                "class_name":    "graphical_scale",
                "confidence":    round(float(scores[i]), 4),
                "x":             int(bx),
                "y":             int(by),
                "width":         int(bww),
                "height":        int(bhh),
                "x_center_norm": (bx + bww / 2) / orig_w,
                "y_center_norm": (by + bhh / 2) / orig_h,
                "width_norm":    bww / orig_w,
                "height_norm":   bhh / orig_h,
            })
        return detections


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
