"""
CNN/yolo_detector.py — YOLOv8 Object Detector
===============================================
Instant object counting & detection using YOLOv8.

Features:
  - Detects: aircraft, vehicles, ships, buildings, trucks,
    storage tanks, bridges, roads, people
  - Draws square bounding boxes on the image
  - Returns normalized [0-1] bboxes + annotated image as base64
  - Lazy-loads model on first call (cached singleton)
"""

import base64
import io
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ─── COCO class IDs that correspond to our categories ────────────────────────
# YOLOv8 (pretrained on COCO-80) label mapping
YOLO_CLASS_MAP = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    6:  "train",
    7:  "truck",
    8:  "boat",
    9:  "traffic light",
    11: "stop sign",
    15: "cat",
    16: "dog",
    17: "horse",
    39: "bottle",
    56: "chair",
    57: "couch",
    59: "bed",
    60: "dining table",
    62: "tv",
    63: "laptop",
    66: "keyboard",
    67: "cell phone",
    72: "refrigerator",
    73: "book",
    79: "toothbrush",
    # Satellite-relevant mappings
    2:  "vehicle",
    7:  "truck",
    8:  "vessel",
    0:  "person",
}

# Human-friendly label overrides for satellite context
SATELLITE_LABELS = {
    "car":        "Vehicle",
    "vehicle":    "Vehicle",
    "truck":      "Truck",
    "bus":        "Vehicle",
    "boat":       "Vessel",
    "ship":       "Vessel",
    "person":     "Person",
    "motorcycle": "Vehicle",
}

# Box colour palette per class (BGR → used with PIL as RGB)
PALETTE = [
    "#00F5FF",  # cyan
    "#FF6B35",  # orange
    "#7FFF00",  # chartreuse
    "#FF1493",  # pink
    "#FFD700",  # gold
    "#DA70D6",  # orchid
    "#00FF7F",  # spring green
    "#FF4500",  # orange-red
    "#1E90FF",  # dodger blue
    "#ADFF2F",  # green-yellow
]


@dataclass
class Detection:
    """Single detected object."""
    label: str
    confidence: float
    bbox_norm: list[float]   # [x1, y1, x2, y2] in 0-1
    bbox_px: list[int]       # [x1, y1, x2, y2] in pixels


@dataclass
class YOLOResult:
    """Full YOLO detection result."""
    detections: list[Detection] = field(default_factory=list)
    object_counts: dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    annotated_image_b64: Optional[str] = None
    confidence: float = 0.0
    duration_ms: float = 0.0
    model_used: str = "yolov8n"
    error: Optional[str] = None


class YOLODetector:
    """Thread-safe lazy-loading YOLOv8 detector."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._model = None
        self._initialized = True

    def _load(self):
        """Lazy-load YOLOv8 nano model (auto-downloads ~6 MB on first run)."""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            logger.info("Loading YOLOv8 model...")
            self._model = YOLO("yolov8n.pt")   # nano — fastest inference
            logger.info("YOLOv8 loaded successfully.")
        except ImportError:
            raise RuntimeError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

    def detect(
        self,
        image: Image.Image,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> YOLOResult:
        """
        Run YOLOv8 detection on a PIL image.

        Args:
            image: Input PIL image (RGB).
            conf_threshold: Minimum confidence to keep a detection.
            iou_threshold: NMS IOU threshold.

        Returns:
            YOLOResult with detections, counts, and annotated image base64.
        """
        t0 = time.perf_counter()
        try:
            self._load()
        except Exception as e:
            return YOLOResult(error=str(e))

        try:
            img_np = np.array(image.convert("RGB"))

            # Run inference
            results = self._model(
                img_np,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )

            result = results[0]
            h, w = img_np.shape[:2]

            detections: list[Detection] = []
            object_counts: dict[str, int] = {}

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf   = float(boxes.conf[i].item())
                    xyxy   = boxes.xyxy[i].tolist()   # pixel coords

                    # Get label
                    raw_label = self._model.names.get(cls_id, f"class_{cls_id}")
                    label = SATELLITE_LABELS.get(raw_label.lower(), raw_label.title())

                    x1_px, y1_px, x2_px, y2_px = [int(v) for v in xyxy]

                    # Make square box (expand the shorter dimension outward)
                    bw = x2_px - x1_px
                    bh = y2_px - y1_px
                    side = max(bw, bh)
                    cx = (x1_px + x2_px) // 2
                    cy = (y1_px + y2_px) // 2
                    sq_x1 = max(0, cx - side // 2)
                    sq_y1 = max(0, cy - side // 2)
                    sq_x2 = min(w, sq_x1 + side)
                    sq_y2 = min(h, sq_y1 + side)

                    bbox_norm = [
                        sq_x1 / w, sq_y1 / h,
                        sq_x2 / w, sq_y2 / h,
                    ]

                    det = Detection(
                        label=label,
                        confidence=conf,
                        bbox_norm=bbox_norm,
                        bbox_px=[sq_x1, sq_y1, sq_x2, sq_y2],
                    )
                    detections.append(det)
                    object_counts[label] = object_counts.get(label, 0) + 1

            # Draw annotated image
            annotated_b64 = self._draw_boxes(image.copy(), detections)

            # Overall confidence = mean of all detected boxes (or 0 if none)
            overall_conf = (
                float(np.mean([d.confidence for d in detections]))
                if detections else 0.0
            )

            duration_ms = (time.perf_counter() - t0) * 1000
            return YOLOResult(
                detections=detections,
                object_counts=object_counts,
                total_count=len(detections),
                annotated_image_b64=annotated_b64,
                confidence=round(overall_conf, 3),
                duration_ms=round(duration_ms, 1),
                model_used="yolov8n",
            )

        except Exception as e:
            logger.exception("YOLO detection failed")
            return YOLOResult(error=str(e), duration_ms=(time.perf_counter() - t0) * 1000)

    def _draw_boxes(self, image: Image.Image, detections: list[Detection]) -> str:
        """
        Draw square bounding boxes with labels on the image.

        Returns base64-encoded PNG string.
        """
        draw = ImageDraw.Draw(image)
        label_set = list({d.label for d in detections})

        # Assign a colour per label class
        colour_map = {
            label: PALETTE[i % len(PALETTE)]
            for i, label in enumerate(sorted(label_set))
        }

        # Try to load a font; fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", size=max(12, image.width // 60))
        except Exception:
            font = ImageFont.load_default()

        w_img, h_img = image.size

        for det in detections:
            x1, y1, x2, y2 = det.bbox_px
            colour = colour_map.get(det.label, "#00F5FF")

            # Draw square box (3px border)
            for thickness in range(3):
                draw.rectangle(
                    [x1 - thickness, y1 - thickness,
                     x2 + thickness, y2 + thickness],
                    outline=colour,
                )

            # Label pill background
            label_text = f"{det.label} {det.confidence:.0%}"
            try:
                bbox_text = draw.textbbox((x1, y1), label_text, font=font)
                tw = bbox_text[2] - bbox_text[0]
                th = bbox_text[3] - bbox_text[1]
            except AttributeError:
                tw, th = len(label_text) * 7, 12

            pad = 3
            lx1 = x1
            ly1 = max(0, y1 - th - pad * 2)
            lx2 = min(w_img, x1 + tw + pad * 2)
            ly2 = y1

            draw.rectangle([lx1, ly1, lx2, ly2], fill=colour)
            draw.text((lx1 + pad, ly1 + pad), label_text, fill="#000000", font=font)

        # Encode to base64 PNG
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# Module-level singleton
yolo_detector = YOLODetector()
