"""
CNN/cnn_pipeline.py — Unified CNN Pipeline
===========================================
Wraps YOLOv8 detection and ResNet/FCN land-cover segmentation
into a single async-friendly interface.

Usage:
    from CNN.cnn_pipeline import cnn_pipeline
    result = await cnn_pipeline.run(image, tasks=["detect", "segment"])
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from CNN.yolo_detector import YOLOResult, yolo_detector
from CNN.segmentation import SegmentationResult, land_cover_segmenter, CLASS_DISPLAY_NAMES

logger = logging.getLogger(__name__)


@dataclass
class CNNResult:
    """
    Combined CNN pipeline result containing:
      - YOLOv8 detection results (counts + bboxes + annotated image)
      - Land-cover segmentation results (percentages + colour mask)
    """
    # Detection
    object_counts: dict[str, int] = field(default_factory=dict)
    total_detections: int = 0
    bboxes: list[list[float]] = field(default_factory=list)      # [[x1,y1,x2,y2], ...]
    bbox_labels: list[str] = field(default_factory=list)
    bbox_confidences: list[float] = field(default_factory=list)
    annotated_image_b64: Optional[str] = None

    # Segmentation
    land_cover_percentages: dict[str, float] = field(default_factory=dict)   # raw key → %
    land_cover_display: dict[str, float] = field(default_factory=dict)        # display name → %
    dominant_land_cover: str = "unknown"
    mask_image_b64: Optional[str] = None

    # Meta
    tasks_run: list[str] = field(default_factory=list)
    overall_confidence: float = 0.0
    total_duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "object_counts":           self.object_counts,
            "total_detections":        self.total_detections,
            "bboxes":                  self.bboxes,
            "bbox_labels":             self.bbox_labels,
            "bbox_confidences":        self.bbox_confidences,
            "annotated_image_b64":     self.annotated_image_b64,
            "land_cover_percentages":  self.land_cover_percentages,
            "land_cover_display":      self.land_cover_display,
            "dominant_land_cover":     self.dominant_land_cover,
            "mask_image_b64":          self.mask_image_b64,
            "tasks_run":               self.tasks_run,
            "overall_confidence":      self.overall_confidence,
            "total_duration_ms":       self.total_duration_ms,
            "errors":                  self.errors,
        }


class CNNPipeline:
    """
    Async-ready CNN pipeline that runs detection and/or segmentation.

    Both models are loaded lazily and cached as singletons.
    Independent tasks are run concurrently using asyncio.
    """

    async def run(
        self,
        image: Image.Image,
        tasks: list[str] | None = None,
        conf_threshold: float = 0.25,
    ) -> CNNResult:
        """
        Run requested CNN tasks on the input image.

        Args:
            image: PIL RGB image.
            tasks: List of tasks to run — any of ["detect", "segment"].
                   Defaults to both if not specified.
            conf_threshold: YOLO detection confidence threshold.

        Returns:
            CNNResult combining results from all requested tasks.
        """
        if tasks is None:
            tasks = ["detect", "segment"]

        t0 = time.perf_counter()
        result = CNNResult(tasks_run=tasks)

        # Run tasks concurrently in the thread executor
        loop = asyncio.get_event_loop()
        coros = []

        if "detect" in tasks:
            coros.append(
                loop.run_in_executor(None, yolo_detector.detect, image, conf_threshold)
            )
        else:
            coros.append(asyncio.coroutine(lambda: None)() if False else self._noop())

        if "segment" in tasks:
            coros.append(
                loop.run_in_executor(None, land_cover_segmenter.segment, image)
            )
        else:
            coros.append(self._noop())

        det_result, seg_result = await asyncio.gather(*coros)

        # ── Unpack detection result ──────────────────────────────────────────
        if "detect" in tasks and isinstance(det_result, YOLOResult):
            if det_result.error:
                result.errors.append(f"Detection: {det_result.error}")
            else:
                result.object_counts       = det_result.object_counts
                result.total_detections    = det_result.total_count
                result.annotated_image_b64 = det_result.annotated_image_b64
                result.bboxes = [
                    d.bbox_norm for d in det_result.detections
                ]
                result.bbox_labels = [d.label for d in det_result.detections]
                result.bbox_confidences = [d.confidence for d in det_result.detections]

        # ── Unpack segmentation result ───────────────────────────────────────
        if "segment" in tasks and isinstance(seg_result, SegmentationResult):
            if seg_result.error:
                result.errors.append(f"Segmentation: {seg_result.error}")
            else:
                result.land_cover_percentages = seg_result.percentages
                result.dominant_land_cover    = seg_result.dominant_class
                result.mask_image_b64         = seg_result.mask_image_b64
                # Build display names mapping
                result.land_cover_display = {
                    CLASS_DISPLAY_NAMES.get(k, k): v
                    for k, v in seg_result.percentages.items()
                }

        # ── Compute overall confidence ───────────────────────────────────────
        conf_values = []
        if "detect" in tasks and isinstance(det_result, YOLOResult) and not det_result.error:
            if det_result.confidence > 0:
                conf_values.append(det_result.confidence)
        if "segment" in tasks and isinstance(seg_result, SegmentationResult) and not seg_result.error:
            conf_values.append(seg_result.confidence)

        result.overall_confidence = round(
            sum(conf_values) / len(conf_values) if conf_values else 0.0, 3
        )
        result.total_duration_ms = round((time.perf_counter() - t0) * 1000, 1)

        logger.info(
            f"CNN pipeline done — tasks={tasks}, "
            f"detections={result.total_detections}, "
            f"conf={result.overall_confidence:.3f}, "
            f"duration={result.total_duration_ms:.0f}ms"
        )
        return result

    @staticmethod
    async def _noop():
        """No-op coroutine for skipped tasks."""
        return None

    def run_sync(
        self,
        image: Image.Image,
        tasks: list[str] | None = None,
        conf_threshold: float = 0.25,
    ) -> CNNResult:
        """Synchronous wrapper for use outside async contexts."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already inside an event loop — use thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.run(image, tasks, conf_threshold))
                    return future.result()
            else:
                return loop.run_until_complete(self.run(image, tasks, conf_threshold))
        except RuntimeError:
            return asyncio.run(self.run(image, tasks, conf_threshold))


# Module-level singleton
cnn_pipeline = CNNPipeline()
