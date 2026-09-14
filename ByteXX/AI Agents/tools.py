"""
AI Agents/tools.py — Tool Registry
=====================================
Defines all callable tools with a standard interface.

Each tool:
  - Accepts a PIL image + optional arguments
  - Returns a ToolResult with: result dict, confidence, duration_ms, tool_name
  - Is async-native (uses asyncio executor for CPU-bound CNN calls)

Available tools:
  cnn_detect(image, target_classes, conf_threshold)  → YOLOv8 detection w/ Gemma 4 fallback
  cnn_segment(image)                                 → Land-cover segmentation
  vlm_vqa(image, question)                           → Gemma 4 VQA (OpenRouter)
  vlm_caption(image)                                 → Gemma 4 captioning (OpenRouter)
  vlm_detect(image, object_label)                    → Gemma 4 detection + textual location
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image

logger = logging.getLogger(__name__)

# ─── Ensure parent directory is on path for both module styles ────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass
class ToolResult:
    """
    Standard result container returned by every tool.
    """
    tool_name: str
    result: dict[str, Any]
    confidence: float              # 0.0 – 1.0
    duration_ms: float
    success: bool = True
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tool_name":   self.tool_name,
            "result":      self.result,
            "confidence":  self.confidence,
            "duration_ms": self.duration_ms,
            "success":     self.success,
            "error":       self.error,
            "metadata":    self.metadata,
        }


# ─── Tool Implementations ────────────────────────────────────────────────────

async def cnn_detect(
    image: Image.Image,
    target_classes: list[str] | None = None,
    conf_threshold: float = 0.25,
) -> ToolResult:
    """
    YOLOv8 object detection tool.

    Primary: YOLOv8 (fast, returns pixel bboxes).
    Fallback: Gemma 4 via OpenRouter (if ultralytics is not installed or YOLO
              returns 0 detections for satellite-specific objects like planes).
    """
    t0 = time.perf_counter()

    # ── Try YOLO first ─────────────────────────────────────────────────────────
    try:
        from CNN.yolo_detector import yolo_detector

        loop = asyncio.get_event_loop()
        det_result = await loop.run_in_executor(
            None, yolo_detector.detect, image, conf_threshold
        )

        if det_result.error:
            raise RuntimeError(det_result.error)

        # Filter to target classes if specified
        detections = det_result.detections
        counts     = det_result.object_counts
        if target_classes:
            tc_lower = [c.lower() for c in target_classes]
            detections = [d for d in detections
                          if d.label.lower() in tc_lower or
                          any(tc in d.label.lower() for tc in tc_lower)]
            counts = {k: v for k, v in counts.items()
                      if k.lower() in tc_lower or
                      any(tc in k.lower() for tc in tc_lower)}

        # ── If YOLO found detections, return them ──────────────────────────────
        if det_result.total_count > 0 or not target_classes:
            result = {
                "object_counts":       counts,
                "total_detections":    len(detections),
                "bboxes":              [d.bbox_norm for d in detections],
                "bbox_labels":         [d.label for d in detections],
                "bbox_confidences":    [d.confidence for d in detections],
                "annotated_image_b64": det_result.annotated_image_b64,
                "model":               "yolov8n",
                "source":              "yolo",
            }
            return ToolResult(
                tool_name="cnn_detect",
                result=result,
                confidence=det_result.confidence,
                duration_ms=det_result.duration_ms,
                metadata={"target_classes": target_classes},
            )

        # YOLO ran but found 0 matching objects for the target class.
        # Fall through to Gemma 4 for satellite-specific object counting.
        logger.info(
            "YOLO found 0 matching detections for %s — falling back to Gemma 4",
            target_classes,
        )
        yolo_zero = True

    except ImportError:
        # ultralytics not installed — go straight to Gemma 4
        logger.warning("ultralytics not installed — using Gemma 4 for detection")
        yolo_zero = False
    except Exception as e:
        logger.warning("YOLO detect failed (%s) — falling back to Gemma 4", e)
        yolo_zero = False

    # ── Gemma 4 fallback for object counting ────────────────────────────────────
    # Gemma 4 is much better than YOLOv8-COCO at counting satellite-specific
    # objects (aircraft, ships, storage tanks, etc.) via visual reasoning.
    try:
        from backend.model import model_manager
        import base64, io as _io

        # Convert PIL image to base64 for Gemma client
        buf = _io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        obj_str = ", ".join(target_classes) if target_classes else "all objects"
        question = (
            f"Count all {obj_str} visible in this satellite/aerial image. "
            f"How many {obj_str} can you see? "
            f"List their approximate locations (e.g. top-left, center, bottom-right). "
            f"Be precise and count carefully."
        )

        loop = asyncio.get_event_loop()
        vqa_result = await loop.run_in_executor(
            None, model_manager.answer_vqa, image_b64, question
        )

        answer = vqa_result.get("answer", "")
        conf   = vqa_result.get("confidence", 0.70)

        # Try to extract a numeric count from the answer
        import re
        count_match = re.search(r'\b(\d+)\b', answer)
        count_val = int(count_match.group(1)) if count_match else 0
        obj_label = target_classes[0] if target_classes else "object"

        result = {
            "object_counts":    {obj_label: count_val} if count_val else {},
            "total_detections": count_val,
            "bboxes":           [],    # Gemma returns text locations, not pixel coords
            "bbox_labels":      [],
            "bbox_confidences": [],
            "answer":           answer,
            "model":            vqa_result.get("model", "gemma-4-31b"),
            "source":           "gemma4_fallback",
            "yolo_zero":        yolo_zero,
        }
        return ToolResult(
            tool_name="cnn_detect",
            result=result,
            confidence=conf,
            duration_ms=(time.perf_counter() - t0) * 1000,
            metadata={"target_classes": target_classes, "fallback": "gemma4"},
        )

    except Exception as e:
        logger.exception("cnn_detect Gemma 4 fallback also failed")
        return ToolResult(
            tool_name="cnn_detect",
            result={},
            confidence=0.0,
            duration_ms=(time.perf_counter() - t0) * 1000,
            success=False,
            error=f"YOLO unavailable and Gemma 4 fallback failed: {e}",
        )


async def cnn_segment(image: Image.Image) -> ToolResult:
    """
    ResNet/FCN land-cover segmentation tool.

    Returns pixel-level percentages for Vegetation, Water, Urban, Agricultural.
    """
    t0 = time.perf_counter()
    try:
        from CNN.segmentation import land_cover_segmenter

        loop = asyncio.get_event_loop()
        seg_result = await loop.run_in_executor(
            None, land_cover_segmenter.segment, image
        )

        if seg_result.error:
            return ToolResult(
                tool_name="cnn_segment",
                result={},
                confidence=0.0,
                duration_ms=(time.perf_counter() - t0) * 1000,
                success=False,
                error=seg_result.error,
            )

        from CNN.segmentation import CLASS_DISPLAY_NAMES
        result = {
            "land_cover_percentages": seg_result.percentages,
            "land_cover_display": {
                CLASS_DISPLAY_NAMES.get(k, k): v
                for k, v in seg_result.percentages.items()
            },
            "dominant_class":     seg_result.dominant_class,
            "mask_image_b64":     seg_result.mask_image_b64,
            "model":              "resnet50-fcn-landcover",
        }
        return ToolResult(
            tool_name="cnn_segment",
            result=result,
            confidence=seg_result.confidence,
            duration_ms=seg_result.duration_ms,
        )

    except Exception as e:
        logger.exception("cnn_segment tool failed")
        return ToolResult(
            tool_name="cnn_segment",
            result={},
            confidence=0.0,
            duration_ms=(time.perf_counter() - t0) * 1000,
            success=False,
            error=str(e),
        )


async def vlm_vqa(image: Image.Image, question: str) -> ToolResult:
    """Qwen2.5-VL Visual Question Answering tool."""
    t0 = time.perf_counter()
    try:
        from backend.model import model_manager

        if not model_manager.is_loaded:
            return ToolResult(
                tool_name="vlm_vqa",
                result={"answer": "[VLM not loaded — stub response]", "confidence": 0.0},
                confidence=0.0,
                duration_ms=(time.perf_counter() - t0) * 1000,
                success=True,
                error=None,
                metadata={"stub": True},
            )

        loop = asyncio.get_event_loop()
        vqa_result = await loop.run_in_executor(
            None, model_manager.answer_vqa, image, question
        )

        return ToolResult(
            tool_name="vlm_vqa",
            result={
                "answer":          vqa_result["answer"],
                "reasoning_trace": vqa_result.get("reasoning_trace", ""),
                "model":           vqa_result.get("model", "gemma-4-31b"),
            },
            confidence=vqa_result.get("confidence", 0.5),
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    except Exception as e:
        logger.exception("vlm_vqa tool failed")
        return ToolResult(
            tool_name="vlm_vqa",
            result={},
            confidence=0.0,
            duration_ms=(time.perf_counter() - t0) * 1000,
            success=False,
            error=str(e),
        )


async def vlm_caption(image: Image.Image) -> ToolResult:
    """Qwen2.5-VL image captioning tool."""
    t0 = time.perf_counter()
    try:
        from backend.model import model_manager

        if not model_manager.is_loaded:
            return ToolResult(
                tool_name="vlm_caption",
                result={"caption": "[VLM not loaded — stub caption]"},
                confidence=0.0,
                duration_ms=(time.perf_counter() - t0) * 1000,
                metadata={"stub": True},
            )

        loop = asyncio.get_event_loop()
        cap_result = await loop.run_in_executor(
            None, model_manager.generate_caption, image
        )

        return ToolResult(
            tool_name="vlm_caption",
            result={
                "caption": cap_result["caption"],
                "model":   cap_result.get("model", "gemma-4-31b"),
            },
            confidence=cap_result.get("confidence", 0.5),
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    except Exception as e:
        logger.exception("vlm_caption tool failed")
        return ToolResult(
            tool_name="vlm_caption",
            result={},
            confidence=0.0,
            duration_ms=(time.perf_counter() - t0) * 1000,
            success=False,
            error=str(e),
        )


async def vlm_detect(image: Image.Image, object_label: str) -> ToolResult:
    """Qwen2.5-VL object detection + bounding boxes tool."""
    t0 = time.perf_counter()
    try:
        from backend.model import model_manager

        if not model_manager.is_loaded:
            return ToolResult(
                tool_name="vlm_detect",
                result={"answer": "[VLM not loaded]", "bboxes": []},
                confidence=0.0,
                duration_ms=(time.perf_counter() - t0) * 1000,
                metadata={"stub": True},
            )

        loop = asyncio.get_event_loop()
        det_result = await loop.run_in_executor(
            None, model_manager.detect_objects, image, object_label
        )

        return ToolResult(
            tool_name="vlm_detect",
            result={
                "answer":       det_result["answer"],
                "bboxes":       det_result.get("bboxes", []),
                "object":       det_result.get("object", object_label),
                "model":        det_result.get("model", "gemma-4-31b"),
            },
            confidence=det_result.get("confidence", 0.5),
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    except Exception as e:
        logger.exception("vlm_detect tool failed")
        return ToolResult(
            tool_name="vlm_detect",
            result={},
            confidence=0.0,
            duration_ms=(time.perf_counter() - t0) * 1000,
            success=False,
            error=str(e),
        )


# ─── Tool Registry ────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, Any] = {
    "cnn_detect":  cnn_detect,
    "cnn_segment": cnn_segment,
    "vlm_vqa":     vlm_vqa,
    "vlm_caption": vlm_caption,
    "vlm_detect":  vlm_detect,
}


async def call_tool(
    tool_name: str,
    image: Image.Image,
    args: dict,
) -> ToolResult:
    """
    Call a tool by name with the given image and args.

    Args:
        tool_name: Name from TOOL_REGISTRY.
        image:     PIL image to analyse.
        args:      Keyword arguments for the tool.

    Returns:
        ToolResult from the called tool.
    """
    if tool_name not in TOOL_REGISTRY:
        return ToolResult(
            tool_name=tool_name,
            result={},
            confidence=0.0,
            duration_ms=0.0,
            success=False,
            error=f"Unknown tool: '{tool_name}'. Available: {list(TOOL_REGISTRY)}",
        )

    fn = TOOL_REGISTRY[tool_name]
    return await fn(image, **args)
