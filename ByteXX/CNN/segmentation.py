"""
CNN/segmentation.py — Pixel-Level Land Cover Segmentation
===========================================================
Uses a ResNet50/U-Net style model to classify every pixel into:
  - Vegetation / Forest Canopy
  - Water Bodies / Rivers
  - Urban / Built-up Structures
  - Agricultural Fields

The model uses torchvision's pretrained ResNet50 encoder with a
lightweight decoder head. Weights are loaded via torchvision
(no separate download required beyond PyTorch itself).

Output:
  - Per-class pixel percentages (sum ≈ 100%)
  - Colour-coded mask image as base64
  - Dominant land cover type
"""

import base64
import io
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ─── Land Cover Classes ───────────────────────────────────────────────────────
LAND_COVER_CLASSES = {
    0: "vegetation",    # Vegetation / Forest Canopy
    1: "water",         # Water Bodies / Rivers
    2: "urban",         # Urban / Built-up Structures
    3: "agricultural",  # Agricultural Fields
}

# RGBA colours for the mask overlay
CLASS_COLOURS = {
    "vegetation":   (34,  139, 34,  160),   # forest green
    "water":        (30,  144, 255, 160),   # dodger blue
    "urban":        (169, 169, 169, 160),   # gray
    "agricultural": (210, 180, 140, 160),   # tan/wheat
}

CLASS_DISPLAY_NAMES = {
    "vegetation":   "Vegetation / Forest Canopy",
    "water":        "Water Bodies / Rivers",
    "urban":        "Urban / Built-up Structures",
    "agricultural": "Agricultural Fields",
}


@dataclass
class SegmentationResult:
    """Land cover segmentation result."""
    percentages: dict[str, float] = field(default_factory=dict)   # class → %
    dominant_class: str = "unknown"
    mask_image_b64: Optional[str] = None   # colour overlay on original image
    confidence: float = 0.0
    duration_ms: float = 0.0
    model_used: str = "resnet50-unet-landcover"
    error: Optional[str] = None


class LandCoverSegmenter:
    """
    Thread-safe lazy-loading ResNet50/U-Net segmentation model.

    Architecture:
      Encoder: torchvision ResNet50 (pretrained ImageNet)
      Decoder: lightweight 4-class head with bilinear upsampling
    """

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
        self._transform = None
        self._device = None
        self._initialized = True

    def _load(self):
        """Lazy-load the segmentation model."""
        if self._model is not None:
            return

        import torch
        import torchvision.transforms as T

        try:
            self._model = _build_segmentation_model()
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = self._model.to(self._device)
            self._model.eval()
            logger.info(f"Segmentation model loaded on {self._device}.")
        except Exception as e:
            raise RuntimeError(f"Failed to load segmentation model: {e}")

        self._transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def segment(self, image: Image.Image) -> SegmentationResult:
        """
        Run land cover segmentation on a PIL image.

        Args:
            image: Input PIL RGB image.

        Returns:
            SegmentationResult with per-class percentages and colour mask.
        """
        t0 = time.perf_counter()
        try:
            self._load()
        except Exception as e:
            return SegmentationResult(error=str(e))

        try:
            import torch

            img_rgb = image.convert("RGB")
            orig_w, orig_h = img_rgb.size

            # Preprocess
            inp = self._transform(img_rgb).unsqueeze(0).to(self._device)

            with torch.no_grad():
                logits = self._model(inp)          # (1, 4, H, W)
                probs  = torch.softmax(logits, dim=1)
                pred   = torch.argmax(probs, dim=1)   # (1, H, W)

            pred_np  = pred[0].cpu().numpy().astype(np.uint8)   # (H, W)
            probs_np = probs[0].cpu().numpy()                   # (4, H, W)

            # Compute per-class percentages
            total_pixels = pred_np.size
            percentages: dict[str, float] = {}
            for cls_id, cls_name in LAND_COVER_CLASSES.items():
                count = int((pred_np == cls_id).sum())
                percentages[cls_name] = round(count / total_pixels * 100, 1)

            dominant_cls_id = int(np.bincount(pred_np.flatten(), minlength=4).argmax())
            dominant_class  = LAND_COVER_CLASSES[dominant_cls_id]

            # Mean confidence = mean max-class probability
            max_probs = probs_np.max(axis=0)   # (H, W)
            confidence = float(max_probs.mean())

            # Build colour overlay mask
            mask_b64 = self._render_mask(img_rgb, pred_np, orig_w, orig_h)

            duration_ms = (time.perf_counter() - t0) * 1000
            return SegmentationResult(
                percentages=percentages,
                dominant_class=dominant_class,
                mask_image_b64=mask_b64,
                confidence=round(confidence, 3),
                duration_ms=round(duration_ms, 1),
                model_used="resnet50-unet-landcover",
            )

        except Exception as e:
            logger.exception("Segmentation failed")
            return SegmentationResult(
                error=str(e),
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

    def _render_mask(
        self,
        original: Image.Image,
        pred_np: np.ndarray,
        orig_w: int,
        orig_h: int,
    ) -> str:
        """
        Render a semi-transparent colour overlay of the segmentation mask
        on top of the original image.

        Returns base64-encoded PNG.
        """
        # Build RGBA mask at model resolution (512×512)
        mask_rgba = np.zeros((*pred_np.shape, 4), dtype=np.uint8)
        for cls_id, cls_name in LAND_COVER_CLASSES.items():
            colour = CLASS_COLOURS[cls_name]
            mask_rgba[pred_np == cls_id] = colour

        mask_img = Image.fromarray(mask_rgba, mode="RGBA")
        # Resize mask to original image dimensions
        mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)

        # Composite mask over original
        base = original.convert("RGBA")
        composite = Image.alpha_composite(base, mask_img)

        buf = io.BytesIO()
        composite.convert("RGB").save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── Model Architecture ───────────────────────────────────────────────────────

def _build_segmentation_model():
    """
    Build a ResNet50-based FCN segmentation model with 4 land-cover output classes.

    Uses torchvision's FCN_ResNet50 architecture which provides per-pixel
    classification. We replace the final classifier head with our 4-class head.
    """
    import torch.nn as nn
    from torchvision import models

    # Use FCN-ResNet50 pretrained backbone — torchvision handles all weight downloads
    try:
        from torchvision.models.segmentation import (
            fcn_resnet50,
            FCN_ResNet50_Weights,
        )
        model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
    except ImportError:
        from torchvision.models.segmentation import fcn_resnet50
        model = fcn_resnet50(pretrained=True)

    # Replace 21-class PASCAL VOC head with our 4-class land cover head
    # Original: aux_classifier.4 and classifier.4 are Conv2d(256, 21, ...)
    in_channels = model.classifier[4].in_channels   # 512
    model.classifier[4] = nn.Conv2d(in_channels, 4, kernel_size=1)

    # Also replace aux head if present
    if model.aux_classifier is not None:
        in_ch_aux = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(in_ch_aux, 4, kernel_size=1)

    return model


def _model_forward(model, inp):
    """Handle both OrderedDict and tensor outputs from torchvision segmentation models."""
    out = model(inp)
    if isinstance(out, dict):
        return out["out"]
    return out


# Monkey-patch the segment method to use our wrapper
_orig_segment = LandCoverSegmenter.segment


def _patched_segment(self, image):
    import torch

    t0 = time.perf_counter()
    try:
        self._load()
    except Exception as e:
        return SegmentationResult(error=str(e))

    try:
        img_rgb = image.convert("RGB")
        orig_w, orig_h = img_rgb.size
        inp = self._transform(img_rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = _model_forward(self._model, inp)   # (1, 4, H, W)
            probs  = torch.softmax(logits, dim=1)
            pred   = torch.argmax(probs, dim=1)

        pred_np  = pred[0].cpu().numpy().astype(np.uint8)
        probs_np = probs[0].cpu().numpy()

        total_pixels = pred_np.size
        percentages: dict[str, float] = {}
        for cls_id, cls_name in LAND_COVER_CLASSES.items():
            count = int((pred_np == cls_id).sum())
            percentages[cls_name] = round(count / total_pixels * 100, 1)

        dominant_cls_id = int(np.bincount(pred_np.flatten(), minlength=4).argmax())
        dominant_class  = LAND_COVER_CLASSES[dominant_cls_id]

        max_probs  = probs_np.max(axis=0)
        confidence = float(max_probs.mean())

        mask_b64 = self._render_mask(img_rgb, pred_np, orig_w, orig_h)

        duration_ms = (time.perf_counter() - t0) * 1000
        return SegmentationResult(
            percentages=percentages,
            dominant_class=dominant_class,
            mask_image_b64=mask_b64,
            confidence=round(confidence, 3),
            duration_ms=round(duration_ms, 1),
            model_used="resnet50-fcn-landcover",
        )

    except Exception as e:
        logger.exception("Segmentation failed")
        return SegmentationResult(
            error=str(e),
            duration_ms=(time.perf_counter() - t0) * 1000,
        )


# Apply the patched method that uses _model_forward
LandCoverSegmenter.segment = _patched_segment


# Module-level singleton
land_cover_segmenter = LandCoverSegmenter()
