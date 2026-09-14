"""
SatQuery AI — Model Manager (OpenRouter / Gemma 4)
===================================================
This module replaces the previous local Qwen2.5-VL loader.

All inference is now handled by the OpenRouter Gemma 4 client.
This file provides a thin compatibility wrapper (SatQueryModel) so that
firebase_listener.py, main.py, and AI Agent tools.py continue to work
without changes to their call sites.

The interface is identical to the old SatQueryModel:
  model_manager.load()                 → no-op (cloud model, always ready)
  model_manager.is_loaded              → True
  model_manager.answer_vqa(img, q)     → dict {answer, confidence, ...}
  model_manager.generate_caption(img)  → dict {caption, confidence, ...}
  model_manager.locate_object(img, e)  → dict {raw_output, bbox, ...}
  model_manager.detect_objects(...)    → dict {answer, bboxes, ...}
  model_manager.stats                  → dict {queries, errors}
  model_manager.get_status()           → dict

Key difference from the old model.py:
  - Methods accept `image_base64: str` (plain b64 string without data: prefix)
    instead of `image: PIL.Image`. The PIL image argument is still accepted
    but silently converted to base64 for backward compatibility.
"""

import base64
import io
import threading

# ── OpenRouter Gemma 4 client ───────────────────────────────────────────────
from backend.openrouter_client import gemma_client, GemmaVQAClient


# ── Optional: convert PIL Image → base64 string (for backward compat) ───────
def _pil_to_b64(image) -> str:
    """Convert a PIL.Image to a plain base64 JPEG string (no data: prefix)."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _ensure_b64(image_or_b64) -> str:
    """
    Accept either:
      - a plain base64 string (new convention)
      - a PIL Image object (old convention — convert on-the-fly)
    Returns a plain base64 string.
    """
    if isinstance(image_or_b64, str):
        return image_or_b64
    # Assume PIL Image
    return _pil_to_b64(image_or_b64)


# ─────────────────────────────────────────────────────────────────────────────
# SatQueryModel — compatibility wrapper
# ─────────────────────────────────────────────────────────────────────────────
class SatQueryModel:
    """
    Thread-safe singleton compatibility wrapper around GemmaVQAClient.

    All existing callers (firebase_listener.py, main.py, AI Agents/tools.py)
    can continue using `model_manager.answer_vqa(image, question)` exactly
    as before — this class handles the PIL→base64 conversion transparently.
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
        self._initialized = True
        self._client: GemmaVQAClient = gemma_client
        # Expose _device for any old code that checks it
        self._device = "openrouter"

    # ─── Properties ─────────────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._client.is_loaded

    @property
    def stats(self) -> dict:
        return self._client.stats

    def get_status(self) -> dict:
        return self._client.get_status()

    # ─── Lifecycle ──────────────────────────────────────────────────────────
    def load(self):
        """No-op: Gemma 4 runs on OpenRouter — nothing to download or load."""
        self._client.load()

    # ─── Inference methods ───────────────────────────────────────────────────
    def answer_vqa(self, image_or_b64, question: str) -> dict:
        """VQA: analyze image and answer the question."""
        b64 = _ensure_b64(image_or_b64)
        return self._client.answer_vqa(b64, question)

    def generate_caption(self, image_or_b64) -> dict:
        """Generate a rich remote-sensing image caption."""
        b64 = _ensure_b64(image_or_b64)
        return self._client.generate_caption(b64)

    def locate_object(self, image_or_b64, expression: str) -> dict:
        """Locate an object via a referring expression (textual output)."""
        b64 = _ensure_b64(image_or_b64)
        return self._client.locate_object(b64, expression)

    def detect_objects(self, image_or_b64, object_label: str, count_hint: int = 0) -> dict:
        """Count and describe all instances of object_label in the image."""
        b64 = _ensure_b64(image_or_b64)
        return self._client.detect_objects(b64, object_label, count_hint)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton instance — imported by main.py, firebase_listener.py, tools.py
# ─────────────────────────────────────────────────────────────────────────────
model_manager = SatQueryModel()
