"""
AI Agents/router.py — Model Router
=====================================
Decides whether to use fast CNN models vs Gemma 4 VLM (via OpenRouter)
(or both in hybrid mode) based on the user's query.

Decision logic:
  - Count/detection keywords      → HYBRID (YOLO + Gemma 4 in parallel)
    Because YOLOv8 COCO-80 misses satellite-specific objects (aircraft
    from overhead, ships, etc.). Gemma 4 fills the gap via visual reasoning.
  - Land-cover/density keywords   → CNN segmentation preferred
  - Complex "describe/analyze"    → VLM (Gemma 4 via OpenRouter)
  - Mixed / compound questions    → Hybrid (CNN + VLM in parallel)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class RouteMode(str, Enum):
    CNN_ONLY   = "cnn_only"     # fast CNN only
    VLM_ONLY   = "vlm_only"     # heavy VLM only
    HYBRID     = "hybrid"       # both in parallel
    CAPTION    = "caption"      # pure image description


@dataclass
class RoutingDecision:
    mode: RouteMode
    use_cnn: bool
    use_vlm: bool
    cnn_tasks: list[str]        # subset of ["detect", "segment"]
    vlm_task: str               # "vqa" | "caption" | "detect" | "none"
    reason: str
    detected_keywords: list[str] = field(default_factory=list)


# ─── Keyword Dictionaries ─────────────────────────────────────────────────────

_COUNT_KEYWORDS = {
    "how many", "count", "number of", "total", "enumerate",
    "aircraft", "airplane", "plane", "jet", "helicopter",
    "vehicle", "car", "truck", "bus",
    "ship", "vessel", "boat",
    "building", "structure", "house",
    "person", "people", "pedestrian",
    "tank", "storage tank",
    "bridge",
}

_SEGMENT_KEYWORDS = {
    "vegetation", "forest", "canopy", "tree",
    "water", "river", "lake", "reservoir", "flood",
    "urban", "built-up", "city", "settlement",
    "agricultural", "farmland", "crop", "field",
    "land cover", "land use", "surface",
    "density", "percentage", "percent", "coverage", "how much",
    "area", "fraction",
}

_VLM_KEYWORDS = {
    "describe", "explain", "analyze", "what is", "why",
    "caption", "tell me about", "summarize", "assess",
    "identify", "what do you see", "detail", "feature",
    "change", "compare", "difference", "before", "after",
    "temporal", "multitemporal", "bi-temporal",
    "sar", "radar", "optical", "cross-modal", "multimodal",
}

_MIXED_TRIGGERS = {
    "and",   # "count planes AND describe the area"
    "also",
    "as well",
    "plus",
    "both",
    "with",
}


def route(question: str) -> RoutingDecision:
    """
    Analyse the user question and return a RoutingDecision.

    Args:
        question: User's natural-language question.

    Returns:
        RoutingDecision describing which models to invoke.
    """
    q_lower = question.lower()
    tokens  = set(re.findall(r"[\w'-]+", q_lower))

    # Detect keyword categories present
    has_count   = _any_phrase(q_lower, tokens, _COUNT_KEYWORDS)
    has_segment = _any_phrase(q_lower, tokens, _SEGMENT_KEYWORDS)
    has_vlm     = _any_phrase(q_lower, tokens, _VLM_KEYWORDS)
    has_mixed   = any(t in q_lower for t in _MIXED_TRIGGERS)

    # Collect triggered keywords for transparency
    triggered = []
    if has_count:
        triggered += [k for k in _COUNT_KEYWORDS if k in q_lower]
    if has_segment:
        triggered += [k for k in _SEGMENT_KEYWORDS if k in q_lower]
    if has_vlm:
        triggered += [k for k in _VLM_KEYWORDS if k in q_lower]

    # ── Routing decision tree ─────────────────────────────────────────────────

    # 1. Pure caption / describe → VLM only
    if not has_count and not has_segment and has_vlm:
        # Check if it's a pure caption question
        caption_words = {"describe", "caption", "tell me about", "what do you see"}
        if _any_phrase(q_lower, tokens, caption_words) and not has_mixed:
            return RoutingDecision(
                mode=RouteMode.CAPTION,
                use_cnn=False,
                use_vlm=True,
                cnn_tasks=[],
                vlm_task="caption",
                reason="Pure description question → VLM caption",
                detected_keywords=triggered,
            )
        return RoutingDecision(
            mode=RouteMode.VLM_ONLY,
            use_cnn=False,
            use_vlm=True,
            cnn_tasks=[],
            vlm_task="vqa",
            reason="Complex analytical question → VLM VQA",
            detected_keywords=triggered,
        )

    # 2. Count + segment + VLM compound → Hybrid
    if (has_count or has_segment) and has_vlm:
        cnn_tasks = []
        if has_count:
            cnn_tasks.append("detect")
        if has_segment:
            cnn_tasks.append("segment")
        return RoutingDecision(
            mode=RouteMode.HYBRID,
            use_cnn=True,
            use_vlm=True,
            cnn_tasks=cnn_tasks,
            vlm_task="vqa",
            reason="Mixed query with counting/segmentation + analysis → Hybrid (CNN + VLM parallel)",
            detected_keywords=triggered,
        )

    # 3. Count + segment (no VLM) → Hybrid CNN
    if has_count and has_segment:
        return RoutingDecision(
            mode=RouteMode.CNN_ONLY,
            use_cnn=True,
            use_vlm=False,
            cnn_tasks=["detect", "segment"],
            vlm_task="none",
            reason="Counting + land-cover query → CNN only (detect + segment in parallel)",
            detected_keywords=triggered,
        )

    # 4. Count only → HYBRID (YOLO + Gemma 4 in parallel)
    # We always include VLM for counting because YOLOv8 (trained on COCO-80
    # ground-level images) misses many satellite objects like overhead aircraft,
    # ships, storage tanks. Gemma 4's visual reasoning is far more reliable here.
    if has_count and not has_segment:
        return RoutingDecision(
            mode=RouteMode.HYBRID,
            use_cnn=True,
            use_vlm=True,
            cnn_tasks=["detect"],
            vlm_task="vqa",
            reason="Object counting query → YOLO + Gemma 4 hybrid (Gemma 4 covers satellite-specific objects YOLO misses)",
            detected_keywords=triggered,
        )

    # 5. Segment only → CNN segment
    if has_segment and not has_count and not has_vlm:
        return RoutingDecision(
            mode=RouteMode.CNN_ONLY,
            use_cnn=True,
            use_vlm=False,
            cnn_tasks=["segment"],
            vlm_task="none",
            reason="Land-cover/density query → ResNet/FCN segmentation",
            detected_keywords=triggered,
        )

    # 6. Default fallback — Hybrid (safest: run everything)
    return RoutingDecision(
        mode=RouteMode.HYBRID,
        use_cnn=True,
        use_vlm=True,
        cnn_tasks=["detect", "segment"],
        vlm_task="vqa",
        reason="General query — running full hybrid pipeline for comprehensive analysis",
        detected_keywords=triggered,
    )


def _any_phrase(text: str, tokens: set, keywords: set) -> bool:
    """Return True if any keyword (phrase or single token) appears in text."""
    for kw in keywords:
        if " " in kw:
            if kw in text:
                return True
        else:
            if kw in tokens:
                return True
    return False
