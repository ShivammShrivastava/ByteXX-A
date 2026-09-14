"""
AI Agents/merger.py — Result Merger
=====================================
Combines all tool outputs from the executor into a single
StructuredReport — merging numeric stats, bounding boxes,
land cover percentages, and natural language into one coherent response.

Output schema:
{
  "summary":             str,           # Natural language paragraph
  "object_counts":       dict,          # {"aircraft": 5, "vehicle": 12}
  "total_detections":    int,
  "bboxes":              list,          # [[x1,y1,x2,y2], ...] normalized 0-1
  "bbox_labels":         list,
  "bbox_confidences":    list,
  "annotated_image_b64": str | None,
  "land_cover":          dict,          # {"vegetation": 42.1, ...} raw keys
  "land_cover_display":  dict,          # {"Vegetation / Forest Canopy": 42.1, ...}
  "dominant_land_cover": str,
  "mask_image_b64":      str | None,
  "vlm_answer":          str | None,
  "caption":             str | None,
  "confidence":          float,
  "tools_used":          list[str],
  "routing_mode":        str,
  "execution_steps":     list[dict],
  "verification_log":    list[dict],
  "reasoning_trace":     str,
  "errors":              list[str],
  "duration_ms":         float,
}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from AI_Agents.executor import StepExecutionRecord
from AI_Agents.verifier import VerificationRecord
from AI_Agents.planner import ExecutionPlan

logger = logging.getLogger(__name__)


@dataclass
class StructuredReport:
    """Complete structured analysis report from the AI Agent."""

    # Natural language
    summary: str = ""
    vlm_answer: Optional[str] = None
    caption: Optional[str] = None

    # Detection
    object_counts: dict[str, int] = field(default_factory=dict)
    total_detections: int = 0
    bboxes: list[list[float]] = field(default_factory=list)
    bbox_labels: list[str] = field(default_factory=list)
    bbox_confidences: list[float] = field(default_factory=list)
    annotated_image_b64: Optional[str] = None

    # Segmentation
    land_cover: dict[str, float] = field(default_factory=dict)
    land_cover_display: dict[str, float] = field(default_factory=dict)
    dominant_land_cover: str = "unknown"
    mask_image_b64: Optional[str] = None

    # Meta
    confidence: float = 0.0
    tools_used: list[str] = field(default_factory=list)
    routing_mode: str = "unknown"
    execution_steps: list[dict] = field(default_factory=list)
    verification_log: list[dict] = field(default_factory=list)
    reasoning_trace: str = ""
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "summary":             self.summary,
            "vlm_answer":          self.vlm_answer,
            "caption":             self.caption,
            "object_counts":       self.object_counts,
            "total_detections":    self.total_detections,
            "bboxes":              self.bboxes,
            "bbox_labels":         self.bbox_labels,
            "bbox_confidences":    self.bbox_confidences,
            "annotated_image_b64": self.annotated_image_b64,
            "land_cover":          self.land_cover,
            "land_cover_display":  self.land_cover_display,
            "dominant_land_cover": self.dominant_land_cover,
            "mask_image_b64":      self.mask_image_b64,
            "confidence":          self.confidence,
            "tools_used":          self.tools_used,
            "routing_mode":        self.routing_mode,
            "execution_steps":     self.execution_steps,
            "verification_log":    self.verification_log,
            "reasoning_trace":     self.reasoning_trace,
            "errors":              self.errors,
            "duration_ms":         self.duration_ms,
        }


class Merger:
    """
    Merges all tool results into a single StructuredReport.

    Combines:
      - Detection counts + bboxes from cnn_detect / vlm_detect
      - Land-cover percentages from cnn_segment
      - Natural language answers from vlm_vqa / vlm_caption
      - Execution + verification metadata
    """

    def merge(
        self,
        plan: ExecutionPlan,
        records: list[StepExecutionRecord],
        verification_log: list[VerificationRecord],
        total_duration_ms: float,
    ) -> StructuredReport:
        """
        Merge all outputs into a StructuredReport.

        Args:
            plan:              The original execution plan.
            records:           Executed step records (may have been updated by verifier).
            verification_log:  Verification records from the verifier.
            total_duration_ms: Total wall-clock time for the full pipeline.

        Returns:
            StructuredReport with all fields populated.
        """
        report = StructuredReport(
            routing_mode=plan.routing_decision.mode.value,
            duration_ms=round(total_duration_ms, 1),
        )

        # ── Collect data from each step ───────────────────────────────────────
        conf_values: list[float] = []
        trace_lines: list[str]  = [
            f"Query: {plan.original_question}",
            f"Route: {plan.routing_decision.reason}",
            f"Plan:  {plan.summary}",
            "",
        ]

        for rec in records:
            if not rec.result:
                continue

            tool = rec.result.tool_name
            r    = rec.result.result
            conf = rec.result.confidence

            if rec.result.success and conf > 0:
                conf_values.append(conf)
                report.tools_used.append(tool)

            trace_lines.append(
                f"[{rec.step.group_id}:{rec.step.index}] {tool} "
                f"→ {'✓' if rec.result.success else '✗'} "
                f"conf={conf:.3f} ({rec.duration_ms:.0f}ms)"
            )

            if not rec.result.success:
                if rec.result.error:
                    report.errors.append(f"{tool}: {rec.result.error}")
                continue

            # ── Detection (YOLO or VLM or Gemma 4 fallback) ──────────────────
            if tool in ("cnn_detect", "vlm_detect"):
                if "object_counts" in r:
                    for k, v in r["object_counts"].items():
                        report.object_counts[k] = report.object_counts.get(k, 0) + v
                if "total_detections" in r:
                    report.total_detections += r["total_detections"]
                if "bboxes" in r:
                    report.bboxes.extend(r["bboxes"])
                    labels = r.get("bbox_labels", ["Object"] * len(r["bboxes"]))
                    confs  = r.get("bbox_confidences", [conf] * len(r["bboxes"]))
                    report.bbox_labels.extend(labels)
                    report.bbox_confidences.extend(confs)
                if r.get("annotated_image_b64") and not report.annotated_image_b64:
                    report.annotated_image_b64 = r["annotated_image_b64"]
                # Promote any natural-language answer to vlm_answer:
                # - vlm_detect always has an "answer"
                # - cnn_detect has an "answer" when Gemma 4 fallback fired
                if "answer" in r and r["answer"] and not report.vlm_answer:
                    report.vlm_answer = r["answer"]

            # ── Segmentation ─────────────────────────────────────────────────
            elif tool == "cnn_segment":
                if "land_cover_percentages" in r:
                    report.land_cover = r["land_cover_percentages"]
                if "land_cover_display" in r:
                    report.land_cover_display = r["land_cover_display"]
                if "dominant_class" in r:
                    report.dominant_land_cover = r["dominant_class"]
                if r.get("mask_image_b64") and not report.mask_image_b64:
                    report.mask_image_b64 = r["mask_image_b64"]

            # ── VLM VQA ──────────────────────────────────────────────────────
            elif tool == "vlm_vqa":
                report.vlm_answer = r.get("answer", "")

            # ── VLM Caption ──────────────────────────────────────────────────
            elif tool == "vlm_caption":
                report.caption = r.get("caption", "")

        # ── Build execution step dicts ────────────────────────────────────────
        report.execution_steps = [rec.to_dict() for rec in records]

        # ── Build verification log ────────────────────────────────────────────
        report.verification_log = [vl.to_dict() for vl in verification_log]

        # ── Compute overall confidence ────────────────────────────────────────
        report.confidence = round(
            sum(conf_values) / len(conf_values) if conf_values else 0.0, 3
        )

        # ── Deduplicate tools_used ────────────────────────────────────────────
        report.tools_used = list(dict.fromkeys(report.tools_used))

        # ── Build reasoning trace ─────────────────────────────────────────────
        report.reasoning_trace = "\n".join(trace_lines)

        # ── Synthesize natural language summary ───────────────────────────────
        report.summary = self._synthesize_summary(report, plan.original_question)

        logger.info(
            f"Merger: report built — "
            f"detections={report.total_detections}, "
            f"land_cover_classes={len(report.land_cover)}, "
            f"tools={report.tools_used}, "
            f"conf={report.confidence:.3f}"
        )
        return report

    @staticmethod
    def _synthesize_summary(report: StructuredReport, question: str) -> str:
        """
        Synthesize a natural-language summary paragraph combining
        all result streams.
        """
        parts: list[str] = []

        # Object detection summary
        if report.object_counts:
            count_parts = []
            for obj, cnt in sorted(report.object_counts.items(), key=lambda x: -x[1]):
                count_parts.append(f"**{cnt} {obj}{'s' if cnt != 1 else ''}**")
            parts.append(
                "**Object Detection:** "
                + f"Identified {', '.join(count_parts)} "
                + f"({report.total_detections} total detections)."
            )
        elif report.total_detections == 0 and any(
            t in report.tools_used for t in ("cnn_detect", "vlm_detect")
        ):
            parts.append("**Object Detection:** No objects detected above confidence threshold."
                         " See Gemma 4 Vision analysis below for details.")

        # Land cover summary
        if report.land_cover:
            sorted_lc = sorted(report.land_cover_display.items(), key=lambda x: -x[1])
            lc_parts = [f"**{name}**: {pct:.1f}%" for name, pct in sorted_lc]
            dom_name = report.land_cover_display.get(
                report.dominant_land_cover, report.dominant_land_cover
            )
            parts.append(
                "**Land Cover (ResNet/FCN):** "
                + " | ".join(lc_parts)
                + f". Dominant cover: **{dom_name}**."
            )

        # VLM natural language answer
        if report.vlm_answer:
            parts.append(f"**Scene Analysis (Gemma 4 Vision):** {report.vlm_answer}")
        elif report.caption:
            parts.append(f"**Image Caption (Gemma 4 Vision):** {report.caption}")

        # Uncertainty notice
        uncertain_steps = [
            s for s in report.execution_steps
            if not s.get("success") or (s.get("confidence") or 1.0) < 0.45
        ]
        if uncertain_steps:
            parts.append(
                f"⚠️ **Note:** {len(uncertain_steps)} step(s) had low confidence "
                "and were re-analysed. Results should be interpreted with caution."
            )

        if report.errors:
            parts.append(f"❌ **Errors:** {'; '.join(report.errors)}")

        return "\n\n".join(parts) if parts else "Analysis complete."


# Module-level singleton
merger = Merger()
