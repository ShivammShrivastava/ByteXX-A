"""
AI Agents/planner.py — Query Decomposer / Planner
===================================================
Breaks a complex user question into an ordered list of
execution steps, each mapped to a specific tool call.

Example:
  Input:  "How many planes are here, and what's the vegetation coverage
           and water area? Also describe the overall scene."
  Output: [
    Step(tool="cnn_detect",  args={"classes": ["aircraft"]}, parallel=True),
    Step(tool="cnn_segment", args={},                        parallel=True),
    Step(tool="vlm_vqa",     args={"question": "..."},       parallel=False, depends_on=[0,1]),
  ]

Steps with parallel=True and the same group_id can run concurrently.
Steps with depends_on must run after all listed step indices complete.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from AI_Agents.router import RoutingDecision, RouteMode


@dataclass
class Step:
    """A single executable step in the plan."""
    index: int
    tool: str                          # Tool name (see tools.py)
    args: dict                         # Tool arguments
    description: str                   # Human-readable description
    parallel: bool = True              # Can run in parallel with sibling steps
    depends_on: list[int] = field(default_factory=list)   # Step indices that must complete first
    group_id: int = 0                  # Steps in the same group run concurrently


@dataclass
class ExecutionPlan:
    """Ordered, optionally parallel execution plan."""
    steps: list[Step]
    original_question: str
    routing_decision: RoutingDecision
    summary: str = ""

    def parallel_groups(self) -> list[list[Step]]:
        """
        Group steps by group_id. Each group runs sequentially,
        but steps within a group run concurrently.
        """
        groups: dict[int, list[Step]] = {}
        for step in self.steps:
            groups.setdefault(step.group_id, []).append(step)
        return [groups[k] for k in sorted(groups.keys())]


# ─── Object classes that YOLO can detect for counting ─────────────────────────
_DETECTABLE_OBJECTS = {
    "aircraft":       ["aircraft", "airplane", "plane", "jet", "helicopter"],
    "vehicle":        ["vehicle", "car", "automobile"],
    "truck":          ["truck"],
    "vessel":         ["ship", "vessel", "boat"],
    "building":       ["building", "structure", "house"],
    "person":         ["person", "people", "pedestrian"],
    "storage tank":   ["tank", "storage tank"],
    "bridge":         ["bridge"],
}


def plan(question: str, routing: RoutingDecision) -> ExecutionPlan:
    """
    Build an execution plan from a question + routing decision.

    Args:
        question: User's natural-language question.
        routing:  RoutingDecision from router.route().

    Returns:
        ExecutionPlan with ordered, optionally parallel steps.
    """
    q_lower   = question.lower()
    steps: list[Step] = []
    idx = 0

    # ── Group 0: Parallel fast tasks (CNN) ───────────────────────────────────
    if routing.use_cnn:
        if "detect" in routing.cnn_tasks:
            # Detect which specific object classes are mentioned
            target_classes = _extract_object_classes(q_lower)
            steps.append(Step(
                index=idx,
                tool="cnn_detect",
                args={"target_classes": target_classes, "conf_threshold": 0.25},
                description=f"YOLOv8 detection"
                            + (f" — targets: {', '.join(target_classes)}" if target_classes else ""),
                parallel=True,
                group_id=0,
            ))
            idx += 1

        if "segment" in routing.cnn_tasks:
            steps.append(Step(
                index=idx,
                tool="cnn_segment",
                args={},
                description="ResNet/FCN land-cover segmentation (Vegetation · Water · Urban · Agricultural)",
                parallel=True,
                group_id=0,
            ))
            idx += 1

    # ── Group 1: VLM tasks (may depend on CNN group) ─────────────────────────
    if routing.use_vlm:
        cnn_deps = [s.index for s in steps]   # VLM step depends on CNN group

        if routing.vlm_task == "caption":
            steps.append(Step(
                index=idx,
                tool="vlm_caption",
                args={},
                description="Qwen2.5-VL image captioning",
                parallel=False,
                depends_on=cnn_deps,
                group_id=1,
            ))
            idx += 1

        elif routing.vlm_task == "vqa":
            # Build a focused VLM question that complements CNN results
            focused_q = _build_vlm_question(question, routing)
            steps.append(Step(
                index=idx,
                tool="vlm_vqa",
                args={"question": focused_q},
                description=f"Qwen2.5-VL VQA — \"{focused_q[:60]}{'...' if len(focused_q)>60 else ''}\"",
                parallel=False,
                depends_on=cnn_deps,
                group_id=1,
            ))
            idx += 1

        elif routing.vlm_task == "detect":
            # Use VLM for fine-grained detection if CNN isn't available
            steps.append(Step(
                index=idx,
                tool="vlm_detect",
                args={"object_label": _primary_object(q_lower)},
                description="Qwen2.5-VL object detection + bounding boxes",
                parallel=False,
                depends_on=cnn_deps,
                group_id=1,
            ))
            idx += 1

    summary_parts = []
    if routing.use_cnn:
        summary_parts.append(f"CNN ({', '.join(routing.cnn_tasks)})")
    if routing.use_vlm:
        summary_parts.append(f"VLM ({routing.vlm_task})")

    return ExecutionPlan(
        steps=steps,
        original_question=question,
        routing_decision=routing,
        summary=f"Plan: {' → '.join(summary_parts)} | {len(steps)} step(s) | "
                + f"Mode: {routing.mode.value}",
    )


def _extract_object_classes(q_lower: str) -> list[str]:
    """Extract which object classes are mentioned in the question."""
    found = []
    for class_key, synonyms in _DETECTABLE_OBJECTS.items():
        if any(syn in q_lower for syn in synonyms):
            found.append(class_key)
    return found  # empty = detect everything YOLO finds


def _primary_object(q_lower: str) -> str:
    """Extract the primary object of interest for VLM detection."""
    for class_key, synonyms in _DETECTABLE_OBJECTS.items():
        if any(syn in q_lower for syn in synonyms):
            return class_key
    return "object"


def _build_vlm_question(original: str, routing: RoutingDecision) -> str:
    """
    Build a focused VLM question that is complementary to the CNN tasks.
    If CNN is also running, steer VLM toward qualitative/contextual analysis.
    """
    if not routing.use_cnn:
        return original

    # CNN handles counting & percentages → ask VLM for scene context
    return (
        f"{original}\n\n"
        "Note: Object counts and land-cover percentages are being computed "
        "separately by CNN models. Please focus on spatial context, scene "
        "characteristics, quality, and any anomalies you observe in the image."
    )
