"""
AI Agents/agent.py — SatAgent Orchestrator
============================================
Top-level agent that wires together all components:
  Router → Planner → Executor (parallel) → Verifier → Merger

Usage:
    from AI_Agents.agent import sat_agent
    report = await sat_agent.run(image, question)

Full pipeline:
  1. route(question)          → RoutingDecision
  2. plan(question, route)    → ExecutionPlan
  3. executor.execute(plan)   → list[StepExecutionRecord]
  4. verifier.verify(records) → (updated_records, verification_log)
  5. merger.merge(...)        → StructuredReport
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# ─── Ensure project root is on sys.path ──────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class SatAgent:
    """
    Satellite Image Analysis Agent.

    Orchestrates the full analysis pipeline:
      Route → Plan → Execute (parallel) → Verify → Merge → Report
    """

    async def run(
        self,
        image: Image.Image,
        question: str,
        conf_threshold: float = 0.25,
    ) -> "StructuredReport":
        """
        Run the full agent pipeline on an image + question.

        Args:
            image:          PIL RGB image to analyse.
            question:       User's natural-language question.
            conf_threshold: YOLOv8 detection confidence threshold.

        Returns:
            StructuredReport — fully merged result from all tools.
        """
        from AI_Agents.router import route
        from AI_Agents.planner import plan
        from AI_Agents.executor import executor
        from AI_Agents.verifier import verifier
        from AI_Agents.merger import merger, StructuredReport

        t_start = time.perf_counter()

        logger.info("=" * 60)
        logger.info(f"SatAgent.run — question: {question[:80]}")

        # ── Step 1: Route ─────────────────────────────────────────────────────
        logger.info("[1/5] Routing...")
        routing = route(question)
        logger.info(
            f"  Mode: {routing.mode.value} | "
            f"CNN: {routing.use_cnn} ({routing.cnn_tasks}) | "
            f"VLM: {routing.use_vlm} ({routing.vlm_task})"
        )
        logger.info(f"  Reason: {routing.reason}")

        # ── Step 2: Plan ──────────────────────────────────────────────────────
        logger.info("[2/5] Planning...")
        execution_plan = plan(question, routing)
        for step in execution_plan.steps:
            logger.info(
                f"  Step {step.index} [group={step.group_id}]: "
                f"{step.tool} — {step.description[:60]}"
            )

        if not execution_plan.steps:
            # Edge case: no tools to run
            logger.warning("  No steps generated — returning empty report.")
            return StructuredReport(
                summary="No analysis steps could be generated for this query.",
                routing_mode=routing.mode.value,
                duration_ms=round((time.perf_counter() - t_start) * 1000, 1),
            )

        # ── Step 3: Execute (parallel within groups) ──────────────────────────
        logger.info(f"[3/5] Executing {len(execution_plan.steps)} step(s)...")
        records = await executor.execute(execution_plan, image)

        # ── Step 4: Verify ────────────────────────────────────────────────────
        logger.info("[4/5] Verifying confidence...")
        records, verification_log = await verifier.verify(records, image, question)

        for vlog in verification_log:
            if vlog.triggered:
                logger.info(
                    f"  Verified step {vlog.original_step_index}: "
                    f"{vlog.verification_note}"
                )

        # ── Step 5: Merge ─────────────────────────────────────────────────────
        logger.info("[5/5] Merging results...")
        total_ms = (time.perf_counter() - t_start) * 1000
        report = merger.merge(execution_plan, records, verification_log, total_ms)

        logger.info(
            f"SatAgent done — "
            f"confidence={report.confidence:.3f}, "
            f"tools={report.tools_used}, "
            f"duration={total_ms:.0f}ms"
        )
        logger.info("=" * 60)

        return report

    def run_sync(
        self,
        image: Image.Image,
        question: str,
        conf_threshold: float = 0.25,
    ) -> "StructuredReport":
        """Synchronous wrapper for use outside async contexts (e.g. tests)."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run, self.run(image, question, conf_threshold)
                    )
                    return future.result()
            return loop.run_until_complete(self.run(image, question, conf_threshold))
        except RuntimeError:
            return asyncio.run(self.run(image, question, conf_threshold))


# Module-level singleton
sat_agent = SatAgent()
