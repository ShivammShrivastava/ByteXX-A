"""
AI Agents/executor.py — Parallel Step Executor
================================================
Executes an ExecutionPlan, running independent steps concurrently
and sequential steps in order.

Concurrency model:
  - Steps within the same group_id run in parallel via asyncio.gather
  - Groups run sequentially (group 0 → group 1 → ...)
  - Results from earlier groups are available when later groups run

Returns a list of ToolResult objects, one per step.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from PIL import Image

from AI_Agents.planner import ExecutionPlan, Step
from AI_Agents.tools import ToolResult, call_tool

logger = logging.getLogger(__name__)


class StepExecutionRecord:
    """Tracks execution metadata for a single step."""

    def __init__(self, step: Step):
        self.step        = step
        self.result: Optional[ToolResult] = None
        self.started_at: Optional[float]  = None
        self.finished_at: Optional[float] = None
        self.status: str = "pending"   # pending | running | done | error

    @property
    def duration_ms(self) -> float:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at) * 1000
        return 0.0

    def to_dict(self) -> dict:
        return {
            "step_index":    self.step.index,
            "tool":          self.step.tool,
            "description":   self.step.description,
            "parallel":      self.step.parallel,
            "group_id":      self.step.group_id,
            "status":        self.status,
            "duration_ms":   round(self.duration_ms, 1),
            "confidence":    self.result.confidence if self.result else None,
            "success":       self.result.success if self.result else False,
            "error":         self.result.error if self.result else None,
        }


class Executor:
    """
    Parallel step executor for agent execution plans.

    Executes steps group by group, with all steps in each group
    running concurrently via asyncio.gather.
    """

    async def execute(
        self,
        plan: ExecutionPlan,
        image: Image.Image,
    ) -> list[StepExecutionRecord]:
        """
        Execute all steps in the plan.

        Args:
            plan:  ExecutionPlan from planner.plan().
            image: PIL image to pass to each tool.

        Returns:
            List of StepExecutionRecord, one per step, in execution order.
        """
        all_records: dict[int, StepExecutionRecord] = {
            step.index: StepExecutionRecord(step)
            for step in plan.steps
        }

        t_total = time.perf_counter()
        groups  = plan.parallel_groups()

        for group in groups:
            if len(group) == 1:
                # Single step — run directly
                rec = all_records[group[0].index]
                await self._run_step(rec, image)
            else:
                # Multiple steps — run concurrently
                await asyncio.gather(*[
                    self._run_step(all_records[step.index], image)
                    for step in group
                ])

        total_ms = (time.perf_counter() - t_total) * 1000
        logger.info(
            f"Executor finished {len(plan.steps)} step(s) "
            f"in {total_ms:.0f}ms across {len(groups)} group(s)."
        )
        # Return in original step order
        return [all_records[step.index] for step in plan.steps]

    @staticmethod
    async def _run_step(record: StepExecutionRecord, image: Image.Image) -> None:
        """Execute a single step and update its record."""
        record.status     = "running"
        record.started_at = time.perf_counter()

        logger.info(
            f"  [{record.step.group_id}:{record.step.index}] "
            f"Running {record.step.tool} — {record.step.description[:60]}"
        )

        try:
            result = await call_tool(record.step.tool, image, record.step.args)
            record.result      = result
            record.status      = "done" if result.success else "error"
        except Exception as e:
            logger.exception(f"Step {record.step.index} ({record.step.tool}) raised exception")
            from AI_Agents.tools import ToolResult
            record.result = ToolResult(
                tool_name=record.step.tool,
                result={},
                confidence=0.0,
                duration_ms=0.0,
                success=False,
                error=str(e),
            )
            record.status = "error"
        finally:
            record.finished_at = time.perf_counter()

        logger.info(
            f"  [{record.step.group_id}:{record.step.index}] "
            f"{record.step.tool} → {record.status} "
            f"({record.duration_ms:.0f}ms, conf={record.result.confidence:.3f})"
        )


# Module-level singleton
executor = Executor()
