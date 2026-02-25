"""Core pass executor for loop-step multi-pass orchestration."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Awaitable, Callable, Dict, List, Optional

from chorus_engine.ens.loop_plugins.contracts import (
    PassExecutionResult,
    StepExecutionAggregate,
    StepOutcomeResolution,
    StepPassPlan,
)

RunPassFn = Callable[[StepPassPlan, Optional[Dict[str, Any]]], Awaitable[PassExecutionResult]]
ResolvePassOutcomeFn = Callable[[StepPassPlan, PassExecutionResult, List[PassExecutionResult]], Awaitable[Optional[StepOutcomeResolution]]]
LoopbackFn = Callable[[StepPassPlan, PassExecutionResult], Awaitable[Optional[Dict[str, Any]]]]


async def execute_step_passes(
    *,
    pass_plans: List[StepPassPlan],
    run_pass: RunPassFn,
    resolve_outcome: Optional[ResolvePassOutcomeFn] = None,
    execute_loopback: Optional[LoopbackFn] = None,
    max_passes_per_step: int = 4,
    allow_single_tool_loopback: bool = True,
) -> StepExecutionAggregate:
    if not pass_plans:
        return StepExecutionAggregate(
            pass_results=[],
            final_visible_text="",
            visible_pass_id=None,
            final_outcome_action="WAIT_FOR_USER",
            final_control_source="default_wait",
            defaulted_wait=True,
        )

    effective_max = max(1, int(max_passes_per_step or 1))
    pass_results: List[PassExecutionResult] = []
    final_visible_text = ""
    visible_pass_id: Optional[str] = None
    final_outcome_action: Optional[str] = None
    final_control_source: Optional[str] = None
    defaulted_wait = False

    for index, pass_plan in enumerate(pass_plans):
        if index >= effective_max:
            pass_results.append(
                PassExecutionResult(
                    pass_id=pass_plan.pass_id,
                    status="failed",
                    output_text="",
                    assistant_result_tier=None,
                    finish_reason=None,
                    tool_calls_count=0,
                    error="max_passes_per_step_exceeded",
                    timing_ms=0,
                    metadata={"guardrail": "max_passes_per_step"},
                )
            )
            final_outcome_action = "WAIT_FOR_USER"
            final_control_source = "default_wait"
            defaulted_wait = True
            break

        started = time.perf_counter()
        result = await run_pass(pass_plan, None)
        result.timing_ms = int((time.perf_counter() - started) * 1000)

        if (
            allow_single_tool_loopback
            and pass_plan.allow_single_tool_loopback
            and execute_loopback is not None
            and result.status == "success"
            and int(result.tool_calls_count or 0) > 0
        ):
            loopback_payload = await execute_loopback(pass_plan, result)
            if loopback_payload is not None:
                result.loopback_invoked = True
                started = time.perf_counter()
                rerun_result = await run_pass(pass_plan, loopback_payload)
                rerun_result.loopback_invoked = True
                rerun_result.timing_ms = int((time.perf_counter() - started) * 1000)
                result = rerun_result

        pass_results.append(result)

        if pass_plan.emit_to_user and result.status == "success":
            final_visible_text = str(result.output_text or "")
            visible_pass_id = pass_plan.pass_id

        if result.status != "success":
            final_outcome_action = "WAIT_FOR_USER"
            final_control_source = "default_wait"
            defaulted_wait = True
            break

        if pass_plan.parse_strategy == "outcome_ladder" and resolve_outcome is not None:
            resolution = await resolve_outcome(pass_plan, result, pass_results)
            if resolution is not None:
                result.metadata["outcome_resolution"] = asdict(resolution)
                final_outcome_action = str(resolution.action or "").strip().upper() or "WAIT_FOR_USER"
                final_control_source = str(resolution.source_stage or "default_wait")
                defaulted_wait = bool(resolution.defaulted_wait)

    if not final_outcome_action:
        final_outcome_action = "WAIT_FOR_USER"
        final_control_source = final_control_source or "default_wait"
        defaulted_wait = True

    return StepExecutionAggregate(
        pass_results=pass_results,
        final_visible_text=final_visible_text,
        visible_pass_id=visible_pass_id,
        final_outcome_action=final_outcome_action,
        final_control_source=final_control_source,
        defaulted_wait=defaulted_wait,
    )
