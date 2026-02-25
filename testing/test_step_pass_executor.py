from __future__ import annotations

import pytest

from chorus_engine.ens.loop_plugins.contracts import PassExecutionResult, StepPassPlan
from chorus_engine.ens.step_execution.pass_executor import execute_step_passes


@pytest.mark.asyncio
async def test_pass_executor_selects_visible_output_and_outcome_resolution():
    plans = [
        StepPassPlan(pass_id="p1", kind="primary_generation", emit_to_user=True),
        StepPassPlan(pass_id="p2", kind="outcome_resolution", parse_strategy="outcome_ladder"),
    ]

    async def run_pass(plan: StepPassPlan, _loopback):
        if plan.pass_id == "p1":
            return PassExecutionResult(
                pass_id=plan.pass_id,
                status="success",
                output_text="beat text",
                assistant_result_tier="tier1",
                finish_reason="stop",
                tool_calls_count=0,
            )
        return PassExecutionResult(
            pass_id=plan.pass_id,
            status="success",
            output_text="",
            assistant_result_tier="tier1",
            finish_reason="stop",
            tool_calls_count=0,
        )

    async def resolve_outcome(_plan, _result, _prior):
        from chorus_engine.ens.loop_plugins.contracts import StepOutcomeResolution

        return StepOutcomeResolution(
            action="CONTINUE",
            ladder_rung_selected="native",
            defaulted_wait=False,
            rung1_native={},
            rung2_parse={},
            rung3_json_schema={},
            source_stage="native",
        )

    aggregate = await execute_step_passes(
        pass_plans=plans,
        run_pass=run_pass,
        resolve_outcome=resolve_outcome,
    )

    assert aggregate.final_visible_text == "beat text"
    assert aggregate.visible_pass_id == "p1"
    assert aggregate.final_outcome_action == "CONTINUE"
    assert aggregate.defaulted_wait is False
    assert len(aggregate.pass_results) == 2
    assert aggregate.pass_results[1].metadata["outcome_resolution"]["action"] == "CONTINUE"


@pytest.mark.asyncio
async def test_pass_executor_fail_closed_wait_on_pass_failure():
    plans = [
        StepPassPlan(pass_id="p1", kind="primary_generation", emit_to_user=True),
        StepPassPlan(pass_id="p2", kind="outcome_resolution", parse_strategy="outcome_ladder"),
    ]

    async def run_pass(plan: StepPassPlan, _loopback):
        if plan.pass_id == "p1":
            return PassExecutionResult(
                pass_id=plan.pass_id,
                status="success",
                output_text="beat text",
                assistant_result_tier="tier1",
                finish_reason="stop",
                tool_calls_count=0,
            )
        return PassExecutionResult(
            pass_id=plan.pass_id,
            status="failed",
            output_text="",
            assistant_result_tier=None,
            finish_reason=None,
            tool_calls_count=0,
            error="boom",
        )

    aggregate = await execute_step_passes(
        pass_plans=plans,
        run_pass=run_pass,
        resolve_outcome=None,
    )

    assert aggregate.final_outcome_action == "WAIT_FOR_USER"
    assert aggregate.defaulted_wait is True
    assert len(aggregate.pass_results) == 2
    assert aggregate.pass_results[1].status == "failed"
