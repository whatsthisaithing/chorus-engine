# Agentic Loop Developer's Guide

This guide is the implementation-level reference for agentic loop behavior in Chorus ENS.

It focuses on loop progression architecture, plugin boundaries, multi-pass step execution, control resolution, diagnostics, and operational guardrails.

## Scope and Philosophy

ENS owns generic loop runtime primitives:

- loop session lifecycle
- scheduler/queue progression
- deterministic step execution scaffolding
- pass orchestration
- state transitions
- persistence and diagnostics

Loop kinds (for example `narrative.v1`) own loop-specific policy:

- prompt addenda
- pass plans
- outcome message construction
- normalization semantics
- loop-specific control policy

This separation is intentional: loop kinds extend ENS capabilities rather than being hard-baked into core orchestration.

## Core Files

Runtime and orchestration:

- `chorus_engine/ens/dispatcher.py`
- `chorus_engine/ens/scheduler.py`
- `chorus_engine/ens/runtime.py`

Loop plugin contracts/registry:

- `chorus_engine/ens/loop_plugins/contracts.py`
- `chorus_engine/ens/loop_plugins/registry.py`
- `chorus_engine/ens/loop_plugins/narrative_v1.py`

Pass execution and outcome resolution:

- `chorus_engine/ens/step_execution/pass_executor.py`
- `chorus_engine/ens/control_resolution/ladders.py`

Invocation and transport:

- `chorus_engine/ens/llm_invocation_service.py`
- `chorus_engine/ens/tool_registry.py`
- provider adapters under `chorus_engine/llm/`

Persistence models:

- `chorus_engine/models/ens.py`

## Data Model Overview

Loop session:

- table/model: `ENSLoopSession`
- carries `loop_id`, `loop_kind`, `state`, `step_index`, `step_count`, budgets, stop reason

Loop step event:

- table/model: `ENSLoopStepEvent`
- one durable record per completed loop step execution
- includes final `control_action`
- `output_json` includes detailed diagnostics and pass traces

Signal queue:

- table/model: `ENSSignalQueue`
- queues `loop_progression` signals
- dedupe and idempotency behavior handled in scheduler enqueue path

## Plugin Contract and Responsibilities

Main contract: `LoopKindPlugin` in `loop_plugins/contracts.py`

Primary responsibilities:

- `build_step_plan(...)`: build loop policy and pass plan
- `build_outcome_messages(...)`: control/outcome pass prompt construction
- `build_outcome_retry_messages(...)`: JSON-schema retry messages
- `outcome_json_schema_response_format(...)`: retry schema contract
- `normalize_step_outcome(...)`: map plugin semantics into runtime control actions
- `post_step_hooks(...)`: optional plugin-specific side effects

Important types:

- `LoopStepPlan`
- `StepPassPlan`
- `StepOutcomePolicy`
- `PassExecutionResult`
- `StepExecutionAggregate`

## Multi-Pass Step Execution

Executor: `execute_step_passes(...)` in `step_execution/pass_executor.py`

Behavior:

- Executes `StepPassPlan[]` in order.
- Supports optional single tool loopback per pass (guarded/capped).
- Selects visible output from pass(es) with `emit_to_user=true`.
- Resolves final step outcome from pass parse strategy (for now: `outcome_ladder`).
- Fail-closed to `WAIT_FOR_USER` on pass failure/guardrail failure.

Guardrails:

- `ens.loop_step_max_passes_per_step` (default `4`)
- `ens.loop_step_allow_single_tool_loopback` (default `true`)
- single loopback max per pass

Progress callbacks:

- pass executor emits pass lifecycle phases: `started`, `completed`, `failed`
- dispatcher consumes these callbacks and updates in-memory loop progress state
- pass status text is plugin-defined via `StepPassPlan.status_text_started` / `status_text_completed`

## Narrative v1 Current Pass Plan

Narrative plugin currently emits two passes:

1. `pass_primary_generation` (`kind=primary_generation`)
- produces the user-visible beat text
- `emit_to_user=true`

2. `pass_outcome_resolution` (`kind=outcome_resolution`)
- evaluates the generated beat and selects next control action
- uses outcome ladder
- `emit_to_user=false`

Current narrative status text mapping:

- primary generation: `Writing next beat...`
- outcome resolution: `Evaluating next action...`

Visible output source is explicit:

- `visible_pass_id=pass_primary_generation`
- `visible_content_source=pass:pass_primary_generation`

## Outcome Resolution Ladder

Core ladder in `control_resolution/ladders.py`:

1. native tool call extraction
2. content parse salvage
3. structured JSON-schema retry
4. default wait (`WAIT_FOR_USER`)

Ambiguity handling:

- ambiguous native/control parse falls through
- ambiguous content parse falls through
- final fallback defaults to wait

Provider capability gating:

- resolved by invoker (`resolve_provider_capabilities`)
- gates native and JSON-schema rungs

## State Transitions

Final `control_action` is normalized then applied to loop state:

- `CONTINUE` => keep running and enqueue next progression (subject to loop policy caps)
- `WAIT_FOR_USER`/`YIELD` => waiting or yielded semantics by loop mode/policy
- `COMPLETE` => stop session

Policy cap support:

- `max_consecutive_continue` can force wait after threshold

## Progression Idempotency and Dedupe

Current behavior:

- auto-enqueued follow-up progression uses deterministic idempotency key:
  - `loop:progression:{loop_id}:{step_index+1}`
- interactive narrative manual tick progression is unkeyed
- scheduler dedupes unkeyed `loop_progression` by existing in-flight (`pending`/`running`) loop signal

Rationale:

- avoids duplicate in-flight execution from repeated manual ticks
- avoids stale-loop deadlock from replaying a consumed deterministic key on damaged historical sessions

Future hardening candidate:

- introduce progression epoch/nonce so manual tick can be deterministically keyed without stale-key lockout

## Diagnostics and Debugging

Primary event:

- `ENSLoopStepEvent.output_json`

Key fields:

- `outcome_*` ladder diagnostics
- `pass_trace` (per-pass status/tiers/timing/outcome info)
- `visible_pass_id`, `visible_content_source`
- `loop_plugin_id`, `control_policy_id`, `control_resolution_engine`

Conversation debug log:

- `data/debug_logs/conversations/{conversation_id}/ens_conversation_YYYY-MM-DD.jsonl`
- event type: `ens_loop_step_turn`

Recommended first checks when something is wrong:

1. `ens_loop_sessions` state/step index/stop reason
2. recent `ens_loop_step_events` for duplicate `step_index_before` or conflicting control actions
3. `ens_signal_queue` for duplicate or racing `loop_progression` rows
4. `pass_trace` + `outcome_*` rung selection to identify control-resolution source

## Live Progress Status (UI Feedback)

Purpose:

- expose in-flight pass progress so users can see what loop execution is currently doing
- avoids silent autoplay/tick behavior during multi-pass steps

Current implementation:

- dispatcher stores latest per-loop progress in `app_state["loop_progress_status"]`
- updated on pass phase transitions (`started`, `completed`, `failed`)
- cleared on step completion

API:

- `GET /interactive-narrative/{loop_id}/progress`
- response shape:
  - `active`
  - `step_index_before`
  - `pass_id`, `pass_kind`
  - `phase`
  - `status_text`
  - `emit_to_user`
  - `error`
  - `updated_at`

Web behavior:

- autoplay path polls progress while loop step is in-flight
- typing indicator renders current `status_text`
- status source is plugin pass config (not hardcoded in frontend)

Cross-surface note:

- this pass implements UI feedback for web only
- same progress events can be mapped later for Discord/Telegram/SMS/background surfaces

## Adding a New Loop Kind

Minimum path:

1. Implement plugin class satisfying `LoopKindPlugin`.
2. Register plugin in `loop_plugins/registry.py`.
3. Define pass plan in `build_step_plan(...)`.
4. If outcome pass is used, define outcome policy + messages + retry schema.
5. Add integration tests for:
- progression behavior
- control action transitions
- pass trace output
- state machine correctness

Design recommendation:

- keep domain policy in plugin
- keep generic orchestration in core executor/dispatcher

## Testing Strategy

Core test files:

- `testing/test_interactive_narrative_v1_integration.py`
- `testing/test_stage_b_control_ladder.py` (outcome ladder behavior)
- `testing/test_step_pass_executor.py`
- scheduler guards: `testing/test_ens_v3_scheduler_scaffold_integration.py`, `testing/test_ens_v3_signal_guardrails.py`

Run with embedded environment:

- `run_test.bat`

## Common Failure Modes

Duplicate progression execution:

- symptom: two step events with same `step_index_before`
- check queue dedupe and manual tick behavior

Tick returns 200 but no step runs:

- symptom: no new step event, state unchanged
- usually stale session/key replay or non-running session state

Control defaults unexpectedly to wait:

- check `outcome_ladder_rung_selected`, `outcome_rung*`, and pass 2 trace
- verify provider capabilities and tool transport mode

Visible output mismatch:

- verify `visible_pass_id` and pass trace emit flags

## Current Limitations

- `pass_trace` is embedded in JSON diagnostics, not normalized in a dedicated table.
- Manual tick progression is deduped but not fully deterministic-idempotent keyed.
- Only narrative loop kind uses multi-pass wiring today.
- live progress status store is in-memory (process-local), not persisted.

## Near-Term Follow-Up

1. adopt pass executor for additional loop kinds
2. add optional epoch-based deterministic keying for manual progression
3. add optional per-pass persistence model if analytics/reporting requires relational querying
