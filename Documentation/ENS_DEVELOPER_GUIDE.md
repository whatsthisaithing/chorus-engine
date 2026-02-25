# ENS Developer Guide (Slices 0-7.5)

This guide is for contributors adding or modifying behavior in ENS-owned codepaths.

Primary files:

- Runtime/planning: `chorus_engine/ens/runtime.py`
- Action execution: `chorus_engine/ens/dispatcher.py`
- Signal/action models: `chorus_engine/ens/models.py`
- Persistence/logging: `chorus_engine/ens/decision_store.py`
- API adapters: `chorus_engine/api/app.py`, `chorus_engine/api/model_routes.py`
- LLM invoke: `chorus_engine/ens/llm_invocation_service.py`
- LLM control plane: `chorus_engine/ens/llm_control_plane_service.py`
- Loop plugins: `chorus_engine/ens/loop_plugins/*`
- Step executor: `chorus_engine/ens/step_execution/pass_executor.py`
- Outcome ladder: `chorus_engine/ens/control_resolution/ladders.py`

Companion guide:

- `Documentation/AGENTIC_LOOP_DEVELOPER_GUIDE.md`

---

## Golden Rules / Invariants

### 1) ENS-owned-first mutation

If code does any of these, route through ENS first:

- DB write affecting continuity/state
- YAML/config write
- production LLM invocation
- LLM engine control operation
- outbound surface intent emission

Do not add new direct endpoint/service writes for these domains under active slice flags.

### 2) Idempotency and replay

- Every mutating action should have a deterministic key.
- Replays should return prior canonical output, not re-run destructive effects.
- For operations where no change applies, return explicit skipped/no_change semantics.

### 3) Planned action <-> result parity

- If an action is planned, there should be an action result (`success`/`skipped`/`failure`).
- Avoid phantom planned actions without output trail.

### 4) Rollback on action exceptions

- On dispatcher action exception, rollback session before continuing.
- Implemented in `chorus_engine/ens/dispatcher.py:302`.
- Without rollback, later persistence/action work can fail with invalid-transaction errors.

### 5) Lock ordering policy

For control-plane operations (slice 7.5):

- order: ENS control mutex -> `llm_usage_lock` -> `comfyui_lock`
- never acquire in reverse order
- if caller already holds shared locks, pass `skip_shared_locks=True` in control metadata to avoid nested lock reacquire stalls/timeouts

Implementation: `chorus_engine/ens/llm_control_plane_service.py`.

---

## Ownership Flags and Reality

Flags are in `ENSConfig` at `chorus_engine/config/models.py:520`.

Use these checks in adapters (`chorus_engine/api/app.py:_ens_flags` and helper booleans) to route behavior.

Important operational flags:

- `enabled`
- `slice1_chat_ownership`
- `nonstream_intake_only`
- `streaming_intake_only`
- `slice2_*`
- `slice25_media_gating_ownership`
- `slice3_continuity_writes_ownership`
- `slice4_config_ownership`
- `slice6_surface_routing_ownership`
- `slice65_egress_outbox_ownership`
- `slice7_unified_llm_invocation`
- `slice75_llm_control_plane_ownership`

---

## Adding New ENS Behavior

### Step 1: Add signal type + adapter

Create/extend endpoint or internal adapter function that emits a `Signal`.

Pattern examples:

- `_ens_thread_chat` in `chorus_engine/api/app.py:4027`
- `_ens_history_message_add` in `chorus_engine/api/app.py:4216`
- `_invoke_llm_control_unified` in `chorus_engine/api/app.py:3940`

Include routing metadata when relevant:

- `surface_id`, `external_thread_id`, `speaker_external_id`, `target_hint`, `relationship_hint`

### Step 2: Plan actions in runtime

Add planning branch in `_propose_actions`:

- `chorus_engine/ens/runtime.py`
- map `signal.type` -> list of `ENSAction`
- include deterministic `idempotency_key` where possible
- gate follow-up actions based on prior results when needed

### Step 3: Implement action handler in dispatcher

Add handler branch in `ENSDispatcher.execute`:

- `chorus_engine/ens/dispatcher.py`
- implement concrete behavior in dedicated private method
- return output dict with stable schema
- include `_ens_action_status="skipped"` with `reason` for no-op/guarded cases

### Step 4: Ensure observability shape is useful

Action output should include enough to debug without leaking secrets:

- operation kind
- identifiers (`message_id`, `tool_call_id`, `model_id`, etc.)
- key result attributes
- normalized error details where relevant

### Step 5: Add integration tests

Use existing integration harness style:

- `testing/conftest.py` fixtures (`app`, `client`, `db`, `helpers.set_ens_flags`)
- FastAPI `TestClient` endpoint tests
- DB assertions on `ens_decisions`, `ens_action_results`, domain tables

---

## Tool-like Flows

### Current pattern

1. LLM emits tool payload (or scene preview generates synthetic tool)
2. ENS adjudicates payload/policy
3. ENS persists pending tool calls (`ENSToolCallRequest`)
4. Confirm endpoint requests execution by `tool_call_id`
5. ENS dispatch executes and updates tool status/result ref

Key code:

- pending persistence: `chorus_engine/ens/dispatcher.py:1139`
- execute hook: `chorus_engine/ens/dispatcher.py:1204`
- app tool executor: `chorus_engine/api/app.py:4644`

### Tool call lifecycle

- `pending` -> `dispatched` -> `completed` or `failed`
- replay returns existing completed result where available

### Scene capture contract

- Preview signal: `scene_capture.preview_requested`
- Preview action: `scene_capture.prompt_generate`
- Persist synthetic tool call `tool_name=scene_capture.generate`
- Confirm path: `tool.execute_requested` with `tool_call_id`
- Prompt override allowed only while `status == pending`

---

## Multi-Surface Handling

### Ingress normalization

Use canonical surface identity:

- `chorus_engine/ens/surface_identity.py`

Resolver:

- `chorus_engine/ens/surface_router.py`

Durable mapping:

- `surface_bindings` + conflict-safe creation via `create_with_retry`
- repository: `chorus_engine/repositories/surface_binding_repository.py`

Notes:

- `surface_instance_id` normalized as string/empty string
- `target_hint` currently accepts `general_chat`; others are ignored and tracked

### Egress outbox

Signal:

- `surface.send_message_requested`

Action:

- `surface.egress.persist_intent`

Persistence/replay/ack/fail:

- `chorus_engine/repositories/surface_egress_intent_repository.py`

Endpoints:

- `/egress/intents`, `/egress/intents/{intent_id}/ack`, `/egress/intents/{intent_id}/fail`, `/debug/egress/send-intent`

---

## Unifying LLM Invocation (Slice 7)

Use `LLMInvocationService` only.

File:

- `chorus_engine/ens/llm_invocation_service.py`

Requirements:

- compute stable `request_fingerprint` from deterministic fields only
- include metadata but strip non-deterministic keys (`trace_id`, timestamps, etc.)
- do retries as internal attempts under one logical request key
- pass effective model config via `resolve_effective_config` precedence:
  - explicit override -> character preferred -> analysis model (for analysis) -> system default

Test guard:

- `testing/conftest.py` fixture `_slice7_strict_direct_generate_guard`
- blocks direct `llm_client.generate*` and `generate_vision` outside invoker context when slice7 enabled

---

## Unifying Control-Plane Calls (Slice 7.5)

Use `LLMControlPlaneService` only for control ops.

File:

- `chorus_engine/ens/llm_control_plane_service.py`

Request contract:

- `op`, `idempotency_key`
- `busy_mode`: `block_with_timeout` or `skip_busy`
- optional `timeout_s`
- optional `metadata.skip_shared_locks` when caller already holds shared locks

Lock policy:

- ENS control mutex first
- then `llm_usage_lock`
- then `comfyui_lock`
- never reverse
- nested contexts should pass `skip_shared_locks=True`

Test guard:

- `testing/conftest.py` fixture `_slice75_strict_direct_control_guard`
- blocks direct control methods outside control-plane service when slice75 enabled

---

## Debugging Playbook

### Given conversation_id or thread_id

1. Find decisions:
- SQL table `ens_decisions` (`signal_type`, `session_id`, `created_at`)
- JSONL `data/debug_logs/ens/decisions.jsonl`

2. Find action trail:
- SQL table `ens_action_results` by `decision_id`
- JSONL `data/debug_logs/ens/action_results.jsonl`

3. Correlate IDs:
- `trace_id`, `signal_id`, `decision_id`, `action_id`, `idempotency_key`

4. For tool/media issues:
- inspect `ENSToolCallRequest` rows (`status`, `args_json`, `result_ref`)

### Common failure modes and what to inspect

- `sqlite3.OperationalError: database is locked`
  - check long-held transactions and concurrent writes
  - verify rollback happened after exception (`dispatcher.py:302`)
- invalid transaction / reconnect until rollback
  - indicates prior unrolled-back failure in same session
- malformed tool payload
  - check `llm.invoke.chat` output metadata + `tool_payload.adjudicate` result
- skipped/busy control-plane ops
  - inspect `llm.control.execute` action output (`reason=busy`, `busy_mode`, `timeout_s`)

---

## ENS Ownership Checklist for PRs

For each new/changed feature, answer yes/no:

- Does it mutate DB state?
- Does it write YAML/config?
- Does it invoke LLM text/vision generation?
- Does it control model engine state?
- Does it emit outbound surface intent?
- If yes, where is the ENS signal?
- Where is action planning?
- Where is action handler?
- What is the idempotency key?
- What are replay semantics?
- What integration tests prove ownership and replay?
- What action outputs make failures diagnosable?

---

## Migration Recipe: Bringing Direct Writes/Calls Under ENS

If you find direct state mutation or `llm_client` calls in production path, do this:

1. Identify trigger:
- endpoint/service/background task entrypoint

2. Define signal:
- `type`, scope, minimal payload, correlation metadata

3. Add runtime branch:
- `_propose_actions` maps signal to ordered actions
- assign deterministic idempotency keys

4. Add dispatcher handler:
- implement action method
- include rollback-safe behavior and explicit skipped cases

5. Preserve response contract:
- build response from `ENSOutcome.response_payload`

6. Add observability:
- ensure action output includes key IDs/status/error
- verify decision/action records in SQL + JSONL

7. Add tests:
- happy path
- replay path
- concurrency/busy path if locks involved
- guard test for forbidden direct calls when slice flag is ON

8. Gate rollout:
- behind appropriate `ens.slice*` flag
- keep fallback path only if explicitly needed

---

## Intentional Current Limitations

- Streaming generation still not fully ENS-owned in all modes.
- Relationship-aware routing is scaffold-level; hints mostly audit-level.
- Provider-model switch endpoint by model id is not yet first-class; `/api/llm/switch-model` remains integrated/file-path oriented.
- Some compatibility branches remain while rollout flags exist.

---

## Appendix: Internal Reference Docs

Implementation evolved alongside planning docs under:

- `Private/InternalPlanning/ENS/`
- `Private/InternalPlanning/ENS/Implementation/`
- `Private/InternalPlanning/ENS/Discussion/`

Use them for historical rationale, but treat code references above as source of truth for behavior.

---

## Agentic Loop Architecture Snapshot

ENS now supports plugin-driven loop behavior with core pass orchestration.

Current pattern:

- core runtime/dispatcher own loop session state machine, queueing, persistence, and observability
- loop plugins define loop-kind policy and pass planning
- pass executor runs ordered passes with guardrails
- outcome ladder resolves control action using native tools/content parse/json-schema retry/default wait

Key contract files:

- `chorus_engine/ens/loop_plugins/contracts.py`
- `chorus_engine/ens/loop_plugins/registry.py`
- `chorus_engine/ens/loop_plugins/narrative_v1.py`
- `chorus_engine/ens/step_execution/pass_executor.py`

Diagnostics naming:

- use `outcome_*` keys (legacy `stage_b_*` removed)
- `pass_trace` is persisted in loop step `output_json`

Important scheduler behavior:

- deterministic progression keys are used for auto-enqueued followups
- manual interactive tick progression is currently unkeyed
- scheduler dedupes unkeyed loop progression signals by existing in-flight (`pending`/`running`) loop progression

## ENS v3 Operational Addendum

This section is additive to all prior guidance and does not replace v2-era behavior notes.

### New v3 Runtime Flags

Set under `ens`:

- `v3_scheduler_enabled`
- `v3_arbitration_enabled`
- `v3_loop_sessions_enabled`
- `v3_structured_control_enabled`
- `v3_assistant_result_enabled`
- `v3_context_compression_enabled`
- `v3_sentinel_fallback_enabled`

Compression knobs:
- `loop_memory_compress_every_n_steps`
- `loop_memory_keep_last_k_steps`

### Adding New Ingress in v3

1. Normalize ingress to `Signal`.
2. Enqueue/ingest via ENS runtime.
3. Add planning in `runtime._propose_actions`.
4. Add dispatcher action implementation.
5. Add replay/idempotency and scheduler-path tests.

### Adding New `loop_kind`

1. Define `loop_kind` name (for policy/tool gating).
2. Choose `loop_mode` (`visible`/`hidden`) at loop creation.
3. Configure loop-kind tool allowlist in dispatcher.
4. Validate control semantics with structured `AssistantResult.control` only.

### Adding New Tool + Loop Gating

1. Implement tool action path.
2. Ensure tool requests come from `AssistantResult.tool_requests`.
3. Gate by `loop_kind` allowlist.
4. Add allowed + blocked test coverage.

### Debug Workflow (v3)

Harness:
- `scripts/ens_v3_harness.bat status`
- `scripts/ens_v3_harness.bat tick --n <N>`
- `scripts/ens_v3_harness.bat verify --profile 3_5|3_6|3_7|3_8`
- `scripts/ens_v3_harness.bat timeline --loop <loop_id>`
- `scripts/ens_v3_harness.bat timeline --relationship <relationship_id>`
- `scripts/ens_v3_harness.bat timeline --last-run`

Replay:
- `chorus_engine/devtools/replay_v3.py`
- `extract_run_signature(...)` in `chorus_engine/ens/decision_store.py`

### Common v3 Footguns

- Running loop steps in-process (must be signal-by-signal).
- Parsing control/tool directives from freeform text.
- Re-introducing surface reopen gating.
- Using unstable idempotency keys.
- Comparing timestamps/raw provider blobs in replay determinism tests.
