# Tool Payload Infrastructure Design

**Phase**: Media Tooling + Moment Pin Cold Recall  
**Created**: February 13, 2026  
**Status**: Implemented, shared infrastructure in active use

---

## Overview

Chorus Engine uses a sentinel-delimited tool payload channel to let the assistant request server-side tool actions while preserving clean user-visible responses.

Current tool families:

- Media generation:
  - `image.generate`
  - `video.generate`
- Moment pin transcript precision:
  - `moment_pin.cold_recall`

The infrastructure is shared, but each tool family applies its own validation and policy gates.

---

## Core Contract

Tool payloads are appended after visible assistant content using sentinels:

```text
---CHORUS_TOOL_PAYLOAD_BEGIN---
{ ...json... }
---CHORUS_TOOL_PAYLOAD_END---
```

The server:

1. strips payload from display text,
2. parses JSON safely,
3. validates per-tool schema and policy,
4. executes allowed actions,
5. never exposes sentinel payload to clients.

---

## Payload Versions and Schemas

### Shared Envelope (v1)

- `version: 1`
- `tool_calls: [ ... ]`

### Media Tools

Required call fields:

- `tool`: one of allowed media tools
- `requires_approval`
- `args.prompt`

### Moment Pin Cold Recall

Required call fields:

- `tool: "moment_pin.cold_recall"`
- `requires_approval: false`
- `args.pin_id`
- `args.reason`

Additional guardrails:

- exactly one cold recall call per turn
- no mixed/chained tool payload with media calls
- `pin_id` must have been injected this turn

---

## Execution Model

### Media Generation

- Tool calls become `pending_tool_calls` in API response.
- UI confirms/executes according to policy.
- Existing workflow orchestration handles generation.

### Cold Recall

- Executed server-side immediately, non-interactive.
- Appends archival transcript wrapper to rerun context.
- Reruns model once and returns rerun answer.

---

## Why This Architecture

### Separation of Concerns

- Assistant decides *intent* ("I need a tool").
- Server decides *authorization + execution*.
- UI only handles approved interactive tools.

### Safety and Determinism

- Strict schema and allowlist validation
- Tool-family-specific gates
- Silent ignore for invalid payloads with structured logs
- No raw tool JSON leaked to user text channel

### Extensibility

New tools can plug into:

1. schema validation path,
2. per-tool policy gates,
3. execution adapter.

---

## Observability

Key diagnostics currently emitted:

- payload presence and validated call counts
- blocked reasons for media payload attempts
- cold recall accepted/rejected reason
- rerun execution details
- per-conversation `media_requests.jsonl` diagnostics

---

## Known Constraints

- Streaming and non-streaming must preserve identical tool safety semantics.
- Payload parsing must tolerate malformed model output without user-visible breakage.
- Tool format quality still depends on prompt adherence; repair loops are limited and explicit.

---

## Related Documents

- `Documentation/Design/COMFYUI_WORKFLOW_SYSTEM.md`
- `Documentation/Design/MOMENT_PIN_SYSTEM.md`
- `Documentation/Design/PROMPT_ASSEMBLY_TOKEN_MANAGEMENT.md`

