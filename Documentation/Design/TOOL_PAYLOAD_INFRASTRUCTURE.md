# Tool Request Transport Infrastructure

**Scope**: ENS chat + loop tool request transport  
**Updated**: February 26, 2026  
**Status**: Native-transport-first architecture in active use

---

## Overview

Chorus Engine now treats provider-native tool transport as the primary request channel for assistant tool actions.

Supported tool families:

- Media generation:
  - `image.generate`
  - `video.generate`
- Moment pin transcript precision:
  - `moment_pin.cold_recall`
- Loop control (loop-only):
  - `chorus.control`

Sentinel-delimited payloads remain as a fallback transport path for compatibility and robustness.

---

## Transport Priority

For chat invocations, tool request extraction follows this priority:

1. Provider-native tool calls (`tool_calls` / function-call channel)
2. Structured retry outputs (JSON-schema constrained retries, where applicable)
3. Sentinel payload fallback in visible content

This precedence is normalized through `AssistantResult` tiering:

- `provider_native`
- `schema_structured`
- `sentinel_fallback`
- `none`

---

## Core Data Contract

Canonical normalized tool request shape:

```json
{
  "id": "call_id",
  "tool": "image.generate",
  "requires_approval": true,
  "args": {
    "prompt": "..."
  }
}
```

For sentinel fallback, the v1 wrapper remains:

```json
{
  "version": 1,
  "tool_calls": [ ... ]
}
```

Sentinel markers (fallback only):

```text
---CHORUS_TOOL_PAYLOAD_BEGIN---
{ ...json... }
---CHORUS_TOOL_PAYLOAD_END---
```

---

## Tool Family Rules

### Media (`image.generate`, `video.generate`)

- Must pass allowlist + turn-level media gating.
- Explicit requests and proactive offers are adjudicated separately.
- Interactive media calls produce `pending_tool_calls`.
- Confirmation/approval behavior is policy-driven.

### Moment Pin Cold Recall (`moment_pin.cold_recall`)

- Non-interactive, server-executed immediately when valid.
- Requires exactly one call (no chaining with other tools).
- `pin_id` must be injected for the current turn.
- Appends archival transcript block and performs one rerun.

### Loop Control (`chorus.control`)

- Loop-only control tool.
- Used by outcome/control passes to set `CONTINUE | YIELD | COMPLETE`.
- Resolved by the outcome ladder in narrative loop flow.

---

## Robustness Ladders

### Loop Outcome Ladder

1. Native control tool call (`chorus.control`)
2. Content parse salvage
3. JSON-schema retry (`response_format`) when supported
4. Safe default (`WAIT_FOR_USER`) if unresolved

### Chat Media Tool Ladder (explicit-required turns)

1. Native tool request extraction
2. JSON-schema retry for tool payload object (`rung3_json_schema_retry`) when supported
3. Sentinel repair retry (`rung4_sentinel_repair`)
4. Block with explicit reason when no valid call is recovered

---

## Capability Gating

Native transport and schema retries are gated by provider capabilities plus ENS config:

- `supports_native_tools`
- `supports_response_format_json_schema`
- `supports_sentinel_retry`

Key ENS controls:

- `native_tool_transport_enabled`
- `native_tool_transport_force_sentinel`
- `native_tool_transport_sentinel_fallback_enabled`
- `v3_sentinel_fallback_enabled`

---

## Safety Model

- Server-owned validation and gating for all tool execution.
- Tool-specific schema and policy checks.
- Unknown/disallowed tools are blocked, not executed.
- Malformed payload-like text is sanitized from user-visible output.
- No raw tool JSON is surfaced in normal client display channels.

---

## Observability

Important diagnostics include:

- `assistant_result_tier`
- native transport plan + provider tool call counts
- media adjudication accepted/blocked reasons
- cold recall requested/executed/rejected reason
- ladder diagnostics:
  - `media_tool_ladder.rung3_json_schema_retry`
  - `media_tool_ladder.rung4_sentinel_repair`

Conversation-level debug logs and tool adjudication logs capture these fields for root-cause analysis.

---

## Migration Notes

- Native transport is now the intended default design.
- Sentinel payloads are retained as fallback only.
- Legacy non-ENS paths may be removed in future cleanup; this document describes the ENS-owned transport architecture.

---

## Related Documents

- `Documentation/Design/MOMENT_PIN_SYSTEM.md`
- `Documentation/Design/PROMPT_ASSEMBLY_TOKEN_MANAGEMENT.md`
- `Documentation/Design/COMFYUI_WORKFLOW_SYSTEM.md`

