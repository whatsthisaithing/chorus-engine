"""Core control-resolution ladder helpers shared by loop-kind plugins."""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable, Dict, Optional, Set

from chorus_engine.ens.assistant_result import AssistantResult
from chorus_engine.ens.loop_plugins.contracts import StepOutcomePolicy, StepOutcomeResolution
from chorus_engine.services.json_extraction import extract_json_block


def normalize_action(value: Any, *, allowed_actions: Set[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    action = value.strip().upper()
    return action if action in allowed_actions else None


def _collect_actions_from_obj(obj: Any, *, allowed_actions: Set[str]) -> list[str]:
    actions: list[str] = []
    if isinstance(obj, dict):
        direct = normalize_action(obj.get("action"), allowed_actions=allowed_actions)
        if direct:
            actions.append(direct)
        parameters = obj.get("parameters")
        if isinstance(parameters, dict):
            nested = normalize_action(parameters.get("action"), allowed_actions=allowed_actions)
            if nested:
                actions.append(nested)
        arguments = obj.get("arguments")
        if isinstance(arguments, dict):
            nested = normalize_action(arguments.get("action"), allowed_actions=allowed_actions)
            if nested:
                actions.append(nested)
        for value in obj.values():
            if isinstance(value, (dict, list)):
                actions.extend(_collect_actions_from_obj(value, allowed_actions=allowed_actions))
    elif isinstance(obj, list):
        for item in obj:
            actions.extend(_collect_actions_from_obj(item, allowed_actions=allowed_actions))
    return actions


def extract_action_from_content(text: str, *, allowed_actions: Set[str]) -> Dict[str, Any]:
    content = str(text or "").strip()
    if not content:
        return {
            "attempted": True,
            "success": False,
            "ambiguous": False,
            "reason": "empty_content",
            "action": None,
        }

    actions: list[str] = []
    try:
        parsed = json.loads(content)
        actions.extend(_collect_actions_from_obj(parsed, allowed_actions=allowed_actions))
    except Exception:
        pass

    parsed_obj, _ = extract_json_block(content, expected_root="object")
    if parsed_obj is not None:
        actions.extend(_collect_actions_from_obj(parsed_obj, allowed_actions=allowed_actions))

    for match in re.finditer(r"```json\s*(.*?)\s*```", content, re.IGNORECASE | re.DOTALL):
        try:
            block = json.loads(match.group(1))
        except Exception:
            block = None
        if block is not None:
            actions.extend(_collect_actions_from_obj(block, allowed_actions=allowed_actions))

    for match in re.finditer(r"\{[^{}]*\}", content, re.DOTALL):
        try:
            frag = json.loads(match.group(0))
        except Exception:
            frag = None
        if frag is not None:
            actions.extend(_collect_actions_from_obj(frag, allowed_actions=allowed_actions))

    # Explicit ambiguity detector for common divergent actions.
    if "CONTINUE" in allowed_actions and "YIELD" in allowed_actions:
        has_continue = bool(re.search(r"\bCONTINUE\b", content, re.IGNORECASE))
        has_yield = bool(re.search(r"\bYIELD\b", content, re.IGNORECASE))
        if has_continue and has_yield:
            return {
                "attempted": True,
                "success": False,
                "ambiguous": True,
                "reason": "ambiguous_actions:CONTINUE|YIELD",
                "action": None,
            }

    uniq = sorted(set(a for a in actions if a in allowed_actions))
    if len(uniq) == 1:
        return {
            "attempted": True,
            "success": True,
            "ambiguous": False,
            "reason": "content_parse_ok",
            "action": uniq[0],
        }
    if len(uniq) > 1:
        return {
            "attempted": True,
            "success": False,
            "ambiguous": True,
            "reason": f"ambiguous_actions:{'|'.join(uniq)}",
            "action": None,
        }
    return {
        "attempted": True,
        "success": False,
        "ambiguous": False,
        "reason": "no_salvageable_action",
        "action": None,
    }


def evaluate_native_rung(
    assistant_result: Optional[AssistantResult],
    *,
    control_tool_name: str,
    allowed_actions: Set[str],
) -> Dict[str, Any]:
    if assistant_result is None:
        return {
            "attempted": False,
            "success": False,
            "ambiguous": False,
            "reason": "outcome_pass_not_invoked",
            "action": None,
        }

    provider_raw = dict(assistant_result.provider_raw or {})
    raw_calls = provider_raw.get("provider_tool_calls_raw")
    control_actions: list[str] = []
    if isinstance(raw_calls, list):
        for item in raw_calls:
            if not isinstance(item, dict):
                continue
            fn = item.get("function") if isinstance(item.get("function"), dict) else {}
            name = (fn.get("name") if isinstance(fn, dict) else None) or item.get("name")
            if str(name or "").strip() != control_tool_name:
                continue
            raw_args = (fn.get("arguments") if isinstance(fn, dict) else None) or item.get("arguments")
            args_obj: Dict[str, Any] = {}
            if isinstance(raw_args, dict):
                args_obj = raw_args
            elif isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args)
                    if isinstance(parsed, dict):
                        args_obj = parsed
                except Exception:
                    args_obj = {}
            action = normalize_action(args_obj.get("action"), allowed_actions=allowed_actions)
            if action:
                control_actions.append(action)

    uniq_actions = sorted(set(control_actions))
    if len(uniq_actions) > 1:
        return {
            "attempted": True,
            "success": False,
            "ambiguous": True,
            "reason": f"ambiguous_native_control_actions:{'|'.join(uniq_actions)}",
            "action": None,
        }
    if len(control_actions) > 1 and len(uniq_actions) == 1:
        return {
            "attempted": True,
            "success": False,
            "ambiguous": True,
            "reason": "ambiguous_native_multiple_control_calls",
            "action": None,
        }

    if assistant_result.control is not None:
        action = normalize_action(assistant_result.control.action, allowed_actions=allowed_actions)
        if action:
            return {
                "attempted": True,
                "success": True,
                "ambiguous": False,
                "reason": "native_control_ok",
                "action": action,
            }

    return {
        "attempted": True,
        "success": False,
        "ambiguous": False,
        "reason": "native_control_missing",
        "action": None,
    }


async def resolve_outcome_with_ladder(
    *,
    outcome_pass_assistant_result: Optional[AssistantResult],
    provider_capabilities: Dict[str, Any],
    outcome_content_source: str,
    policy: StepOutcomePolicy,
    invoke_json_retry: Callable[[], Awaitable[Dict[str, Any]]],
) -> StepOutcomeResolution:
    rung1_native: Dict[str, Any] = {
        "attempted": False,
        "success": False,
        "ambiguous": False,
        "reason": "not_invoked",
        "action": None,
    }
    rung2_parse: Dict[str, Any] = {
        "attempted": False,
        "success": False,
        "ambiguous": False,
        "reason": "not_invoked",
        "action": None,
    }
    rung3_json_schema: Dict[str, Any] = {
        "attempted": False,
        "success": False,
        "reason": "not_invoked",
        "action": None,
    }
    selected = "not_invoked"
    action: Optional[str] = None
    source_result: Optional[AssistantResult] = None

    if bool(provider_capabilities.get("supports_native_tools")):
        rung1_native = evaluate_native_rung(
            outcome_pass_assistant_result,
            control_tool_name=policy.control_tool_name,
            allowed_actions=set(policy.allowed_actions),
        )
        if bool(rung1_native.get("success")):
            action = str(rung1_native.get("action") or "").strip().upper()
            source_result = outcome_pass_assistant_result
            selected = "native"
    else:
        rung1_native = {
            "attempted": False,
            "success": False,
            "ambiguous": False,
            "reason": "skipped_unsupported_native_tools",
            "action": None,
        }

    if not action:
        rung2_parse = extract_action_from_content(
            outcome_content_source or "",
            allowed_actions=set(policy.allowed_actions),
        )
        if bool(rung2_parse.get("success")):
            action = str(rung2_parse.get("action") or "").strip().upper()
            source_result = outcome_pass_assistant_result
            selected = "content_parse"

    should_try_rung3 = False
    if not action:
        if bool(rung1_native.get("ambiguous")):
            should_try_rung3 = True
        elif bool(rung2_parse.get("ambiguous")):
            should_try_rung3 = True
        elif bool(rung2_parse.get("attempted")):
            should_try_rung3 = True
        else:
            should_try_rung3 = not bool(rung1_native.get("attempted"))

    if should_try_rung3 and bool(provider_capabilities.get("supports_response_format_json_schema")):
        rung3_json_schema["attempted"] = True
        retry_result = await invoke_json_retry()
        if str(retry_result.get("status") or "").lower() == "success":
            retry_assistant_result = retry_result.get("assistant_result")
            retry_content = ""
            if isinstance(retry_assistant_result, AssistantResult):
                retry_content = (
                    retry_assistant_result.display_text
                    or retry_assistant_result.raw_content
                    or ""
                )
            retry_parse = extract_action_from_content(
                retry_content,
                allowed_actions=set(policy.allowed_actions),
            )
            rung3_json_schema["success"] = bool(retry_parse.get("success"))
            rung3_json_schema["reason"] = str(retry_parse.get("reason") or "json_schema_retry_failed")
            rung3_json_schema["action"] = retry_parse.get("action")
            if bool(retry_parse.get("success")):
                action = str(retry_parse.get("action") or "").strip().upper()
                source_result = retry_assistant_result if isinstance(retry_assistant_result, AssistantResult) else None
                selected = "json_schema_retry"
        else:
            rung3_json_schema["success"] = False
            rung3_json_schema["reason"] = str(retry_result.get("error") or "json_schema_retry_invocation_failed")
            rung3_json_schema["action"] = None
    elif should_try_rung3:
        rung3_json_schema = {
            "attempted": False,
            "success": False,
            "reason": "skipped_unsupported_response_format_json_schema",
            "action": None,
        }

    defaulted_wait = False
    source_stage = selected
    if not action:
        action = str(policy.default_action or "WAIT_FOR_USER").strip().upper() or "WAIT_FOR_USER"
        defaulted_wait = True
        source_stage = "default_wait"

    return StepOutcomeResolution(
        action=action,
        ladder_rung_selected=(selected if selected != "not_invoked" else "default_wait"),
        defaulted_wait=defaulted_wait,
        rung1_native=rung1_native,
        rung2_parse=rung2_parse,
        rung3_json_schema=rung3_json_schema,
        source_stage=source_stage,
        source_assistant_result=source_result,
        error=None,
    )
