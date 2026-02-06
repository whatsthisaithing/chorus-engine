"""
Helpers for extracting and parsing JSON blocks from LLM responses.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Tuple


ExpectedRoot = Literal["array", "object"]


def extract_json_block(text: str, expected_root: ExpectedRoot) -> Tuple[Any | None, str]:
    """
    Extract and parse JSON from text, returning (parsed, parse_mode).

    parse_mode:
        - "raw" if full text parses directly
        - "fenced" if parsed from first fenced block
        - "balanced" if parsed from first balanced [] or {} substring
        - "failed" if parsing fails
    """
    if text is None:
        return None, "failed"

    try:
        parsed = json.loads(text)
        if _matches_root(parsed, expected_root):
            return parsed, "raw"
    except Exception:
        pass

    stripped = text.strip()

    fenced = _extract_first_fenced_block(stripped)
    if fenced is not None:
        try:
            parsed = json.loads(fenced)
            if _matches_root(parsed, expected_root):
                return parsed, "fenced"
        except Exception:
            pass

    balanced = _extract_first_balanced(stripped, expected_root)
    if balanced is not None:
        try:
            parsed = json.loads(balanced)
            if _matches_root(parsed, expected_root):
                return parsed, "balanced"
        except Exception:
            pass

    return None, "failed"


def _matches_root(parsed: Any, expected_root: ExpectedRoot) -> bool:
    if expected_root == "array":
        return isinstance(parsed, list)
    return isinstance(parsed, dict)


def _extract_first_fenced_block(text: str) -> str | None:
    # Match ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def _extract_first_balanced(text: str, expected_root: ExpectedRoot) -> str | None:
    if expected_root == "array":
        opener = "["
        closer = "]"
    else:
        opener = "{"
        closer = "}"

    start = text.find(opener)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue

        if ch == "\"":
            in_string = True
            continue

        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None
