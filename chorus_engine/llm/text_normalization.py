"""Utilities for normalizing LLM text output."""

from __future__ import annotations


_MOJIBAKE_REPLACEMENTS = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": "\"",
    "â€�": "\"",
    "â€”": "--",
    "â€“": "-",
    "â€¦": "...",
    "Â ": " ",
    "Â": "",
}


def normalize_mojibake(text: str) -> str:
    """Normalize common mojibake sequences in LLM output.

    This is a conservative pass: if the text doesn't contain the common
    mojibake marker "â" or the replacement character, it returns unchanged.
    """
    if not text:
        return text

    if "â" not in text and "\uFFFD" not in text:
        return text

    # Try a latin-1 -> utf-8 roundtrip if it looks like mojibake.
    try:
        recovered = text.encode("latin-1").decode("utf-8")
        if recovered and recovered.count("\uFFFD") <= text.count("\uFFFD"):
            return recovered
    except Exception:
        pass

    # Fall back to common replacements.
    cleaned = text
    for bad, good in _MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(bad, good)
    return cleaned
