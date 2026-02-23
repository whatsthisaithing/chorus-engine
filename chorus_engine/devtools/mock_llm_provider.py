"""Deterministic mock LLM provider for ENS replay/testing."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class _MockResponse:
    content: str
    finish_reason: str = "stop"
    usage: Optional[Dict[str, Any]] = None


class DeterministicMockLLMProvider:
    """Mock provider that deterministically maps prompt/messages -> response."""

    base_url = "mock://deterministic-llm"

    async def health_check(self) -> bool:
        return True

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None, **kwargs: Any) -> _MockResponse:
        _ = (system_prompt, model, kwargs)
        return _MockResponse(content=self._content_for_prompt(prompt or ""))

    async def generate_with_history(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> _MockResponse:
        _ = (temperature, max_tokens, model)
        last = ""
        for item in reversed(messages or []):
            if str(item.get("role")) == "user":
                last = str(item.get("content") or "")
                break
        digest = hashlib.sha256(last.encode("utf-8")).hexdigest()[:12]
        return _MockResponse(content=f"Mock chat reply [{digest}]")

    async def generate_vision(self, **kwargs: Any) -> _MockResponse:
        _ = kwargs
        return _MockResponse(content='{"mock":"vision"}')

    @staticmethod
    def _extract_step(prompt: str) -> int:
        match = re.search(r"Loop progression step\s+(\d+)", prompt or "")
        if not match:
            return 1
        try:
            return max(1, int(match.group(1)))
        except ValueError:
            return 1

    def _content_for_prompt(self, prompt: str) -> str:
        step = self._extract_step(prompt)
        if "MOCK_TOOL" in prompt:
            return (
                "Tool step.\n"
                "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
                '{"version":1,"control":{"action":"COMPLETE","args":{}},'
                '"tool_calls":[{"id":"mock_tool_1","tool":"image.generate","requires_approval":true,"args":{"prompt":"mock tool"}}]}\n'
                "---CHORUS_TOOL_PAYLOAD_END---"
            )
        # Deterministic finite loop progression.
        if step >= 12:
            control = "COMPLETE"
        else:
            control = "CONTINUE"
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]
        return (
            f"Deterministic loop step {step} [{digest}].\n"
            "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
            f'{{"version":1,"control":{{"action":"{control}","args":{{}}}},"tool_calls":[]}}\n'
            "---CHORUS_TOOL_PAYLOAD_END---"
        )
