from types import SimpleNamespace

from chorus_engine.config.models import CharacterConfig
from chorus_engine.services.moment_pin_retrieval_service import (
    MomentPinRetrievalService,
    RetrievedMomentPin,
)
from chorus_engine.services.system_prompt_generator import SystemPromptGenerator


def _character() -> CharacterConfig:
    return CharacterConfig(
        id="test_char",
        name="Test Character",
        role="assistant",
        system_prompt="You are a concise assistant that answers directly.",
        image_generation={"enabled": True},
        video_generation={"enabled": True},
    )


def _pin_prompt_block() -> str:
    pin = SimpleNamespace(
        id="pin-1",
        what_happened="Shared a specific quote",
        why_user="Need exact wording",
        why_model="Precision recall",
        quote_snippet="You said hello there.",
        tags=["quote", "continuity"],
    )
    retrieved = [RetrievedMomentPin(pin=pin, similarity=0.9, score=0.9)]
    return MomentPinRetrievalService.format_for_prompt(retrieved, tool_transport_mode="native")


def test_moment_pin_policy_block_is_transport_agnostic():
    block = _pin_prompt_block()
    assert "Tool payload template:" not in block
    assert "Native tool args template:" not in block
    assert "---CHORUS_TOOL_PAYLOAD_BEGIN---" not in block
    assert (
        "If the user asks for exact transcript wording (verbatim, exact quote, etc.), "
        "you MUST call `moment_pin.cold_recall` before answering."
    ) in block
    assert "Do NOT guess exact quotes." in block
    assert "Maximum one cold recall per turn." in block


def test_native_prompt_without_pins_has_no_cold_recall_tool_docs():
    generator = SystemPromptGenerator()
    prompt = generator.generate(
        _character(),
        tool_transport_mode="native",
        allowed_media_tools={"image.generate"},
        contract_tools={"image.generate"},
    )
    assert "- `moment_pin.cold_recall`:" not in prompt
    assert "- `moment_pin.cold_recall`: Use only when transcript precision is required for an injected moment pin." not in prompt


def test_native_prompt_with_pins_includes_cold_recall_tool_docs_and_policy_without_templates():
    generator = SystemPromptGenerator()
    system_prompt = generator.generate(
        _character(),
        tool_transport_mode="native",
        allowed_media_tools={"image.generate"},
        contract_tools={"image.generate", "moment_pin.cold_recall"},
    )
    moment_pin_block = _pin_prompt_block()
    assembled = f"{system_prompt}\n\n{moment_pin_block}"
    assert "- `moment_pin.cold_recall`:" in assembled
    assert "Tool payload template:" not in assembled
    assert "Native tool args template:" not in assembled
