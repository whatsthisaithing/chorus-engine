from chorus_engine.config.models import CharacterConfig
from chorus_engine.services.system_prompt_generator import SystemPromptGenerator


def _character() -> CharacterConfig:
    return CharacterConfig(
        id="test_char",
        name="Test Character",
        role="assistant",
        role_type="companion",
        system_prompt="You are a concise assistant that answers directly.",
    )


def test_non_loop_turn_does_not_inject_loop_blocks():
    generator = SystemPromptGenerator()
    character = _character()

    prompt = generator.generate(character, loop_step=False, loop_kind=None)

    assert "**Loop Step Mode (Mandatory):**" not in prompt
    assert "**Interactive Narrative Control Selection Rules:**" not in prompt


def test_loop_step_injects_control_contract_and_loop_block():
    generator = SystemPromptGenerator()
    character = _character()

    prompt = generator.generate(character, loop_step=True, loop_kind="generic")

    assert "**Control / Tool Payload Contract (Mandatory When Requested):**" in prompt
    assert "\"control\":" in prompt
    assert "- `control` is REQUIRED for loop steps." in prompt
    assert "**Loop Step Mode (Mandatory):**" in prompt
    assert prompt.count("**Loop Step Mode (Mandatory):**") == 1
    assert "Keep `tool_calls` empty unless explicitly allowed." not in prompt


def test_narrative_v1_loop_injection_order_and_rules():
    generator = SystemPromptGenerator()
    character = _character()

    prompt = generator.generate(character, loop_step=True, loop_kind="narrative.v1")

    contract_idx = prompt.find("**Control / Tool Payload Contract (Mandatory When Requested):**")
    loop_idx = prompt.find("**Loop Step Mode (Mandatory):**")
    narrative_idx = prompt.find("**Interactive Narrative Control Selection Rules:**")

    assert contract_idx != -1
    assert loop_idx != -1
    assert narrative_idx != -1
    assert contract_idx < loop_idx < narrative_idx
    assert "**Narrative.v1 Media Safeguard:**" in prompt
    assert "emit `control.action = YIELD`" in prompt


def test_native_loop_step_uses_native_contract_and_chorus_control_guidance():
    generator = SystemPromptGenerator()
    character = _character()

    prompt = generator.generate(
        character,
        loop_step=True,
        loop_kind="narrative.v1",
        tool_transport_mode="native",
    )

    assert "**Native Tool Call Contract (Provider Transport):**" in prompt
    assert "`chorus.control`" in prompt
    assert "---CHORUS_TOOL_PAYLOAD_BEGIN---" not in prompt
    assert "emit `control.action = YIELD`" not in prompt
    assert "call `chorus.control` with action YIELD" in prompt
    assert "Never emit sentinel payload markers in message text." not in prompt
    assert "Keep `tool_calls` empty unless explicitly allowed." not in prompt
    assert "Even if the user message contains the tool name" in prompt
    assert "Tool calls are emitted separately via the provider tool-call mechanism." in prompt


def test_native_narrative_beat_stage_removes_mandatory_control_requirements():
    generator = SystemPromptGenerator()
    character = _character()

    prompt = generator.generate(
        character,
        loop_step=True,
        loop_kind="narrative.v1",
        tool_transport_mode="native",
        loop_stage="beat",
    )

    assert "**Loop Step Mode (Mandatory):**" in prompt
    assert "This is the beat-generation stage." in prompt
    assert "Do not emit loop control payloads or control tool calls in this stage." in prompt
    assert "Emit exactly one `chorus.control` tool call." not in prompt
    assert "Set `chorus.control.action` to one of: CONTINUE, YIELD, COMPLETE." not in prompt
