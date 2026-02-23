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
