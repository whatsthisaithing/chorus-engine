from chorus_engine.config.models import CharacterConfig
from chorus_engine.services.system_prompt_generator import SystemPromptGenerator


def _character() -> CharacterConfig:
    return CharacterConfig(
        id="test_char",
        name="Test Character",
        role="assistant",
        system_prompt="You are a concise assistant that answers directly.",
    )


def test_general_chat_modifier_only_for_general_chat_kind():
    generator = SystemPromptGenerator()
    character = _character()

    general_prompt = generator.generate(character, conversation_kind="general_chat")
    standard_prompt = generator.generate(character, conversation_kind="standard")

    assert "## General Chat Conversation Stance" in general_prompt
    assert "## General Chat Conversation Stance" not in standard_prompt
