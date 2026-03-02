from chorus_engine.config.models import CharacterConfig
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


def test_archival_rerun_prompt_excludes_tool_and_media_affordances():
    generator = SystemPromptGenerator()
    prompt = generator.generate(
        _character(),
        tool_transport_mode="native",
        allowed_media_tools={"image.generate", "video.generate"},
        media_gate_context={
            "media_tool_calls_allowed": True,
            "allowed_tools": ["image.generate", "video.generate"],
            "requested_media_type": "none",
            "is_iteration_request": False,
        },
        contract_tools={"image.generate", "video.generate", "moment_pin.cold_recall"},
        prompt_mode="archival_rerun",
    )

    assert "## Native Tool Call Contract (Provider Transport)" not in prompt
    assert "Available tools:" not in prompt
    assert "## Media Tooling Runtime Gate (Authoritative)" not in prompt
    assert "## Prompt Mode Switch (Mandatory When Emitting a Media Tool Call)" not in prompt
    assert "moment_pin.cold_recall" not in prompt
    assert "image.generate" not in prompt
    assert "video.generate" not in prompt
    assert "Injected Moment Pins:" not in prompt


def test_archival_rerun_prompt_includes_interpretation_mode_rules():
    generator = SystemPromptGenerator()
    prompt = generator.generate(
        _character(),
        tool_transport_mode="native",
        prompt_mode="archival_rerun",
    )

    assert "## ARCHIVAL INTERPRETATION MODE (Mandatory)" in prompt
    assert "Tools are not available in this pass." in prompt
    assert "Do not say you are retrieving the transcript." in prompt
    assert "[CHORUS_END]" in prompt
    assert prompt.rfind("## ARCHIVAL INTERPRETATION MODE (Mandatory)") > prompt.rfind("[CHORUS_END]")
