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


def _media_guidance_block(prompt: str) -> str:
    start = prompt.find("## Character Capabilities")
    assert start != -1
    end = prompt.find("**Control / Tool Payload Contract", start)
    if end == -1:
        end = prompt.find("**Native Tool Call Contract", start)
    if end == -1:
        end = len(prompt)
    return prompt[start:end]


def test_native_media_guidance_has_no_payload_or_sentinel_language():
    generator = SystemPromptGenerator()
    prompt = generator.generate(
        _character(),
        tool_transport_mode="native",
        allowed_media_tools={"image.generate"},
        media_gate_context={
            "media_tool_calls_allowed": True,
            "allowed_tools": ["image.generate"],
            "requested_media_type": "image",
            "is_iteration_request": False,
        },
    )
    media_block = _media_guidance_block(prompt).lower()
    assert "payload" not in media_block
    assert "sentinel" not in media_block
    assert "you can retrieve exact transcript details with `moment_pin.cold_recall`." in media_block
    assert "tool's `prompt` argument" in media_block
    assert "must make exactly one valid media tool call" in media_block


def test_sentinel_mode_keeps_sentinel_contract_block():
    generator = SystemPromptGenerator()
    prompt = generator.generate(
        _character(),
        tool_transport_mode="sentinel",
        allowed_media_tools={"image.generate"},
    )
    assert "## Control / Tool Payload Contract (Mandatory When Requested)" in prompt
    assert "---CHORUS_TOOL_PAYLOAD_BEGIN---" in prompt


def test_no_contract_tools_omits_all_contract_sections():
    generator = SystemPromptGenerator()
    prompt = generator.generate(
        _character(),
        tool_transport_mode="native",
        allowed_media_tools=set(),
        contract_tools=set(),
    )
    assert "## Native Tool Call Contract (Provider Transport)" not in prompt
    assert "## Control / Tool Payload Contract (Mandatory When Requested)" not in prompt
    assert "Available tools:" not in prompt
    assert "Supported tools:" not in prompt


def test_native_prompt_has_no_payload_or_sentinel_markers_anywhere():
    generator = SystemPromptGenerator()
    prompt = generator.generate(
        _character(),
        tool_transport_mode="native",
        allowed_media_tools={"image.generate"},
        media_gate_context={
            "media_tool_calls_allowed": True,
            "allowed_tools": ["image.generate"],
            "requested_media_type": "image",
            "is_iteration_request": False,
        },
    ).lower()
    assert "payload" not in prompt
    assert "sentinel" not in prompt
    assert "begin_tool_call" not in prompt


def test_native_media_gate_restriction_is_media_scoped_when_non_media_tools_exist():
    generator = SystemPromptGenerator()
    prompt = generator.generate(
        _character(),
        tool_transport_mode="native",
        allowed_media_tools={"image.generate", "video.generate"},
        contract_tools={"image.generate", "video.generate", "moment_pin.cold_recall"},
        media_gate_context={
            "media_tool_calls_allowed": True,
            "allowed_tools": ["image.generate", "video.generate"],
            "requested_media_type": "none",
            "is_iteration_request": False,
        },
    )
    assert "- ALLOWED_MEDIA_TOOLS: ['image.generate', 'video.generate']" in prompt
    assert "moment_pin.cold_recall" in prompt
    assert "You may only call media tools listed in ALLOWED_MEDIA_TOOLS." in prompt
    assert "You may only call tools listed in ALLOWED_TOOLS." not in prompt
