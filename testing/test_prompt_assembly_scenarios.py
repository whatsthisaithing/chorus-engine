from chorus_engine.config.models import CharacterConfig
from chorus_engine.services.system_prompt_generator import SystemPromptGenerator
from chorus_engine.models.conversation import Conversation
from chorus_engine.services.prompt_assembly import PromptAssemblyService


def test_prompt_assembly_builds_scenario_block_when_snapshot_present(db):
    service = PromptAssemblyService(db=db, character_id="test_char")
    conversation = Conversation(
        character_id="test_char",
        title="Scenario Prompt Test",
        source="web",
        scenario_source="library",
        scenario_title="Arrival",
        scenario_text="You arrive in a rain-soaked city at dawn.",
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    result = service._build_scenario_snapshot_block(
        conversation_id=conversation.id,
    )

    assert "## Scenario: Arrival" in result
    assert "You arrive in a rain-soaked city at dawn." in result


def test_prompt_assembly_omits_scenario_block_for_none_source(db):
    service = PromptAssemblyService(db=db, character_id="test_char")
    conversation = Conversation(
        character_id="test_char",
        title="Scenario Prompt None",
        source="web",
        scenario_source="none",
        scenario_title="Ignored",
        scenario_text="Should not be inserted",
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    result = service._build_scenario_snapshot_block(
        conversation_id=conversation.id,
    )

    assert result is None


def test_system_prompt_generator_places_scenario_before_role_guidance():
    generator = SystemPromptGenerator()
    character = CharacterConfig(
        id="scenario_order_test",
        name="Scenario Order Test",
        role="Roleplayer",
        role_type="roleplayer",
        system_prompt="You are a roleplay character.",
    )
    prompt = generator.generate(
        character=character,
        scenario_block="## Scenario: Storm Night\nRain pounds the windows.",
    )
    scenario_idx = prompt.find("## Scenario: Storm Night")
    role_idx = prompt.find("## Roleplayer Role")
    assert scenario_idx != -1
    assert role_idx != -1
    assert scenario_idx < role_idx
