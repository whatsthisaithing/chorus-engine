from chorus_engine.services.media_turn_classifier import classify_media_turn


def test_lexical_iteration_requires_media_anchor():
    signals = classify_media_turn(
        message="I noticed another bug and fixed it.",
        semantic_intents=[],
    )
    assert signals.is_iteration_request is False
    assert signals.requested_media_type == "none"
    assert signals.explicit_media_request is False


def test_lexical_iteration_with_media_anchor_triggers():
    signals = classify_media_turn(
        message="Can you do another image with softer lighting?",
        semantic_intents=[],
    )
    assert signals.is_iteration_request is True
    assert signals.requested_media_type in {"either", "image"}
