from pathlib import Path


def test_collect_form_preserves_nested_keys_and_vision_tokens_wiring():
    """
    Regression guard for UI config-save behavior.

    We don't have a JS runtime test harness in pytest yet, so this asserts the
    critical wiring and nested-preservation hooks are present in source.
    """
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "web/js/system_settings.js").read_text(encoding="utf-8")

    # Vision max_response_tokens must round-trip through load + save.
    assert "vision_max_response_tokens" in source
    assert "vision.max_response_tokens" in source
    assert "max_response_tokens: parseInt(document.getElementById('vision_max_response_tokens').value)" in source

    # collectFormData must preserve unknown nested keys to avoid destructive saves.
    assert "const preserveMissingNestedKeys = (target, source) =>" in source
    assert "preserveMissingNestedKeys(data, loaded);" in source
