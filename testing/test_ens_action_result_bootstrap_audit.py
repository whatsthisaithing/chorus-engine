from chorus_engine.ens.runtime import ENSRuntime


def test_sql_safe_action_results_preserve_general_chat_bootstrap_audit_fields():
    action_results = [
        {
            "kind": "llm.invoke.chat",
            "status": "success",
            "output": {
                "content": "assistant response content",
                "assistant_metadata": {"general_chat_bootstrap_injected": True},
                "general_chat_bootstrap_injected": True,
                "general_chat_bootstrap_fingerprint": "fp-123",
            },
        }
    ]

    sql_docs = ENSRuntime._sql_safe_action_results(action_results)
    output = sql_docs[0]["output"]

    assert "content" not in output
    assert output["content_length"] == len("assistant response content")
    assert output["general_chat_bootstrap_injected"] is True
    assert output["general_chat_bootstrap_fingerprint"] == "fp-123"

