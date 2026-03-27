from types import SimpleNamespace

from opentelemetry.instrumentation.anthropic.streaming import _process_response_item


def test_message_delta_usage_merges_cache_fields_without_overwriting_with_none():
    complete_response = {
        "events": [{"type": "text", "text": ""}],
        "model": "claude-opus-4-5-20251101",
        "usage": {
            "input_tokens": 13889,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        "id": "msg_123",
    }

    item = SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason="tool_use"),
        usage={
            "input_tokens": None,
            "output_tokens": 235,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": 1163,
        },
    )

    _process_response_item(item, complete_response)

    assert complete_response["events"][0]["finish_reason"] == "tool_use"
    assert complete_response["usage"] == {
        "input_tokens": 13889,
        "output_tokens": 235,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 1163,
    }
