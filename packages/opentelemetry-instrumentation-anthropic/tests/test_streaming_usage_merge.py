from types import SimpleNamespace

from opentelemetry.instrumentation.anthropic.streaming import (
    _merge_final_message,
    _process_response_item,
)


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


def test_final_message_usage_backfills_cache_fields_for_streams():
    complete_response = {
        "events": [{"type": "text", "text": "hello"}],
        "model": "claude-opus-4-5-20251101",
        "usage": {
            "input_tokens": 13889,
            "output_tokens": 235,
        },
        "id": "msg_stream",
    }

    final_message = SimpleNamespace(
        id="msg_final",
        model="claude-opus-4-5-20251101",
        usage={
            "input_tokens": 2201,
            "output_tokens": 4281,
            "cache_creation_input_tokens": 79097,
            "cache_read_input_tokens": 197584,
        },
    )

    _merge_final_message(complete_response, final_message)

    assert complete_response["id"] == "msg_final"
    assert complete_response["usage"] == {
        "input_tokens": 2201,
        "output_tokens": 4281,
        "cache_creation_input_tokens": 79097,
        "cache_read_input_tokens": 197584,
    }
