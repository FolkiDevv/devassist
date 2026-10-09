"""Офлайн-тесты сериализации/разбора GigaChat-провайдера (без сети)."""

from __future__ import annotations

import json

from devassist.config import Config
from devassist.llm.gigachat import GigaChatProvider
from devassist.llm.types import FunctionCall, Message


def _provider() -> GigaChatProvider:
    # Не делает сетевых запросов до вызова complete(); ключ нужен лишь для конструктора.
    return GigaChatProvider(Config(access_key="dummy"))


def test_function_content_wrapped_as_json():
    p = _provider()
    # произвольный текст -> JSON-объект
    payload = p._message_to_payload(
        Message(role="function", name="read_file", content="1\tversion = 1.0.0")
    )
    parsed = json.loads(payload["content"])  # не должно бросить
    assert parsed["result"] == "1\tversion = 1.0.0"
    assert payload["role"] == "function"
    assert payload["name"] == "read_file"
    p.close()


def test_function_content_valid_json_passthrough():
    p = _provider()
    payload = p._message_to_payload(Message(role="function", name="x", content='{"a": 1}'))
    assert json.loads(payload["content"]) == {"a": 1}
    p.close()


def test_assistant_function_call_serialized():
    p = _provider()
    msg = Message(
        role="assistant",
        content="",
        function_call=FunctionCall(name="read_file", arguments={"path": "a.py"}),
        functions_state_id="sid-123",
    )
    payload = p._message_to_payload(msg)
    assert payload["function_call"] == {"name": "read_file", "arguments": {"path": "a.py"}}
    assert payload["functions_state_id"] == "sid-123"
    p.close()


def test_parse_response_tool_call():
    p = _provider()
    data = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "role": "assistant",
                    "function_call": {"name": "read_file", "arguments": {"path": "x"}},
                    "functions_state_id": "sid",
                },
                "finish_reason": "function_call",
            }
        ],
        "usage": {"total_tokens": 10},
    }
    turn = p._parse_response(data)
    assert turn.wants_tool
    assert turn.message.function_call.name == "read_file"
    assert turn.message.function_call.arguments == {"path": "x"}
    assert turn.message.functions_state_id == "sid"
    p.close()


class _FakeSSE:
    """Минимальный двойник httpx.Response для проверки разбора SSE."""

    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self):
        return iter(self._lines)


def test_consume_sse_text_stream():
    p = _provider()
    deltas = []
    lines = [
        'data: {"choices":[{"delta":{"content":"При","role":"assistant"},"index":0}]}',
        'data: {"choices":[{"delta":{"content":"вет"},"index":0}]}',
        'data: {"choices":[{"delta":{"content":""},"index":0,"finish_reason":"stop"}],'
        '"usage":{"total_tokens":5}}',
        "data: [DONE]",
    ]
    turn = p._consume_sse(_FakeSSE(lines), deltas.append)
    assert deltas == ["При", "вет"]
    assert turn.message.content == "Привет"
    assert turn.finish_reason == "stop"
    assert turn.usage.total_tokens == 5
    assert not turn.wants_tool
    p.close()


def test_consume_sse_function_call_stream():
    p = _provider()
    deltas = []
    lines = [
        'data: {"choices":[{"delta":{"content":"","role":"assistant",'
        '"function_call":{"name":"get_weather","arguments":{"city":"Казань"}},'
        '"functions_state_id":"sid-9"},"index":0,"finish_reason":"function_call"}]}',
        "data: [DONE]",
    ]
    turn = p._consume_sse(_FakeSSE(lines), deltas.append)
    assert deltas == []  # текста не было, только вызов функции
    assert turn.wants_tool
    assert turn.message.function_call.name == "get_weather"
    assert turn.message.function_call.arguments == {"city": "Казань"}
    assert turn.message.functions_state_id == "sid-9"
    assert turn.finish_reason == "function_call"
    p.close()


def test_parse_response_arguments_as_string():
    """Подстраховка: если arguments прилетит строкой, парсим её."""
    p = _provider()
    data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "function_call": {"name": "f", "arguments": '{"k": "v"}'},
                },
                "finish_reason": "function_call",
            }
        ]
    }
    turn = p._parse_response(data)
    assert turn.message.function_call.arguments == {"k": "v"}
    p.close()
