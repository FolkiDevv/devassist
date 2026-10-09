"""Тесты истории диалога, окна контекста и ограничителей цикла."""

from __future__ import annotations

import json

import pytest

from devassist.agent.context_window import estimate_tokens, fit_history
from devassist.agent.conversation import Conversation
from devassist.agent.guard import LoopGuard
from devassist.llm.types import FunctionCall, Message, Usage


def _call(name="read_file", **args):
    return Message(
        role="assistant",
        function_call=FunctionCall(name=name, arguments=args),
        functions_state_id="s",
    )


def _result(name="read_file", content="ok"):
    return Message(role="function", name=name, content=content)


# ------------------------------ Conversation ------------------------------ #
def test_round_trip_preserves_everything():
    conv = Conversation()
    conv.add_user("привет")
    conv.add_assistant(_call(path="a.py"), Usage(prompt_tokens=10, completion_tokens=2))
    conv.add_function_result("read_file", "1\tx")
    conv.add_assistant(Message(role="assistant", content="готово"))

    data = json.loads(json.dumps(conv.to_dict(), ensure_ascii=False))
    restored = Conversation.from_dict(data)
    assert restored.messages == conv.messages
    assert restored.last_usage == Usage(prompt_tokens=10, completion_tokens=2)


def test_from_dict_rejects_unknown_version():
    with pytest.raises(ValueError):
        Conversation.from_dict({"version": 999, "messages": []})


def test_repair_closes_pending_call():
    conv = Conversation()
    conv.add_user("x")
    assert conv.repair() is False
    conv.add_assistant(_call(path="a"))
    assert conv.pending_call().name == "read_file"
    assert conv.repair() is True
    assert conv.messages[-1].role == "function" and conv.messages[-1].name == "read_file"
    assert conv.pending_call() is None
    assert conv.repair() is False


# ------------------------------ fit_history ------------------------------ #
def _history():
    big = "x" * 3000  # ~1000 токенов
    return [
        Message(role="user", content="старая задача " + big),
        Message(role="assistant", content="старый ответ " + big),
        Message(role="user", content="ТЕКУЩАЯ ЗАДАЧА"),
        _call(path="1"),
        _result(content=big),
        _call(path="2"),
        _result(content=big),
        _call(path="3"),
        _result(content="последний результат"),
    ]


def test_fit_history_unchanged_under_budget():
    history = _history()
    assert fit_history(history, 10**6) == history


def test_fit_history_pins_task_and_keeps_pairs():
    history = _history()
    fitted = fit_history(history, 1500)
    assert estimate_tokens(fitted) <= 1500
    assert any(m.content == "ТЕКУЩАЯ ЗАДАЧА" for m in fitted)
    assert fitted[-1].content == "последний результат"
    assert fitted[0].role != "function"
    for i, m in enumerate(fitted):
        if m.role == "function":
            prev = fitted[i - 1]
            assert prev.role == "assistant" and prev.function_call is not None


def test_fit_history_keeps_pinned_even_over_budget():
    history = _history()
    fitted = fit_history(history, 10)
    assert [m.content for m in fitted if m.role == "user"] == ["ТЕКУЩАЯ ЗАДАЧА"]
    assert fitted[-1].content == "последний результат"


def test_fit_history_drops_leading_orphan_result():
    assert fit_history([_result(), Message(role="user", content="q")], 10**6)[0].role == "user"


# ------------------------------ LoopGuard ------------------------------ #
def test_guard_step_limit():
    guard = LoopGuard(max_steps=2, max_failures=10)
    assert guard.before_step() is None
    assert guard.before_step() is None
    stop = guard.before_step()
    assert stop is not None and stop.kind == "max_steps"


def test_guard_failures_reset_on_success():
    guard = LoopGuard(max_steps=100, max_failures=3)
    call = FunctionCall(name="x")
    assert guard.after_tool(call, False) is None
    assert guard.after_tool(call, False) is None
    assert guard.after_tool(call, True) is None  # успех сбрасывает серию
    assert guard.after_tool(call, False) is None
    assert guard.after_tool(call, False) is None
    stop = guard.after_tool(call, False)
    assert stop is not None and stop.kind == "tool_failures"
