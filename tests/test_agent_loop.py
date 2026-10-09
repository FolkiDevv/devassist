"""Офлайн-тесты агентного цикла (без сети): поток, подтверждения, анти-залипание."""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeTool, RecordingEvents, ScriptedProvider, text_turn, tool_turn

from devassist.agent.loop import Agent
from devassist.config import Config
from devassist.errors import ToolError
from devassist.llm.types import Usage
from devassist.permissions import PermissionMode, ToolKind
from devassist.tools.base import Display, ToolRegistry, ToolResult, build_default_registry
from devassist.tools.plan import APPROVE_EDITS
from devassist.tools.questions import Answer


def _agent(provider, tmp_path: Path, *, events=None, registry=None, **cfg_kw) -> Agent:
    cfg_kw.setdefault("auto_approve", True)
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, **cfg_kw)
    return Agent(
        provider,
        registry or build_default_registry(),
        cfg,
        events or RecordingEvents(),
    )


def _registry_with(tool) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(tool)
    return reg


def test_loop_stops_after_repeated_failures(tmp_path):
    # модель упорно вызывает read_file на несуществующих файлах (разных — иначе
    # первым сработает детектор повторов)
    bad = [tool_turn("read_file", {"path": f"nope{i}.txt"}) for i in range(20)]
    provider = ScriptedProvider(bad)
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, max_tool_failures=4)
    final = agent.run_turn("сделай что-нибудь")
    assert "Прервано" in final
    # должно остановиться на пороге, а не крутить 50 шагов
    assert provider.calls == 4
    assert events.stats[-1].stop_reason == "tool_failures"
    assert events.notices and events.notices[-1][0] == "error"


def _function_results(agent) -> list[str]:
    return [m.content for m in agent.conversation.messages if m.role == "function"]


def test_loop_stops_on_repeated_identical_calls(tmp_path):
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    provider = ScriptedProvider([tool_turn("read_file", {"path": "f.txt"})] * 10)
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)
    final = agent.run_turn("прочитай")
    assert "зациклился" in final
    assert provider.calls == 4
    assert events.stats[-1].stop_reason == "tool_repeats"
    assert events.stats[-1].tool_calls == 3  # 4-й вызов не выполнялся
    assert [level for level, _ in events.notices] == ["warn", "error"]
    results = _function_results(agent)
    assert len(results) == 4
    assert ["ВНИМАНИЕ" in r for r in results] == [False, False, True, False]
    assert "hello" in results[2]  # предупреждение дописано к результату
    assert "зацикливания" in results[3]
    assert agent.conversation.pending_call() is None
    _assert_well_formed(agent.conversation.messages)


def test_loop_stops_on_repeated_identical_changes(tmp_path):
    # изменяющая операция без изменений между повторами (как pytest без правок)
    tool = FakeTool()
    provider = ScriptedProvider([tool_turn("fake_write", {})] * 10)
    events = RecordingEvents()
    _agent(provider, tmp_path, events=events, registry=_registry_with(tool)).run_turn("x")
    assert tool.runs == 3
    assert events.stats[-1].stop_reason == "tool_repeats"


def test_change_between_repeats_is_not_a_loop(tmp_path):
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    read = tool_turn("read_file", {"path": "f.txt"})
    write = tool_turn("write_file", {"path": "g.txt", "content": "x"})
    provider = ScriptedProvider([read, read, read, write, read, read, text_turn("ок")])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)
    assert agent.run_turn("x") == "ок"
    assert events.stats[-1].stop_reason is None
    assert sum("ВНИМАНИЕ" in r for r in _function_results(agent)) == 1  # только 3-е чтение


def test_repeated_rejected_call_is_not_asked_again(tmp_path):
    tool = FakeTool()
    call = tool_turn("fake_write", {})
    provider = ScriptedProvider([call, call, text_turn("ок"), call, text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(
        provider, tmp_path, events=events, registry=_registry_with(tool), auto_approve=False
    )
    assert agent.run_turn("x") == "ок"
    assert len(events.confirms) == 1 and tool.runs == 0
    assert "уже ОТКЛОНИЛ" in _function_results(agent)[-1]
    # в следующем ходе пользователя спрашивают снова
    agent.run_turn("всё же сделай")
    assert len(events.confirms) == 2


def test_rejections_do_not_count_as_failures(tmp_path):
    tool = FakeTool()
    calls = [tool_turn("fake_write", {"path": p}) for p in "abc"]
    provider = ScriptedProvider([*calls, text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(
        provider,
        tmp_path,
        events=events,
        registry=_registry_with(tool),
        auto_approve=False,
        max_tool_failures=2,
    )
    assert agent.run_turn("x") == "ок"
    assert len(events.confirms) == 3
    assert events.stats[-1].stop_reason is None


def test_loop_completes_on_text(tmp_path):
    provider = ScriptedProvider([text_turn("Готово!")])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)
    assert agent.run_turn("привет") == "Готово!"
    assert ("text", "Готово!") in events.events


def test_loop_runs_tool_then_finishes(tmp_path):
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    provider = ScriptedProvider([tool_turn("read_file", {"path": "f.txt"}), text_turn("прочитал")])
    agent = _agent(provider, tmp_path)
    assert agent.run_turn("прочитай f.txt") == "прочитал"
    assert provider.calls == 2


def test_streaming_events_wrap_each_request(tmp_path):
    provider = ScriptedProvider([text_turn("поток")])
    events = RecordingEvents()
    cfg = Config(access_key="x", project_root=tmp_path, stream=True)
    agent = Agent(provider, build_default_registry(), cfg, events)
    agent.run_turn("hi")
    kinds = [k for k, _ in events.events]
    assert kinds == ["stream_start", "delta", "stream_end"]


def test_write_outside_sandbox_does_not_crash_without_auto_approve(tmp_path):
    # Раньше SandboxError из preview() пробивал run_turn и ронял REPL.
    root = tmp_path / "proj"
    root.mkdir()
    provider = ScriptedProvider(
        [tool_turn("write_file", {"path": "../escape.txt", "content": "x"}), text_turn("не вышло")]
    )
    events = RecordingEvents()
    agent = _agent(provider, root, events=events, auto_approve=False)
    assert agent.run_turn("запиши файл") == "не вышло"
    assert not (tmp_path / "escape.txt").exists()
    assert events.confirms == []  # невыполнимую операцию не предлагаем подтверждать
    results = [m for m in agent.conversation.messages if m.role == "function"]
    assert "за пределы" in results[-1].content


@pytest.mark.parametrize("exc", [ToolError("нельзя"), ValueError("сбой превью")])
def test_preview_failure_never_runs_tool(tmp_path, exc):
    def boom():
        raise exc

    tool = FakeTool(preview=boom)
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents()
    agent = _agent(
        provider, tmp_path, events=events, registry=_registry_with(tool), auto_approve=False
    )
    agent.run_turn("x")
    assert tool.runs == 0
    assert events.confirms == []
    assert events.results[-1][1].ok is False


def test_rejection_does_not_run_and_tells_model(tmp_path):
    tool = FakeTool()
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(
        provider, tmp_path, events=events, registry=_registry_with(tool), auto_approve=False
    )
    agent.run_turn("x")
    assert tool.runs == 0
    call, preview, dangerous = events.confirms[0]
    assert call.name == "fake_write" and preview.kind == "diff" and dangerous is False
    function_msgs = [m for m in agent.conversation.messages if m.role == "function"]
    assert "ОТКЛОНИЛ" in function_msgs[-1].content


def test_auto_approve_skips_confirmation(tmp_path):
    tool = FakeTool()
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(provider, tmp_path, events=events, registry=_registry_with(tool))
    agent.run_turn("x")
    assert tool.runs == 1
    assert events.confirms == []


def test_custom_tool_display_reaches_ui(tmp_path):
    display = Display("+new line", kind="diff", title="x")
    tool = FakeTool(run=lambda: ToolResult(content="ok", summary="готово", display=display))
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=True)
    agent = _agent(
        provider, tmp_path, events=events, registry=_registry_with(tool), auto_approve=False
    )
    agent.run_turn("x")
    call, result, previewed = events.results[-1]
    assert result.display == display and previewed is True


def test_disallowed_git_subcommand_is_not_confirmed(tmp_path):
    provider = ScriptedProvider([tool_turn("git", {"subcommand": "push"}), text_turn("ок")])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, auto_approve=False)
    agent.run_turn("запушь")
    assert events.confirms == []
    assert events.results[-1][1].ok is False


def test_turn_stats_split_billed_and_context(tmp_path):
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    provider = ScriptedProvider(
        [
            tool_turn(
                "read_file", {"path": "f.txt"}, Usage(prompt_tokens=100, completion_tokens=10)
            ),
            text_turn("ок", Usage(prompt_tokens=130, completion_tokens=20)),
        ]
    )
    events = RecordingEvents()
    _agent(provider, tmp_path, events=events).run_turn("x")
    stats = events.stats[-1]
    assert (stats.steps, stats.tool_calls) == (2, 1)
    assert stats.billed_tokens == 260
    assert stats.context_tokens == 150


def _assert_well_formed(messages):
    for i, m in enumerate(messages):
        if m.role == "function":
            prev = messages[i - 1]
            assert prev.role == "assistant" and prev.function_call is not None


def test_ctrl_c_during_tool_repairs_history(tmp_path):
    def interrupted():
        raise KeyboardInterrupt

    tool = FakeTool(run=interrupted)
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("снова тут")])
    agent = _agent(provider, tmp_path, registry=_registry_with(tool))
    with pytest.raises(KeyboardInterrupt):
        agent.run_turn("x")
    assert agent.conversation.pending_call() is None
    # следующий ход отправляет корректную историю
    assert agent.run_turn("продолжай") == "снова тут"
    _assert_well_formed(provider.requests[-1]["messages"])


def test_set_model_is_passed_to_provider(tmp_path):
    provider = ScriptedProvider([text_turn("a"), text_turn("b")])
    agent = _agent(provider, tmp_path)
    agent.run_turn("1")
    agent.set_model("GigaChat-2-Max")
    agent.run_turn("2")
    assert [r["model"] for r in provider.requests] == [agent._cfg.model, "GigaChat-2-Max"]
    with pytest.raises(ValueError):
        agent.set_model("  ")


def test_reset_rebuilds_system_prompt(tmp_path):
    provider = ScriptedProvider([text_turn("a"), text_turn("b")])
    agent = _agent(provider, tmp_path)
    agent.run_turn("1")
    (tmp_path / "new_file.py").write_text("x", encoding="utf-8")
    agent.reset()
    assert len(agent.conversation) == 0
    agent.run_turn("2")
    system = provider.requests[-1]["messages"][0]
    assert system.role == "system" and "new_file.py" in system.content
    assert [m.role for m in provider.requests[-1]["messages"]] == ["system", "user"]


def test_reset_with_saved_conversation_continues_it(tmp_path):
    from devassist.agent.conversation import Conversation

    saved = Conversation()
    saved.add_user("старый запрос")
    saved.add_assistant(
        text_turn("старый ответ").message, Usage(prompt_tokens=50, completion_tokens=5)
    )
    provider = ScriptedProvider([text_turn("новый ответ")])
    agent = _agent(provider, tmp_path)
    agent.reset(saved)
    assert agent.conversation is saved and agent.context_tokens == 55
    agent.run_turn("дальше")
    roles = [(m.role, m.content) for m in provider.requests[-1]["messages"][1:]]
    assert roles == [
        ("user", "старый запрос"),
        ("assistant", "старый ответ"),
        ("user", "дальше"),
    ]


def test_constructor_has_no_filesystem_side_effects(tmp_path, monkeypatch):
    import devassist.agent.loop as loop

    def boom(_ws):
        raise AssertionError("системный промпт собран в конструкторе")

    monkeypatch.setattr(loop, "build_system_prompt", boom)
    _agent(ScriptedProvider(), tmp_path)  # не падает


def test_long_turn_keeps_current_task_in_context(tmp_path):
    (tmp_path / "big.txt").write_text("y" * 5000, encoding="utf-8")
    steps = [tool_turn("read_file", {"path": "big.txt"}) for _ in range(6)] + [text_turn("ок")]
    provider = ScriptedProvider(steps)
    agent = _agent(provider, tmp_path, context_budget_tokens=4000)
    agent.run_turn("ТЕКУЩАЯ ЗАДАЧА")
    for request in provider.requests:
        contents = [m.content for m in request["messages"] if m.role == "user"]
        assert contents == ["ТЕКУЩАЯ ЗАДАЧА"]
        _assert_well_formed(request["messages"])
    assert len(provider.requests[-1]["messages"]) < len(agent.conversation) + 1


def test_empty_conversation_instance_is_used(tmp_path):
    from devassist.agent.conversation import Conversation

    conv = Conversation()  # пустой — ложен по __len__
    cfg = Config(access_key="x", project_root=tmp_path, stream=False)
    agent = Agent(
        ScriptedProvider([text_turn("ок")]), build_default_registry(), cfg, conversation=conv
    )
    agent.run_turn("x")
    assert agent.conversation is conv and len(conv) == 2


# ------------------------- события для индикаторов ------------------------- #
def _kinds(events) -> list[str]:
    return [k for k, _ in events.events]


@pytest.mark.parametrize("error", [None, ToolError("нельзя"), ValueError("сбой")])
def test_tool_start_end_bracket_run(tmp_path, error):
    def run():
        if error is not None:
            raise error
        return ToolResult(content="ok", summary="ok")

    tool = FakeTool(run=run)
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents()
    _agent(provider, tmp_path, events=events, registry=_registry_with(tool)).run_turn("x")
    kinds = _kinds(events)
    start = kinds.index("tool_start")
    assert kinds[start : start + 3] == ["tool_start", "tool_end", "tool_result"]


def test_tool_end_called_on_ctrl_c(tmp_path):
    def interrupted():
        raise KeyboardInterrupt

    tool = FakeTool(run=interrupted)
    events = RecordingEvents()
    agent = _agent(
        ScriptedProvider([tool_turn("fake_write", {})]),
        tmp_path,
        events=events,
        registry=_registry_with(tool),
    )
    with pytest.raises(KeyboardInterrupt):
        agent.run_turn("x")
    assert _kinds(events)[-2:] == ["tool_start", "tool_end"]


def test_rejected_tool_is_not_started(tmp_path):
    tool = FakeTool()
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    _agent(
        provider, tmp_path, events=events, registry=_registry_with(tool), auto_approve=False
    ).run_turn("x")
    assert "tool_start" not in _kinds(events)


def test_non_stream_mode_signals_waiting(tmp_path):
    events = RecordingEvents()
    _agent(ScriptedProvider([text_turn("ответ")]), tmp_path, events=events).run_turn("x")
    assert _kinds(events) == ["stream_start", "stream_end", "text"]


def test_non_stream_error_still_ends_waiting(tmp_path):
    from devassist.llm.base import LLMError

    events = RecordingEvents()
    agent = _agent(ScriptedProvider([LLMError("сеть")]), tmp_path, events=events)
    with pytest.raises(LLMError):
        agent.run_turn("x")
    assert _kinds(events) == ["stream_start", "stream_end"]


def test_session_counters(tmp_path):
    provider = ScriptedProvider(
        [
            text_turn("a", Usage(prompt_tokens=100, completion_tokens=10)),
            text_turn("b", Usage(prompt_tokens=200, completion_tokens=20)),
        ]
    )
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)
    assert agent.context_tokens == 0
    agent.run_turn("1")
    assert agent.context_tokens == 110 and agent.billed_tokens == 110
    agent.reset()
    assert agent.context_tokens == 0 and agent.billed_tokens == 110
    agent.run_turn("2")
    assert agent.context_tokens == 220 and agent.billed_tokens == 330
    assert events.stats[-1].duration_s >= 0
    assert agent.context_budget > 0


def test_ask_user_reaches_ui_and_model(tmp_path):
    from devassist.tools.questions import Answer

    question = {
        "question": "Какой формат?",
        "options": [{"label": "JSON"}, {"label": "YAML", "description": "читаемее"}],
    }
    provider = ScriptedProvider(
        [tool_turn("ask_user", {"questions": [question]}), text_turn("Сделаю YAML")]
    )
    events = RecordingEvents(answers=[Answer(("YAML",))])
    agent = _agent(provider, tmp_path, events=events, auto_approve=False)
    assert agent.run_turn("сохрани конфиг") == "Сделаю YAML"
    assert events.questions[0][0].options[1].description == "читаемее"
    assert events.confirms == []  # вопрос — не изменяющая операция
    sent = provider.requests[-1]["messages"][-1]
    assert sent.role == "function" and "Какой формат? → YAML" in sent.content
    kinds = _kinds(events)
    assert kinds[kinds.index("tool_start") + 1] == "tool_end"


def test_ask_user_without_ui_tells_model(tmp_path):
    from devassist.agent.events import AgentEvents

    question = {"question": "Да?", "options": [{"label": "да"}, {"label": "нет"}]}
    provider = ScriptedProvider(
        [tool_turn("ask_user", {"questions": [question]}), text_turn("решу сам")]
    )
    cfg = Config(access_key="x", project_root=tmp_path, stream=False)
    agent = Agent(provider, build_default_registry(), cfg, AgentEvents())
    assert agent.run_turn("x") == "решу сам"
    assert "Нельзя задать вопрос" in provider.requests[-1]["messages"][-1].content


# ------------------------------ окно контекста ------------------------------ #
def _probe(model: str, window: int):
    from devassist.llm.context_probe import ProbeResult

    return ProbeResult(model, window, window + 1000, False, 5, window)


def test_context_budget_follows_model_window(tmp_path):
    from devassist.agent.context_window import DEFAULT_CONTEXT_WINDOW, budget_for_window
    from devassist.llm.model_windows import ModelWindows

    windows = ModelWindows()
    windows.record(_probe("big", 128_000))
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, model="small")
    agent = Agent(ScriptedProvider(), build_default_registry(), cfg, windows=windows)
    assert agent.windows is windows
    assert agent.context_window is None
    assert agent.context_budget == budget_for_window(DEFAULT_CONTEXT_WINDOW)
    agent.set_model("big")
    assert agent.context_window == 128_000
    assert agent.context_budget == budget_for_window(128_000) < 128_000 - 8_000

    explicit = Config(access_key="x", project_root=tmp_path, context_budget_tokens=5_000)
    agent = Agent(ScriptedProvider(), build_default_registry(), explicit, windows=windows)
    agent.set_model("big")
    assert agent.context_budget == 5_000


def test_request_budget_includes_tool_schemas(tmp_path):
    from devassist.agent.conversation import Conversation
    from devassist.llm.types import Message, ToolSpec

    conv = Conversation()
    for i in range(20):
        conv.add_user(f"вопрос {i} " + "x" * 1500)
        conv.add_assistant(Message(role="assistant", content=f"ответ {i} " + "y" * 1500))
    agent = _agent(ScriptedProvider(), tmp_path, context_budget_tokens=12_000)
    agent.reset(conv)
    heavy = [ToolSpec(name="t", description="d" * 15_000, parameters={})]  # ~5000 токенов
    without = agent._build_request([])
    with_specs = agent._build_request(heavy)
    assert len(with_specs) < len(without) < len(conv) + 1
    assert with_specs[-1].content.startswith("ответ 19")


# ----------------------------- режимы разрешений ----------------------------- #
def _system_prompts(provider) -> list[str]:
    return [request["messages"][0].content for request in provider.requests]


def test_plan_mode_blocks_edits_without_asking(tmp_path):
    tool = FakeTool(kind=ToolKind.EDIT)
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=True)
    agent = _agent(
        provider,
        tmp_path,
        events=events,
        registry=_registry_with(tool),
        auto_approve=False,
        mode=PermissionMode.PLAN,
    )
    agent.run_turn("поправь")
    assert tool.runs == 0 and events.confirms == []
    assert events.results[-1][1].summary == "заблокировано: режим планирования"
    assert "режим планирования" in _function_results(agent)[-1]
    assert "exit_plan_mode" in _function_results(agent)[-1]


def test_plan_mode_blocks_even_with_auto_approve(tmp_path):
    tool = FakeTool(kind=ToolKind.EDIT)
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    agent = _agent(provider, tmp_path, registry=_registry_with(tool), mode=PermissionMode.PLAN)
    agent.run_turn("поправь")
    assert tool.runs == 0


def test_plan_mode_asks_for_commands(tmp_path):
    tool = FakeTool(kind=ToolKind.COMMAND)
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=True)
    agent = _agent(
        provider,
        tmp_path,
        events=events,
        registry=_registry_with(tool),
        auto_approve=False,
        mode=PermissionMode.PLAN,
    )
    agent.run_turn("запусти тесты")
    assert tool.runs == 1 and len(events.confirms) == 1


def test_plan_mode_stops_model_that_keeps_editing(tmp_path):
    edits = [tool_turn("write_file", {"path": f"f{i}.txt", "content": "x"}) for i in range(10)]
    provider = ScriptedProvider(edits)
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, mode=PermissionMode.PLAN)
    agent.run_turn("сделай")
    assert events.stats[-1].stop_reason == "tool_failures"
    assert list(tmp_path.iterdir()) == []


def test_accept_edits_applies_edits_without_asking(tmp_path):
    tool = FakeTool(kind=ToolKind.EDIT)
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(
        provider,
        tmp_path,
        events=events,
        registry=_registry_with(tool),
        auto_approve=False,
        mode=PermissionMode.ACCEPT_EDITS,
    )
    agent.run_turn("поправь")
    assert tool.runs == 1 and events.confirms == []


@pytest.mark.parametrize("kind", [ToolKind.COMMAND, ToolKind.OTHER])
def test_accept_edits_still_asks_for_the_rest(tmp_path, kind):
    tool = FakeTool(kind=kind)
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(
        provider,
        tmp_path,
        events=events,
        registry=_registry_with(tool),
        auto_approve=False,
        mode=PermissionMode.ACCEPT_EDITS,
    )
    agent.run_turn("x")
    assert tool.runs == 0 and len(events.confirms) == 1


def test_plan_rules_are_sent_only_in_plan_mode(tmp_path):
    provider = ScriptedProvider([text_turn("план"), text_turn("готово")])
    agent = _agent(provider, tmp_path, mode=PermissionMode.PLAN)
    agent.run_turn("спланируй")
    agent.set_mode(PermissionMode.MANUAL)
    agent.run_turn("делай")
    planning, manual = _system_prompts(provider)
    assert "РЕЖИМ ПЛАНИРОВАНИЯ" in planning
    assert "РЕЖИМ ПЛАНИРОВАНИЯ" not in manual


def test_mode_survives_reset_and_cycles(tmp_path):
    agent = _agent(ScriptedProvider(), tmp_path)
    assert agent.mode is PermissionMode.MANUAL
    assert agent.cycle_mode() is PermissionMode.ACCEPT_EDITS
    agent.reset()
    assert agent.mode is PermissionMode.ACCEPT_EDITS
    assert [agent.cycle_mode(), agent.cycle_mode()] == [PermissionMode.PLAN, PermissionMode.MANUAL]


def test_approved_plan_switches_mode_within_the_turn(tmp_path):
    provider = ScriptedProvider(
        [
            tool_turn("exit_plan_mode", {"plan": "# План\n1. создать a.txt"}),
            tool_turn("write_file", {"path": "a.txt", "content": "x"}),
            text_turn("готово"),
        ]
    )
    events = RecordingEvents(confirm_answer=False, answers=[Answer((APPROVE_EDITS,))])
    agent = _agent(provider, tmp_path, events=events, auto_approve=False, mode=PermissionMode.PLAN)
    assert agent.run_turn("спланируй и сделай") == "готово"
    assert agent.mode is PermissionMode.ACCEPT_EDITS
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "x"
    assert events.confirms == []  # правка после одобрения — уже без вопроса
    (question,) = events.questions[0]
    assert question.body == "# План\n1. создать a.txt"
    # правила плана — только до одобрения
    assert ["РЕЖИМ ПЛАНИРОВАНИЯ" in p for p in _system_prompts(provider)] == [True, False, False]
