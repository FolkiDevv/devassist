"""Офлайн-тесты агентного цикла (без сети): поток, подтверждения, анти-залипание."""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeTool, RecordingEvents, ScriptedProvider, text_turn, tool_turn

from devassist.agent.loop import Agent
from devassist.agent.session import Session
from devassist.config import Config
from devassist.errors import ToolError
from devassist.llm.types import Usage
from devassist.tools.base import Display, ToolRegistry, ToolResult, build_default_registry


def _agent(provider, tmp_path: Path, *, events=None, registry=None, **cfg_kw) -> Agent:
    cfg_kw.setdefault("auto_approve", True)
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, **cfg_kw)
    return Agent(
        provider,
        registry or build_default_registry(),
        cfg,
        events or RecordingEvents(),
        session=Session(tmp_path),
    )


def _registry_with(tool) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(tool)
    return reg


def test_loop_stops_after_repeated_failures(tmp_path):
    # модель упорно вызывает read_file на несуществующем файле
    bad = tool_turn("read_file", {"path": "nope.txt"})
    provider = ScriptedProvider([bad] * 20)
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, max_tool_failures=4)
    final = agent.run_turn("сделай что-нибудь")
    assert "Прервано" in final
    # должно остановиться на пороге, а не крутить 50 шагов
    assert provider.calls == 4
    assert events.stats[-1].stop_reason == "tool_failures"
    assert events.notices and events.notices[-1][0] == "error"


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
    agent = Agent(provider, build_default_registry(), cfg, events, session=Session(tmp_path))
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
    results = [m for m in agent.session.messages() if m.role == "function"]
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
    function_msgs = [m for m in agent.session.messages() if m.role == "function"]
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
