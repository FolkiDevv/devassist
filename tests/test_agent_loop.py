"""Офлайн-тесты агентного цикла (без сети): анти-залипание и базовый поток."""

from __future__ import annotations

from pathlib import Path

from devassist.agent.loop import Agent
from devassist.agent.session import Session
from devassist.config import Config
from devassist.llm.base import LLMProvider
from devassist.llm.types import AssistantTurn, FunctionCall, Message
from devassist.ui.console import Console


class ScriptedProvider(LLMProvider):
    """Провайдер-заглушка: выдаёт заранее заданную последовательность ходов."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.calls = 0

    @property
    def model(self) -> str:
        return "scripted"

    def complete(self, messages, tools=None, *, temperature=0.2) -> AssistantTurn:
        self.calls += 1
        if self._turns:
            return self._turns.pop(0)
        return AssistantTurn(message=Message(role="assistant", content="конец"))

    def stream(self, messages, tools=None, *, temperature=0.2, on_delta=None):
        return self.complete(messages, tools, temperature=temperature)


def _agent(provider, tmp_path: Path, **cfg_kw) -> Agent:
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, **cfg_kw)
    from devassist.tools.base import build_default_registry

    ui = Console(no_color=True, assume_yes=True)
    return Agent(provider, build_default_registry(), cfg, ui, session=Session(tmp_path))


def _tool_turn(name, args):
    return AssistantTurn(
        message=Message(
            role="assistant",
            content="",
            function_call=FunctionCall(name=name, arguments=args),
        ),
        finish_reason="function_call",
    )


def test_loop_stops_after_repeated_failures(tmp_path):
    # модель упорно вызывает read_file на несуществующем файле
    bad = _tool_turn("read_file", {"path": "nope.txt"})
    provider = ScriptedProvider([bad] * 20)
    agent = _agent(provider, tmp_path, max_tool_failures=4)
    final = agent.run_turn("сделай что-нибудь")
    assert "Прервано" in final
    # должно остановиться примерно на пороге, а не крутить 50 шагов
    assert provider.calls <= 5


def test_loop_completes_on_text(tmp_path):
    provider = ScriptedProvider(
        [AssistantTurn(message=Message(role="assistant", content="Готово!"))]
    )
    agent = _agent(provider, tmp_path)
    final = agent.run_turn("привет")
    assert final == "Готово!"


def test_loop_runs_tool_then_finishes(tmp_path):
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    provider = ScriptedProvider(
        [
            _tool_turn("read_file", {"path": "f.txt"}),
            AssistantTurn(message=Message(role="assistant", content="прочитал")),
        ]
    )
    agent = _agent(provider, tmp_path)
    final = agent.run_turn("прочитай f.txt")
    assert final == "прочитал"
    assert provider.calls == 2


def test_write_outside_sandbox_does_not_crash_without_auto_approve(tmp_path):
    # Раньше SandboxError из preview() пробивал run_turn и ронял REPL.
    root = tmp_path / "proj"
    root.mkdir()
    provider = ScriptedProvider(
        [
            _tool_turn("write_file", {"path": "../escape.txt", "content": "x"}),
            AssistantTurn(message=Message(role="assistant", content="не вышло")),
        ]
    )
    agent = _agent(provider, root, auto_approve=False)
    assert agent.run_turn("запиши файл") == "не вышло"
    assert not (tmp_path / "escape.txt").exists()
    results = [m for m in agent.session.messages() if m.role == "function"]
    assert "за пределы" in results[-1].content
