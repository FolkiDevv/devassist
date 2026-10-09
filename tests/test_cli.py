"""Тесты CLI (без сети): разбор команд, реестр, REPL, одноразовый режим."""

from __future__ import annotations

import pytest
from fakes import ScriptedProvider, text_turn, tool_turn

import devassist.cli.app as app
from devassist.agent.loop import Agent
from devassist.cli.commands import (
    CommandContext,
    SlashCommand,
    default_commands,
    is_repl_command,
)
from devassist.cli.repl import run_repl
from devassist.config import Config
from devassist.tools.base import build_default_registry
from devassist.ui.console import Console


@pytest.mark.parametrize(
    "line",
    ["/help", "/exit", "/q", "/clear", "/model GigaChat-2-Max", "/model"],
)
def test_recognised_commands(line):
    assert is_repl_command(line) is True


@pytest.mark.parametrize(
    "line",
    [
        "/home/kestrel/repos/devassist",  # абсолютный путь — НЕ команда
        "/usr/bin/python3 запусти это",
        "/",  # просто слеш
        "проанализируй /home/kestrel/x",  # путь внутри запроса
        "посмотри код",
        "",
    ],
)
def test_non_commands_go_to_agent(line):
    assert is_repl_command(line) is False


# ------------------------------ реестр ------------------------------ #
@pytest.fixture
def cli_env(tmp_path, capsys):
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, auto_approve=True)
    provider = ScriptedProvider()
    ui = Console(no_color=True)
    agent = Agent(provider, build_default_registry(), cfg, ui)
    commands = default_commands()
    ctx = CommandContext(agent=agent, ui=ui, commands=commands)
    return agent, ui, commands, ctx, provider


def _out(capsys) -> str:
    return capsys.readouterr().out


def test_help_lists_all_commands(cli_env, capsys):
    _, _, commands, ctx, _ = cli_env
    assert commands.dispatch("/help", ctx) is True
    out = _out(capsys)
    for cmd in commands:
        assert cmd.name in out
    assert "/quit" in out


def test_aliases_and_case_insensitive(cli_env, capsys):
    _, _, commands, ctx, _ = cli_env
    assert commands.get("/Q") is commands.get("/exit")
    assert commands.dispatch("/QUIT", ctx) is False


def test_unknown_command_reports_and_continues(cli_env, capsys):
    _, _, commands, ctx, _ = cli_env
    assert commands.dispatch("/nope", ctx) is True
    assert "неизвестная команда" in _out(capsys)


def test_duplicate_registration_rejected(cli_env):
    _, _, commands, _, _ = cli_env
    with pytest.raises(ValueError):
        commands.register(SlashCommand("/x", "x", lambda c, a: True, aliases=("/help",)))


def test_model_and_clear_commands(cli_env, capsys):
    agent, _, commands, ctx, _ = cli_env
    commands.dispatch("/model GigaChat-2-Max", ctx)
    assert agent.model == "GigaChat-2-Max"
    commands.dispatch("/model", ctx)
    assert "GigaChat-2-Max" in _out(capsys)
    agent.conversation.add_user("старое")
    commands.dispatch("/clear", ctx)
    assert len(agent.conversation) == 0


# ------------------------------- REPL ------------------------------- #
def _reader(*items):
    queue = list(items)

    def read():
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return read


def test_repl_ctrl_c_at_prompt_does_not_exit(cli_env, capsys):
    agent, ui, commands, _, _ = cli_env
    read = _reader(KeyboardInterrupt(), "/exit")
    assert run_repl(agent, ui, commands, read_input=read) == 0
    out = _out(capsys)
    assert "Ctrl+D" in out and "/model" in out  # подсказка + баннер из реестра


def test_repl_survives_errors_in_turn(cli_env, capsys, monkeypatch):
    agent, ui, commands, _, provider = cli_env

    def broken(_line):
        raise RuntimeError("что-то сломалось")

    monkeypatch.setattr(agent, "run_turn", broken)
    read = _reader("сделай", "ещё раз", EOFError())
    assert run_repl(agent, ui, commands, read_input=read) == 0
    assert _out(capsys).count("что-то сломалось") == 2


# ---------------------------- one-shot e2e ---------------------------- #
@pytest.fixture
def oneshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIGACHAT_ACCESS_KEY", "dummy")
    for name in ("GIGACHAT_TIMEOUT", "DEVASSIST_TEMPERATURE", "DEVASSIST_CONTEXT_TOKENS"):
        monkeypatch.delenv(name, raising=False)

    def use(provider):
        monkeypatch.setattr(app, "_make_provider", lambda _cfg: provider)

    return use


def test_oneshot_end_to_end(tmp_path, oneshot, capsys):
    (tmp_path / "hello.txt").write_text("привет из файла", encoding="utf-8")
    provider = ScriptedProvider(
        [tool_turn("read_file", {"path": "hello.txt"}), text_turn("В файле приветствие.")]
    )
    oneshot(provider)
    code = app.main(["-C", str(tmp_path), "-p", "что в hello.txt?", "--no-color", "--no-stream"])
    assert code == 0
    out = capsys.readouterr().out
    assert "В файле приветствие." in out and "read_file" in out
    assert "привет из файла" in provider.requests[-1]["messages"][-1].content
    assert not (tmp_path / ".devassist").exists()  # одноразовый запуск ничего не пишет


def test_oneshot_ctrl_c_exit_code(tmp_path, oneshot):
    oneshot(ScriptedProvider([KeyboardInterrupt()]))
    assert app.main(["-C", str(tmp_path), "-p", "x", "--no-color"]) == 130


def test_oneshot_llm_error_exit_code(tmp_path, oneshot):
    from devassist.llm.base import LLMError

    oneshot(ScriptedProvider([LLMError("сеть упала")]))
    assert app.main(["-C", str(tmp_path), "-p", "x", "--no-color"]) == 2


def test_missing_project_dir(tmp_path, oneshot):
    assert app.main(["-C", str(tmp_path / "nope"), "-p", "x", "--no-color"]) == 1


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        app.main(["--version"])
    assert exc.value.code == 0
    assert "devassist" in capsys.readouterr().out


def test_repl_status_and_auto_approve_banner(cli_env, capsys):
    from devassist.cli.repl import status_of

    agent, ui, commands, _, _ = cli_env  # cli_env собран с auto_approve=True
    status = status_of(agent)
    assert status.model == agent.model and status.auto_approve is True
    assert status.context_budget == agent.config.context_budget_tokens
    assert run_repl(agent, ui, commands, read_input=_reader("/exit")) == 0
    assert "авто-подтверждение" in _out(capsys)


def test_repl_prefills_typeahead_and_uses_esc(cli_env, capsys):
    import contextlib

    agent, ui, commands, _, provider = cli_env

    class FakeEsc:
        enabled = True
        entered = 0

        def __enter__(self):
            self.entered += 1
            return self

        def __exit__(self, *exc):
            return None

        def paused(self):
            return contextlib.nullcontext()

        def take_typeahead(self):
            return "набрано во время хода"

    calls = []
    answers = ["привет", "/exit"]

    def read(**kwargs):
        calls.append(kwargs)
        return answers.pop(0)

    esc = FakeEsc()
    assert run_repl(agent, ui, commands, read_input=read, interrupt=esc) == 0
    assert esc.entered == 1  # ход — внутри перехвата Esc, команды — нет
    assert calls == [{}, {"default": "набрано во время хода"}]
    assert ui._interrupt_hint == "Esc — прервать"


def test_ctrl_c_discards_prefilled_typeahead(cli_env):
    import contextlib

    agent, ui, commands, _, _ = cli_env

    class FakeEsc:
        enabled = True

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def paused(self):
            return contextlib.nullcontext()

        def take_typeahead(self):
            return "набрано"

    calls = []
    answers = ["привет", KeyboardInterrupt(), "/exit"]

    def read(**kwargs):
        calls.append(kwargs)
        item = answers.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    run_repl(agent, ui, commands, read_input=read, interrupt=FakeEsc())
    assert calls == [{}, {"default": "набрано"}, {}]


def test_plain_reader_keeps_typeahead(monkeypatch):
    from devassist.cli import prompt

    monkeypatch.setattr("builtins.input", lambda p: " и ещё")
    assert prompt._read_plain("набрано") == "набрано и ещё"
