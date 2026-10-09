"""Строка ввода: автодополнение, история в .devassist/, клавиши, статус-строка."""

from __future__ import annotations

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from devassist.cli.commands import default_commands
from devassist.cli.prompt import (
    LazyFileHistory,
    SlashCommandCompleter,
    StatusInfo,
    create_prompt_session,
    toolbar_fragments,
)
from devassist.project.workspace import Workspace


def _complete(text: str) -> list[tuple[str, str]]:
    completer = SlashCommandCompleter(default_commands())
    return [
        (c.text, c.display_meta_text)
        for c in completer.get_completions(Document(text), CompleteEvent())
    ]


def test_completes_commands_with_descriptions():
    assert _complete("/mo") == [("/model", "сменить модель")]
    assert {name for name, _ in _complete("/")} == {
        "/help",
        "/model",
        "/clear",
        "/resume",
        "/index",
        "/exit",
    }
    assert _complete("/q") == [("/quit", "выход")]  # по алиасу
    assert _complete("/MO") == [("/model", "сменить модель")]


@pytest.mark.parametrize("text", ["", "привет", "/home/x", "/model Giga", "текст /mo"])
def test_no_completion_outside_command_name(text):
    assert _complete(text) == []


def _complete_args(text: str) -> list[tuple[str, str]]:
    models = [
        ("GigaChat-2-Max", "текущая · окно 128k"),
        ("GigaChat-2-Pro", "окно не замерено"),
        ("Qwen-Max", "окно 32k"),
    ]
    completer = SlashCommandCompleter(default_commands(), {"/model": lambda: models})
    return [
        (c.text, c.display_meta_text)
        for c in completer.get_completions(Document(text), CompleteEvent())
    ]


def test_completes_model_argument():
    assert [name for name, _ in _complete_args("/model ")] == [
        "GigaChat-2-Max",
        "GigaChat-2-Pro",
        "Qwen-Max",
    ]
    assert _complete_args("/model giga") == [
        ("GigaChat-2-Max", "текущая · окно 128k"),
        ("GigaChat-2-Pro", "окно не замерено"),
    ]
    # сначала совпадения по началу, затем по подстроке
    assert [name for name, _ in _complete_args("/MODEL max")] == ["GigaChat-2-Max", "Qwen-Max"]
    assert _complete_args("/model Qwen-Max") == [("Qwen-Max", "окно 32k")]


@pytest.mark.parametrize("text", ["/model", "/model a b", "/help ", "/nope Giga", "x /model "])
def test_no_model_completion_elsewhere(text):
    assert [name for name, _ in _complete_args(text)] == (["/model"] if text == "/model" else [])


def test_history_creates_data_dir_lazily(tmp_path):
    ws = Workspace(tmp_path)
    history = LazyFileHistory(ws)
    assert list(history.load_history_strings()) == []
    assert not ws.data_dir.exists()
    history.append_string("первый запрос")
    assert (ws.data_dir / ".gitignore").is_file()
    assert list(LazyFileHistory(ws).load_history_strings()) == ["первый запрос"]


def test_history_write_errors_are_swallowed(tmp_path):
    class ReadOnly(Workspace):
        def ensure_data_dir(self):
            raise PermissionError("только чтение")

    history = LazyFileHistory(ReadOnly(tmp_path))
    history.append_string("a")
    history.append_string("b")  # не падает и не пытается снова
    assert list(history.get_strings()) == ["a", "b"]  # в памяти осталась


def _prompt(tmp_path, keys: str) -> str:
    with create_pipe_input() as pipe:
        session = create_prompt_session(
            commands=default_commands(),
            workspace=Workspace(tmp_path),
            status=lambda: StatusInfo(model="m"),
            input=pipe,
            output=DummyOutput(),
        )
        pipe.send_text(keys)
        return session.prompt()


def test_backslash_enter_inserts_newline(tmp_path):
    assert _prompt(tmp_path, "строка 1\\\rстрока 2\r") == "строка 1\nстрока 2"


def test_alt_enter_inserts_newline(tmp_path):
    assert _prompt(tmp_path, "a\x1b\rb\r") == "a\nb"


def test_plain_enter_submits(tmp_path):
    assert _prompt(tmp_path, "привет\r") == "привет"
    assert (tmp_path / ".devassist" / "history").is_file()


def _styles(status: StatusInfo) -> dict[str, str]:
    return {text: style for style, text in toolbar_fragments(status)}


@pytest.mark.parametrize(
    ("used", "style"),
    [(1_000, "class:toolbar.ok"), (35_000, "class:toolbar.warn"), (55_000, "class:toolbar.danger")],
)
def test_toolbar_context_fill_colors(used, style):
    status = StatusInfo(model="GigaChat", context_tokens=used, context_budget=60_000)
    fill = [s for s, t in toolbar_fragments(status) if "/60k" in t]
    assert fill == [style]


def test_toolbar_contents():
    text = "".join(
        t
        for _, t in toolbar_fragments(
            StatusInfo(
                model="GigaChat-2-Max",
                context_tokens=12_345,
                context_budget=60_000,
                billed_tokens=45_100,
                auto_approve=True,
            )
        )
    )
    assert "GigaChat-2-Max" in text and "12.3k/60k (21%)" in text
    assert "потрачено 45.1k" in text and "авто-подтверждение" in text
    assert "авто" not in "".join(t for _, t in toolbar_fragments(StatusInfo(model="m")))


def test_double_escape_clears_input_into_history(tmp_path):
    with create_pipe_input() as pipe:
        session = create_prompt_session(
            commands=default_commands(),
            workspace=Workspace(tmp_path),
            status=lambda: StatusInfo(model="m"),
            input=pipe,
            output=DummyOutput(),
        )
        pipe.send_text("черновик\x1b\x1bok\r")
        assert session.prompt() == "ok"
    assert list(session.history.get_strings()) == ["черновик", "ok"]


def test_default_text_is_prefilled(tmp_path):
    with create_pipe_input() as pipe:
        session = create_prompt_session(
            commands=default_commands(),
            workspace=Workspace(tmp_path),
            status=lambda: StatusInfo(model="m"),
            input=pipe,
            output=DummyOutput(),
        )
        pipe.send_text(" ещё\r")
        assert session.prompt(default="набрано") == "набрано ещё"
