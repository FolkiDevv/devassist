"""Тесты парсинга ввода REPL (без сети)."""

from __future__ import annotations

import pytest

from devassist.cli import is_repl_command


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
