"""Интерактивный режим (REPL).

Семантика клавиш:
  * Ctrl+C на приглашении — сбросить ввод (не выход);
  * Ctrl+C во время ответа — прервать текущий ход агента;
  * Ctrl+D (EOF) или /exit — выход.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from devassist import __version__
from devassist.agent.loop import Agent
from devassist.cli.commands import CommandContext, CommandRegistry, is_repl_command
from devassist.llm.base import LLMError
from devassist.ui.console import Console

InputReader = Callable[[], str]


def make_input_reader() -> InputReader:
    """Чтение строки: prompt_toolkit (история, редактирование) или input().

    Если ввод не из терминала (pipe, CI), используется простой input().
    """

    def read_plain() -> str:
        return input("devassist> ")

    if not sys.stdin.isatty():
        return read_plain
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import InMemoryHistory

        session = PromptSession(history=InMemoryHistory())

        def read_input() -> str:
            return session.prompt("devassist› ")

        return read_input
    except Exception:  # prompt_toolkit недоступен или терминал не поддерживается
        return read_plain


def run_repl(
    agent: Agent,
    ui: Console,
    commands: CommandRegistry,
    *,
    read_input: InputReader | None = None,
) -> int:
    read = read_input or make_input_reader()
    ui.banner(
        version=__version__,
        model=agent.model,
        root=str(agent.workspace.root),
        hints=[(cmd.name, cmd.summary) for cmd in commands],
    )
    ctx = CommandContext(agent=agent, ui=ui, commands=commands)

    while True:
        try:
            line = read().strip()
        except EOFError:
            ui.system("\nдо встречи!")
            return 0
        except KeyboardInterrupt:
            ui.system("(Ctrl+D или /exit — выход)")
            continue
        if not line:
            continue

        if is_repl_command(line):
            if not commands.dispatch(line, ctx):
                return 0
            continue

        try:
            ui.print()
            agent.run_turn(line)
            ui.print()
        except LLMError as e:
            ui.error(str(e))
        except KeyboardInterrupt:
            ui.system("\n(прервано)")
        except Exception as e:  # ошибка внутри хода не должна закрывать REPL
            ui.error(f"внутренняя ошибка: {type(e).__name__}: {e}")
