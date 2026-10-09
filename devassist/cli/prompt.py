"""Строка ввода REPL на prompt_toolkit.

* автодополнение слеш-команд из :class:`~devassist.cli.commands.CommandRegistry`
  (имя + краткое описание), меню появляется при вводе ``/``;
* история ввода в ``.devassist/history`` (папка создаётся при первом сохранённом
  вводе) и подсказки из истории;
* многострочный ввод: ``\\`` в конце строки + Enter или Alt+Enter — новая строка;
  Esc Esc — очистить ввод (черновик сохраняется в истории);
  вставка многострочного текста не отправляет его;
* статус-строка: модель, заполнение контекста, потраченные токены, режим ``-y``.

Если stdin/stdout не терминал (pipe, CI) или prompt_toolkit не смог запуститься,
используется обычный ``input()``.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import Style

from devassist.project.workspace import Workspace
from devassist.ui.format import format_tokens
from devassist.ui.theme import PROMPT_STYLES

if TYPE_CHECKING:  # commands импортирует KEY_HELP отсюда
    from devassist.cli.commands import CommandRegistry

# Чтение строки; ``default`` — текст, которым заполнить ввод (набранный во время хода).
InputReader = Callable[..., str]

HISTORY_FILE = "history"
PROMPT = "❯ "
PLAIN_PROMPT = "devassist> "
PLACEHOLDER = "сообщение…  / — команды · \\ и Enter — новая строка"

# Подсказки по клавишам для /help.
KEY_HELP: tuple[tuple[str, str], ...] = (
    ("Enter", "отправить"),
    ("\\ + Enter, Alt+Enter", "новая строка (вставка многострочного текста не отправляет)"),
    ("Tab, ↑ ↓", "автодополнение команд, история ввода"),
    ("→", "принять подсказку из истории"),
    ("Esc", "во время ответа — прервать ход"),
    ("Esc Esc", "очистить ввод (черновик остаётся в истории, ↑ вернёт его)"),
    ("Ctrl+C", "сбросить ввод; во время ответа — прервать ход"),
    ("Ctrl+D", "выход"),
)

_COMMAND_PREFIX_RE = re.compile(r"/[\w-]*")
_CONTEXT_WARN = 50  # % заполнения контекста — жёлтый
_CONTEXT_DANGER = 80  # % — красный


# ------------------------------ статус-строка ------------------------------ #
@dataclass(frozen=True)
class StatusInfo:
    model: str
    context_tokens: int = 0
    context_budget: int = 0
    billed_tokens: int = 0
    auto_approve: bool = False


def toolbar_fragments(status: StatusInfo) -> list[tuple[str, str]]:
    """Статус-строка в формате prompt_toolkit: список пар (стиль, текст)."""
    sep = ("", "  ·  ")
    parts: list[tuple[str, str]] = [("class:toolbar.model", f" {status.model}")]
    if status.context_budget > 0:
        percent = round(100 * status.context_tokens / status.context_budget)
        if percent >= _CONTEXT_DANGER:
            style = "class:toolbar.danger"
        elif percent >= _CONTEXT_WARN:
            style = "class:toolbar.warn"
        else:
            style = "class:toolbar.ok"
        used, budget = format_tokens(status.context_tokens), format_tokens(status.context_budget)
        parts += [sep, ("", "контекст "), (style, f"{used}/{budget} ({percent}%)")]
    parts += [sep, ("", f"потрачено {format_tokens(status.billed_tokens)}")]
    if status.auto_approve:
        parts += [sep, ("class:toolbar.danger", "⚠ авто-подтверждение")]
    return parts


# ------------------------------ автодополнение ----------------------------- #
class SlashCommandCompleter(Completer):
    """Дополняет первое слово, если это начало слеш-команды (не путь ``/home/...``)."""

    def __init__(self, commands: CommandRegistry):
        self._commands = commands

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not _COMMAND_PREFIX_RE.fullmatch(text):
            return
        prefix = text.lower()
        for cmd in self._commands:
            for name in (cmd.name, *cmd.aliases):
                if name.lower().startswith(prefix):
                    yield Completion(name, start_position=-len(text), display_meta=cmd.summary)
                    break


# --------------------------------- история --------------------------------- #
class LazyFileHistory(FileHistory):
    """История в ``.devassist/history``; папка создаётся при первом сохранении.

    Если записать не удалось (проект только для чтения), история остаётся в памяти
    до конца сессии — ошибка не должна закрывать REPL.
    """

    def __init__(self, workspace: Workspace):
        self._workspace = workspace
        self._persist = True
        super().__init__(workspace.data_dir / HISTORY_FILE)

    def store_string(self, string: str) -> None:
        if not self._persist:
            return
        try:
            self._workspace.ensure_data_dir()
            super().store_string(string)
        except OSError:
            self._persist = False


# --------------------------------- клавиши --------------------------------- #
def _key_bindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("escape", "escape")
    def _clear(event: KeyPressEvent) -> None:
        buffer = event.current_buffer
        if buffer.text.strip():
            buffer.reset(append_to_history=True)  # черновик не теряется: ↑ вернёт его
        else:
            buffer.reset()

    @kb.add("escape", "enter")
    def _newline(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    @Condition
    def _line_continues() -> bool:
        return get_app().current_buffer.document.text_before_cursor.endswith("\\")

    @kb.add("enter", filter=_line_continues)
    def _continue_line(event: KeyPressEvent) -> None:
        buffer = event.current_buffer
        buffer.delete_before_cursor(1)
        buffer.insert_text("\n")

    return kb


def _continuation(width: int, line_number: int, wrap_count: int) -> list[tuple[str, str]]:
    return [("class:continuation", "·".rjust(max(width - 1, 1)) + " ")]


# --------------------------------- сборка ---------------------------------- #
def create_prompt_session(
    *,
    commands: CommandRegistry,
    workspace: Workspace,
    status: Callable[[], StatusInfo],
    no_color: bool = False,
    **kwargs: Any,
) -> PromptSession[str]:
    """Сессия ввода. ``kwargs`` (``input``/``output``) подменяются в тестах."""

    @Condition
    def _typing_command() -> bool:
        return get_app().current_buffer.text.startswith("/")

    return PromptSession(
        message=[("class:prompt", PROMPT)],
        history=LazyFileHistory(workspace),
        auto_suggest=AutoSuggestFromHistory(),
        completer=SlashCommandCompleter(commands),
        complete_while_typing=_typing_command,
        reserve_space_for_menu=min(8, len(list(commands)) + 1),
        key_bindings=_key_bindings(),
        bottom_toolbar=lambda: toolbar_fragments(status()),
        prompt_continuation=_continuation,
        placeholder=[("class:placeholder", PLACEHOLDER)],
        style=Style.from_dict(PROMPT_STYLES),
        color_depth=ColorDepth.MONOCHROME if no_color else None,
        **kwargs,
    )


def _read_plain(default: str = "") -> str:
    return input(PLAIN_PROMPT)


def make_input_reader(
    *,
    commands: CommandRegistry,
    workspace: Workspace,
    status: Callable[[], StatusInfo],
    no_color: bool = False,
) -> InputReader:
    """Функция чтения строки: prompt_toolkit в терминале, иначе ``input()``."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return _read_plain
    try:
        session = create_prompt_session(
            commands=commands, workspace=workspace, status=status, no_color=no_color
        )
    except Exception:  # терминал не поддерживается (например, mintty без консоли)
        return _read_plain

    def read(default: str = "") -> str:
        return session.prompt(default=default)

    return read
