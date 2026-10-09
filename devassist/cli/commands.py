"""Слеш-команды REPL (/help, /model, ...).

Команды регистрируются в :class:`CommandRegistry`; из него же строятся справка,
подсказки в баннере и автодополнение ввода (:mod:`devassist.cli.prompt`). Новая команда
(``/compact``, ``/resume``, ``/index``) — это один :class:`SlashCommand`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from devassist.agent.loop import Agent
from devassist.cli.prompt import KEY_HELP
from devassist.ui.console import Console


@dataclass
class CommandContext:
    agent: Agent
    ui: Console
    commands: CommandRegistry


# Обработчик получает контекст и аргумент (текст после имени команды).
# Возвращает False, если REPL нужно завершить.
Handler = Callable[[CommandContext, str], bool]


@dataclass(frozen=True)
class SlashCommand:
    name: str  # с ведущим слешем: "/model"
    summary: str  # кратко — для баннера и справки
    handler: Handler
    usage: str = ""  # синтаксис для справки: "/model [имя]"
    aliases: tuple[str, ...] = ()


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}
        self._lookup: dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand) -> None:
        for key in (command.name, *command.aliases):
            key = key.lower()
            if key in self._lookup:
                raise ValueError(f"Команда {key} уже зарегистрирована")
            self._lookup[key] = command
        self._commands[command.name] = command

    def get(self, name: str) -> SlashCommand | None:
        return self._lookup.get(name.lower())

    def __iter__(self) -> Iterator[SlashCommand]:
        return iter(self._commands.values())

    def dispatch(self, line: str, ctx: CommandContext) -> bool:
        """Выполняет команду из строки. False — завершить REPL."""
        parts = line.strip().split(maxsplit=1)
        name = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""
        command = self.get(name)
        if command is None:
            ctx.ui.error(f"неизвестная команда: {name} (список — /help)")
            return True
        return command.handler(ctx, arg)


# Команда REPL — это слеш + слово (/help, /model ...). Чистый путь
# (/home/...) или начало регэкспа содержит дополнительные слеши и НЕ считается
# командой, а уходит обычным запросом агенту.
_REPL_COMMAND_RE = re.compile(r"^/[a-zA-Zа-яА-Я][\w-]*$")


def is_repl_command(line: str) -> bool:
    if not line.startswith("/"):
        return False
    first = line.split(maxsplit=1)[0]
    return bool(_REPL_COMMAND_RE.match(first))


# ------------------------------- команды ------------------------------- #
def _help(ctx: CommandContext, _arg: str) -> bool:
    rows = []
    for cmd in ctx.commands:
        alias = f" (также {', '.join(cmd.aliases)})" if cmd.aliases else ""
        rows.append((cmd.usage or cmd.name, f"{cmd.summary}{alias}"))
    ctx.ui.help(rows, KEY_HELP)
    return True


def _model(ctx: CommandContext, arg: str) -> bool:
    if arg:
        ctx.agent.set_model(arg)
        ctx.ui.info(f"модель теперь: {arg} (применится со следующего запроса)")
    else:
        ctx.ui.info(f"текущая модель: {ctx.agent.model}")
    return True


def _clear(ctx: CommandContext, _arg: str) -> bool:
    ctx.agent.reset()
    ctx.ui.info("история очищена")
    return True


def _exit(ctx: CommandContext, _arg: str) -> bool:
    ctx.ui.system("до встречи!")
    return False


def default_commands() -> CommandRegistry:
    registry = CommandRegistry()
    registry.register(SlashCommand("/help", "справка", _help))
    registry.register(SlashCommand("/model", "сменить модель", _model, usage="/model [имя]"))
    registry.register(SlashCommand("/clear", "очистить историю", _clear))
    registry.register(SlashCommand("/exit", "выход", _exit, aliases=("/quit", "/q")))
    return registry
