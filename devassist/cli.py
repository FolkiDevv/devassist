"""CLI-точка входа devassist.

Режимы:
  * интерактивный REPL (по умолчанию);
  * одноразовый запрос: ``devassist -p "сделай X"``;
  * диагностика: ``devassist --list-models``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from devassist import __version__
from devassist.agent.loop import Agent
from devassist.config import Config
from devassist.llm.base import LLMError
from devassist.llm.gigachat import GigaChatProvider
from devassist.tools.base import build_default_registry
from devassist.ui.console import Console


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="devassist",
        description="Локальный CLI-агент для программирования на базе GigaChat.",
    )
    p.add_argument("-p", "--prompt", help="Одноразовый запрос (без интерактива)")
    p.add_argument("-m", "--model", help="Модель GigaChat (переопределяет конфиг)")
    p.add_argument("-C", "--dir", default=".", help="Корень проекта (по умолчанию текущая папка)")
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Авто-подтверждение всех операций (используйте осознанно)",
    )
    p.add_argument("--no-color", action="store_true", help="Отключить цвет")
    p.add_argument(
        "--no-stream",
        action="store_true",
        help="Отключить потоковый вывод (ответ печатается целиком в конце)",
    )
    p.add_argument("--list-models", action="store_true", help="Показать доступные модели и выйти")
    p.add_argument("--version", action="version", version=f"devassist {__version__}")
    return p


def _make_provider(config: Config) -> GigaChatProvider:
    return GigaChatProvider(config)


def run_oneshot(agent: Agent, ui: Console, prompt: str) -> int:
    try:
        agent.run_turn(prompt)
    except LLMError as e:
        ui.error(str(e))
        return 2
    return 0


def run_repl(agent: Agent, ui: Console, config: Config) -> int:
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import InMemoryHistory

        psession = PromptSession(history=InMemoryHistory())

        def read_input() -> str:
            return psession.prompt("devassist› ")
    except Exception:  # prompt_toolkit недоступен — простой input()

        def read_input() -> str:
            return input("devassist> ")

    ui.banner(version=__version__, model=config.model, root=str(config.project_root))

    while True:
        try:
            line = read_input().strip()
        except (EOFError, KeyboardInterrupt):
            ui.system("\nдо встречи!")
            return 0
        if not line:
            continue

        if is_repl_command(line):
            if not _handle_command(line, agent, ui, config):
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


# Команда REPL — это слеш + слово (/help, /model ...). Чистый путь
# (/home/...) или начало регэкспа содержит дополнительные слеши и НЕ считается
# командой, а уходит обычным запросом агенту.
_REPL_COMMAND_RE = re.compile(r"^/[a-zA-Zа-яА-Я][\w-]*$")


def is_repl_command(line: str) -> bool:
    if not line.startswith("/"):
        return False
    first = line.split(maxsplit=1)[0]
    return bool(_REPL_COMMAND_RE.match(first))


def _handle_command(line: str, agent: Agent, ui: Console, config: Config) -> bool:
    """Обрабатывает /команды. Возвращает False, если нужно выйти."""
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit", "/q"):
        ui.system("до встречи!")
        return False
    if cmd == "/help":
        ui.info(
            "/help — помощь\n"
            "/model <имя> — сменить модель\n"
            "/clear — очистить историю диалога\n"
            "/exit — выход"
        )
    elif cmd == "/model":
        if arg:
            config.model = arg
            ui.info(f"модель теперь: {arg} (применится со следующего запроса)")
        else:
            ui.info(f"текущая модель: {config.model}")
    elif cmd == "/clear":
        from devassist.agent.session import Session

        agent._session = Session(config.project_root)  # noqa: SLF001
        ui.info("история очищена")
    else:
        ui.error(f"неизвестная команда: {cmd}")
    return True


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.dir).resolve()
    config = Config.load(
        project_root=root,
        model=args.model,
        auto_approve=args.yes,
        stream=not args.no_stream,
    )
    ui = Console(no_color=args.no_color, assume_yes=args.yes)

    try:
        config.require_credentials()
    except RuntimeError as e:
        ui.error(str(e))
        return 1

    try:
        provider = _make_provider(config)
    except Exception as e:
        ui.error(f"не удалось инициализировать провайдера: {e}")
        return 1

    try:
        if args.list_models:
            try:
                models = provider.list_models()
                ui.info("Доступные модели:\n" + "\n".join(f"  • {m}" for m in models))
                return 0
            except Exception as e:
                ui.error(f"не удалось получить список моделей: {e}")
                return 2

        registry = build_default_registry()
        agent = Agent(provider, registry, config, ui)

        if args.prompt:
            return run_oneshot(agent, ui, args.prompt)
        return run_repl(agent, ui, config)
    finally:
        provider.close()


if __name__ == "__main__":
    raise SystemExit(main())
