"""Точка входа devassist и сборка компонентов.

Режимы:
  * интерактивный REPL (по умолчанию);
  * одноразовый запрос: ``devassist -p "сделай X"``;
  * диагностика: ``devassist --list-models``.

Коды возврата: 0 — успех, 1 — ошибка конфигурации/внутренняя, 2 — ошибка LLM,
130 — прервано пользователем (Ctrl+C).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from devassist import __version__
from devassist.agent.loop import Agent
from devassist.cli.commands import default_commands
from devassist.cli.repl import run_repl
from devassist.config import Config, ConfigError
from devassist.llm.base import LLMError, LLMProvider
from devassist.llm.gigachat import GigaChatProvider
from devassist.tools.base import build_default_registry
from devassist.ui.console import Console

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_LLM_ERROR = 2
EXIT_INTERRUPTED = 130


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


def _make_provider(config: Config) -> LLMProvider:
    return GigaChatProvider(config)


def run_oneshot(agent: Agent, ui: Console, prompt: str) -> int:
    try:
        agent.run_turn(prompt)
    except LLMError as e:
        ui.error(str(e))
        return EXIT_LLM_ERROR
    except KeyboardInterrupt:
        ui.system("\n(прервано)")
        return EXIT_INTERRUPTED
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ui = Console(no_color=args.no_color)
    root = Path(args.dir)
    if not root.is_dir():
        ui.error(f"каталог проекта не найден: {root}")
        return EXIT_ERROR
    try:
        config = Config.load(
            project_root=root,
            model=args.model,
            auto_approve=args.yes,
            stream=not args.no_stream,
        )
        config.require_credentials()
    except ConfigError as e:
        ui.error(f"ошибка конфигурации: {e}")
        return EXIT_ERROR
    except RuntimeError as e:
        ui.error(str(e))
        return EXIT_ERROR

    try:
        provider = _make_provider(config)
    except Exception as e:
        ui.error(f"не удалось инициализировать провайдера: {e}")
        return EXIT_ERROR

    try:
        if args.list_models:
            try:
                models = provider.list_models()
            except LLMError as e:
                ui.error(f"не удалось получить список моделей: {e}")
                return EXIT_LLM_ERROR
            ui.info("Доступные модели:\n" + "\n".join(f"  • {m}" for m in models))
            return EXIT_OK

        agent = Agent(provider, build_default_registry(), config, ui)
        if args.prompt:
            return run_oneshot(agent, ui, args.prompt)
        return run_repl(agent, ui, default_commands())
    finally:
        provider.close()
