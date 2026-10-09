"""Терминальный UI devassist.

Обёртка над rich, изолирующая остальной код от деталей вывода. Задаёт единый
визуальный язык: брендовые цвета, панели, рендеринг вызовов инструментов,
потоковый вывод ответа модели, диффы и подтверждения.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import IO

from rich.box import HEAVY, ROUNDED
from rich.console import Console as RichConsole
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from devassist.agent.events import AgentEvents, NoticeLevel, ToolCallInfo, TurnStats
from devassist.tools.base import Display, ToolResult
from devassist.ui.theme import (
    ACCENT,
    BRAND,
    DANGER,
    ICON_ARROW,
    ICON_BRAND,
    ICON_FAIL,
    ICON_OK,
    ICON_TOOL,
    MUTED,
    OK,
    SPINNER,
    WARN,
)

# Управляющие символы, которые нельзя пропускать в терминал из текста модели и
# вывода команд: ESC-последовательности могут перекрасить/стереть экран,
# подменить заголовок окна и т.п. Оставляем только \n и \t.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_OUTPUT_PREVIEW_CHARS = 2000


def sanitize(text: str) -> str:
    """Убирает управляющие символы (кроме переводов строк и табуляции)."""
    return _CONTROL_RE.sub("", text.replace("\r\n", "\n"))


class Console(AgentEvents):
    """Терминальный интерфейс на rich; реализует события агента."""

    def __init__(self, *, no_color: bool = False, file: IO[str] | None = None):
        self._c = RichConsole(no_color=no_color, highlight=False, emoji=False, file=file)
        self._ansi = self._c.is_terminal and not no_color
        self._streaming = False
        self._got_token = False

    # ----------------------------- базовое ----------------------------- #
    def print(self, *args, **kwargs) -> None:
        self._c.print(*args, **kwargs)

    def blank(self) -> None:
        self._c.print()

    def rule(self, title: str = "") -> None:
        self._c.rule(Text(title, style=MUTED) if title else "", style=MUTED)

    def system(self, text: str) -> None:
        self._c.print(Text(text, style=MUTED))

    def error(self, text: str) -> None:
        self._c.print(Text(f"{ICON_FAIL} {text}", style=f"bold {DANGER}"))

    def warn(self, text: str) -> None:
        self._c.print(Text(f"⚠ {text}", style=WARN))

    def info(self, text: str) -> None:
        self._c.print(Text(text, style=ACCENT))

    def success(self, text: str) -> None:
        self._c.print(Text(f"{ICON_OK} {text}", style=f"bold {OK}"))

    # ------------------------------ баннер ------------------------------ #
    def banner(
        self,
        *,
        version: str,
        model: str,
        root: str,
        hints: Sequence[tuple[str, str]] = (),
    ) -> None:
        """Приветственная панель; ``hints`` — пары (команда, краткое описание)."""
        logo = Text()
        logo.append(f"{ICON_BRAND} ", style=f"bold {BRAND}")
        logo.append("dev", style=f"bold {BRAND}")
        logo.append("assist", style="bold white")
        logo.append(f"  v{version}", style=MUTED)

        subtitle = Text("AI-ассистент разработчика · работает на GigaChat", style=MUTED)

        meta = Table.grid(padding=(0, 1))
        meta.add_column(style=MUTED, justify="right")
        meta.add_column()
        meta.add_row("модель", Text(model, style=f"bold {ACCENT}"))
        meta.add_row("проект", Text(root, style="white"))

        body = Group(logo, subtitle, Text(""), meta)
        self._c.print(
            Panel(
                body,
                box=ROUNDED,
                border_style=BRAND,
                padding=(1, 2),
                expand=False,
            )
        )
        hint = Text("  ", style=MUTED)
        for i, (cmd, desc) in enumerate(hints):
            if i:
                hint.append("   ", style=MUTED)
            hint.append(cmd, style=ACCENT)
            hint.append(f" {desc}", style=MUTED)
        self._c.print(hint)
        self._c.print()

    # --------------------------- сообщения LLM -------------------------- #
    def on_assistant_text(self, text: str) -> None:
        """Печать ответа модели целиком (не потоковый режим)."""
        if not text.strip():
            return
        self._assistant_label()
        self._c.print(Markdown(sanitize(text)))

    def _assistant_label(self) -> None:
        self._c.print(Text(f"{ICON_BRAND} devassist", style=f"bold {BRAND}"))

    # ----------------------- потоковый вывод LLM ------------------------ #
    def on_stream_start(self) -> None:
        """Начинает потоковый вывод: индикатор ожидания первого токена."""
        self._streaming = True
        self._got_token = False
        if self._ansi:
            self._c.file.write(f"\x1b[38;2;124;124;138m{SPINNER} думаю…\x1b[0m")
            self._c.file.flush()

    def on_stream_delta(self, text: str) -> None:
        """Печатает очередной кусок текста модели по мере поступления."""
        text = sanitize(text)
        if not text:
            return
        if self._streaming and not self._got_token:
            self._got_token = True
            if self._ansi:
                self._c.file.write("\r\x1b[K")  # стереть индикатор
            self._assistant_label()
        self._c.file.write(text)
        self._c.file.flush()

    def on_stream_end(self) -> None:
        """Завершает потоковый вывод (перевод строки/очистка индикатора)."""
        if not self._streaming:
            return
        if self._got_token:
            self._c.file.write("\n")
        elif self._ansi:
            self._c.file.write("\r\x1b[K")  # ничего не пришло — убрать индикатор
        self._c.file.flush()
        self._streaming = False
        self._got_token = False

    # --------------------------- инструменты ---------------------------- #
    def on_tool_call(self, call: ToolCallInfo) -> None:
        line = Text()
        line.append(f"{ICON_TOOL} ", style=f"bold {OK}")
        line.append(call.name, style="bold white")
        if call.summary:
            line.append("  ", style=MUTED)
            line.append(sanitize(call.summary), style=ACCENT)
        self._c.print(line)

    def on_tool_result(self, call: ToolCallInfo, result: ToolResult, *, previewed: bool) -> None:
        self.tool_result(result.summary or ("готово" if result.ok else "ошибка"), ok=result.ok)
        shown = result.display
        if not shown or not shown.text.strip():
            return
        if shown.kind == "diff":
            # дифф уже показан при подтверждении — не дублируем
            if not previewed:
                self.diff(shown.text, title=shown.title or None)
        else:
            self.output_block(shown.text[:_OUTPUT_PREVIEW_CHARS], title=shown.title or "вывод")

    def tool_result(self, summary: str, ok: bool = True) -> None:
        summary = sanitize(summary)
        line = Text("  ")
        if ok:
            line.append(f"{ICON_ARROW} ", style=OK)
            line.append(summary, style=MUTED)
        else:
            line.append(f"{ICON_ARROW} {ICON_FAIL} ", style=f"bold {DANGER}")
            line.append(summary, style=DANGER)
        self._c.print(line)

    def diff(self, diff_text: str, *, title: str | None = None) -> None:
        if not diff_text.strip():
            return
        syntax = Syntax(
            sanitize(diff_text).rstrip(),
            "diff",
            theme="ansi_dark",
            background_color="default",
            word_wrap=True,
        )
        self._c.print(
            Panel(
                syntax,
                title=Text(title or "изменения", style=ACCENT),
                title_align="left",
                box=ROUNDED,
                border_style=MUTED,
                padding=(0, 1),
                expand=False,
            )
        )

    def output_block(self, text: str, *, title: str = "вывод") -> None:
        if not text.strip():
            return
        self._c.print(
            Panel(
                Text(sanitize(text).rstrip(), style="white"),
                title=Text(title, style=MUTED),
                title_align="left",
                box=ROUNDED,
                border_style=MUTED,
                padding=(0, 1),
                expand=False,
            )
        )

    # ---------------------------- статистика ---------------------------- #
    def on_notice(self, text: str, *, level: NoticeLevel = "info") -> None:
        {"info": self.info, "warn": self.warn, "error": self.error}[level](text)

    def on_turn_end(self, stats: TurnStats) -> None:
        steps = stats.steps
        plural = (
            "шаг"
            if steps % 10 == 1 and steps % 100 != 11
            else ("шага" if 2 <= steps % 10 <= 4 and not 12 <= steps % 100 <= 14 else "шагов")
        )
        parts = [f"{steps} {plural}"]
        if stats.tool_calls:
            parts.append(f"инструментов: {stats.tool_calls}")
        if stats.context_tokens:
            parts.append(f"контекст ~{stats.context_tokens} ток.")
        if stats.billed_tokens:
            parts.append(f"потрачено {stats.billed_tokens} ток.")
        self._c.print(Text("  " + "  ·  ".join(parts), style=MUTED))

    # -------------------------- подтверждения --------------------------- #
    def confirm(self, call: ToolCallInfo, preview: Display | None, *, dangerous: bool) -> bool:
        if preview and preview.text.strip():
            if preview.kind == "diff":
                self.diff(preview.text, title=preview.title or None)
            else:
                self.output_block(preview.text, title=preview.title or "превью")
        question = (
            f"Выполнить опасную операцию '{call.name}'?"
            if dangerous
            else f"Применить '{call.name}'?"
        )
        return self.ask(question, dangerous=dangerous)

    def ask(self, question: str, *, dangerous: bool = False) -> bool:
        """Вопрос да/нет. Ctrl+C/Ctrl+D — «нет»."""
        color = DANGER if dangerous else WARN
        title = "⚠ ОПАСНАЯ ОПЕРАЦИЯ" if dangerous else "Подтверждение"
        self._c.print(
            Panel(
                Text(question, style="white"),
                title=Text(title, style=f"bold {color}"),
                title_align="left",
                box=HEAVY if dangerous else ROUNDED,
                border_style=color,
                padding=(0, 1),
                expand=False,
            )
        )
        try:
            answer = self._c.input(Text("  выполнить? [y/N] ", style=f"bold {color}"))
        except (EOFError, KeyboardInterrupt):
            self._c.print()
            return False
        return answer.strip().lower() in ("y", "yes", "д", "да")
