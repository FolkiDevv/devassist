"""Терминальный UI devassist.

Обёртка над rich, изолирующая остальной код от деталей вывода. Задаёт единый
визуальный язык: брендовые цвета, панели, рендеринг вызовов инструментов,
потоковый вывод ответа модели, диффы и подтверждения.
"""

from __future__ import annotations

from rich.box import HEAVY, ROUNDED
from rich.console import Console as RichConsole
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# ------------------------------- палитра ------------------------------- #
BRAND = "#A78BFA"  # фиолетовый — бренд
ACCENT = "#22D3EE"  # бирюзовый — акценты/пути
OK = "#34D399"  # зелёный — успех
WARN = "#FBBF24"  # жёлтый — предупреждение/подтверждение
DANGER = "#F87171"  # красный — ошибки/опасность
MUTED = "#7C7C8A"  # серый — второстепенное
USER = "#93C5FD"  # голубой — пользователь

ICON_TOOL = "●"
ICON_OK = "✔"
ICON_FAIL = "✘"
ICON_BRAND = "✦"
ICON_ARROW = "↳"
SPINNER = "✦"


class Console:
    def __init__(self, *, no_color: bool = False, assume_yes: bool = False):
        self._c = RichConsole(no_color=no_color, highlight=False, emoji=False)
        self._assume_yes = assume_yes
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
    def banner(self, *, version: str, model: str, root: str) -> None:
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
        for i, (cmd, desc) in enumerate(
            [
                ("/help", "справка"),
                ("/model", "сменить модель"),
                ("/clear", "сброс"),
                ("/exit", "выход"),
            ]
        ):
            if i:
                hint.append("   ", style=MUTED)
            hint.append(cmd, style=ACCENT)
            hint.append(f" {desc}", style=MUTED)
        self._c.print(hint)
        self._c.print()

    # --------------------------- сообщения LLM -------------------------- #
    def assistant(self, text: str) -> None:
        """Печать ответа модели целиком (не потоковый режим)."""
        if not text.strip():
            return
        self._assistant_label()
        self._c.print(Markdown(text))

    def _assistant_label(self) -> None:
        self._c.print(Text(f"{ICON_BRAND} devassist", style=f"bold {BRAND}"))

    # ----------------------- потоковый вывод LLM ------------------------ #
    def begin_stream(self) -> None:
        """Начинает потоковый вывод: индикатор ожидания первого токена."""
        self._streaming = True
        self._got_token = False
        if self._ansi:
            self._c.file.write(f"\x1b[38;2;124;124;138m{SPINNER} думаю…\x1b[0m")
            self._c.file.flush()

    def stream_write(self, text: str) -> None:
        """Печатает очередной кусок текста модели по мере поступления."""
        if not text:
            return
        if self._streaming and not self._got_token:
            self._got_token = True
            if self._ansi:
                self._c.file.write("\r\x1b[K")  # стереть индикатор
            self._assistant_label()
        self._c.file.write(text)
        self._c.file.flush()

    def end_stream(self) -> None:
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
    def tool_call(self, name: str, summary: str) -> None:
        line = Text()
        line.append(f"{ICON_TOOL} ", style=f"bold {OK}")
        line.append(name, style="bold white")
        if summary:
            line.append("  ", style=MUTED)
            line.append(summary, style=ACCENT)
        self._c.print(line)

    def tool_result(self, summary: str, ok: bool = True) -> None:
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
            diff_text.rstrip(),
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
                Text(text.rstrip(), style="white"),
                title=Text(title, style=MUTED),
                title_align="left",
                box=ROUNDED,
                border_style=MUTED,
                padding=(0, 1),
                expand=False,
            )
        )

    # ---------------------------- статистика ---------------------------- #
    def turn_stats(self, *, steps: int, tokens: int, tools: int) -> None:
        parts = []
        plural = "шаг" if steps == 1 else ("шага" if 2 <= steps <= 4 else "шагов")
        parts.append(f"{steps} {plural}")
        if tools:
            parts.append(f"{tools} вызов(ов) инструментов")
        if tokens:
            parts.append(f"{tokens} токенов")
        self._c.print(Text("  " + "  ·  ".join(parts), style=MUTED))

    # -------------------------- подтверждения --------------------------- #
    def confirm(self, question: str, *, dangerous: bool = False) -> bool:
        if self._assume_yes:
            self._c.print(Text(f"  {ICON_ARROW} {question} → авто-подтверждено", style=MUTED))
            return True
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
