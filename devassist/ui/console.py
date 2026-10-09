"""Терминальный UI devassist.

Обёртка над rich, изолирующая остальной код от деталей вывода. Задаёт единый
визуальный язык: брендовые цвета, панели, рендеринг вызовов инструментов,
потоковый вывод ответа модели с живым Markdown, индикаторы ожидания, диффы и
подтверждения.

Временная область внизу экрана (``rich.live.Live``) — одна на всю консоль: в ней
крутится индикатор ожидания модели или работы инструмента и рисуется ещё растущий
хвост ответа. Перед запуском новой области предыдущая всегда останавливается,
перед вопросом пользователю — тоже. Анимацию двигает собственный поток под общей
с печатью блокировкой: встроенное авто-обновление rich перерисовывает область
параллельно с печатью над ней, и при гонке стирает уже напечатанные строки.

В не-терминале (pipe, файл, «глупый» терминал) живой области нет: ответ модели
печатается как есть, по мере поступления — так его удобно перенаправлять в файл.
"""

from __future__ import annotations

import contextlib
import re
import sys
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import IO

from rich.box import ROUNDED
from rich.console import Console as RichConsole
from rich.console import ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.panel import Panel
from rich.segment import Segment, SegmentLines
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from devassist.agent.chat_store import ChatInfo
from devassist.agent.events import AgentEvents, NoticeLevel, ToolCallInfo, TurnStats
from devassist.llm.types import Message
from devassist.tools.ask_user import format_answer
from devassist.tools.base import Display, ToolResult
from devassist.tools.questions import Answer, Question, QuestionsUnavailable
from devassist.ui.format import SKIPPED_MARK, clip_lines, format_tokens, format_when, plural
from devassist.ui.markdown import Markdown
from devassist.ui.markdown_stream import MarkdownStream
from devassist.ui.theme import (
    ACCENT,
    BRAND,
    DANGER,
    ICON_ARROW,
    ICON_BRAND,
    ICON_FAIL,
    ICON_OK,
    ICON_TOOL,
    MARKDOWN_STYLES,
    MUTED,
    OK,
    USER,
    WARN,
)

# Управляющие символы, которые нельзя пропускать в терминал из текста модели и
# вывода команд: ESC-последовательности могут перекрасить/стереть экран,
# подменить заголовок окна и т.п. Оставляем только \n и \t.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_SKIPPED_RE = re.compile(rf"^{SKIPPED_MARK} пропущено .*$", re.MULTILINE)
_REFRESH_PER_SECOND = 12


def sanitize(text: str) -> str:
    """Убирает управляющие символы (кроме переводов строк и табуляции)."""
    return _CONTROL_RE.sub("", text.replace("\r\n", "\n"))


class _LiveView:
    """Содержимое временной области: хвост ответа (если есть) + строка индикатора.

    Вызывается из потока обновления ``Live``. Сама обрезает себя по высоте экрана:
    то, что ушло за верх экрана, временная область уже не сможет стереть.
    """

    def __init__(
        self,
        label: str,
        tail: Callable[[], list[list[Segment]]] | None = None,
        *,
        hint: str = "Ctrl+C — прервать",
    ):
        self._label = label
        self._tail = tail
        self._hint = hint
        self._started = time.monotonic()
        self._spinner = Spinner("dots", style=BRAND)

    def __rich_console__(self, console: RichConsole, options: ConsoleOptions) -> RenderResult:
        elapsed = int(time.monotonic() - self._started)
        self._spinner.update(
            text=Text.assemble(
                (f" {self._label}… ", MUTED),
                (f"{elapsed} с", MUTED),
                (f"  ·  {self._hint}", MUTED),
            )
        )
        lines = self._tail() if self._tail else []
        if lines:
            room = max(console.size.height - 3, 1)  # индикатор + отступ + запас
            yield SegmentLines(lines[-room:], new_lines=True)
            yield Text("")
        yield self._spinner


class _LiveArea:
    """Временная область + поток анимации; печать над ней — только через :meth:`print`."""

    def __init__(self, console: RichConsole, view: _LiveView, *, animate: bool):
        self._live = Live(
            view,
            console=console,
            auto_refresh=False,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
            vertical_overflow="crop",
        )
        self._lock = threading.RLock()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._animate, daemon=True) if animate else None

    @property
    def active(self) -> bool:
        return not self._stopped.is_set()

    def start(self) -> None:
        self._live.start(refresh=True)
        if self._thread is not None:
            self._thread.start()

    def _animate(self) -> None:
        while not self._stopped.wait(1 / _REFRESH_PER_SECOND):
            with self._lock:
                if not self._stopped.is_set():
                    self._live.refresh()

    def print(self, renderable) -> None:
        with self._lock:
            self._live.console.print(renderable)

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1)
        with self._lock:
            self._live.stop()


class Console(AgentEvents):
    """Терминальный интерфейс на rich; реализует события агента.

    ``force_terminal``/``width``/``auto_refresh`` нужны тестам: живую область
    можно проверить без настоящего терминала и без фонового потока обновления.
    """

    def __init__(
        self,
        *,
        no_color: bool = False,
        file: IO[str] | None = None,
        force_terminal: bool | None = None,
        width: int | None = None,
        auto_refresh: bool = True,
    ):
        self._c = RichConsole(
            no_color=no_color,
            highlight=False,
            emoji=False,
            file=file,
            force_terminal=force_terminal,
            width=width,
            theme=Theme(MARKDOWN_STYLES),
        )
        self._no_color = no_color
        self._live_ok = self._c.is_terminal and not self._c.is_dumb_terminal
        self._auto_refresh = auto_refresh
        self._live: _LiveArea | None = None
        self._md: MarkdownStream | None = None  # потоковый ответ (живой режим)
        self._raw_started = False  # потоковый ответ (не-терминал): метка уже напечатана
        self._interrupt_hint = "Ctrl+C — прервать"
        self._input_guard: Callable[[], AbstractContextManager[None]] = contextlib.nullcontext

    def set_interrupt_keys(
        self, hint: str, input_guard: Callable[[], AbstractContextManager[None]]
    ) -> None:
        """Подсказка о прерывании в индикаторе и защита ввода во время хода.

        ``input_guard`` оборачивает вопрос пользователю (подтверждение) — например,
        чтобы на это время отпустить терминал, который слушает клавишу Esc.
        """
        self._interrupt_hint = hint
        self._input_guard = input_guard

    @property
    def no_color(self) -> bool:
        return self._no_color

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

    # ------------------------- временная область ------------------------ #
    def _start_live(self, view: _LiveView) -> None:
        self.stop_live()
        if not self._live_ok:
            return
        area = _LiveArea(self._c, view, animate=self._auto_refresh)
        area.start()
        self._live = area

    def _print_above_live(self, renderable) -> None:
        if self._live is not None:
            self._live.print(renderable)
        else:
            self._c.print(renderable)

    def stop_live(self) -> None:
        """Убирает временную область (идемпотентно). Безопасно вызывать всегда."""
        live, self._live = self._live, None
        if live is not None:
            live.stop()

    # ------------------------------ баннер ------------------------------ #
    def banner(
        self,
        *,
        version: str,
        model: str,
        root: str,
        hints: Sequence[tuple[str, str]] = (),
        auto_approve: bool = False,
    ) -> None:
        """Приветственная панель; ``hints`` — пары (команда, краткое описание)."""
        logo = Text()
        logo.append(f"{ICON_BRAND} ", style=f"bold {BRAND}")
        logo.append("dev", style=f"bold {BRAND}")
        logo.append("assist", style="bold")
        logo.append(f"  v{version}", style=MUTED)

        subtitle = Text("AI-ассистент разработчика · работает на GigaChat", style=MUTED)

        meta = Table.grid(padding=(0, 1))
        meta.add_column(style=MUTED, justify="right")
        meta.add_column()
        meta.add_row("модель", Text(model, style=f"bold {ACCENT}"))
        meta.add_row("проект", Text(root))

        body = Group(logo, subtitle, Text(""), meta)
        self._c.print(Panel(body, box=ROUNDED, border_style=BRAND, padding=(1, 2), expand=False))
        hint = Text("  ", style=MUTED)
        for i, (cmd, desc) in enumerate(hints):
            if i:
                hint.append("   ", style=MUTED)
            hint.append(cmd, style=ACCENT)
            hint.append(f" {desc}", style=MUTED)
        self._c.print(hint)
        if auto_approve:
            self.warn("авто-подтверждение включено (-y): изменения применяются без вопросов")
        self._c.print()

    def help(
        self, commands: Sequence[tuple[str, str]], keys: Sequence[tuple[str, str]] = ()
    ) -> None:
        """Справка: таблица команд и клавиш."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style=ACCENT, no_wrap=True)
        table.add_column(style=MUTED)
        for name, desc in commands:
            table.add_row(name, desc)
        if keys:
            table.add_row("", "")
            for key, desc in keys:
                table.add_row(Text(key, style="bold"), desc)
        self._c.print(table)

    # --------------------------- сообщения LLM -------------------------- #
    def on_assistant_text(self, text: str) -> None:
        """Печать ответа модели целиком (не потоковый режим)."""
        if not text.strip():
            return
        self._assistant_label()
        self._c.print(Markdown(sanitize(text)))

    def _assistant_label(self) -> None:
        self._print_above_live(Text(f"{ICON_BRAND} devassist", style=f"bold {BRAND}"))

    # ----------------------- потоковый вывод LLM ------------------------ #
    def on_stream_start(self) -> None:
        """Запрос отправлен: индикатор ожидания до первого токена."""
        self._md = None
        self._raw_started = False
        self._start_live(_LiveView("думаю", hint=self._interrupt_hint))

    def on_stream_delta(self, text: str) -> None:
        """Очередной кусок текста модели."""
        text = sanitize(text)
        if not text:
            return
        if not self._live_ok:
            if not self._raw_started:
                self._raw_started = True
                self._assistant_label()
            self._c.file.write(text)
            self._c.file.flush()
            return
        if self._md is None:
            self._md = MarkdownStream(self._c)
            self._assistant_label()  # при активной области печатается над ней
            self._start_live(
                _LiveView("печатает", tail=self._md.tail_lines, hint=self._interrupt_hint)
            )
        lines = self._md.feed(text)
        if lines:
            self._print_above_live(SegmentLines(lines, new_lines=True))

    def on_stream_end(self) -> None:
        """Ответ закончен (или прерван): убрать индикатор, дорисовать хвост."""
        md, self._md = self._md, None
        self.stop_live()
        if md is not None:
            lines = md.finish()
            if lines:
                self._c.print(SegmentLines(lines, new_lines=True))
        if self._raw_started:
            self._raw_started = False
            self._c.file.write("\n")
            self._c.file.flush()

    # --------------------------- инструменты ---------------------------- #
    def on_tool_call(self, call: ToolCallInfo) -> None:
        line = Text()
        line.append(f"{ICON_TOOL} ", style=f"bold {OK}")
        line.append(call.name, style="bold")
        if call.summary:
            line.append("  ", style=MUTED)
            line.append(sanitize(call.summary), style=ACCENT)
        self._c.print(line)

    def on_tool_start(self, call: ToolCallInfo) -> None:
        self._start_live(_LiveView("выполняется", hint=self._interrupt_hint))

    def on_tool_end(self, call: ToolCallInfo) -> None:
        self.stop_live()

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
            self.output_block(clip_lines(shown.text), title=shown.title or "вывод")

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
        body = Text(sanitize(text).rstrip())
        body.highlight_regex(_SKIPPED_RE, f"italic {MUTED}")
        self._c.print(
            Panel(
                body,
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
        parts = [f"{steps} {plural(steps, 'шаг', 'шага', 'шагов')}"]
        if stats.tool_calls:
            n = stats.tool_calls
            parts.append(f"{n} {plural(n, 'инструмент', 'инструмента', 'инструментов')}")
        if stats.duration_s:
            parts.append(f"{stats.duration_s:.1f} с")
        if stats.context_tokens:
            parts.append(f"контекст ~{format_tokens(stats.context_tokens)}")
        if stats.billed_tokens:
            parts.append(f"потрачено {format_tokens(stats.billed_tokens)} ток.")
        self._c.print(Text("  " + "  ·  ".join(parts), style=MUTED))

    # -------------------------- подтверждения --------------------------- #
    def confirm(self, call: ToolCallInfo, preview: Display | None, *, dangerous: bool) -> bool:
        if preview and preview.text.strip():
            if preview.kind == "diff":
                self.diff(preview.text, title=preview.title or None)
            else:
                self.output_block(preview.text, title=preview.title or "превью")
        action = "Выполнить" if dangerous else "Применить"
        target = f" ({sanitize(call.summary)})" if call.summary else ""
        return self.ask(f"{action} {call.name}{target}?", dangerous=dangerous)

    def ask(self, question: str, *, dangerous: bool = False) -> bool:
        """Вопрос да/нет. Ctrl+C/Ctrl+D — «нет»."""
        self.stop_live()
        color = DANGER if dangerous else WARN
        prompt = Text("  ")
        prompt.append("⚠ ОПАСНО: " if dangerous else "? ", style=f"bold {color}")
        prompt.append(question, style="bold")
        prompt.append(" [y/N] ", style=MUTED)
        try:
            with self._input_guard():
                answer = self._c.input(prompt)
        except (EOFError, KeyboardInterrupt):
            self._c.print()
            return False
        return answer.strip().lower() in ("y", "yes", "д", "да")

    # ---------------------------- вопросы агента ---------------------------- #
    def ask_user(self, questions: Sequence[Question]) -> list[Answer] | None:
        """Задаёт вопросы по очереди (меню со стрелками или ввод номера). None — отказ."""
        self.stop_live()
        if not _stdin_is_terminal():
            raise QuestionsUnavailable("ввод не с клавиатуры")
        answers: list[Answer] = []
        with self._input_guard():
            for i, question in enumerate(questions, 1):
                answer = self._ask_one(question, i, len(questions))
                if answer is None:
                    self._c.print(Text("  ? пользователь отказался отвечать", style=MUTED))
                    return None
                line = Text("  ? ", style=f"bold {BRAND}")
                line.append(sanitize(question.text), style="bold")
                line.append(f" {ICON_ARROW} ", style=MUTED)
                line.append(sanitize(format_answer(answer)), style=ACCENT)
                self._c.print(line)
                answers.append(answer)
        return answers

    def _ask_one(self, question: Question, index: int, total: int) -> Answer | None:
        if self._live_ok:
            try:
                from devassist.ui.choice import choose

                return choose(question, index, total, no_color=self._no_color)
            except (KeyboardInterrupt, EOFError):
                return None
        return self._ask_plain(question, index, total)

    def _ask_plain(self, question: Question, index: int, total: int) -> Answer | None:
        """Запасной режим без интерактивного меню: номер(а) варианта или свой текст."""
        title = Text(f"Вопрос {index} из {total}", style=f"bold {BRAND}")
        if question.header:
            title.append(f" · {question.header}", style=ACCENT)
        self._c.print(title)
        self._c.print(Text(question.text, style="bold"))
        options = question.options
        for k, option in enumerate(options, 1):
            self._c.print(Text(f"  {k}. {option.label}"))
            if option.description:
                self._c.print(Text(f"     {option.description}", style=MUTED))
        custom = len(options) + 1
        self._c.print(Text(f"  {custom}. Свой ответ (или просто напишите текст)"))
        hint = "номера через пробел" if question.multi_select else "номер"
        while True:
            try:
                raw = self._c.input(Text(f"  {hint} или ответ: ", style=MUTED)).strip()
                if not raw:
                    continue
                numbers = raw.replace(",", " ").split()
                if not numbers:
                    continue  # одни запятые — спросить заново
                # isdecimal, а не isdigit: «²» — цифра, но int() её не разберёт
                if not all(n.isdecimal() and 1 <= int(n) <= custom for n in numbers):
                    return Answer(custom=raw)  # не номера — это свой ответ
                picked = sorted({int(n) for n in numbers})
                if len(picked) > 1 and not question.multi_select:
                    self.warn("здесь можно выбрать только один вариант")
                    continue
                text = ""
                if custom in picked:
                    text = self._c.input(Text("  ваш ответ: ", style=MUTED)).strip()
                    if not text:
                        continue
                labels = tuple(options[n - 1].label for n in picked if n != custom)
                return Answer(labels, text)
            except (EOFError, KeyboardInterrupt):
                self._c.print()
                return None

    # ---------------------------- сохранённые чаты ---------------------------- #
    def pick_chat(self, chats: Sequence[ChatInfo], current_id: str = "") -> ChatInfo | None:
        """Выбор чата: меню со стрелками и поиском, без терминала — номер. None — отмена."""
        self.stop_live()
        if not chats:
            return None
        if self._live_ok and _stdin_is_terminal():
            try:
                from devassist.ui.chat_picker import pick_chat

                return pick_chat(chats, current_id, no_color=self._no_color)
            except (KeyboardInterrupt, EOFError):
                return None
        return self._pick_chat_plain(chats, current_id)

    def _pick_chat_plain(self, chats: Sequence[ChatInfo], current_id: str) -> ChatInfo | None:
        table = Table.grid(padding=(0, 2))
        table.add_column(style=ACCENT, justify="right", no_wrap=True)
        table.add_column(style=MUTED, no_wrap=True)
        table.add_column()
        for k, chat in enumerate(chats, 1):
            n = chat.requests
            meta = (
                f"{format_when(chat.updated_at)} · {n} {plural(n, 'запрос', 'запроса', 'запросов')}"
            )
            title = Text(sanitize(chat.title))
            if chat.id == current_id:
                title.append(" (текущий)", style=MUTED)
            table.add_row(f"{k}.", meta, title)
        self._c.print(Text(f"Чаты проекта · {len(chats)}", style=f"bold {BRAND}"))
        self._c.print(table)
        while True:
            try:
                raw = self._c.input(Text("  номер чата (Enter — отмена): ", style=MUTED))
            except (EOFError, KeyboardInterrupt):
                self._c.print()
                return None
            raw = raw.strip()
            if not raw:
                return None
            if raw.isdecimal() and 1 <= int(raw) <= len(chats):
                return chats[int(raw) - 1]
            self.warn(f"введите номер от 1 до {len(chats)}")

    def chat_resumed(self, info: ChatInfo, messages: Sequence[Message]) -> None:
        """Сообщение о продолжении чата и последний обмен репликами."""
        n = info.requests
        line = Text("↺ продолжаем чат ", style=MUTED)
        line.append(f"«{sanitize(info.title)}»", style=f"bold {ACCENT}")
        line.append(
            f" · {n} {plural(n, 'запрос', 'запроса', 'запросов')} · {format_when(info.updated_at)}",
            style=MUTED,
        )
        self._c.print(line)
        last = next(
            (
                k
                for k in range(len(messages) - 1, -1, -1)
                if messages[k].role == "user" and messages[k].content.strip()
            ),
            None,
        )
        if last is not None:
            shown = clip_lines(messages[last].content, head=3, tail=2, max_chars=600)
            self._c.print(Text("› ", style=f"bold {USER}").append(sanitize(shown), style=USER))
            # Ответ ищется только после этого запроса: у прерванного хода его нет, и
            # ответ на предыдущий запрос здесь выглядел бы ответом на этот.
            answer = next(
                (
                    m.content
                    for m in reversed(messages[last + 1 :])
                    if m.role == "assistant" and m.function_call is None and m.content.strip()
                ),
                "",
            )
            if answer:
                self._assistant_label()
                self._c.print(Markdown(sanitize(answer)))
            else:
                self._c.print(Text("  (ответа нет — ход не был завершён)", style=MUTED))
        self._c.print()


def _stdin_is_terminal() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False
