"""Потоковый рендеринг Markdown: окончательные строки + изменяемый хвост.

Ответ модели приходит кусками, а Markdown нельзя форматировать по кускам: строка
``**жирный`` станет жирной только когда придёт закрывающее ``**``, таблица меняет
ширину колонок с каждой строкой. Поэтому текст делится на две части:

* **завершённые блоки** верхнего уровня (абзац, список, таблица, блок кода...),
  после которых уже начался следующий блок, — их вид больше не изменится; они
  печатаются один раз и навсегда;
* **хвост** — последний, ещё растущий блок; его показывают во временной области
  (``rich.live.Live``) и перерисовывают по мере поступления текста.

Границы блоков определяет тот же парсер, которым rich рендерит Markdown
(``Markdown.parsed``), поэтому они совпадают с тем, как блок будет нарисован.
Чтобы отступы между блоками совпали с рендером целиком, завершённая часть каждый
раз рендерится заново как префикс документа, и наружу отдаются только ещё не
напечатанные строки. В итоге «поток» печатает ровно то же, что рендер всего
ответа сразу (проверяется тестом).

Известное ограничение: ссылки-сноски (``[1]: url``), определённые ниже места
использования, в уже напечатанной части останутся текстом.

Модуль не управляет терминалом: :class:`MarkdownStream` только считает строки.
:meth:`MarkdownStream.tail_lines` вызывается из потока обновления ``Live``, поэтому
состояние хранится в одном неизменяемом кортеже, который заменяется целиком.
"""

from __future__ import annotations

from rich.console import Console
from rich.segment import Segment

from devassist.ui.markdown import Markdown

Lines = list[list[Segment]]


def block_start_lines(source: str) -> list[int]:
    """Номера строк, с которых начинаются блоки верхнего уровня."""
    starts = []
    for token in Markdown(source).parsed:
        # Открывающие (nesting=1) и самодостаточные (fence, hr: nesting=0) токены
        # верхнего уровня; у закрывающих нет map.
        if token.level == 0 and token.map and token.nesting != -1:
            starts.append(token.map[0])
    return starts


def complete_prefix_length(text: str) -> int:
    """Длина (в символах) префикса ``text``, состоящего из завершённых блоков.

    Разбирается только часть до последнего перевода строки: незаконченная строка
    может поменять смысл предыдущих (``#`` — заголовок, ``#тег`` — продолжение абзаца).
    """
    full = text[: text.rfind("\n") + 1]
    if not full:
        return 0
    starts = block_start_lines(full)
    if len(starts) < 2:
        return 0
    last = starts[-1]
    offset = 0
    for _ in range(last):
        offset = full.index("\n", offset) + 1
    return offset


def _is_blank(line: list[Segment]) -> bool:
    return not "".join(segment.text for segment in line).strip()


class MarkdownStream:
    """Накопитель потокового Markdown-ответа.

    ``width`` фиксируется на весь ответ: при другой ширине уже напечатанные строки
    не совпали бы с перерендеренным префиксом.
    """

    def __init__(self, console: Console, *, width: int | None = None):
        self._console = console
        self._options = console.options.update_width(width or console.width)
        # (весь текст, длина завершённой части, сколько строк уже отдано)
        self._state: tuple[str, int, int] = ("", 0, 0)
        self._tail_cache: tuple[str, int, Lines] | None = None

    @property
    def text(self) -> str:
        return self._state[0]

    def render(self, source: str) -> Lines:
        return self._console.render_lines(Markdown(source), self._options, pad=False)

    def feed(self, delta: str) -> Lines:
        """Добавляет кусок текста. Возвращает новые окончательные строки (часто — ни одной)."""
        text, committed, printed = self._state
        text += delta
        if "\n" in delta:
            boundary = complete_prefix_length(text)
            if boundary > committed:
                lines = self.render(text[:boundary])
                self._state = (text, boundary, len(lines))
                return lines[printed:]
        self._state = (text, committed, printed)
        return []

    def tail_lines(self) -> Lines:
        """Строки незавершённого хвоста (для временной области). Потокобезопасно."""
        text, committed, printed = self._state
        tail = text[committed:]
        cached = self._tail_cache
        if cached is not None and cached[0] == tail and cached[1] == printed:
            return cached[2]
        lines = self.render(tail) if tail.strip() else []
        while lines and _is_blank(lines[0]):
            lines.pop(0)
        if printed and lines:
            lines.insert(0, [])  # отступ от уже напечатанного
        self._tail_cache = (tail, printed, lines)
        return lines

    def finish(self) -> Lines:
        """Завершает ответ: оставшиеся ненапечатанные строки."""
        text, _committed, printed = self._state
        if not text.strip():
            self._state = (text, len(text), printed)
            return []
        lines = self.render(text)
        self._state = (text, len(text), len(lines))
        return lines[printed:]
