"""Форматирование чисел и текста для терминала (без зависимостей от rich)."""

from __future__ import annotations

SKIPPED_MARK = "…"


def plural(n: int, one: str, few: str, many: str) -> str:
    """Форма слова для числа: 1 шаг, 2 шага, 5 шагов."""
    n = abs(n)
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def format_tokens(n: int) -> str:
    """Компактная запись числа токенов: 950, 12.3k, 120k, 1.2M."""
    if n < 1000:
        return str(n)
    value, suffix = n / 1000, "k"
    if round(value) >= 1000:
        value, suffix = n / 1_000_000, "M"
    text = f"{value:.1f}" if value < 99.95 else f"{value:.0f}"
    return text.removesuffix(".0") + suffix


def skipped_line(count: int) -> str:
    return f"{SKIPPED_MARK} пропущено {count} {plural(count, 'строка', 'строки', 'строк')} …"


def clip_lines(
    text: str, *, head: int = 6, tail: int = 10, max_line: int = 300, max_chars: int = 4000
) -> str:
    """Сокращает длинный вывод: начало + конец (итог pytest/компилятора обычно в конце).

    Длинные строки обрезаются по ``max_line`` (одна минифицированная строка иначе
    обошла бы лимит по строкам), общий размер — не больше ``max_chars``.
    """
    lines = text.rstrip("\n").split("\n")
    if len(lines) > head + tail + 1:
        lines = [*lines[:head], skipped_line(len(lines) - head - tail), *lines[-tail:]]
    lines = [line if len(line) <= max_line else line[: max_line - 1] + "…" for line in lines]
    result = "\n".join(lines)
    if len(result) > max_chars:
        # Конец важнее начала (итог команды) — делим бюджет, пропуск посередине.
        head_chars = (max_chars - 1) // 2
        tail_chars = max_chars - 1 - head_chars
        result = result[:head_chars] + "…" + result[len(result) - tail_chars :]
    return result
