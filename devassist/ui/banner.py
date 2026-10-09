"""Арт стартового экрана: пиксельная ракета среди звёзд и крупное название."""

from __future__ import annotations

from rich.text import Text

from devassist.ui.theme import ACCENT, BRAND, BRAND_SOFT, DANGER, FLAME, MUTED, WARN

# Ракета: строки символов и маска цветов той же длины (пробел — без цвета).
_ROCKET: tuple[tuple[str, str], ...] = (
    (" ✦    ▄    · ", " b    b    m "),
    ("     ▟█▙     ", "     bbb     "),
    (" ·   █◉█   ✧ ", " m   bab   l "),
    ("     ███     ", "     bbb     "),
    ("    ▟███▙  · ", "    bbbbb  m "),
    ("    ▀ ▀ ▀    ", "    b b b    "),
    (" ✧   ▓▓▓     ", " l   hyh     "),
    ("     ▒▓▒   ✦ ", "     ryr   b "),
    ("  ·   ░      ", "  m   m      "),
)
_COLORS = {"b": BRAND, "l": BRAND_SOFT, "a": ACCENT, "h": FLAME, "y": WARN, "r": DANGER, "m": MUTED}

# Контурный шрифт (в духе figlet «future»): три строки на букву.
_FONT: dict[str, tuple[str, str, str]] = {
    "d": ("╺┳┓", " ┃┃", "╺┻┛"),
    "e": ("┏━╸", "┣╸ ", "┗━╸"),
    "v": ("╻ ╻", "┃┏┛", "┗┛ "),
    "a": ("┏━┓", "┣━┫", "╹ ╹"),
    "s": ("┏━┓", "┗━┓", "┗━┛"),
    "i": ("╻", "┃", "╹"),
    "t": ("╺┳╸", " ┃ ", " ╹ "),
}


def rocket() -> Text:
    """Ракета B5: корпус фирменного цвета, иллюминатор, пламя и звёзды вокруг."""
    art = Text()
    for i, (line, mask) in enumerate(_ROCKET):
        if i:
            art.append("\n")
        for ch, key in zip(line, mask, strict=True):
            art.append(ch, style=_COLORS.get(key, ""))
    return art


def title() -> Text:
    """Название «devassist» крупными буквами: «dev» фирменным цветом, «assist» — светлым."""
    word = (("dev", f"bold {BRAND}"), ("assist", "bold"))
    text = Text()
    for row in range(3):
        if row:
            text.append("\n")
        for part, style in word:
            text.append("".join(_FONT[ch][row] for ch in part), style=style)
    return text
