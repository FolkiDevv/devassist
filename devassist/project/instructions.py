"""Файлы инструкций проекта, которые агент держит в контексте.

Пользователь кладёт в корень проекта соглашения, команды сборки и заметки;
они добавляются в системный промпт. ``DEVASSIST.local.md`` — личные заметки,
обычно в ``.gitignore``. Поддержка новых соглашений (например, ``AGENTS.md``)
сводится к добавлению имени в :data:`INSTRUCTION_FILES`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from devassist.project.files import is_within

INSTRUCTION_FILES: tuple[str, ...] = ("DEVASSIST.md", "DEVASSIST.local.md")
MAX_INSTRUCTION_CHARS = 20_000


@dataclass(frozen=True)
class InstructionFile:
    name: str
    content: str
    truncated: bool = False


def load_instructions(
    root: Path,
    names: Sequence[str] = INSTRUCTION_FILES,
    max_chars: int = MAX_INSTRUCTION_CHARS,
) -> list[InstructionFile]:
    """Читает существующие файлы инструкций в порядке ``names``.

    Пустые и нечитаемые файлы пропускаются; слишком длинные обрезаются до
    ``max_chars``. Симлинки, ведущие за пределы корня, игнорируются.
    """
    root = root.resolve()
    result: list[InstructionFile] = []
    for name in names:
        path = root / name
        if not path.is_file() or not is_within(root, path):
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if not text:
            continue
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars]
        result.append(InstructionFile(name=name, content=text, truncated=truncated))
    return result
