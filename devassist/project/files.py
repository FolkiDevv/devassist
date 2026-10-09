"""Файлы проекта: единые правила игнорирования, обход, glob-сопоставление, дерево.

Единственный источник правды о том, какие каталоги агент не обходит. Им
пользуются дерево проекта в системном промпте, find_files и search_content
(а в будущем — индекс проекта; сюда же ляжет учёт .gitignore).
"""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Iterator
from pathlib import Path

from devassist.project.workspace import DATA_DIR_NAME

IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        ".eggs",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".idea",
        DATA_DIR_NAME,
    }
)


def is_ignored_dir(name: str) -> bool:
    """Каталог служебный (VCS, кеши, окружения, сборка) — не обходим его."""
    return name in IGNORE_DIRS or name.endswith(".egg-info")


def is_secret_file(name: str) -> bool:
    """Файл с секретами (``.env``, ``.env.local``…), кроме шаблона ``.env.example``."""
    return name == ".env" or (name.startswith(".env.") and name != ".env.example")


def is_within(root: Path, path: Path) -> bool:
    """``path`` после раскрытия симлинков лежит внутри ``root``."""
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents


def walk_files(root: Path, base: Path | None = None) -> Iterator[Path]:
    """Обходит файлы под ``base`` (по умолчанию — ``root``) в отсортированном порядке.

    Служебные каталоги отсекаются до спуска в них (обход ``node_modules`` не
    тратит время). Симлинки на каталоги не раскрываются, симлинки на файлы
    вне ``root`` пропускаются.
    """
    root = root.resolve()
    start = base if base is not None else root
    for dirpath, dirnames, filenames in os.walk(start, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not is_ignored_dir(d))
        current = Path(dirpath)
        for name in sorted(filenames):
            path = current / name
            if path.is_symlink() and not is_within(root, path):
                continue
            yield path


@functools.lru_cache(maxsize=256)
def _compile_glob(pattern: str) -> re.Pattern[str]:
    """glob → regex: ``**`` — любое число каталогов, ``*``/``?`` — в пределах сегмента."""
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern.startswith("**", i):
                i += 2
                if i < n and pattern[i] == "/":
                    i += 1
                    out.append("(?:.*/)?")
                else:
                    out.append(".*")
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = i + 1
            if j < n and pattern[j] in "!^":
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            while j < n and pattern[j] != "]":
                j += 1
            if j >= n:
                out.append(re.escape(c))
            else:
                body = pattern[i + 1 : j]
                if body[0] in "!^":
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = j
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("".join(out) + r"\Z")


def glob_match(rel_posix: str, pattern: str) -> bool:
    """Сопоставляет относительный POSIX-путь с glob-шаблоном.

    Шаблон без ``/`` сравнивается с именем файла (``*.py`` — любой .py в дереве),
    шаблон с ``/`` — с путём от корня (``src/**/*.py``).
    """
    pattern = pattern.strip()
    while pattern.startswith("./"):
        pattern = pattern[2:]
    target = rel_posix if "/" in pattern else rel_posix.rsplit("/", 1)[-1]
    return _compile_glob(pattern).match(target) is not None


def build_file_tree(root: Path, max_entries: int = 200) -> str:
    """Компактное дерево проекта (отсортированное, с обрезкой).

    Скрытые файлы и каталоги (``.github``, ``.gitignore``) показываются,
    служебные каталоги (``.git``, ``node_modules``…) — нет.
    """
    lines: list[str] = []
    count = 0

    def is_real_dir(p: Path) -> bool:
        # симлинки на каталоги не раскрываем (возможны циклы и выход из корня)
        return p.is_dir() and not p.is_symlink()

    def walk(directory: Path, prefix: str) -> None:
        nonlocal count
        try:
            entries = sorted(
                directory.iterdir(), key=lambda e: (not is_real_dir(e), e.name.lower())
            )
        except OSError:
            return
        for e in entries:
            if count >= max_entries:
                return
            is_dir = is_real_dir(e)
            if is_dir and is_ignored_dir(e.name):
                continue
            count += 1
            if is_dir:
                lines.append(f"{prefix}{e.name}/")
                walk(e, prefix + "  ")
            else:
                lines.append(f"{prefix}{e.name}")

    walk(root, "")
    if count >= max_entries:
        lines.append("... (дерево обрезано)")
    return "\n".join(lines)
