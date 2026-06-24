"""Понимание контекста проекта.

Собирает компактное описание проекта для системного промпта:
  * краткое дерево файлов (с игнорированием служебных директорий);
  * содержимое файла-памяти DEVASSIST.md (если есть) — заметки, которые
    пользователь/агент хочет держать в контексте между сессиями.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

MEMORY_FILENAME = "DEVASSIST.md"
_IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".pytest_cache", ".mypy_cache", "dist", "build", ".idea", ".devassist",
}
_MAX_ENTRIES = 200


def build_file_tree(root: Path, max_entries: int = _MAX_ENTRIES) -> str:
    """Компактное дерево проекта (отсортированное, с обрезкой)."""
    lines: List[str] = []
    count = 0

    def walk(directory: Path, prefix: str) -> None:
        nonlocal count
        try:
            entries = sorted(
                directory.iterdir(), key=lambda e: (e.is_file(), e.name.lower())
            )
        except OSError:
            return
        for e in entries:
            if count >= max_entries:
                return
            if e.name.startswith(".") and e.name != ".env.example":
                continue
            if e.is_dir() and e.name in _IGNORE_DIRS:
                continue
            count += 1
            if e.is_dir():
                lines.append(f"{prefix}{e.name}/")
                walk(e, prefix + "  ")
            else:
                lines.append(f"{prefix}{e.name}")

    walk(root, "")
    if count >= max_entries:
        lines.append("... (дерево обрезано)")
    return "\n".join(lines)


def read_memory(root: Path) -> str:
    """Читает файл-память проекта, если он существует."""
    path = root / MEMORY_FILENAME
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
    return ""


def build_project_context(root: Path) -> str:
    """Формирует блок контекста проекта для системного промпта."""
    parts = [f"Корень проекта: {root}"]
    memory = read_memory(root)
    if memory:
        parts.append(f"\nЗаметки проекта ({MEMORY_FILENAME}):\n{memory}")
    tree = build_file_tree(root)
    if tree:
        parts.append(f"\nСтруктура проекта:\n{tree}")
    return "\n".join(parts)
