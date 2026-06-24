"""Поиск по содержимому кодовой базы (аналог grep)."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from devassist.security import resolve_in_root
from devassist.tools.base import Tool, ToolContext, ToolError, ToolResult

_IGNORE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".pytest_cache"}
_MAX_MATCHES = 200


class SearchContentParams(BaseModel):
    pattern: str = Field(description="Регулярное выражение для поиска по содержимому")
    path: str = Field(default=".", description="Директория для поиска (от корня проекта)")
    glob: str | None = Field(
        default=None, description="Ограничить поиск файлами по glob, например '*.py'"
    )
    ignore_case: bool = Field(default=False, description="Игнорировать регистр")
    max_results: int = Field(default=100, description="Максимум совпадений")


class SearchContentTool(Tool):
    name = "search_content"
    description = (
        "Ищет строки, соответствующие регулярному выражению, по файлам проекта. "
        "Возвращает совпадения в формате path:line:текст. Аналог grep -rn."
    )
    Params = SearchContentParams

    def run(self, params: SearchContentParams, ctx: ToolContext) -> ToolResult:
        base = resolve_in_root(ctx.root, params.path)
        if not base.exists():
            raise ToolError(f"Путь не найден: {params.path}")
        flags = re.IGNORECASE if params.ignore_case else 0
        try:
            regex = re.compile(params.pattern, flags)
        except re.error as e:
            raise ToolError(f"Некорректное регулярное выражение: {e}") from e

        import fnmatch

        files = base.rglob("*") if base.is_dir() else [base]
        matches: list[str] = []
        scanned = 0
        for f in files:
            if not f.is_file():
                continue
            if any(part in _IGNORE_DIRS for part in f.parts):
                continue
            if params.glob and not fnmatch.fnmatch(f.name, params.glob):
                continue
            scanned += 1
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    rel = f.relative_to(ctx.root)
                    snippet = line.strip()[:200]
                    matches.append(f"{rel}:{i}:{snippet}")
                    if len(matches) >= params.max_results:
                        break
            if len(matches) >= params.max_results:
                break

        body = "\n".join(matches) if matches else "(совпадений не найдено)"
        return ToolResult(
            content=body,
            summary=f"совпадений: {len(matches)} (файлов просмотрено: {scanned})",
        )
