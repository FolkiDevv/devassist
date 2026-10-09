"""Поиск по содержимому кодовой базы (аналог grep)."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from devassist.project.files import glob_match, is_secret_file, walk_files
from devassist.security import resolve_in_root
from devassist.tools.base import Tool, ToolContext, ToolError, ToolResult

_MAX_MATCHES = 500  # потолок max_results, который может запросить модель
_MAX_FILE_BYTES = 1_000_000  # файлы крупнее не читаем (сгенерированное, дампы)
_BINARY_PROBE = 8192  # NUL в первых байтах — бинарный файл
_MAX_SNIPPET = 200


class SearchContentParams(BaseModel):
    pattern: str = Field(description="Регулярное выражение для поиска по содержимому")
    path: str = Field(default=".", description="Директория или файл для поиска (от корня проекта)")
    glob: str | None = Field(
        default=None,
        description="Ограничить поиск файлами по glob: '*.py' (по имени) или 'src/**/*.py'",
    )
    ignore_case: bool = Field(default=False, description="Игнорировать регистр")
    max_results: int = Field(default=100, description="Максимум совпадений")


class SearchContentTool(Tool):
    name = "search_content"
    description = (
        "Ищет строки, соответствующие регулярному выражению, по файлам проекта. "
        "Возвращает совпадения в формате path:line:текст. Аналог grep -rn. "
        "Пропускает служебные каталоги, бинарные и очень большие файлы, а также .env."
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
        limit = min(max(params.max_results, 1), _MAX_MATCHES)

        walking = base.is_dir()
        files = walk_files(ctx.root, base) if walking else [base]
        matches: list[str] = []
        scanned = skipped = 0
        for f in files:
            rel = f.relative_to(ctx.root).as_posix()
            if params.glob and not glob_match(rel, params.glob):
                continue
            # секреты не ищем при обходе каталога; явно указанный файл — можно
            if walking and is_secret_file(f.name):
                continue
            try:
                if f.stat().st_size > _MAX_FILE_BYTES:
                    skipped += 1
                    continue
                data = f.read_bytes()
            except OSError:
                continue
            if b"\0" in data[:_BINARY_PROBE]:
                continue
            scanned += 1
            text = data.decode("utf-8", errors="replace")
            for i, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{rel}:{i}:{line.strip()[:_MAX_SNIPPET]}")
                    if len(matches) >= limit:
                        break
            if len(matches) >= limit:
                break

        body = "\n".join(matches) if matches else "(совпадений не найдено)"
        if len(matches) >= limit:
            body += f"\n… достигнут лимит {limit} совпадений; уточните запрос"
        summary = f"совпадений: {len(matches)} (файлов просмотрено: {scanned})"
        if skipped:
            summary += f", пропущено больших файлов: {skipped}"
        return ToolResult(content=body, summary=summary)
