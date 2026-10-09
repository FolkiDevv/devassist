"""Инструменты навигации по индексу проекта: поиск определений и оглавление файла.

Перед каждым запросом индекс инкрементально обновляется (перечитываются только
изменённые файлы), поэтому результаты отражают текущее состояние файлов, включая
правки самого агента.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

from pydantic import BaseModel, Field

from devassist.project.files import is_excluded, is_secret_path
from devassist.project.index import (
    INDEX_ERRORS,
    MAX_INDEX_FILE_BYTES,
    STATUS_BINARY,
    STATUS_ERROR,
    STATUS_LARGE,
    FileEntry,
    ProjectIndex,
    RefreshStats,
)
from devassist.project.symbols import Symbol
from devassist.security import resolve_in_root
from devassist.tools.base import Tool, ToolContext, ToolError, ToolResult

MAX_SYMBOL_RESULTS = 200
MAX_OUTLINE_FILES = 200  # больше файлов в каталоге — сводка по подкаталогам


@contextmanager
def _open_index(ctx: ToolContext) -> Iterator[ProjectIndex]:
    try:
        with ProjectIndex(ctx.workspace) as index:
            yield index
    except INDEX_ERRORS as e:
        raise ToolError(f"Индекс проекта недоступен: {e}") from e


def _refresh_note(stats: RefreshStats) -> str:
    return f"; индекс обновлён (файлов: {stats.changed})" if stats.changed else ""


def _lines(symbol: Symbol) -> str:
    if symbol.end_line and symbol.end_line != symbol.line:
        return f"{symbol.line}-{symbol.end_line}"
    return str(symbol.line)


# --------------------------------------------------------------------------- #
# find_symbol
# --------------------------------------------------------------------------- #
class FindSymbolParams(BaseModel):
    query: str = Field(
        description=(
            "Имя или часть имени определения (без учёта регистра): 'Agent', 'run_turn'; "
            "метод класса — 'Agent.run_turn'"
        )
    )
    kind: str | None = Field(
        default=None,
        description=(
            "Вид определения: function (включая методы), class (включая struct, interface, "
            "enum…), method, constant, section (заголовок Markdown) и т.п."
        ),
    )
    glob: str | None = Field(
        default=None, description="Ограничить файлами по glob: '*.py' или 'src/**/*.ts'"
    )
    max_results: int = Field(default=50, description="Максимум результатов")


class FindSymbolTool(Tool):
    name = "find_symbol"
    description = (
        "Ищет определения (классы, функции, методы, типы, константы, заголовки Markdown) "
        "по имени в индексе проекта. Возвращает path:строки вид имя — сигнатура. "
        "Быстрее search_content, когда нужно найти, где что-то определено; затем читайте "
        "нужный диапазон строк через read_file."
    )
    Params = FindSymbolParams

    def describe(self, params: FindSymbolParams) -> str:
        return f"{params.query} ({params.kind})" if params.kind else params.query

    def run(self, params: FindSymbolParams, ctx: ToolContext) -> ToolResult:
        if not params.query.strip():
            raise ToolError("Пустой запрос.")
        limit = min(max(params.max_results, 1), MAX_SYMBOL_RESULTS)
        with _open_index(ctx) as index:
            refreshed = index.refresh()
            hits, total = index.find_symbols(
                params.query, kind=params.kind, path_glob=params.glob, limit=limit
            )
        if not hits:
            body = (
                "(определений не найдено). Индекс знает только определения; "
                "использования ищите через search_content."
            )
        else:
            body = "\n".join(
                f"{h.path}:{_lines(h.symbol)} {h.symbol.kind} {h.symbol.qualname} — "
                f"{h.symbol.signature}"
                for h in hits
            )
            if total > len(hits):
                body += f"\n… показано {len(hits)} из {total}; уточните запрос, kind или glob"
        return ToolResult(
            content=body,
            summary=f"найдено определений: {total}{_refresh_note(refreshed)}",
        )


# --------------------------------------------------------------------------- #
# file_outline
# --------------------------------------------------------------------------- #
class FileOutlineParams(BaseModel):
    path: str = Field(description="Файл или каталог относительно корня проекта")


def _entry_line(entry: FileEntry, shown_path: str) -> str:
    details = [entry.language or "?", f"строк: {entry.lines}"]
    if entry.symbols:
        details.append(f"определений: {entry.symbols}")
    if entry.status == STATUS_LARGE:
        details.append("слишком большой для разбора")
    elif entry.status == STATUS_BINARY:
        details.append("бинарный")
    return f"{shown_path} ({', '.join(details)})"


class FileOutlineTool(Tool):
    name = "file_outline"
    description = (
        "Оглавление по индексу проекта. Для файла — его определения (классы, функции, "
        "методы, заголовки) с номерами строк и вложенностью: позволяет прочитать через "
        "read_file только нужный фрагмент большого файла. Для каталога — файлы с языком, "
        "числом строк и определений (при большом числе файлов — сводка по подкаталогам)."
    )
    Params = FileOutlineParams

    def run(self, params: FileOutlineParams, ctx: ToolContext) -> ToolResult:
        target = resolve_in_root(ctx.root, params.path)
        if not target.exists():
            raise ToolError(f"Путь не найден: {params.path}")
        if target.is_file():
            if is_secret_path(target):
                raise ToolError(f"Файлы с секретами не индексируются: {params.path}")
            if is_excluded(ctx.root, target):
                raise ToolError(
                    f"Файл исключён из индекса (.gitignore или служебный каталог): "
                    f"{params.path}. Используйте read_file."
                )
        with _open_index(ctx) as index:
            refreshed = index.refresh(target)
            rel = target.relative_to(ctx.root).as_posix()
            rel = "" if rel == "." else rel
            if target.is_dir():
                return self._directory(index, rel, refreshed)
            return self._file(index, rel, params.path, refreshed)

    def _file(
        self, index: ProjectIndex, rel: str, shown: str, refreshed: RefreshStats
    ) -> ToolResult:
        entry = index.file_entry(rel)
        if entry is None:
            raise ToolError(f"Файл не попал в индекс: {shown}. Используйте read_file.")
        if entry.status == STATUS_LARGE:
            raise ToolError(
                f"Файл слишком большой для разбора (>{MAX_INDEX_FILE_BYTES} байт): {shown}. "
                "Используйте search_content или read_file с диапазоном строк."
            )
        if entry.status == STATUS_BINARY:
            raise ToolError(f"Бинарный файл: {shown}")
        if entry.status == STATUS_ERROR:
            raise ToolError(f"Не удалось разобрать файл: {shown}. Используйте read_file.")
        symbols = index.outline(rel)
        header = _entry_line(entry, rel)
        if not symbols:
            body = f"{header}\n(определений не найдено)"
        else:
            body = (
                header
                + "\n"
                + "\n".join(
                    f"{'  ' * s.depth}{_lines(s)}  {s.signature or s.kind + ' ' + s.name}"
                    for s in symbols
                )
            )
        return ToolResult(
            content=body,
            summary=f"{rel}: определений {len(symbols)}{_refresh_note(refreshed)}",
        )

    def _directory(self, index: ProjectIndex, rel: str, refreshed: RefreshStats) -> ToolResult:
        entries = list(index.files_under(rel))
        prefix = f"{rel}/" if rel else ""
        if not entries:
            body = "(в индексе нет файлов этого каталога)"
        elif len(entries) <= MAX_OUTLINE_FILES:
            body = "\n".join(_entry_line(e, e.path[len(prefix) :]) for e in entries)
        else:
            body = self._grouped(entries, prefix)
        symbols = sum(e.symbols for e in entries)
        where = rel or "."
        return ToolResult(
            content=body,
            summary=f"{where}: файлов {len(entries)}, определений {symbols}"
            f"{_refresh_note(refreshed)}",
        )

    @staticmethod
    def _grouped(entries: list[FileEntry], prefix: str) -> str:
        groups: dict[str, list[FileEntry]] = defaultdict(list)
        direct: list[FileEntry] = []
        for e in entries:
            rest = e.path[len(prefix) :]
            if "/" in rest:
                groups[rest.split("/", 1)[0]].append(e)
            else:
                direct.append(e)
        lines = [
            f"{name}/ (файлов: {len(items)}, определений: {sum(i.symbols for i in items)})"
            for name, items in sorted(groups.items())
        ]
        lines += [_entry_line(e, e.path[len(prefix) :]) for e in direct[:MAX_OUTLINE_FILES]]
        if len(direct) > MAX_OUTLINE_FILES:
            lines.append(f"… и ещё {len(direct) - MAX_OUTLINE_FILES} файлов")
        lines.append(
            f"(всего файлов: {len(entries)} — показана сводка по подкаталогам; "
            "уточните путь, чтобы увидеть файлы)"
        )
        return "\n".join(lines)
