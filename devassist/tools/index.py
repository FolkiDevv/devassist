"""Инструменты навигации по индексу проекта: поиск определений, использований и
оглавление файла.

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
    RESOLVED_IMPORT,
    RESOLVED_NAME,
    RESOLVED_SAME_FILE,
    STATUS_BINARY,
    STATUS_ERROR,
    STATUS_LARGE,
    FileEntry,
    ProjectIndex,
    RefHit,
    RefreshStats,
    SymbolHit,
)
from devassist.project.symbols import Symbol, language_of
from devassist.security import resolve_in_root
from devassist.tools.base import Tool, ToolContext, ToolError, ToolResult

MAX_SYMBOL_RESULTS = 200
MAX_OUTLINE_FILES = 200  # больше файлов в каталоге — сводка по подкаталогам
MAX_LISTED_FILES = 15  # файлов в строках «импортирует» / «импортируется в»
MAX_SOURCE_LINE = 160


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


def _doc_note(symbol: Symbol) -> str:
    return f"  # {symbol.doc}" if symbol.doc else ""


def _definition_line(hit: SymbolHit) -> str:
    s = hit.symbol
    return f"{hit.path}:{_lines(s)} {s.kind} {s.qualname} — {s.signature}{_doc_note(s)}"


def _short_list(items: list[str]) -> str:
    text = ", ".join(items[:MAX_LISTED_FILES])
    if len(items) > MAX_LISTED_FILES:
        text += f", … и ещё {len(items) - MAX_LISTED_FILES}"
    return text


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
            body = "\n".join(_definition_line(h) for h in hits)
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
            # импорты Python-файла сопоставляются с другими файлами проекта — нужен весь индекс
            python = target.is_file() and language_of(target.name) == "python"
            refreshed = index.refresh() if python else index.refresh(target)
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
        header = _entry_line(entry, rel)
        if entry.language == "python":
            header += self._imports(index, rel)
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
        if not symbols:
            body = f"{header}\n(определений не найдено)"
        else:
            body = (
                header
                + "\n"
                + "\n".join(
                    f"{'  ' * s.depth}{_lines(s)}  {s.signature or s.kind + ' ' + s.name}"
                    f"{_doc_note(s)}"
                    for s in symbols
                )
            )
        return ToolResult(
            content=body,
            summary=f"{rel}: определений {len(symbols)}{_refresh_note(refreshed)}",
        )

    @staticmethod
    def _imports(index: ProjectIndex, rel: str) -> str:
        targets = sorted({e.target for e in index.imports_of(rel) if e.target not in (None, rel)})
        users = [path for path, _ in index.imported_by(rel)]
        text = ""
        if targets:
            text += f"\nимпортирует из проекта: {_short_list(targets)}"  # type: ignore[arg-type]
        if users:
            text += f"\nимпортируется в ({len(users)}): {_short_list(users)}"
        return text

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


# --------------------------------------------------------------------------- #
# find_references
# --------------------------------------------------------------------------- #
REF_KINDS = ("call", "attr", "name", "import")
_KIND_LABELS = {"call": "вызов", "attr": "атрибут", "name": "имя", "import": "импорт"}
_RESOLUTION_LABELS = {
    RESOLVED_IMPORT: "импортирует модуль с определением",
    RESOLVED_SAME_FILE: "файл определения",
    RESOLVED_NAME: "совпадение только по имени — может быть другое определение",
}


class FindReferencesParams(BaseModel):
    query: str = Field(
        description=(
            "Имя определения (точное, без учёта регистра): 'ProjectIndex', 'refresh'; "
            "метод класса — 'ProjectIndex.refresh'"
        )
    )
    kind: str | None = Field(
        default=None,
        description=(
            "Вид использования: call (вызовы), attr (обращения к атрибуту), name "
            "(упоминания имени), import (строки импорта); по умолчанию — все"
        ),
    )
    glob: str | None = Field(
        default=None, description="Ограничить файлами по glob: '*.py' или 'src/**'"
    )
    max_results: int = Field(default=50, description="Максимум результатов")


class _SourceLines:
    """Строки исходников для показа — читаются по файлу один раз."""

    def __init__(self, ctx: ToolContext):
        self._root = ctx.root
        self._cache: dict[str, list[str]] = {}

    def get(self, path: str, line: int) -> str:
        if path not in self._cache:
            try:
                text = (self._root / path).read_text(encoding="utf-8", errors="replace")
                self._cache[path] = text.splitlines()
            except OSError:
                self._cache[path] = []
        lines = self._cache[path]
        text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        if len(text) > MAX_SOURCE_LINE:
            text = text[: MAX_SOURCE_LINE - 1] + "…"
        return text


def format_ref_hits(hits: list[RefHit], ctx: ToolContext, labels: dict[str, str]) -> list[str]:
    """Использования, сгруппированные по файлам: заголовок файла — уровень разрешения."""
    source = _SourceLines(ctx)
    out: list[str] = []
    current = None
    for h in hits:
        if (h.path, h.resolution) != current:
            current = (h.path, h.resolution)
            out.append(f"{h.path} — {labels.get(h.resolution, h.resolution)}:")
        where = f" в {h.scope}" if h.scope else ""
        out.append(
            f"  {h.line} [{_KIND_LABELS.get(h.kind, h.kind)}]{where} — {source.get(h.path, h.line)}"
        )
    return out


class FindReferencesTool(Tool):
    name = "find_references"
    description = (
        "Ищет использования определения по индексу проекта: вызовы, обращения к "
        "атрибуту, упоминания, импорты — с функцией или классом, где они встречаются. "
        "Отвечает на вопросы «кто вызывает», «где используется», «что сломается при "
        "изменении». Сначала идут файлы, импортирующие модуль определения, затем сам "
        "файл определения, затем совпадения только по имени. Использования известны "
        "для Python; для других языков — search_content."
    )
    Params = FindReferencesParams

    def describe(self, params: FindReferencesParams) -> str:
        return f"{params.query} ({params.kind})" if params.kind else params.query

    def run(self, params: FindReferencesParams, ctx: ToolContext) -> ToolResult:
        if not params.query.strip(" ."):
            raise ToolError("Пустой запрос.")
        kind = params.kind.strip().lower() if params.kind else None
        if kind and kind not in REF_KINDS:
            raise ToolError(
                f"Неизвестный вид использования {params.kind!r}; допустимые: {', '.join(REF_KINDS)}"
            )
        limit = min(max(params.max_results, 1), MAX_SYMBOL_RESULTS)
        with _open_index(ctx) as index:
            refreshed = index.refresh()
            result = index.find_refs(params.query, kind=kind, path_glob=params.glob, limit=limit)
        lines = [f"определение: {_definition_line(d)}" for d in result.definitions]
        if not result.definitions:
            lines.append(
                f"(определение «{params.query.strip()}» в индексе не найдено — "
                "ищу использования по имени)"
            )
        if not result.hits:
            lines.append("(использований не найдено)")
        else:
            lines.append(f"использования ({result.total}):")
            lines += format_ref_hits(result.hits, ctx, _RESOLUTION_LABELS)
            if result.total > len(result.hits):
                lines.append(
                    f"… показано {len(result.hits)} из {result.total}; "
                    "уточните запрос, kind или glob"
                )
        return ToolResult(
            content="\n".join(lines),
            summary=f"найдено использований: {result.total}{_refresh_note(refreshed)}",
        )
