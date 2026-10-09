"""Файловые инструменты: чтение, запись, точечное редактирование, листинг, поиск файлов."""

from __future__ import annotations

import difflib
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

from pydantic import BaseModel, Field

from devassist.project.files import glob_match, walk_files
from devassist.project.workspace import DATA_DIR_NAME
from devassist.security import RiskLevel, resolve_in_root
from devassist.tools.base import Display, Tool, ToolContext, ToolError, ToolResult

# Жёсткий предел размера читаемого файла (дальше — только search_content).
MAX_READ_BYTES = 20_000_000
# Сколько отдаётся модели за один вызов: строки, символы на строку, символы всего.
MAX_READ_LINES = 1000
MAX_LINE_CHARS = 1000
MAX_READ_CHARS = 60_000
_BINARY_PROBE = 8192


def _rel(ctx: ToolContext, path: Path) -> str:
    try:
        return str(path.relative_to(ctx.root))
    except ValueError:
        return str(path)


def _writable_path(ctx: ToolContext, path: str) -> Path:
    """Путь для записи: внутри корня и не в служебной папке ``.devassist/``.

    Там лежат индекс, история ввода и чаты агента — модель не должна их править.
    Проверка срабатывает и в превью, то есть до вопроса о подтверждении.
    """
    p = resolve_in_root(ctx.root, path)
    data_dir = ctx.workspace.data_dir.resolve()  # .devassist может быть симлинком
    # is_data_path — ещё и без учёта регистра: на macOS/Windows .DEVASSIST — та же папка.
    if p == data_dir or data_dir in p.parents or ctx.workspace.is_data_path(p):
        raise ToolError(
            f"Служебная папка {DATA_DIR_NAME}/ (индекс, история, чаты агента) "
            f"недоступна для записи: {path}"
        )
    return p


def make_diff(old: str, new: str, path: str) -> str:
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff)


def _unescape_simple(s: str) -> str:
    """Разэкранирует частые последовательности (\\n, \\t, \\", \\') без условий.

    Используется как один из вариантов-кандидатов при поиске old_string —
    применяется только если разэкранированная форма реально нашлась в файле.
    """
    return (
        s.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\'", "'")
    )


_LINE_NUM_RE = re.compile(r"^\s*\d+\t")


def _strip_line_numbers(s: str) -> str:
    """Удаляет префиксы-номера строк (``N\\t``), если ими помечено большинство строк.

    read_file показывает содержимое с номерами (как ``cat -n``); слабые модели
    иногда копируют эти префиксы прямо в old_string/new_string. Снимаем их,
    чтобы фрагмент совпал с реальным файлом.
    """
    lines = s.split("\n")
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return s
    marked = sum(1 for ln in nonempty if _LINE_NUM_RE.match(ln))
    if marked < max(2, 0.6 * len(nonempty)):
        return s
    return "\n".join(_LINE_NUM_RE.sub("", ln) for ln in lines)


def repair_escaped_content(content: str) -> tuple[str, bool]:
    """Чинит «двойную экранизацию», которую иногда выдаёт модель.

    Слабые модели порой кладут в write_file одну физическую строку, где все
    переносы — это литералы ``\\n`` (а кавычки — ``\\"``), иногда ещё и с
    префиксами номеров строк (``1\\timport ...``), скопированными из подсказки.
    Такой контент однозначно битый: ноль настоящих переносов при множестве
    литеральных ``\\n``. В этом случае разэкранируем и снимаем нумерацию.

    Возвращает (исправленный_текст, был_ли_ремонт).
    """
    if content.count("\n") == 0 and content.count("\\n") >= 2:
        fixed = (
            content.replace("\\r\\n", "\n")
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace('\\"', '"')
            .replace("\\'", "'")
            .replace("\\\\", "\\")
        )
        lines = fixed.split("\n")
        nonempty = [ln for ln in lines if ln.strip()]
        if nonempty and all(re.match(r"^\d+\t", ln) for ln in nonempty):
            lines = [re.sub(r"^\d+\t", "", ln) for ln in lines]
            fixed = "\n".join(lines)
        return fixed, True
    return content, False


def _numbered_excerpt(text: str, max_lines: int = 60) -> str:
    """Содержимое файла с номерами строк (для подсказки при неудачном edit)."""
    lines = text.splitlines()
    shown = lines[:max_lines]
    width = len(str(len(shown)))
    body = "\n".join(f"{str(i + 1).rjust(width)}\t{ln}" for i, ln in enumerate(shown))
    if len(lines) > max_lines:
        body += f"\n… (ещё {len(lines) - max_lines} строк)"
    return body


def _tolerant_find(text: str, pattern: str):
    """Ищет блок строк, совпадающий с pattern с точностью до пробелов/отступов.

    Возвращает (start_offset, end_offset) в исходном тексте при ЕДИНСТВЕННОМ
    совпадении, иначе None. Используется как запасной вариант, когда точное
    совпадение не найдено (модель часто слегка путает отступы/хвостовые пробелы).
    """
    raw = text.splitlines(keepends=True)
    if not raw:
        return None
    offsets = []
    pos = 0
    for ln in raw:
        offsets.append(pos)
        pos += len(ln)
    contents = [ln.rstrip("\r\n") for ln in raw]

    pat_lines = pattern.splitlines()
    while pat_lines and not pat_lines[0].strip():
        pat_lines.pop(0)
    while pat_lines and not pat_lines[-1].strip():
        pat_lines.pop()
    n = len(pat_lines)
    if n == 0:
        return None

    # От более строгой нормализации (только хвостовые пробелы) к более мягкой
    # (полный strip — игнор отступов). Берём первый режим с уникальным совпадением.
    for norm in (lambda s: s.rstrip(), lambda s: s.strip()):
        target = [norm(ln) for ln in pat_lines]
        hits = [
            i
            for i in range(len(contents) - n + 1)
            if [norm(contents[j]) for j in range(i, i + n)] == target
        ]
        if len(hits) == 1:
            i = hits[0]
            start = offsets[i]
            end = offsets[i + n - 1] + len(contents[i + n - 1])
            return (start, end)
    return None


# --------------------------------------------------------------------------- #
# read_file
# --------------------------------------------------------------------------- #
class ReadFileParams(BaseModel):
    path: str = Field(description="Путь к файлу относительно корня проекта")
    start_line: int | None = Field(
        default=None, description="Начальная строка (1-индексация), включительно"
    )
    end_line: int | None = Field(
        default=None, description="Конечная строка (1-индексация), включительно"
    )


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Читает содержимое текстового файла. Можно указать диапазон строк "
        "(start_line/end_line). Возвращает текст с номерами строк; за один вызов — "
        f"не более {MAX_READ_LINES} строк, для продолжения укажите start_line."
    )
    Params = ReadFileParams

    def run(self, params: ReadFileParams, ctx: ToolContext) -> ToolResult:
        p = resolve_in_root(ctx.root, params.path)
        if not p.exists():
            raise ToolError(f"Файл не найден: {params.path}")
        if p.is_dir():
            raise ToolError(f"Это директория, а не файл: {params.path}")
        if not p.is_file():  # FIFO/сокет/устройство — чтение может заблокироваться
            raise ToolError(f"Не обычный файл (FIFO, сокет или устройство): {params.path}")
        if p.stat().st_size > MAX_READ_BYTES:
            raise ToolError(
                f"Файл слишком большой (>{MAX_READ_BYTES} байт). Используйте search_content."
            )
        with p.open("rb") as fh:
            if b"\0" in fh.read(_BINARY_PROBE):
                raise ToolError(f"Бинарный файл, чтение не поддерживается: {params.path}")

        start = max(params.start_line or 1, 1)
        end = params.end_line
        selected: list[tuple[int, str]] = []
        budget = MAX_READ_CHARS
        total = 0
        stopped_at: int | None = None  # первая строка, которая не поместилась
        try:
            with p.open(encoding="utf-8") as fh:
                for total, raw in enumerate(fh, start=1):
                    if total < start or (end is not None and total > end):
                        continue
                    if stopped_at is not None:
                        continue  # досчитываем общее число строк
                    line = raw.rstrip("\r\n")
                    if len(line) > MAX_LINE_CHARS:
                        line = line[:MAX_LINE_CHARS] + " …[строка обрезана]"
                    if len(selected) >= MAX_READ_LINES or len(line) + 1 > budget:
                        stopped_at = total
                        continue
                    selected.append((total, line))
                    budget -= len(line) + 1
        except UnicodeDecodeError as e:
            raise ToolError(f"Файл не является текстовым (UTF-8): {params.path}") from e

        if not selected:
            numbered = "(пусто)" if total == 0 else f"(нет строк в диапазоне; всего строк: {total})"
        else:
            width = len(str(selected[-1][0]))
            numbered = "\n".join(f"{str(n).rjust(width)}\t{line}" for n, line in selected)
        if stopped_at is not None:
            numbered += (
                f"\n… показаны строки {selected[0][0] if selected else start}–"
                f"{stopped_at - 1} из {total}. Продолжение: start_line={stopped_at}."
            )
        return ToolResult(
            content=numbered,
            summary=f"прочитан {_rel(ctx, p)} ({len(selected)} строк)",
        )


# --------------------------------------------------------------------------- #
# write_file
# --------------------------------------------------------------------------- #
class WriteFileParams(BaseModel):
    path: str = Field(description="Путь к файлу относительно корня проекта")
    content: str = Field(description="Полное новое содержимое файла")


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Создаёт новый файл или полностью перезаписывает существующий заданным "
        "содержимым. Создаёт родительские директории при необходимости."
    )
    Params = WriteFileParams

    def risk(self, params: WriteFileParams, ctx: ToolContext) -> RiskLevel:
        return RiskLevel.WRITE

    def _old_content(self, ctx: ToolContext, params: WriteFileParams) -> str:
        p = _writable_path(ctx, params.path)
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return ""
        return ""

    def preview(self, params: WriteFileParams, ctx: ToolContext) -> Display | None:
        old = self._old_content(ctx, params)
        content, _ = repair_escaped_content(params.content)
        diff = make_diff(old, content, params.path) or "(новый пустой файл)"
        return Display(diff, kind="diff", title=params.path)

    def run(self, params: WriteFileParams, ctx: ToolContext) -> ToolResult:
        p = _writable_path(ctx, params.path)
        if p.is_dir():
            raise ToolError(f"Это директория: {params.path}")
        existed = p.is_file()
        old = self._old_content(ctx, params)
        content, repaired = repair_escaped_content(params.content)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        verb = "перезаписан" if existed else "создан"
        n = len(content.splitlines())
        note = " (автокоррекция экранирования)" if repaired else ""
        return ToolResult(
            content=f"Файл {verb}: {params.path} ({n} строк).{note}",
            summary=f"{verb} {_rel(ctx, p)}{note}",
            display=Display(make_diff(old, content, params.path), kind="diff", title=params.path),
        )


# --------------------------------------------------------------------------- #
# edit_file
# --------------------------------------------------------------------------- #
class EditFileParams(BaseModel):
    path: str = Field(description="Путь к файлу относительно корня проекта")
    old_string: str = Field(
        description="Точный фрагмент, который нужно заменить (должен встречаться)"
    )
    new_string: str = Field(description="Текст замены")
    replace_all: bool = Field(default=False, description="Заменить все вхождения, а не только одно")


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Точечно редактирует файл: заменяет old_string на new_string. "
        "old_string должен встречаться ровно один раз (иначе добавьте контекста "
        "или укажите replace_all=true для замены всех вхождений). Совпадение "
        "устойчиво к незначительным различиям в отступах и хвостовых пробелах, "
        "но фрагмент лучше копировать дословно из содержимого файла."
    )
    Params = EditFileParams

    def risk(self, params: EditFileParams, ctx: ToolContext) -> RiskLevel:
        return RiskLevel.WRITE

    def _compute(self, params: EditFileParams, ctx: ToolContext):
        p = _writable_path(ctx, params.path)
        if not p.is_file():
            raise ToolError(f"Файл не найден: {params.path}")
        try:
            old = p.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ToolError(f"Файл не текстовый: {params.path}") from e
        if params.old_string == params.new_string:
            raise ToolError("old_string и new_string совпадают — нечего менять.")

        # Слабые модели часто слегка искажают old_string: экранируют переносы
        # (литералы \n, \"), либо копируют префиксы-номера строк (N\t) прямо из
        # вывода read_file. Пробуем набор согласованных преобразований к ОБОИМ
        # фрагментам и берём первое, которое реально находится в файле — это
        # безопасно: преобразованная форма применяется, только если совпала.
        def both(s):
            return _strip_line_numbers(_unescape_simple(s))

        transforms = [
            lambda s: s,  # как прислано
            _unescape_simple,  # снять экранизацию \n/\"
            _strip_line_numbers,  # снять префиксы номеров строк
            both,  # и то, и другое
        ]

        seen = set()
        for tf in transforms:
            old_string = tf(params.old_string)
            if old_string in seen:
                continue
            seen.add(old_string)
            new_string = tf(params.new_string)

            # Тир 1: точное совпадение.
            count = old.count(old_string)
            if count == 1 or (count > 1 and params.replace_all):
                new = old.replace(old_string, new_string)
                return p, old, new, count
            if count > 1 and not params.replace_all:
                raise ToolError(
                    f"old_string встречается {count} раз. Добавьте контекста, чтобы "
                    "фрагмент стал уникальным, или укажите replace_all=true."
                )
            # Тир 2: совпадение с поправкой на пробелы/отступы (одиночная замена).
            span = _tolerant_find(old, old_string)
            if span is not None:
                start, end = span
                new = old[:start] + new_string + old[end:]
                return p, old, new, 1

        # Не найдено — даём модели контекст файла, чтобы скопировать точно.
        raise ToolError(
            "old_string не найден в файле (ни точно, ни с поправкой на пробелы). "
            "Скопируйте фрагмент дословно из содержимого ниже:\n\n" + _numbered_excerpt(old)
        )

    def preview(self, params: EditFileParams, ctx: ToolContext) -> Display | None:
        _, old, new, _ = self._compute(params, ctx)
        return Display(make_diff(old, new, params.path), kind="diff", title=params.path)

    def run(self, params: EditFileParams, ctx: ToolContext) -> ToolResult:
        p, old, new, count = self._compute(params, ctx)
        p.write_text(new, encoding="utf-8")
        replaced = count if params.replace_all else 1
        return ToolResult(
            content=f"Отредактирован {params.path}: заменено вхождений — {replaced}.",
            summary=f"изменён {_rel(ctx, p)}",
            display=Display(make_diff(old, new, params.path), kind="diff", title=params.path),
        )


# --------------------------------------------------------------------------- #
# list_dir
# --------------------------------------------------------------------------- #
MAX_LIST_ENTRIES = 500


class ListDirParams(BaseModel):
    path: str = Field(default=".", description="Путь к директории (по умолчанию корень)")


class ListDirTool(Tool):
    name = "list_dir"
    description = "Выводит список файлов и поддиректорий (включая скрытые) в указанной директории."
    Params = ListDirParams

    def run(self, params: ListDirParams, ctx: ToolContext) -> ToolResult:
        p = resolve_in_root(ctx.root, params.path)
        if not p.is_dir():
            raise ToolError(f"Не директория: {params.path}")
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        lines = [f"{e.name}/" if e.is_dir() else e.name for e in entries[:MAX_LIST_ENTRIES]]
        body = "\n".join(lines) if lines else "(пусто)"
        if len(entries) > MAX_LIST_ENTRIES:
            body += f"\n… показано {MAX_LIST_ENTRIES} из {len(entries)}"
        return ToolResult(content=body, summary=f"{_rel(ctx, p)}: {len(entries)} элементов")


# --------------------------------------------------------------------------- #
# find_files
# --------------------------------------------------------------------------- #
MAX_FIND_RESULTS = 1000


class FindFilesParams(BaseModel):
    pattern: str = Field(
        description="Glob-шаблон: '*.py' — по имени в любом каталоге, 'src/**/*.ts' — путь от корня"
    )
    max_results: int = Field(default=200, description="Максимум результатов")


def _check_pattern(pattern: str) -> None:
    """Шаблон должен быть относительным и не подниматься выше корня проекта."""
    if not pattern:
        raise ToolError("Пустой шаблон.")
    if PurePosixPath(pattern).is_absolute() or PureWindowsPath(pattern).is_absolute():
        raise ToolError("Шаблон должен быть относительным путём от корня проекта.")
    if ".." in re.split(r"[\\/]", pattern):
        raise ToolError("Шаблон не может содержать '..' — поиск только внутри проекта.")


class FindFilesTool(Tool):
    name = "find_files"
    description = (
        "Ищет файлы по glob-шаблону (рекурсивно, внутри проекта). Шаблон без '/' "
        "сравнивается с именем файла, с '/' — с путём от корня ('**' — любые каталоги). "
        "Игнорирует служебные директории (.git, node_modules и т.п.) и файлы из .gitignore."
    )
    Params = FindFilesParams

    def run(self, params: FindFilesParams, ctx: ToolContext) -> ToolResult:
        pattern = params.pattern.strip()
        _check_pattern(pattern)
        limit = min(max(params.max_results, 1), MAX_FIND_RESULTS)
        root = ctx.root
        matches = sorted(
            rel
            for rel in (path.relative_to(root).as_posix() for path in walk_files(root))
            if glob_match(rel, pattern)
        )
        shown = matches[:limit]
        body = "\n".join(shown) if shown else "(ничего не найдено)"
        if len(matches) > limit:
            body += f"\n… показано {limit} из {len(matches)}; уточните шаблон"
        return ToolResult(content=body, summary=f"найдено файлов: {len(matches)}")
