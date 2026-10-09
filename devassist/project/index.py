"""Индекс проекта: файлы и определения (символы) в ``.devassist/index/``.

Индекс — база SQLite (stdlib): выборки и точечные правки не требуют загружать его
целиком в память, поэтому он годится и для больших кодовых баз. Обновление
инкрементальное: обходятся файлы проекта (с теми же правилами игнорирования, что
у поиска, — :func:`~devassist.project.files.walk_files`), перечитываются только
файлы с изменившимися размером или временем изменения, исчезнувшие — удаляются.

Изменения фиксируются пачками: прерванное построение (Ctrl+C) сохраняет уже
сделанное, а следующее обновление его продолжает. Признак полноты
(:meth:`ProjectIndex.is_complete`) выставляется только в конце полного обхода.

Файлы с секретами (``.env``) не индексируются; большие и бинарные файлы попадают
в список файлов без разбора содержимого.
"""

from __future__ import annotations

import sqlite3
import stat
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from devassist.project.files import glob_match, is_excluded, is_secret_file, walk_files
from devassist.project.symbols import Symbol, extract_symbols, language_of
from devassist.project.workspace import Workspace

SCHEMA_VERSION = 1
INDEX_FILE_NAME = "index.sqlite3"
MAX_INDEX_FILE_BYTES = 1_000_000  # крупнее — сгенерированное/дампы: без разбора
_BINARY_PROBE = 8192
_BATCH = 200  # изменённых файлов на одну транзакцию
_COUNT_CAP = 10_000  # дальше совпадения не досчитываются

# Ошибки, которые вызывающий код ловит при работе с индексом.
INDEX_ERRORS: tuple[type[BaseException], ...] = (sqlite3.Error, OSError)

# Обобщённые виды для фильтра: "function" находит и методы и т.п.
KIND_GROUPS: dict[str, tuple[str, ...]] = {
    "function": ("function", "method"),
    "class": (
        "class",
        "struct",
        "interface",
        "trait",
        "record",
        "object",
        "protocol",
        "enum",
        "union",
        "module",
    ),
}

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE files (
    path TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    language TEXT,
    lines INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL
);
CREATE TABLE symbols (
    path TEXT NOT NULL,
    name TEXT NOT NULL,
    name_lower TEXT NOT NULL,
    parent TEXT NOT NULL,
    parent_lower TEXT NOT NULL,
    kind TEXT NOT NULL,
    line INTEGER NOT NULL,
    end_line INTEGER,
    depth INTEGER NOT NULL,
    signature TEXT NOT NULL
);
CREATE INDEX symbols_name ON symbols (name_lower);
CREATE INDEX symbols_path ON symbols (path);
"""

# Состояния файла в индексе.
STATUS_INDEXED = "indexed"
STATUS_LARGE = "large"
STATUS_BINARY = "binary"
STATUS_ERROR = "error"


@dataclass(frozen=True)
class RefreshStats:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    duration_s: float = 0.0

    @property
    def changed(self) -> int:
        return self.added + self.updated + self.removed


@dataclass(frozen=True)
class IndexStats:
    files: int
    symbols: int
    languages: list[tuple[str, int]]  # (язык, файлов) по убыванию
    complete: bool
    updated_at: float | None  # время последнего полного обновления (epoch)
    db_bytes: int


@dataclass(frozen=True)
class FileEntry:
    path: str
    language: str | None
    lines: int
    status: str
    symbols: int = 0


@dataclass(frozen=True)
class SymbolHit:
    path: str
    symbol: Symbol


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _under(prefix: str) -> tuple[str, list[str]]:
    """SQL-условие «путь равен ``prefix`` или лежит под ним» (пустой — весь проект)."""
    if not prefix:
        return "1", []
    return "(path = ? OR path LIKE ? ESCAPE '\\')", [prefix, _escape_like(prefix) + "/%"]


def _count_lines(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


class ProjectIndex:
    """Индекс проекта. Конструктор не трогает ФС; база открывается в :meth:`open`.

    Использование::

        with ProjectIndex(workspace) as index:
            index.refresh()
            hits, total = index.find_symbols("Agent")
    """

    def __init__(self, workspace: Workspace):
        self._ws = workspace
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------ #
    @property
    def path(self) -> Path:
        return self._ws.index_dir / INDEX_FILE_NAME

    @property
    def root(self) -> Path:
        return self._ws.root

    def __enter__(self) -> ProjectIndex:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def open(self) -> ProjectIndex:
        """Открывает (создаёт) базу; несовместимая или повреждённая — пересоздаётся."""
        if self._conn is not None:
            return self
        self._ws.ensure_data_dir()
        self._ws.index_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = self._connect_checked()
        except sqlite3.DatabaseError:
            self._recreate_file()
            self._conn = self._connect_checked()
        return self

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()

    @property
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self.open()
        assert self._conn is not None
        return self._conn

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError:
            pass  # ФС без поддержки WAL — работаем в режиме по умолчанию
        except BaseException:
            conn.close()
            raise
        return conn

    def _connect_checked(self) -> sqlite3.Connection:
        conn = self._connect()
        try:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
            if not tables:
                conn.executescript(_SCHEMA)
                with conn:
                    conn.execute(
                        "INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
                    )
                return conn
            row = None
            if "meta" in tables:
                row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if row is not None and row[0] == str(SCHEMA_VERSION):
                return conn
        except BaseException:
            conn.close()
            raise
        conn.close()
        raise sqlite3.DatabaseError("несовместимая версия индекса")

    def _recreate_file(self) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(f"{self.path}{suffix}").unlink(missing_ok=True)

    # ------------------------------ meta ------------------------------- #
    def _meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    def is_complete(self) -> bool:
        """Индекс существует и полностью построен (не создаёт базу, если её нет)."""
        if not self.path.is_file():
            return False
        return self._meta("complete") == "1"

    # ---------------------------- обновление ---------------------------- #
    def _rel(self, path: Path) -> str:
        rel = path.relative_to(self.root).as_posix()
        return "" if rel == "." else rel

    def _candidates(self, base: Path) -> Iterable[Path]:
        if is_excluded(self.root, base):
            return ()
        if base.is_dir():
            return walk_files(self.root, base)
        return (base,) if base.exists() else ()

    def refresh(
        self, base: Path | None = None, *, progress: Callable[[int], None] | None = None
    ) -> RefreshStats:
        """Синхронизирует индекс с файлами под ``base`` (по умолчанию — весь проект).

        ``progress`` получает число просмотренных файлов. Прерывание (в т.ч. Ctrl+C)
        откатывает только незафиксированную пачку изменений.
        """
        started = time.monotonic()
        db = self._db
        base = self.root if base is None else base
        full = base == self.root
        prefix = self._rel(base)
        cond, args = _under(prefix)
        known = {
            path: (size, mtime)
            for path, size, mtime in db.execute(
                f"SELECT path, size, mtime_ns FROM files WHERE {cond}", args
            )
        }
        scanned = added = updated = pending = 0
        seen: set[str] = set()
        try:
            if full:
                with db:
                    self._set_meta("complete", "0")
            for path in self._candidates(base):
                if is_secret_file(path.name):
                    continue
                try:
                    st = path.stat()
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                rel = self._rel(path)
                seen.add(rel)
                scanned += 1
                if progress is not None:
                    progress(scanned)
                old = known.get(rel)
                if old == (st.st_size, st.st_mtime_ns):
                    continue
                self._index_file(rel, path, st)
                if old is None:
                    added += 1
                else:
                    updated += 1
                pending += 1
                if pending >= _BATCH:
                    db.commit()
                    pending = 0
            gone = [p for p in known if p not in seen]
            for chunk in _chunks(gone, 500):
                marks = ",".join("?" * len(chunk))
                db.execute(f"DELETE FROM symbols WHERE path IN ({marks})", chunk)
                db.execute(f"DELETE FROM files WHERE path IN ({marks})", chunk)
            if full:
                self._set_meta("complete", "1")
                self._set_meta("updated_at", str(time.time()))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        return RefreshStats(
            scanned=scanned,
            added=added,
            updated=updated,
            removed=len(gone),
            duration_s=time.monotonic() - started,
        )

    def _index_file(self, rel: str, path: Path, st) -> None:
        language = language_of(rel)
        status, lines, symbols = STATUS_INDEXED, 0, []
        if st.st_size > MAX_INDEX_FILE_BYTES:
            status = STATUS_LARGE
        else:
            try:
                data = path.read_bytes()
            except OSError:
                data, status = b"", STATUS_ERROR
            if b"\0" in data[:_BINARY_PROBE]:
                status = STATUS_BINARY
            elif status == STATUS_INDEXED:
                text = data.decode("utf-8", errors="replace")
                lines = _count_lines(text)
                try:
                    symbols = extract_symbols(text, language)
                except Exception:  # разбор — эвристика: сбой на одном файле не валит индекс
                    status = STATUS_ERROR
        db = self._db
        db.execute("DELETE FROM symbols WHERE path = ?", (rel,))
        db.execute(
            "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?, ?)",
            (rel, st.st_size, st.st_mtime_ns, language, lines, status),
        )
        db.executemany(
            "INSERT INTO symbols VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    rel,
                    s.name,
                    s.name.lower(),
                    s.parent,
                    s.parent.lower(),
                    s.kind,
                    s.line,
                    s.end_line,
                    s.depth,
                    s.signature,
                )
                for s in symbols
            ],
        )

    def rebuild(self, *, progress: Callable[[int], None] | None = None) -> RefreshStats:
        """Удаляет базу и строит индекс заново."""
        self.close()
        self._recreate_file()
        self.open()
        return self.refresh(progress=progress)

    # ------------------------------ запросы ----------------------------- #
    def stats(self) -> IndexStats:
        db = self._db
        files = db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        symbols = db.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        languages = [
            (lang, n)
            for lang, n in db.execute(
                "SELECT language, COUNT(*) AS n FROM files WHERE language IS NOT NULL "
                "GROUP BY language ORDER BY n DESC, language"
            )
        ]
        updated = self._meta("updated_at")
        try:
            size = sum(
                Path(f"{self.path}{s}").stat().st_size
                for s in ("", "-wal")
                if Path(f"{self.path}{s}").exists()
            )
        except OSError:
            size = 0
        return IndexStats(
            files=files,
            symbols=symbols,
            languages=languages,
            complete=self._meta("complete") == "1",
            updated_at=float(updated) if updated else None,
            db_bytes=size,
        )

    def find_symbols(
        self,
        query: str,
        *,
        kind: str | None = None,
        path_glob: str | None = None,
        limit: int = 50,
    ) -> tuple[list[SymbolHit], int]:
        """Определения, чьё имя содержит ``query`` (без учёта регистра).

        ``Class.method`` — метод с именем ``method`` у владельца, оканчивающегося на
        ``Class``. Порядок: точное совпадение имени, начало имени, вхождение; затем
        короткие имена и путь. Возвращает (первые ``limit``, сколько всего — не
        больше :data:`_COUNT_CAP`).
        """
        query = query.strip().lower()
        if not query:
            raise ValueError("пустой запрос")
        parent = ""
        if "." in query.strip("."):
            parent, query = query.rsplit(".", 1)
        where = ["name_lower LIKE ? ESCAPE '\\'"]
        args: list[object] = [f"%{_escape_like(query)}%"]
        if parent:
            where.append("(parent_lower = ? OR parent_lower LIKE ? ESCAPE '\\')")
            args += [parent, f"%.{_escape_like(parent)}"]
        if kind:
            kinds = KIND_GROUPS.get(kind.strip().lower(), (kind.strip().lower(),))
            where.append(f"kind IN ({','.join('?' * len(kinds))})")
            args += list(kinds)
        sql = (
            "SELECT path, name, kind, line, end_line, parent, depth, signature, "
            "CASE WHEN name_lower = ? THEN 0 WHEN name_lower LIKE ? ESCAPE '\\' THEN 1 "
            "ELSE 2 END AS rank "
            f"FROM symbols WHERE {' AND '.join(where)} "
            "ORDER BY rank, length(name), path, line"
        )
        params = [query, _escape_like(query) + "%", *args]
        hits: list[SymbolHit] = []
        total = 0
        for row in self._db.execute(sql, params):
            if path_glob and not glob_match(row[0], path_glob):
                continue
            total += 1
            if len(hits) < limit:
                hits.append(SymbolHit(row[0], _symbol(row[1:8])))
            if total >= _COUNT_CAP:
                break
        return hits, total

    def file_entry(self, rel_path: str) -> FileEntry | None:
        row = self._db.execute(
            "SELECT path, language, lines, status FROM files WHERE path = ?", (rel_path,)
        ).fetchone()
        if row is None:
            return None
        count = self._db.execute(
            "SELECT COUNT(*) FROM symbols WHERE path = ?", (rel_path,)
        ).fetchone()[0]
        return FileEntry(*row, symbols=count)

    def outline(self, rel_path: str) -> list[Symbol]:
        """Определения файла в порядке строк."""
        rows = self._db.execute(
            "SELECT name, kind, line, end_line, parent, depth, signature FROM symbols "
            "WHERE path = ? ORDER BY line, depth",
            (rel_path,),
        )
        return [_symbol(r) for r in rows]

    def files_under(self, rel_dir: str) -> Iterator[FileEntry]:
        """Файлы каталога (рекурсивно) с числом символов, по пути."""
        cond, args = _under(rel_dir)
        cond = cond.replace("path", "f.path")
        rows = self._db.execute(
            "SELECT f.path, f.language, f.lines, f.status, "
            "(SELECT COUNT(*) FROM symbols s WHERE s.path = f.path) "
            f"FROM files f WHERE {cond} ORDER BY f.path",
            args,
        )
        for path, language, lines, status, count in rows:
            yield FileEntry(path, language, lines, status, symbols=count)


def _symbol(row) -> Symbol:
    name, kind, line, end_line, parent, depth, signature = row
    return Symbol(
        name=name,
        kind=kind,
        line=line,
        end_line=end_line,
        parent=parent,
        depth=depth,
        signature=signature,
    )


def _chunks(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
