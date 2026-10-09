"""Тесты индекса проекта (SQLite в .devassist/index)."""

from __future__ import annotations

import os
import sqlite3

import pytest

from devassist.project import index as index_mod
from devassist.project.index import INDEX_FILE_NAME, ProjectIndex
from devassist.project.workspace import Workspace


def _make(root, rel, text="x = 1\n"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _bump(path, text):
    """Перезаписать файл так, чтобы изменились размер и mtime."""
    st = path.stat()
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


@pytest.fixture
def project(tmp_path):
    _make(tmp_path, "app/agent.py", "class Agent:\n    def run_turn(self):\n        pass\n")
    _make(
        tmp_path, "app/events.py", "class AgentEvents:\n    def on_turn_end(self):\n        pass\n"
    )
    _make(tmp_path, "web/store.ts", "export class Store {}\nexport function runTurn() {}\n")
    _make(tmp_path, "README.md", "# Проект\n## Установка\n")
    return tmp_path


@pytest.fixture
def index(project):
    with ProjectIndex(Workspace(project)) as ix:
        yield ix


def _paths(ix):
    return [e.path for e in ix.files_under("")]


# ------------------------------ построение ------------------------------ #
def test_constructor_has_no_side_effects(project):
    ix = ProjectIndex(Workspace(project))
    assert not (project / ".devassist").exists()
    assert ix.is_complete() is False
    assert not (project / ".devassist").exists()


def test_build_creates_data_dir_and_marks_complete(index, project):
    stats = index.refresh()
    assert (stats.scanned, stats.added, stats.updated, stats.removed) == (4, 4, 0, 0)
    assert (project / ".devassist" / ".gitignore").is_file()
    assert (project / ".devassist" / "index" / INDEX_FILE_NAME).is_file()
    assert index.is_complete()
    info = index.stats()
    assert info.files == 4 and info.symbols == 8 and info.complete
    assert dict(info.languages) == {"python": 2, "typescript": 1, "markdown": 1}
    assert info.updated_at is not None and info.db_bytes > 0


def test_incremental_refresh_reparses_only_changes(index, project, monkeypatch):
    index.refresh()
    parsed = []
    real = index_mod.extract_symbols
    monkeypatch.setattr(
        index_mod, "extract_symbols", lambda text, lang: parsed.append(lang) or real(text, lang)
    )
    assert index.refresh().changed == 0 and parsed == []

    _bump(project / "app/agent.py", "class Agent:\n    def renamed(self):\n        pass\n")
    _make(project, "app/new.py", "def fresh():\n    pass\n")
    (project / "web/store.ts").unlink()
    stats = index.refresh()
    assert (stats.added, stats.updated, stats.removed) == (1, 1, 1)
    assert parsed == ["python", "python"]
    assert "web/store.ts" not in _paths(index)
    assert [s.qualname for s in index.outline("app/agent.py")] == ["Agent", "Agent.renamed"]
    assert index.find_symbols("runTurn")[1] == 0


def test_refresh_subtree_leaves_rest_untouched(index, project):
    index.refresh()
    (project / "web/store.ts").unlink()
    _make(project, "app/more.py", "def more():\n    pass\n")
    stats = index.refresh(project / "app")
    assert (stats.added, stats.removed) == (1, 0)
    assert "web/store.ts" in _paths(index)  # вне поддерева — не трогаем
    assert index.is_complete()  # частичное обновление не снимает признак полноты


def test_refresh_single_file(index, project):
    index.refresh(project / "app" / "agent.py")
    assert _paths(index) == ["app/agent.py"]
    assert not index.is_complete()


def test_secrets_binary_large_and_ignored_files(index, project, monkeypatch):
    monkeypatch.setattr(index_mod, "MAX_INDEX_FILE_BYTES", 100)
    _make(project, ".env", "GIGACHAT_ACCESS_KEY=secret\n")
    _make(project, ".gitignore", "generated/\n")
    _make(project, "generated/big.py", "def hidden(): pass\n")
    (project / "logo.png").write_bytes(b"\x89PNG\0\0\0")
    _make(project, "huge.py", "def huge(): pass\n" * 20)
    index.refresh()
    paths = _paths(index)
    assert ".env" not in paths and "generated/big.py" not in paths
    assert index.file_entry("logo.png").status == "binary"
    big = index.file_entry("huge.py")
    assert big.status == "large" and big.symbols == 0
    # явное обновление исключённого каталога не протаскивает его в индекс
    index.refresh(project / "generated")
    assert "generated/big.py" not in _paths(index)


def test_interrupted_build_keeps_committed_batches(project, monkeypatch):
    monkeypatch.setattr(index_mod, "_BATCH", 2)
    for i in range(5):
        _make(project, f"many/m{i}.py", f"def f{i}(): pass\n")

    def stop_at(n):
        if n == 6:
            raise KeyboardInterrupt

    ws = Workspace(project)
    with ProjectIndex(ws) as ix, pytest.raises(KeyboardInterrupt):
        ix.refresh(progress=stop_at)
    with ProjectIndex(ws) as ix:
        assert not ix.is_complete()
        assert len(_paths(ix)) == 4  # две зафиксированные пачки по 2 файла
        stats = ix.refresh()
        assert stats.added == 5 and ix.is_complete()


def test_rebuild(index, project):
    index.refresh()
    stats = index.rebuild()
    assert stats.added == 4 and index.is_complete()


# ------------------------------ совместимость ------------------------------ #
def test_old_schema_is_recreated(project):
    ws = Workspace(project)
    with ProjectIndex(ws) as ix:
        ix.refresh()
    conn = sqlite3.connect(ws.index_dir / INDEX_FILE_NAME)
    with conn:
        conn.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
    conn.close()
    with ProjectIndex(ws) as ix:
        assert not ix.is_complete() and ix.stats().files == 0


def test_corrupted_database_is_recreated(project):
    ws = Workspace(project)
    ws.ensure_data_dir()
    ws.index_dir.mkdir()
    (ws.index_dir / INDEX_FILE_NAME).write_bytes(b"not a database at all" * 100)
    with ProjectIndex(ws) as ix:
        assert ix.refresh().added == 4


# ------------------------------ поиск ------------------------------ #
def test_find_symbols_ranking(index):
    index.refresh()
    hits, total = index.find_symbols("agent")
    assert total == 2
    assert [h.symbol.name for h in hits] == ["Agent", "AgentEvents"]  # точное — первым


def test_find_symbols_qualified_kind_glob_and_limit(index):
    index.refresh()
    hits, _ = index.find_symbols("Agent.run_turn")
    assert [(h.path, h.symbol.qualname) for h in hits] == [("app/agent.py", "Agent.run_turn")]
    # регистр не важен, 'function' включает методы
    hits, total = index.find_symbols("RUNTURN", kind="function")
    assert total == 1 and hits[0].path == "web/store.ts"
    hits, total = index.find_symbols("run", kind="function", path_glob="app/**")
    assert [h.symbol.qualname for h in hits] == ["Agent.run_turn"]
    hits, total = index.find_symbols("e", limit=1)
    assert len(hits) == 1 and total > 1
    assert index.find_symbols("100%_")[1] == 0  # спецсимволы LIKE экранируются
    with pytest.raises(ValueError):
        index.find_symbols("  ")


@pytest.mark.skipif(os.name == "nt", reason="регистр путей")
def test_subtree_is_case_sensitive(index, project):
    _make(project, "Src/upper.py", "def upper(): pass\n")
    _make(project, "src/lower.py", "def lower(): pass\n")
    index.refresh()
    index.refresh(project / "src")
    assert "Src/upper.py" in _paths(index)  # соседний каталог в другом регистре не тронут
    assert [e.path for e in index.files_under("src")] == ["src/lower.py"]


def test_unreadable_file_is_retried_when_unchanged(index, project, monkeypatch):
    real = index_mod.Path.read_bytes
    broken = {"app/agent.py"}

    def read_bytes(self):
        if self.relative_to(project.resolve()).as_posix() in broken:
            raise PermissionError("нет доступа")
        return real(self)

    monkeypatch.setattr(index_mod.Path, "read_bytes", read_bytes)
    index.refresh()
    assert index.file_entry("app/agent.py").status == "error"
    broken.clear()  # права починили, файл не менялся
    assert index.refresh().updated == 1
    assert index.file_entry("app/agent.py").status == "indexed"
    assert index.refresh().changed == 0
