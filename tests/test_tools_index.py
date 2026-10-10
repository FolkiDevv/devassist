"""Тесты инструментов find_symbol и file_outline."""

from __future__ import annotations

import pytest

from devassist.project.workspace import Workspace
from devassist.tools import index as tools_index
from devassist.tools.base import ToolContext, ToolError, build_default_registry
from devassist.tools.index import (
    FileOutlineParams,
    FileOutlineTool,
    FindReferencesParams,
    FindReferencesTool,
    FindSymbolParams,
    FindSymbolTool,
)


def _make(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def ctx(tmp_path):
    _make(
        tmp_path,
        "app/agent.py",
        "class Agent:\n    def run_turn(self, text):\n        return text\n\n\ndef helper():\n    pass\n",
    )
    _make(tmp_path, "docs/guide.md", "# Гайд\n## Установка\n")
    return ToolContext(workspace=Workspace(tmp_path))


def _find(ctx, **kw):
    return FindSymbolTool().run(FindSymbolParams(**kw), ctx)


def _outline(ctx, path):
    return FileOutlineTool().run(FileOutlineParams(path=path), ctx)


def test_registered_in_default_registry():
    reg = build_default_registry()
    assert "find_symbol" in reg and "file_outline" in reg and "find_references" in reg


def test_find_symbol_lists_definitions(ctx):
    result = _find(ctx, query="run_turn")
    assert result.ok
    assert result.content == ("app/agent.py:2-3 method Agent.run_turn — def run_turn(self, text):")
    assert result.summary.startswith("найдено определений: 1; индекс обновлён (файлов: 2)")
    # второй вызов: индекс уже актуален
    assert _find(ctx, query="run_turn").summary == "найдено определений: 1"


def test_find_symbol_sees_fresh_edits(ctx):
    _find(ctx, query="Agent")
    _make(ctx.root, "app/new.py", "def brand_new():\n    pass\n")
    assert "app/new.py:1-2 function brand_new" in _find(ctx, query="brand").content


def test_find_symbol_filters_and_limit(ctx):
    assert "helper" in _find(ctx, query="e", kind="function", glob="app/*.py").content
    assert "Гайд" not in _find(ctx, query="е", kind="function").content
    limited = _find(ctx, query="e", max_results=1)
    assert "… показано 1 из" in limited.content


def test_find_symbol_nothing_found_and_empty_query(ctx):
    assert "search_content" in _find(ctx, query="nope_nothing").content
    with pytest.raises(ToolError):
        _find(ctx, query="   ")


def test_find_symbol_describe():
    tool = FindSymbolTool()
    assert tool.describe(FindSymbolParams(query="Agent")) == "Agent"
    assert tool.describe(FindSymbolParams(query="Agent", kind="class")) == "Agent (class)"


def test_file_outline_of_file(ctx):
    result = _outline(ctx, "app/agent.py")
    assert result.content.splitlines() == [
        "app/agent.py (python, строк: 7, определений: 3)",
        "1-3  class Agent:",
        "  2-3  def run_turn(self, text):",
        "6-7  def helper():",
    ]
    md = _outline(ctx, "docs/guide.md").content.splitlines()
    assert md[1:] == ["1  # Гайд", "  2  ## Установка"]


def test_file_outline_of_directory(ctx):
    result = _outline(ctx, ".")
    assert result.content.splitlines() == [
        "app/agent.py (python, строк: 7, определений: 3)",
        "docs/guide.md (markdown, строк: 2, определений: 2)",
    ]
    assert result.summary.startswith(".: файлов 2, определений 5")
    assert _outline(ctx, "app").content.startswith("agent.py (python")


def test_file_outline_groups_large_directories(ctx, monkeypatch):
    monkeypatch.setattr(tools_index, "MAX_OUTLINE_FILES", 1)
    _make(ctx.root, "top.py", "def top():\n    pass\n")
    lines = _outline(ctx, ".").content.splitlines()
    assert lines[:3] == [
        "app/ (файлов: 1, определений: 3)",
        "docs/ (файлов: 1, определений: 2)",
        "top.py (python, строк: 2, определений: 1)",
    ]
    assert "всего файлов: 3" in lines[-1]


def test_file_outline_errors(ctx):
    _make(ctx.root, ".env", "KEY=1\n")
    _make(ctx.root, ".gitignore", "build.py\n")
    _make(ctx.root, "build.py", "def x():\n    pass\n")
    (ctx.root / "blob.bin").write_bytes(b"\0\1\2")
    for path, needle in [
        ("missing.py", "не найден"),
        (".env", "секрет"),
        ("build.py", "исключён"),
        ("blob.bin", "Бинарный"),
        ("../outside.py", "за пределы"),
    ]:
        with pytest.raises(ToolError, match=needle):
            _outline(ctx, path)


def test_large_file_outline_suggests_ranges(ctx, monkeypatch):
    from devassist.project import index as index_mod

    monkeypatch.setattr(index_mod, "MAX_INDEX_FILE_BYTES", 10)
    with pytest.raises(ToolError, match="диапазоном строк"):
        _outline(ctx, "app/agent.py")


def test_broken_index_becomes_tool_error(ctx, monkeypatch):
    def boom(self):
        raise OSError("диск недоступен")

    monkeypatch.setattr(tools_index.ProjectIndex, "open", boom)
    with pytest.raises(ToolError, match="Индекс проекта недоступен"):
        _find(ctx, query="Agent")


def test_file_outline_reports_parse_errors(ctx, monkeypatch):
    from devassist.project import index as index_mod

    def boom(text, language, path):
        raise RuntimeError("сбой разбора")

    monkeypatch.setattr(index_mod, "extract", boom)
    with pytest.raises(ToolError, match="Не удалось разобрать"):
        _outline(ctx, "app/agent.py")


# ------------------------------ find_references ------------------------------ #
def _refs(ctx, **kw):
    return FindReferencesTool().run(FindReferencesParams(**kw), ctx)


def _cli(ctx):
    _make(
        ctx.root,
        "app/cli.py",
        "from app.agent import Agent, helper\n\n\n"
        "def main(text):\n"
        "    Agent().run_turn(text)\n"
        "    helper()\n",
    )
    _make(ctx.root, "scripts/x.py", "def other(bot):\n    bot.run_turn()\n")


def test_find_references_groups_by_file_and_resolution(ctx):
    _cli(ctx)
    result = _refs(ctx, query="Agent.run_turn")
    assert result.content.splitlines() == [
        "определение: app/agent.py:2-3 method Agent.run_turn — def run_turn(self, text):",
        "использования (2):",
        "app/cli.py — импортирует модуль с определением:",
        "  5 [вызов] в main — Agent().run_turn(text)",
        "scripts/x.py — совпадение только по имени — может быть другое определение:",
        "  2 [вызов] в other — bot.run_turn()",
    ]
    assert result.summary.startswith("найдено использований: 2; индекс обновлён")


def test_find_references_kind_limit_and_errors(ctx):
    _cli(ctx)
    imports = _refs(ctx, query="helper", kind="IMPORT").content.splitlines()
    assert imports[1:] == [
        "использования (1):",
        "app/cli.py — импортирует модуль с определением:",
        "  1 [импорт] — from app.agent import Agent, helper",
    ]
    limited = _refs(ctx, query="run_turn", max_results=1).content
    assert "… показано 1 из 2; уточните запрос, kind или glob" in limited
    missing = _refs(ctx, query="nowhere").content.splitlines()
    assert missing == [
        "(определение «nowhere» в индексе не найдено — ищу использования по имени)",
        "(использований не найдено)",
    ]
    with pytest.raises(ToolError, match="допустимые: call, attr, name, import"):
        _refs(ctx, query="helper", kind="usage")
    with pytest.raises(ToolError, match="Пустой запрос"):
        _refs(ctx, query=" . ")


def test_file_outline_shows_project_imports(ctx):
    _cli(ctx)
    cli = _outline(ctx, "app/cli.py").content.splitlines()
    assert cli[:2] == [
        "app/cli.py (python, строк: 6, определений: 1)",
        "импортирует из проекта: app/agent.py",
    ]
    agent = _outline(ctx, "app/agent.py").content.splitlines()
    assert agent[1] == "импортируется в (1): app/cli.py"


def test_find_symbol_and_outline_show_docstring(ctx):
    _make(ctx.root, "app/doc.py", 'def documented():\n    """Делает дело.\n\n    Подробно."""\n')
    assert _find(ctx, query="documented").content.endswith("  # Делает дело.")
    assert _outline(ctx, "app/doc.py").content.splitlines()[1] == (
        "1-4  def documented():  # Делает дело."
    )
