"""Тесты файловых инструментов и песочницы."""

from __future__ import annotations

import pytest

from devassist.security import SandboxError
from devassist.tools.base import ToolError
from devassist.tools.fs import (
    EditFileTool,
    FindFilesTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
    repair_escaped_content,
)


def test_write_then_read(ctx):
    w = WriteFileTool()
    res = w.run(w.parse({"path": "a/b.txt", "content": "привет\nмир"}), ctx)
    assert res.ok
    assert (ctx.root / "a" / "b.txt").read_text(encoding="utf-8") == "привет\nмир"

    r = ReadFileTool()
    out = r.run(r.parse({"path": "a/b.txt"}), ctx)
    assert "привет" in out.content
    assert "1\t" in out.content  # номера строк


def test_write_overwrite_and_diff(ctx):
    w = WriteFileTool()
    w.run(w.parse({"path": "f.txt", "content": "old\n"}), ctx)
    params = w.parse({"path": "f.txt", "content": "new\n"})
    preview = w.preview(params, ctx)
    assert preview.kind == "diff"
    assert "-old" in preview.text and "+new" in preview.text
    res = w.run(params, ctx)
    assert "перезаписан" in res.summary


def test_read_missing(ctx):
    r = ReadFileTool()
    with pytest.raises(ToolError):
        r.run(r.parse({"path": "nope.txt"}), ctx)


def test_read_line_range(ctx):
    w = WriteFileTool()
    w.run(w.parse({"path": "n.txt", "content": "l1\nl2\nl3\nl4"}), ctx)
    r = ReadFileTool()
    out = r.run(r.parse({"path": "n.txt", "start_line": 2, "end_line": 3}), ctx)
    assert "l2" in out.content and "l3" in out.content
    assert "l1" not in out.content and "l4" not in out.content


def test_edit_single_occurrence(ctx):
    w = WriteFileTool()
    w.run(w.parse({"path": "c.py", "content": "x = 1\ny = 2\n"}), ctx)
    e = EditFileTool()
    res = e.run(e.parse({"path": "c.py", "old_string": "y = 2", "new_string": "y = 3"}), ctx)
    assert res.ok
    assert (ctx.root / "c.py").read_text() == "x = 1\ny = 3\n"


def test_edit_ambiguous_without_replace_all(ctx):
    w = WriteFileTool()
    w.run(w.parse({"path": "d.txt", "content": "a\na\n"}), ctx)
    e = EditFileTool()
    with pytest.raises(ToolError, match="встречается"):
        e.run(e.parse({"path": "d.txt", "old_string": "a", "new_string": "b"}), ctx)


def test_edit_replace_all(ctx):
    w = WriteFileTool()
    w.run(w.parse({"path": "d.txt", "content": "a\na\n"}), ctx)
    e = EditFileTool()
    res = e.run(
        e.parse({"path": "d.txt", "old_string": "a", "new_string": "b", "replace_all": True}),
        ctx,
    )
    assert res.ok
    assert (ctx.root / "d.txt").read_text() == "b\nb\n"


def test_edit_tolerant_trailing_whitespace(ctx):
    # в файле есть хвостовые пробелы, модель прислала фрагмент без них
    w = WriteFileTool()
    w.run(w.parse({"path": "t.py", "content": "def f():   \n    return 1   \n"}), ctx)
    e = EditFileTool()
    res = e.run(
        e.parse(
            {
                "path": "t.py",
                "old_string": "def f():\n    return 1",
                "new_string": "def f():\n    return 2",
            }
        ),
        ctx,
    )
    assert res.ok
    assert "return 2" in (ctx.root / "t.py").read_text()


def test_edit_tolerant_indentation(ctx):
    # модель прислала фрагмент с другим отступом
    w = WriteFileTool()
    w.run(w.parse({"path": "i.py", "content": "class A:\n        x = 1\n        y = 2\n"}), ctx)
    e = EditFileTool()
    res = e.run(
        e.parse({"path": "i.py", "old_string": "x = 1\ny = 2", "new_string": "x = 10\ny = 20"}),
        ctx,
    )
    assert res.ok
    text = (ctx.root / "i.py").read_text()
    assert "x = 10" in text and "y = 20" in text


def test_edit_with_double_escaped_old_string(ctx):
    # модель прислала old_string с литеральными \n (двойная экранизация)
    w = WriteFileTool()
    w.run(
        w.parse({"path": "m.py", "content": "def a():\n    return 1\n\ndef b():\n    return 2\n"}),
        ctx,
    )
    e = EditFileTool()
    res = e.run(
        e.parse(
            {
                "path": "m.py",
                "old_string": "def b():\\n    return 2",
                "new_string": "def b():\\n    return 22",
            }
        ),
        ctx,
    )
    assert res.ok
    assert "return 22" in (ctx.root / "m.py").read_text()


def test_edit_old_string_with_line_number_prefixes(ctx):
    # модель скопировала old_string ВМЕСТЕ с префиксами номеров строк read_file
    w = WriteFileTool()
    w.run(
        w.parse(
            {
                "path": "k.py",
                "content": "def add(a, b):\n    return a + b\n\ndef divide(a, b):\n    return a / b\n",
            }
        ),
        ctx,
    )
    e = EditFileTool()
    res = e.run(
        e.parse(
            {
                "path": "k.py",
                "old_string": "4\tdef divide(a, b):\n5\t    return a / b",
                "new_string": "4\tdef divide(a, b):\n5\t    return a / b\n6\t\n7\tdef power(b, e):\n8\t    return b ** e",
            }
        ),
        ctx,
    )
    assert res.ok
    text = (ctx.root / "k.py").read_text()
    assert "def power(b, e):" in text
    assert "\t" not in text  # префиксы номеров не попали в файл


def test_edit_not_found_shows_context(ctx):
    w = WriteFileTool()
    w.run(w.parse({"path": "c.py", "content": "alpha\nbeta\ngamma\n"}), ctx)
    e = EditFileTool()
    with pytest.raises(ToolError) as ei:
        e.run(e.parse({"path": "c.py", "old_string": "zzz", "new_string": "q"}), ctx)
    # сообщение об ошибке содержит пронумерованный контекст файла
    msg = str(ei.value)
    assert "alpha" in msg and "1\t" in msg


def test_edit_not_found(ctx):
    w = WriteFileTool()
    w.run(w.parse({"path": "e.txt", "content": "hello"}), ctx)
    e = EditFileTool()
    with pytest.raises(ToolError, match="не найден"):
        e.run(e.parse({"path": "e.txt", "old_string": "zzz", "new_string": "q"}), ctx)


def test_list_dir(ctx):
    w = WriteFileTool()
    w.run(w.parse({"path": "src/main.py", "content": "pass"}), ctx)
    w.run(w.parse({"path": "readme.md", "content": "# hi"}), ctx)
    lister = ListDirTool()
    out = lister.run(lister.parse({"path": "."}), ctx)
    assert "src/" in out.content
    assert "readme.md" in out.content


def test_find_files(ctx):
    w = WriteFileTool()
    for p in ["a.py", "b.py", "c.txt", "sub/d.py"]:
        w.run(w.parse({"path": p, "content": "x"}), ctx)
    f = FindFilesTool()
    out = f.run(f.parse({"pattern": "*.py"}), ctx)
    assert "a.py" in out.content and "sub/d.py" in out.content
    assert "c.txt" not in out.content


def test_repair_double_escaped_content():
    # одна физическая строка с литеральными \n и \" — типичный брак слабой модели
    broken = 'def f():\\n    return "x"\\n'
    fixed, repaired = repair_escaped_content(broken)
    assert repaired is True
    assert fixed == 'def f():\n    return "x"\n'


def test_repair_strips_line_numbers():
    broken = "1\\timport os\\n2\\tx = 1\\n3\\tprint(x)"
    fixed, repaired = repair_escaped_content(broken)
    assert repaired is True
    assert fixed == "import os\nx = 1\nprint(x)"


def test_repair_leaves_normal_content_untouched():
    normal = "def f():\n    return 1\n"
    fixed, repaired = repair_escaped_content(normal)
    assert repaired is False
    assert fixed == normal
    # строка с настоящими переносами и одним литералом \n внутри тоже не трогается
    code = 'print("a\\nb")\nx = 1\n'
    fixed2, repaired2 = repair_escaped_content(code)
    assert repaired2 is False
    assert fixed2 == code


def test_write_file_autorepairs_escaped(ctx):
    w = WriteFileTool()
    broken = 'print("hi")\\nx = 1\\n'
    res = w.run(w.parse({"path": "g.py", "content": broken}), ctx)
    assert "автокоррекция" in res.summary
    assert (ctx.root / "g.py").read_text() == 'print("hi")\nx = 1\n'


def test_sandbox_escape_blocked(ctx):
    w = WriteFileTool()
    with pytest.raises(SandboxError):
        w.run(w.parse({"path": "../escape.txt", "content": "x"}), ctx)
    r = ReadFileTool()
    with pytest.raises(SandboxError):
        r.run(r.parse({"path": "/etc/passwd"}), ctx)


def test_find_files_rejects_escape_patterns(ctx):
    f = FindFilesTool()
    for pattern in ("../*.toml", "src/../../x", "/etc/*"):
        with pytest.raises(ToolError):
            f.run(f.parse({"pattern": pattern}), ctx)


def test_find_files_glob_semantics_and_ignored_dirs(ctx):
    w = WriteFileTool()
    for path in ("src/a.py", "src/x/b.py", "node_modules/c.py", "z.py"):
        w.run(w.parse({"path": path, "content": "x"}), ctx)
    f = FindFilesTool()
    out = f.run(f.parse({"pattern": "src/**/*.py"}), ctx).content
    assert out.splitlines() == ["src/a.py", "src/x/b.py"]
    out = f.run(f.parse({"pattern": "*.py"}), ctx).content
    assert "node_modules" not in out
    assert out.splitlines() == ["src/a.py", "src/x/b.py", "z.py"]  # отсортировано


def test_find_files_truncation_is_reported(ctx):
    w = WriteFileTool()
    for i in range(5):
        w.run(w.parse({"path": f"f{i}.txt", "content": "x"}), ctx)
    f = FindFilesTool()
    out = f.run(f.parse({"pattern": "*.txt", "max_results": 2}), ctx).content
    assert out.splitlines()[:2] == ["f0.txt", "f1.txt"]
    assert "показано 2 из 5" in out


def test_list_dir_shows_hidden(ctx):
    (ctx.root / ".github").mkdir()
    (ctx.root / ".gitignore").write_text("x", encoding="utf-8")
    lister = ListDirTool()
    out = lister.run(lister.parse({}), ctx).content
    assert ".github/" in out and ".gitignore" in out


@pytest.mark.parametrize("path", [".devassist/index/index.sqlite3", ".devassist", "./.devassist/x"])
def test_data_dir_is_not_writable(ctx, path):
    """Индекс, история и чаты агента — не для правки моделью (отказ ещё в превью)."""
    (ctx.root / ".devassist" / "index").mkdir(parents=True)
    (ctx.root / ".devassist" / "x").write_text("old", encoding="utf-8")
    w, e = WriteFileTool(), EditFileTool()
    write = w.parse({"path": path, "content": "new"})
    edit = e.parse({"path": path, "old_string": "old", "new_string": "new"})
    for tool, params in ((w, write), (e, edit)):
        with pytest.raises(ToolError, match="Служебная папка"):
            tool.preview(params, ctx)
        with pytest.raises(ToolError, match="Служебная папка"):
            tool.run(params, ctx)
    assert (ctx.root / ".devassist" / "x").read_text(encoding="utf-8") == "old"


def test_similar_names_outside_data_dir_are_writable(ctx):
    w = WriteFileTool()
    assert w.run(w.parse({"path": ".devassist.md", "content": "ok"}), ctx).ok


def test_data_dir_symlink_target_is_not_writable(ctx):
    """.devassist — симлинк на каталог проекта: запись по любому из путей запрещена."""
    import os

    if os.name == "nt":
        pytest.skip("симлинки")
    (ctx.root / "agent-data").mkdir()
    (ctx.root / ".devassist").symlink_to(ctx.root / "agent-data", target_is_directory=True)
    w = WriteFileTool()
    for path in (".devassist/index.sqlite3", "agent-data/index.sqlite3"):
        with pytest.raises(ToolError, match="Служебная папка"):
            w.preview(w.parse({"path": path, "content": "x"}), ctx)
