"""Тесты инструментов: поиск по содержимому, shell, git."""

from __future__ import annotations

import subprocess

import pytest

from devassist.devassist.tools.base import ToolError
from devassist.devassist.tools.fs import WriteFileTool
from devassist.devassist.tools.git import GitTool
from devassist.devassist.tools.search import SearchContentTool
from devassist.devassist.tools.shell import RunShellTool


def _write(ctx, path, content):
    w = WriteFileTool()
    w.run(w.parse({"path": path, "content": content}), ctx)


# ------------------------------- search ------------------------------- #
def test_search_content_finds_matches(ctx):
    _write(ctx, "a.py", "def foo():\n    return 42\n")
    _write(ctx, "b.py", "x = foo()\n")
    s = SearchContentTool()
    out = s.run(s.parse({"pattern": r"foo"}), ctx)
    assert "a.py:1:" in out.content
    assert "b.py:1:" in out.content


def test_search_glob_filter(ctx):
    _write(ctx, "a.py", "needle")
    _write(ctx, "a.txt", "needle")
    s = SearchContentTool()
    out = s.run(s.parse({"pattern": "needle", "glob": "*.py"}), ctx)
    assert "a.py" in out.content and "a.txt" not in out.content


def test_search_invalid_regex(ctx):
    s = SearchContentTool()
    with pytest.raises(ToolError):
        s.run(s.parse({"pattern": "([unclosed"}), ctx)


# ------------------------------- shell -------------------------------- #
def test_shell_echo(ctx):
    sh = RunShellTool()
    res = sh.run(sh.parse({"command": "echo hello"}), ctx)
    assert res.ok
    assert "hello" in res.content
    assert "exit code: 0" in res.content


def test_shell_nonzero_exit(ctx):
    sh = RunShellTool()
    res = sh.run(sh.parse({"command": "exit 3"}), ctx)
    assert not res.ok
    assert "exit code: 3" in res.content


def test_shell_runs_in_project_root(ctx):
    sh = RunShellTool()
    res = sh.run(sh.parse({"command": "pwd"}), ctx)
    assert str(ctx.root) in res.content


def test_shell_dangerous_classification(ctx):
    from devassist.devassist.security import RiskLevel

    sh = RunShellTool()
    safe = sh.parse({"command": "ls -la"})
    danger = sh.parse({"command": "rm -rf /tmp/x"})
    assert sh.risk(safe, ctx) == RiskLevel.WRITE
    assert sh.risk(danger, ctx) == RiskLevel.DANGEROUS


# -------------------------------- git --------------------------------- #
@pytest.fixture
def git_repo(ctx):
    root = ctx.root
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    return ctx


def test_git_status(git_repo):
    ctx = git_repo
    _write(ctx, "f.txt", "hi")
    g = GitTool()
    res = g.run(g.parse({"subcommand": "status"}), ctx)
    assert res.ok
    assert "f.txt" in res.content


def test_git_add_and_commit(git_repo):
    ctx = git_repo
    _write(ctx, "f.txt", "hi")
    g = GitTool()
    g.run(g.parse({"subcommand": "add", "args": ["f.txt"]}), ctx)
    res = g.run(g.parse({"subcommand": "commit", "args": ["-m", "init"]}), ctx)
    assert res.ok
    log = g.run(g.parse({"subcommand": "log"}), ctx)
    assert "init" in log.content


def test_git_disallowed_subcommand(git_repo):
    g = GitTool()
    with pytest.raises(ToolError):
        g.run(g.parse({"subcommand": "push"}), git_repo)
